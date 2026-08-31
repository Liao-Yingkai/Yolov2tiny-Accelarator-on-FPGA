#include "im2row.h"

#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace {

// 把一个 pixel_t 的全部物理通道写成量化零点。
void clear_pixel(pixel_t& pixel) {
#pragma HLS INLINE
    for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
#pragma HLS UNROLL
        pixel.channel[channel] = 0;
    }
}

// 把 source 的有效输入通道复制到 destination，并清零其余物理通道。
// Cin 是本帧有效通道数；
void copy_valid_channels(
    pixel_t& destination,
    const pixel_t& source,
    int Cin
) {
#pragma HLS INLINE
    for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
#pragma HLS UNROLL
        destination.channel[channel] =
            (channel < Cin) ? source.channel[channel] : activation_t(0);
    }
}

// 读取指定物理行 Bank 中的一个列像素。
// column%P 选择列 Bank，column/P 选择该 Bank 内地址。
pixel_t read_line_pixel(
    const pixel_t line_buffer[2][IM2ROW_P][IM2ROW_MAX_BLOCKS_PER_ROW],
    int row_bank,
    int column
) {
#pragma HLS INLINE
    return line_buffer[row_bank][column % IM2ROW_P]
                      [column / IM2ROW_P];
}

// 把一个像素写入指定物理行 Bank 和列位置。
// 写地址与读取采用相同的 P 路列 Bank 映射。
void write_line_pixel(
    pixel_t line_buffer[2][IM2ROW_P][IM2ROW_MAX_BLOCKS_PER_ROW],
    int row_bank,
    int column,
    const pixel_t& pixel
) {
#pragma HLS INLINE
    line_buffer[row_bank][column % IM2ROW_P]
               [column / IM2ROW_P] = pixel;
}

// 计算当前 Tile 需要从输入流读取的第一个 Bottom 列。
// 行首 Tile 从 x=0 开始；后续 Tile 的前两列已由 Strip 重叠保留，因此从 base_x+1 开始。
int bottom_first_x(int tile_x, int tile_base_x) {
#pragma HLS INLINE
    return (tile_x == 0) ? 0 : (tile_base_x + 1);
}

// 计算第一个 Bottom 输入在 current Strip 中的写入列。
// 行首 Strip 的列0是左 Padding，故从列1写入；同行后续 Strip 的列0、1是重叠列，故从列2写入。
int bottom_first_strip_column(int tile_x) {
#pragma HLS INLINE
    return (tile_x == 0) ? 1 : 2;
}

// 计算当前 Tile 实际需要突发读取的 Bottom 像素数量。
// 行首最多为 P+1，后续最多为 P，并按右边界截断。
int bottom_load_count(int tile_x, int tile_base_x, int W) {
#pragma HLS INLINE
    const int first_x = bottom_first_x(tile_x, tile_base_x);
    const int maximum = (tile_x == 0) ? (IM2ROW_P + 1) : IM2ROW_P;
    const int remaining = W - first_x;
    if (remaining <= 0) {
        return 0;
    }
    return (remaining < maximum) ? remaining : maximum;
}

// 组合辅助函数：为当前 Tile 生成 P 路空间有效掩码。
// 有效 Lane 即使窗口元素落在卷积 Padding 上也保持有效；行尾不存在的 Lane 才置0，且其9拍输出均清零。
ap_uint<IM2ROW_P> make_spatial_valid(int tile_base_x, int W) {
#pragma HLS INLINE
    ap_uint<IM2ROW_P> spatial_valid = 0;
    for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
        spatial_valid[lane] = (tile_base_x + lane < W) ? 1 : 0;
    }
    return spatial_valid;
}

}  // 匿名命名空间

// Im2Row 顶层：把 H×W 个行优先、通道打包的标量像素转换成 Systolic 所需的 P 路窗口流。
// 输入每次成功读取一个 pixel_t；输出每次成功写入一个 activation_pack_t，包含同一 (kh,kw) 下的 P 个空间 Lane。
// 函数先预装首行，再进入双 Strip 的稳态 Token 调度；每个 Tile 按 kh→kw 连续产生9个 Pack。
// 全局循环以 II=1 为综合目标，同行 Tile 和换行 Tile 都在 current 输出期间准备 next；实际 II 仍需综合确认。
void im2row(
    hls::stream<pixel_t>& input_stream,
    hls::stream<activation_pack_t>& output_stream,
    int H,
    int W,
    int Cin
) {
    // 非法参数无法安全解释输入流，因此不消费也不产生任何令牌。
    if ((H <= 0) || (H > IM2ROW_MAX_H) ||
        (W <= 0) || (W > IM2ROW_MAX_W) ||
        (Cin <= 0) || (Cin > IM2ROW_MAX_CHANNELS)) {
        return;
    }

    // Line Buffer 的三个维度依次表示：2个循环行 Bank、P个按列取模的独立列 Bank、
    // 每个列 Bank 内的块地址。处理输出行时，一个行 Bank 保存 Middle，另一个先保存 Top，
    // 待旧 Top 已进入 Strip 后再按列覆盖为下一输入行，避免整行搬移。
    pixel_t line_buffer[2][IM2ROW_P][IM2ROW_MAX_BLOCKS_PER_ROW];
    // 双 Strip 的维度依次为：2个 Ping-Pong 缓冲、Top/Middle/Bottom 三行、P+2个逻辑列。
    // 对 base_x 的 Tile，逻辑列覆盖 base_x-1...base_x+P；current 只读输出，next 只写准备。
    pixel_t strip[2][IM2ROW_KERNEL][IM2ROW_STRIP_WIDTH];
    // 同行预取保存 P 个新列；换行时也保存第一个 Bank Beat 的列0...P-1。
    pixel_t prefetch_top[IM2ROW_P];
    pixel_t prefetch_middle[IM2ROW_P];
    // 换行首 Tile 的列P与列0属于同一列 Bank，第二个 Beat 单独暂存在这里。
    pixel_t boundary_top;
    pixel_t boundary_middle;

#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_t2p impl=bram
#pragma HLS ARRAY_PARTITION variable=strip complete dim=1
#pragma HLS ARRAY_PARTITION variable=strip complete dim=2
#pragma HLS ARRAY_PARTITION variable=strip complete dim=3
#pragma HLS ARRAY_PARTITION variable=prefetch_top complete dim=1
#pragma HLS ARRAY_PARTITION variable=prefetch_middle complete dim=1

    pixel_t zero_pixel;
    clear_pixel(zero_pixel);

#ifndef __SYNTHESIS__
    // 仅用于非综合构建的接口守恒检查：输入必须恰好 H×W，输出必须恰好 H×ceil(W/P)×9。
    int input_read_count = 0;
    int output_token_count = 0;
#endif

    // 启动准备：首输出行的 Top 是上 Padding，Bottom 将在稳态前6拍填入；
    // Middle 的左/右 Padding 在此写零，真实列随后随首行输入直接进入 Strip。
    for (int column = 0; column < IM2ROW_STRIP_WIDTH; ++column) {
#pragma HLS UNROLL
        strip[0][0][column] = zero_pixel;
        strip[0][2][column] = zero_pixel;
        if ((column == 0) || (column > W)) {
            strip[0][1][column] = zero_pixel;
        }
    }

    // 简单启动：连续预装完整首行到行 Bank 0，并同时建立首 Tile 的 Middle。
    // 启动完成后立即进入连续 Token 调度；该启动循环不属于稳态输出流水线。
    for (int input_x = 0; input_x < W; ++input_x) {
#pragma HLS PIPELINE II=1
        const pixel_t input_pixel = input_stream.read();
        write_line_pixel(line_buffer, 0, input_x, input_pixel);
        if (input_x <= IM2ROW_P) {
            strip[0][1][input_x + 1] = input_pixel;
        }
#ifndef __SYNTHESIS__
        ++input_read_count;
#endif
    }

    const int tiles_per_row = (W + IM2ROW_P - 1) / IM2ROW_P;
    const int total_tokens = H * tiles_per_row * IM2ROW_KERNEL * IM2ROW_KERNEL;

    int output_y = 0;     // 当前输出空间行，决定行 Bank 的逻辑角色。
    int tile_x = 0;       // 当前行内 Tile 编号；每个 Tile 包含最多 P 个相邻输出位置。
    int token = 0;        // 当前 Tile 内的0...8拍，按 kh→kw 顺序推进。
    int current_strip = 0;  // 当前9拍只读并送往 Systolic 的 Strip 编号。
    int next_strip = 1;     // 与 current 并行构造、在 token=8 后接管输出的 Strip 编号。

    // 稳态唯一调度器：每次迭代恰好执行一次 Pack 写入，每个 Tile 连续执行9次。
    // 同行和换行所需的 next Strip 都在这9次迭代内提前准备，设计目标是 Tile 边界无输出气泡。
    // PIPELINE II=1 仅声明综合目标，实际 II、RAM 端口和读延迟必须由综合报告确认。
    // write 位于状态更新之前；下游阻塞时，阻塞写使 Token、Strip、行 Bank 和输入前沿均不推进。
    for (int global_token = 0; global_token < total_tokens; ++global_token) {
#pragma HLS PIPELINE II=1
        const int kh = token / IM2ROW_KERNEL;  // 当前卷积核行：token 0...2/3...5/6...8分别对应0/1/2。
        const int kw = token % IM2ROW_KERNEL;  // 当前卷积核列：每三个 Token 依次为0、1、2。
        const int tile_base_x = tile_x * IM2ROW_P;  // 当前 Tile 的第一个输出位置横坐标。
        const bool final_tile_in_row = (tile_x + 1 == tiles_per_row);
        const bool final_output_row = (output_y + 1 == H);
        const int middle_bank = output_y & 1;  // 保存当前输出行 y 对应输入行的物理行 Bank。
        const int top_bank = 1 - middle_bank;  // 先保存上一行 Top，并作为下一输入行 Bottom 的逐列写入 Bank。

        activation_pack_t output_pack;
#pragma HLS ARRAY_PARTITION variable=output_pack.pixel complete dim=1
        // spatial_valid 描述输出位置是否存在，而不是窗口元素是否属于 Padding。
        output_pack.spatial_valid = make_spatial_valid(tile_base_x, W);
        for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
            if (output_pack.spatial_valid[lane]) {
                copy_valid_channels(
                    output_pack.pixel[lane],
                    strip[current_strip][kh][lane + kw],
                    Cin
                );
            } else {
                // 行尾无效 Lane 不属于输出特征图，全部9拍强制为零且 valid=0。
                clear_pixel(output_pack.pixel[lane]);
            }
        }

        // 此循环中唯一的输出事务：每次成功写入向 Systolic 提供 P 个同一 (kh,kw) 的空间 Lane。
        output_stream.write(output_pack);

#ifndef __SYNTHESIS__
        ++output_token_count;
#endif

        // Token 0...5 正在输出 Top/Middle，同时按需填充当前 Tile 尚未使用的 Bottom。
        // load_count 是本 Tile 的真实读取总数；token 同时作为输入偏移，结合起始真实列和 Strip 起始列定位写入。
        // 每拍最多读取一个像素，并在旧 Top 已锁存于 current Strip 后覆盖 top_bank 的同列槽位。
        const int load_count = bottom_load_count(tile_x, tile_base_x, W);
        if ((!final_output_row) && (token < load_count)) {
            const int input_x = bottom_first_x(tile_x, tile_base_x) + token;
            const int strip_column = bottom_first_strip_column(tile_x) + token;
            const pixel_t input_pixel = input_stream.read();
            strip[current_strip][2][strip_column] = input_pixel;
            write_line_pixel(line_buffer, top_bank, input_x, input_pixel);
#ifndef __SYNTHESIS__
            ++input_read_count;
#endif
        }

        if (!final_tile_in_row) {
            // 同行下一 Tile 保留 current 最后两列。Top 在 token0...2 使用完后，于 token3复制。
            if (token == 3) {
                strip[next_strip][0][0] = strip[current_strip][0][IM2ROW_P];
                strip[next_strip][0][1] = strip[current_strip][0][IM2ROW_P + 1];
            }
            // Middle 在 token3...5 使用完后，于 token6复制；此时 current 已开始输出 Bottom。
            if (token == 6) {
                strip[next_strip][1][0] = strip[current_strip][1][IM2ROW_P];
                strip[next_strip][1][1] = strip[current_strip][1][IM2ROW_P + 1];
            }
            if (token == 7) {
                // 同行下一 Tile 还需要 P 个不重叠的新列，它们分布在 P 个不同列 Bank。
                // token7 发出 Top/Middle 的并行读取，按一拍 RAM 延迟在 token8 写入 next。
                for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
                    const int input_x = tile_base_x + IM2ROW_P + 1 + lane;
                    pixel_t top_pixel = zero_pixel;
                    pixel_t middle_pixel = zero_pixel;
                    if (input_x < W) {
                        middle_pixel = read_line_pixel(line_buffer, middle_bank, input_x);
                        if (output_y > 0) {
                            top_pixel = read_line_pixel(line_buffer, top_bank, input_x);
                        }
                    }
                    prefetch_top[lane] = top_pixel;
                    prefetch_middle[lane] = middle_pixel;
                }
            }
            if (token == 8) {
                // token8 已成功输出 current 的最后一个 Bottom Pack，因此其最后两列不再被 current 使用。
                // 此时补齐 next 的 Top/Middle，复制 Bottom 重叠两列，并把待突发填入的新 Bottom 槽位初始化为零。
                for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
                    strip[next_strip][0][2 + lane] = prefetch_top[lane];
                    strip[next_strip][1][2 + lane] = prefetch_middle[lane];
                    strip[next_strip][2][2 + lane] = zero_pixel;
                }
                strip[next_strip][2][0] = strip[current_strip][2][IM2ROW_P];
                strip[next_strip][2][1] = strip[current_strip][2][IM2ROW_P + 1];
            }
        } else if ((!final_output_row) && (token == 6)) {
            // 换行首 Tile 不复用同行重叠列。其真实列0...P中，列0和列P映射到同一个列 Bank，
            // 因而必须分成两个 Bank Beat；token6先并行读取列0...P-1的 Top 和新行 Middle。
            for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
                pixel_t top_pixel = zero_pixel;
                pixel_t middle_pixel = zero_pixel;
                if (lane < W) {
                    top_pixel = read_line_pixel(line_buffer, middle_bank, lane);
                    middle_pixel = read_line_pixel(line_buffer, top_bank, lane);
                }
                prefetch_top[lane] = top_pixel;
                prefetch_middle[lane] = middle_pixel;
            }
        } else if ((!final_output_row) && (token == 7)) {
            // token7接收第一个 Beat，并发出列P的第二个 Beat；Strip 列0直接构造为左 Padding 零。
            strip[next_strip][0][0] = zero_pixel;
            strip[next_strip][1][0] = zero_pixel;
            for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
                strip[next_strip][0][1 + lane] = prefetch_top[lane];
                strip[next_strip][1][1 + lane] = prefetch_middle[lane];
            }
            boundary_top = zero_pixel;
            boundary_middle = zero_pixel;
            if (IM2ROW_P < W) {
                boundary_top = read_line_pixel(line_buffer, middle_bank, IM2ROW_P);
                boundary_middle = read_line_pixel(line_buffer, top_bank, IM2ROW_P);
            }
        } else if ((!final_output_row) && (token == 8)) {
            // token8接收第二个 Beat，下一行首 Tile 的 Top/Middle 至此完整。
            // Bottom 先按 Padding 构造，并在该 Tile 自己的 token0...5 中读取真实列，因此下一拍即可输出 Top。
            strip[next_strip][0][IM2ROW_P + 1] = boundary_top;
            strip[next_strip][1][IM2ROW_P + 1] = boundary_middle;
            for (int column = 0; column < IM2ROW_STRIP_WIDTH; ++column) {
#pragma HLS UNROLL
                strip[next_strip][2][column] = zero_pixel;
            }
        }

        // token8的 Pack 已成功写出，current 的9拍全部结束，next 的首个 Top Pack 也已准备完成，故可安全交换。
        if (token == IM2ROW_KERNEL * IM2ROW_KERNEL - 1) {
            token = 0;
            const int previous_current = current_strip;
            current_strip = next_strip;
            next_strip = previous_current;
            if (final_tile_in_row) {
                tile_x = 0;
                ++output_y;
            } else {
                ++tile_x;
            }
        } else {
            ++token;
        }
    }

#ifndef __SYNTHESIS__
    // 非综合断言只核对接口事务数量，不参与硬件逻辑，也不替代后续功能仿真和综合验证。
    assert(input_read_count == H * W &&
           "图像行转矩阵模块必须恰好消费 H×W 个真实输入像素");
    assert(output_token_count == total_tokens &&
           "图像行转矩阵模块必须恰好输出 H×ceil(W/P)×9 个令牌");
#endif
}
