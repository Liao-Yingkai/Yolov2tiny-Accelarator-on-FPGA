#include "im2row.h"
#include "../channel_pack.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace {

// Line Buffer 的一个存储字完整包含一个空间位置的全部物理通道。
// IM2ROW_MAX_CHANNELS 对应当前通道并行度 C_PAR；默认配置下字宽为 8×8=64 位。
constexpr int IM2ROW_LINE_WORD_BITS =
    ACTIVATION_BITS * IM2ROW_MAX_CHANNELS;
using line_word_t = ap_uint<IM2ROW_LINE_WORD_BITS>;

// 把一个 pixel_t 的全部物理通道写成量化零点。
void clear_pixel(pixel_t& pixel) {
#pragma HLS INLINE
    for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
#pragma HLS UNROLL
        pixel.channel[channel] = 0;
    }
}

// 输入一被消费就规范为固定物理8通道：Conv1 仅保留 channel0，Conv2 保留全部8通道。
// 此后 Strip、Line Buffer 和输出热路径都只处理已规范化的 pixel_t。
pixel_t normalize_input_pixel(const pixel_t& source, int logical_cin) {
#pragma HLS INLINE
    pixel_t destination;
    for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
#pragma HLS UNROLL
        destination.channel[channel] =
            (channel < logical_cin) ? source.channel[channel] :
                                         activation_t(0);
    }
    return destination;
}

// 读取指定物理行 Bank 中的一个完整通道打包字。
// column&3 选择列 Bank，column>>2 选择该 Bank 内地址；Line Buffer 只发生整字读取。
line_word_t read_line_word(
    const line_word_t line_buffer[2][IM2ROW_P][IM2ROW_MAX_BLOCKS_PER_ROW],
    int row_bank,
    int column
) {
#pragma HLS INLINE
    return line_buffer[row_bank][column & 3][column >> 2];
}

// 将一个完整通道打包字写入指定物理行 Bank 和列位置。
// 写地址与读取采用相同的 P 路列 Bank 映射；Line Buffer 禁止通道级部分写。
void write_line_word(
    line_word_t line_buffer[2][IM2ROW_P][IM2ROW_MAX_BLOCKS_PER_ROW],
    int row_bank,
    int column,
    const line_word_t& packed_pixel
) {
#pragma HLS INLINE
    line_buffer[row_bank][column & 3][column >> 2] = packed_pixel;
}

// 当前输出的 Strip 访问只允许 Ping-Pong 的2选1。Strip 行和列偏移由模板
// 固化，lane 循环 UNROLL 后也为常量，避免运行时 kh/kw/列索引形成完整 Strip
// crossbar。
template <int StripRow, int ColumnOffset>
line_word_t select_fixed_strip_word(
    const line_word_t strip[2][IM2ROW_KERNEL][IM2ROW_STRIP_WIDTH],
    ap_uint<1> current_strip,
    int lane
) {
#pragma HLS INLINE
    return (current_strip == 0)
        ? strip[0][StripRow][lane + ColumnOffset]
        : strip[1][StripRow][lane + ColumnOffset];
}

// Token 顺序固定为 kh-major、kw-minor。每个 case 只引用一个确定的窗口
// 位置，因此 token 选择不再参与多维数组下标。
line_word_t select_output_word(
    const line_word_t strip[2][IM2ROW_KERNEL][IM2ROW_STRIP_WIDTH],
    ap_uint<1> current_strip,
    ap_uint<4> token,
    int lane
) {
#pragma HLS INLINE
    line_word_t selected_word = 0;
    switch (token.to_uint()) {
        case 0: selected_word = select_fixed_strip_word<0, 0>(strip, current_strip, lane); break;
        case 1: selected_word = select_fixed_strip_word<0, 1>(strip, current_strip, lane); break;
        case 2: selected_word = select_fixed_strip_word<0, 2>(strip, current_strip, lane); break;
        case 3: selected_word = select_fixed_strip_word<1, 0>(strip, current_strip, lane); break;
        case 4: selected_word = select_fixed_strip_word<1, 1>(strip, current_strip, lane); break;
        case 5: selected_word = select_fixed_strip_word<1, 2>(strip, current_strip, lane); break;
        case 6: selected_word = select_fixed_strip_word<2, 0>(strip, current_strip, lane); break;
        case 7: selected_word = select_fixed_strip_word<2, 1>(strip, current_strip, lane); break;
        case 8: selected_word = select_fixed_strip_word<2, 2>(strip, current_strip, lane); break;
        default: break;  // 状态机只产生 token 0...8。
    }
    return selected_word;
}

// Strip 的 Bottom 行只会写逻辑列1...5。显式列译码避免 complete-partitioned
// Strip 因运行时列下标构建大范围写使能网络。
void write_current_bottom(
    line_word_t strip[2][IM2ROW_KERNEL][IM2ROW_STRIP_WIDTH],
    ap_uint<1> current_strip,
    int strip_column,
    const line_word_t& packed_pixel
) {
#pragma HLS INLINE
    if (current_strip == 0) {
        switch (strip_column) {
            case 1: strip[0][2][1] = packed_pixel; break;
            case 2: strip[0][2][2] = packed_pixel; break;
            case 3: strip[0][2][3] = packed_pixel; break;
            case 4: strip[0][2][4] = packed_pixel; break;
            case 5: strip[0][2][5] = packed_pixel; break;
            default: break;
        }
    } else {
        switch (strip_column) {
            case 1: strip[1][2][1] = packed_pixel; break;
            case 2: strip[1][2][2] = packed_pixel; break;
            case 3: strip[1][2][3] = packed_pixel; break;
            case 4: strip[1][2][4] = packed_pixel; break;
            case 5: strip[1][2][5] = packed_pixel; break;
            default: break;
        }
    }
}

// 启动阶段仅写 Strip0 Middle 的逻辑列1...5，也以固定译码保留该性质。
void write_initial_middle(
    line_word_t strip[2][IM2ROW_KERNEL][IM2ROW_STRIP_WIDTH],
    int strip_column,
    const line_word_t& packed_pixel
) {
#pragma HLS INLINE
    switch (strip_column) {
        case 1: strip[0][1][1] = packed_pixel; break;
        case 2: strip[0][1][2] = packed_pixel; break;
        case 3: strip[0][1][3] = packed_pixel; break;
        case 4: strip[0][1][4] = packed_pixel; break;
        case 5: strip[0][1][5] = packed_pixel; break;
        default: break;
    }
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

// 两种固定网络模式下各 Tile 的 Bottom 输入数量。首Tile需P+1个真实列；
// 最后Tile从固定几何派生剩余列数，其他Tile需P个。Bottom输入只在不是最后输出行时读取。
int bottom_load_count(const ConvGeometry& geometry, int tile_x) {
#pragma HLS INLINE
    if (tile_x == 0) {
        return IM2ROW_P + 1;
    }
    return tile_x == geometry.last_tile_x ? geometry.last_bottom_loads : IM2ROW_P;
}

// Conv1 的7个 Tile 均为4个真实空间位置；Conv2 的末 Tile 只有 lane0、1有效。
ap_uint<IM2ROW_P> make_spatial_valid(const ConvGeometry& geometry, int tile_x) {
#pragma HLS INLINE
    return tile_x == geometry.last_tile_x
        ? ap_uint<IM2ROW_P>(geometry.tail_spatial_valid_mask)
        : ap_uint<IM2ROW_P>(CNN_FULL_SPATIAL_VALID_MASK);
}

// 同行预取下一 Tile 的4个新列。仅为即将到来的最后 Tile 需要屏蔽越界列：
// Conv1、Conv2均仅在当前Tile是倒数第二个时屏蔽越界的新列。
ap_uint<IM2ROW_P> next_new_column_valid(const ConvGeometry& geometry, int tile_x) {
#pragma HLS INLINE
    return tile_x == geometry.prelast_tile_x
        ? ap_uint<IM2ROW_P>(geometry.prelast_new_column_valid_mask)
        : ap_uint<IM2ROW_P>(CNN_FULL_SPATIAL_VALID_MASK);
}

}  // 匿名命名空间

// Im2Row 顶层：把 H×W 个行优先、通道打包的标量像素转换成 Systolic 所需的 P 路窗口流。
// 输入每次成功读取一个 pixel_t；输出每次成功写入一个 activation_pack_t，包含同一 (kh,kw) 下的 P 个空间 Lane。
// 函数先预装首行，再进入双 Strip 的稳态 Token 调度；每个 Tile 按 kh→kw 连续产生9个 Pack。
// 全局循环以 II=1 为综合目标，同行 Tile 和换行 Tile 都在 current 输出期间准备 next；实际 II 仍需综合确认。
void im2row(
    hls::stream<pixel_t>& input_stream,
    hls::stream<activation_pack_t>& output_stream,
    layer_mode_t layer_mode
) {
    const ConvGeometry geometry = conv_geometry(layer_mode);
    // mode在调用期间保持稳定；所有尺寸均由固定网络配置派生。
    const int feature_height = geometry.h;
    const int feature_width = geometry.w;
    const int last_tile_x = geometry.last_tile_x;
    const int total_tokens = geometry.im2row_tokens;

    // Line Buffer 的三个维度依次表示：2个循环行 Bank、P个按列取模的独立列 Bank、
    // 每个列 Bank 内的块地址。每个元素是包含全部 C_PAR 通道的无符号打包字，读写始终覆盖整字；
    // 打包只改变内部存储位型，不改变 pixel_t、Strip 或模块间 Stream 接口。
    // 处理输出行时，一个行 Bank 保存 Middle，另一个先保存 Top；待旧 Top 已进入 Strip 后，
    // 再按列覆盖为下一输入行，避免整行搬移。
    line_word_t line_buffer[2][IM2ROW_P][IM2ROW_MAX_BLOCKS_PER_ROW];
    // 双 Strip 的维度依次为：2个 Ping-Pong 缓冲、Top/Middle/Bottom 三行、P+2个逻辑列。
    // 对 base_x 的 Tile，逻辑列覆盖 base_x-1...base_x+P；current 只读输出，next 只写准备。
    line_word_t strip[2][IM2ROW_KERNEL][IM2ROW_STRIP_WIDTH];
    // 同行预取保存 P 个新列；换行时也保存第一个 Bank Beat 的列0...P-1。
    line_word_t prefetch_top[IM2ROW_P];
    line_word_t prefetch_middle[IM2ROW_P];
    // 换行首 Tile 的列P与列0属于同一列 Bank，第二个 Beat 单独暂存在这里。
    line_word_t boundary_top;
    line_word_t boundary_middle;

#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_t2p impl=bram
#pragma HLS ARRAY_PARTITION variable=strip complete dim=1
#pragma HLS ARRAY_PARTITION variable=strip complete dim=2
#pragma HLS ARRAY_PARTITION variable=strip complete dim=3
#pragma HLS ARRAY_PARTITION variable=prefetch_top complete dim=1
#pragma HLS ARRAY_PARTITION variable=prefetch_middle complete dim=1

    const line_word_t zero_word = 0;

#ifndef __SYNTHESIS__
    // 仅用于非综合构建的接口守恒检查：输入必须恰好 H×W，输出必须恰好 H×ceil(W/P)×9。
    int input_read_count = 0;
    int output_token_count = 0;
#endif

    // 启动准备：首输出行的 Top 是上 Padding，Bottom 将在稳态前6拍填入；
    // Middle 的左/右 Padding 在此写零，真实列随后随首行输入直接进入 Strip。
    for (int column = 0; column < IM2ROW_STRIP_WIDTH; ++column) {
#pragma HLS UNROLL
        strip[0][0][column] = zero_word;
        strip[0][2][column] = zero_word;
        if (column == 0) {
            strip[0][1][column] = zero_word;
        }
    }

    // 简单启动：连续预装完整首行到行 Bank 0，并同时建立首 Tile 的 Middle。
    // 启动完成后立即进入连续 Token 调度；该启动循环不属于稳态输出流水线。
    for (int input_x = 0; input_x < feature_width; ++input_x) {
#pragma HLS PIPELINE II=1
        const pixel_t input_pixel =
            normalize_input_pixel(input_stream.read(), geometry.cin);
        const line_word_t input_word = pack_activation_lane(input_pixel);
        write_line_word(line_buffer, 0, input_x, input_word);
        if (input_x <= IM2ROW_P) {
            write_initial_middle(strip, input_x + 1, input_word);
        }
#ifndef __SYNTHESIS__
        ++input_read_count;
#endif
    }

    // 固定网络状态保持最小位宽；输出路径通过固定 Token selector 消除其作为
    // 多维 Strip 下标的需求。
    ap_uint<5> output_y = 0;
    ap_uint<3> tile_x = 0;
    ap_uint<4> token = 0;
    ap_uint<1> current_strip = 0;
    ap_uint<1> next_strip = 1;

    // 稳态唯一调度器：每次迭代恰好执行一次 Pack 写入，每个 Tile 连续执行9次。
    // 同行和换行所需的 next Strip 都在这9次迭代内提前准备，设计目标是 Tile 边界无输出气泡。
    // PIPELINE II=1 仅声明综合目标，实际 II、RAM 端口和读延迟必须由综合报告确认。
    // write 位于状态更新之前；下游阻塞时，阻塞写使 Token、Strip、行 Bank 和输入前沿均不推进。
    for (int global_token = 0; global_token < total_tokens; ++global_token) {
#pragma HLS PIPELINE II=1
        const int tile_base_x = tile_x.to_int() << 2;  // 当前 Tile 的第一个输出位置横坐标。
        const bool final_tile_in_row = (tile_x == last_tile_x);
        const bool final_output_row = (output_y == feature_height - 1);
        const int middle_bank = output_y & 1;  // 保存当前输出行 y 对应输入行的物理行 Bank。
        const int top_bank = 1 - middle_bank;  // 先保存上一行 Top，并作为下一输入行 Bottom 的逐列写入 Bank。

        activation_pack_t output_pack;
#pragma HLS ARRAY_PARTITION variable=output_pack.pixel complete dim=1
        // spatial_valid 描述输出位置是否存在，而不是窗口元素是否属于 Padding。
        output_pack.spatial_valid = make_spatial_valid(geometry, tile_x);
        for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
            if (output_pack.spatial_valid[lane]) {
                const line_word_t selected_word =
                    select_output_word(strip, current_strip, token, lane);
                output_pack.pixel[lane] = unpack_activation_lane(selected_word);
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
        const int load_count = bottom_load_count(geometry, tile_x);
        if ((!final_output_row) && (token < load_count)) {
            const int input_x = bottom_first_x(tile_x, tile_base_x) + token;
            const int strip_column = bottom_first_strip_column(tile_x) + token;
            const pixel_t input_pixel =
                normalize_input_pixel(input_stream.read(), geometry.cin);
            const line_word_t input_word = pack_activation_lane(input_pixel);
            write_current_bottom(
                strip, current_strip, strip_column, input_word);
            write_line_word(line_buffer, top_bank, input_x, input_word);
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
                const ap_uint<IM2ROW_P> new_column_mask =
                    next_new_column_valid(geometry, tile_x);
                for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
                    const int input_x = tile_base_x + IM2ROW_P + 1 + lane;
                    line_word_t top_word = zero_word;
                    line_word_t middle_word = zero_word;
                    if (new_column_mask[lane]) {
                        middle_word = read_line_word(line_buffer, middle_bank, input_x);
                        if (output_y > 0) {
                            top_word = read_line_word(line_buffer, top_bank, input_x);
                        }
                    }
                    prefetch_top[lane] = top_word;
                    prefetch_middle[lane] = middle_word;
                }
            }
            if (token == 8) {
                // token8 已成功输出 current 的最后一个 Bottom Pack，因此其最后两列不再被 current 使用。
                // 此时补齐 next 的 Top/Middle，复制 Bottom 重叠两列，并把待突发填入的新 Bottom 槽位初始化为零。
                for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
                    strip[next_strip][0][2 + lane] = prefetch_top[lane];
                    strip[next_strip][1][2 + lane] = prefetch_middle[lane];
                    strip[next_strip][2][2 + lane] = zero_word;
                }
                strip[next_strip][2][0] = strip[current_strip][2][IM2ROW_P];
                strip[next_strip][2][1] = strip[current_strip][2][IM2ROW_P + 1];
            }
        } else if ((!final_output_row) && (token == 6)) {
            // 换行首 Tile 不复用同行重叠列。其真实列0...P中，列0和列P映射到同一个列 Bank，
            // 因而必须分成两个 Bank Beat；token6先并行读取列0...P-1的 Top 和新行 Middle。
            for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
                const line_word_t top_word =
                    read_line_word(line_buffer, middle_bank, lane);
                const line_word_t middle_word =
                    read_line_word(line_buffer, top_bank, lane);
                prefetch_top[lane] = top_word;
                prefetch_middle[lane] = middle_word;
            }
        } else if ((!final_output_row) && (token == 7)) {
            // token7接收第一个 Beat，并发出列P的第二个 Beat；Strip 列0直接构造为左 Padding 零。
            strip[next_strip][0][0] = zero_word;
            strip[next_strip][1][0] = zero_word;
            for (int lane = 0; lane < IM2ROW_P; ++lane) {
#pragma HLS UNROLL
                strip[next_strip][0][1 + lane] = prefetch_top[lane];
                strip[next_strip][1][1 + lane] = prefetch_middle[lane];
            }
            boundary_top = read_line_word(line_buffer, middle_bank, IM2ROW_P);
            boundary_middle = read_line_word(line_buffer, top_bank, IM2ROW_P);
        } else if ((!final_output_row) && (token == 8)) {
            // token8接收第二个 Beat，下一行首 Tile 的 Top/Middle 至此完整。
            // Bottom 先按 Padding 构造，并在该 Tile 自己的 token0...5 中读取真实列，因此下一拍即可输出 Top。
            strip[next_strip][0][IM2ROW_P + 1] = boundary_top;
            strip[next_strip][1][IM2ROW_P + 1] = boundary_middle;
            for (int column = 0; column < IM2ROW_STRIP_WIDTH; ++column) {
#pragma HLS UNROLL
                strip[next_strip][2][column] = zero_word;
            }
        }

        // token8的 Pack 已成功写出，current 的9拍全部结束，next 的首个 Top Pack 也已准备完成，故可安全交换。
        if (token == IM2ROW_KERNEL * IM2ROW_KERNEL - 1) {
            token = 0;
            const ap_uint<1> previous_current = current_strip;
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
    assert(input_read_count == feature_height * feature_width &&
           "图像行转矩阵模块必须恰好消费 H×W 个真实输入像素");
    assert(output_token_count == total_tokens &&
           "图像行转矩阵模块必须恰好输出 H×ceil(W/P)×9 个令牌");
#endif
}
