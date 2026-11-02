#include "im2row.h"
#include "../channel_pack.h"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace {
// 内部64-bit字始终只含一个位置的一组8通道；G保留在RAM地址中。
using word_t = packed_activation_lane_t;
using coordinate_t = ap_uint<bits_for_max(IM2ROW_MAX_W + 8)>;
using tile_t = ap_uint<bits_for_max(IM2ROW_MAX_BLOCKS_PER_ROW)>;
using group_t = ap_uint<bits_for_max(IM2ROW_MAX_GROUPS)>;
using strip_address_t = ap_uint<bits_for_max(IM2ROW_STRIP_BANK_DEPTH - 1)>;
using line_address_t = ap_uint<bits_for_max(IM2ROW_LINE_BANK_DEPTH)>;
using input_count_t = ap_uint<bits_for_max(yolo_max_field(&ConvGeometry::input_transactions))>;
using output_count_t = ap_uint<bits_for_max(yolo_max_field(&ConvGeometry::im2row_tokens))>;
constexpr int STRIP_PONG_SHIFT = bits_for_max(IM2ROW_MAX_GROUPS - 1);
static_assert((IM2ROW_MAX_GROUPS & (IM2ROW_MAX_GROUPS - 1)) == 0,
              "Strip pong concatenation requires a power-of-two group capacity");
word_t normalized_word(const channel_pack_t& source, int group, int cin) {
#pragma HLS INLINE
    channel_pack_t normalized;
    for (int c = 0; c < FEATURE_PACK_CHANNELS; ++c) {
#pragma HLS UNROLL
        normalized.channel[c] = group * FEATURE_PACK_CHANNELS + c < cin
            ? source.channel[c] : activation_t(0);
    }
    return pack_activation_lane(normalized);
}

// UNROLL固定空间Bank选择，onehot掩码避免优先级选择链；
// 只有pong/group进入RAM地址，不把通道组展开成寄存器阵列。
void output_words(const word_t banks[3][6], ap_uint<4> kernel_pos, word_t words[4]) {
#pragma HLS INLINE
    // 独立one-hot掩码后做OR，不生成9级有优先级的switch选择链。
    for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
        word_t row_word[3];
#pragma HLS ARRAY_PARTITION variable=row_word complete
        for (int row = 0; row < 3; ++row) {
#pragma HLS UNROLL
            const word_t a = banks[row][lane] & word_t(kernel_pos == row * 3 ? ~word_t(0) : word_t(0));
            const word_t b = banks[row][lane + 1] & word_t(kernel_pos == row * 3 + 1 ? ~word_t(0) : word_t(0));
            const word_t c = banks[row][lane + 2] & word_t(kernel_pos == row * 3 + 2 ? ~word_t(0) : word_t(0));
            row_word[row] = a | b | c;
        }
        words[lane] = row_word[0] | row_word[1] | row_word[2];
    }
}

// 有限的当前事务写命令，不按G扩展。copy/prefetch/写零/输入写入
// 均先收集目的地址和数据，再从每Bank唯一入口提交。BRAM实际返回寄存器
// 与标签对齐由HLS按latency=1及真实RAW排程，不人为建立跨迭代提前消费。
struct WriteBeat {
    bool enable[3][6];
    strip_address_t address[3][6];
    word_t data[3][6];
};
void commit_strip_writes(word_t strip[3][6][IM2ROW_STRIP_BANK_DEPTH],
                         const WriteBeat& beat) {
#pragma HLS INLINE
    for (int row = 0; row < 3; ++row) {
#pragma HLS UNROLL
        for (int column = 0; column < 6; ++column) {
#pragma HLS UNROLL
            if (beat.enable[row][column])
                strip[row][column][beat.address[row][column].to_uint()] = beat.data[row][column];
        }
    }
}
} // 匿名命名空间

void im2row(hls::stream<channel_pack_t>& input_stream,
            hls::stream<activation_pack_t>& output_stream,
            const ConvGeometry& geometry) {
#ifndef __SYNTHESIS__
    // 表内配置是调用契约。严格校验只用于非综合构建，不产生运行时硬件
    // 拒绝分支，以免静默返回后让DATAFLOW下游永久等待本应出现的事务。
    assert(im2row_geometry_supported(geometry) && "Im2Row仅接受固定YOLOv2-tiny层表配置");
#endif
    // 两行循环×四列Bank，地址=(x/4)*G+group，总数据容量32KiB。
    // Strip物理18Bank，地址=pong*128+group，总数据容量36KiB。
    // 每Bank只允许一读一写；Group和pong维不作complete partition。
    word_t line[2][4][IM2ROW_LINE_BANK_DEPTH];
    word_t strip[3][6][IM2ROW_STRIP_BANK_DEPTH];
#pragma HLS ARRAY_PARTITION variable=line complete dim=1
#pragma HLS ARRAY_PARTITION variable=line complete dim=2
#pragma HLS BIND_STORAGE variable=line type=ram_s2p impl=bram latency=1
#pragma HLS ARRAY_PARTITION variable=strip complete dim=1
#pragma HLS ARRAY_PARTITION variable=strip complete dim=2
#pragma HLS BIND_STORAGE variable=strip type=ram_s2p impl=bram latency=1
    const int groups = geometry.cin_groups, width = geometry.w, height = geometry.h;
    // G只取1/2/.../128；调用级译码，扫描中不做通用乘法。
    ap_uint<3> group_shift = 0;
    for (int bit = 0; bit < 8; ++bit) {
#pragma HLS UNROLL
        if (groups == (1 << bit)) group_shift = bit;
    }
    const group_t last_group = groups - 1;
    const coordinate_t image_width = width, last_y = height - 1;
    const tile_t last_tile = geometry.last_tile_x;
    const coordinate_t first_end = width < 5 ? width : 5;
    const input_count_t input_transactions = geometry.input_transactions;
    const output_count_t output_transactions = geometry.im2row_tokens;
#ifndef __SYNTHESIS__
    int reads = 0, writes = 0;
#endif
    if (geometry.kernel == 1) {
        // 1×1满Tile只缓存前三位置，第四位置到来时组合输出；尾Tile的r=1
        // 不读任何缓存。direct、mask和缓存返回同属一次阻塞事务，由HLS
        // 随RAM读延迟对齐；不保证在输入气泡期间仍持续输出。
        // 前r-1个真实位置复用Strip的三个固定Bank，不新增Feature缓存。
        // 最后位置输入与前三位置缓存的RAM返回组成同一个输出事务，
        // 避免读完4G拍后再单独花G拍输出。
        tile_t tile = 0;
        ap_uint<2> position = 0;
        group_t group = 0;
        const ap_uint<2> tail_last_position = (width - 1) & 3;
        // 跨位置、Tile和行的单连续流水；不逐Tile排空/重新启动。
        for (input_count_t transaction = 0; transaction < input_transactions; ++transaction) {
#pragma HLS PIPELINE II=1
            const ap_uint<2> final_position = tile == last_tile ? tail_last_position : ap_uint<2>(3);
            const word_t direct = normalized_word(input_stream.read(), group, geometry.cin);
#ifndef __SYNTHESIS__
            ++reads;
#endif
            if (position != final_position) {
                for (int bank = 0; bank < 3; ++bank) {
#pragma HLS UNROLL
                    if (position == bank) strip[0][bank][group] = direct;
                }
            } else {
                activation_pack_t output;
                output.spatial_valid = tile == last_tile ? geometry.tail_spatial_valid_mask : 15;
                for (int lane = 0; lane < 4; ++lane) {
#pragma HLS UNROLL
                    word_t word = 0;
                    if (lane == final_position) word = direct;
                    else if (lane < final_position && lane < 3) word = strip[0][lane][group];
                    output.pixel[lane] = unpack_activation_lane(word);
                }
                output_stream.write(output);
#ifndef __SYNTHESIS__
                ++writes;
#endif
            }
            if (group == last_group) {
                group = 0;
                if (position == final_position) {
                    position = 0;
                    tile = tile == last_tile ? tile_t(0) : tile_t(tile + 1);
                } else ++position;
            } else ++group;
        }
    } else {
        // 仅初始化首个活跃窗口；后续Padding在预取/复制前在线准备为零，
        // 不能只在最终输出选择器上抹零而允许复制未初始化RAM内容。
        for (group_t group = 0; group <= last_group; ++group) {
#pragma HLS PIPELINE II=1
            for (int row = 0; row < 3; ++row) {
#pragma HLS UNROLL
                for (int c = 0; c < 6; ++c) {
#pragma HLS UNROLL
                    strip[row][c][group] = 0;
                }
            }
        }
        coordinate_t preload_x = 0;
        group_t preload_group = 0;
        line_address_t preload_block = 0;
        // 单输入事务循环：不让flatten用W*G重建边界乘法。
        while (preload_x < image_width) {
#pragma HLS PIPELINE II=1
            const word_t word = normalized_word(input_stream.read(), preload_group, geometry.cin);
            const line_address_t preload_address = preload_block + preload_group;
            for (int bank = 0; bank < 4; ++bank) {
#pragma HLS UNROLL
                if ((preload_x.to_uint() & 3u) == unsigned(bank))
                    line[0][bank][preload_address.to_uint()] = word;
            }
            for (int c = 1; c < 6; ++c) {
#pragma HLS UNROLL
                if (preload_x == c - 1) strip[1][c][preload_group] = word;
            }
#ifndef __SYNTHESIS__
            ++reads;
#endif
            if (preload_group == last_group) {
                preload_group = 0;
                if ((preload_x & 3) == 3) preload_block += groups;
                ++preload_x;
            } else ++preload_group;
        }
        coordinate_t y = 0, base = 0, bottom_x = 0, bottom_end = first_end;
        tile_t tile = 0;
        ap_uint<4> kernel_pos = 0;
        group_t group = 0, bottom_group = 0;
        ap_uint<3> bottom_column = 1;
        ap_uint<1> pong = 0;
        // 调度端口表（按每个group）：q0..4逐列填当前Bottom；q3读Top
        // 列4/5并写next列0/1；q4从两行四Bank各读一次，返回后写next
        // Top/Middle列2..5；q6复制Middle；q8复用输出已读Bottom列4/5，
        // 写next列0/1并清零列2..5。换行则q6请求列0..3、q7请求列4，
        // 返回后回写。写零同样计入每Bank的一次写，不靠输出抹零替代。
        // next.address包含本事务目标pong与group，后续状态切换不改变在途标签。
        // 所有真实RAW保留给调度器；请求、返回、store的实际周期由HLS确定，
        // 不用伪依赖声明或跨迭代手工寄存器强制调度。
        // 全部Tile和输出行共用一个连续流水循环，不插入逐Tile准备或排空。
        // BRAM显式绑定一拍读延迟，预取经有限返回寄存器传递稳定标签；
        // 保留真实读后写依赖，II=1仅为调度目标，不代表已验证的实现吞吐。
        for (output_count_t emitted = 0; emitted < output_transactions; ++emitted) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=strip inter RAW distance=2 true
            // 仅说明真实RAW的最短迭代距离，不删除依赖。固定层表G>=1：
            // G=1时Bottom最后新输入q4写→q6首次读，以及换行Top列5
            // q7写→下一Tile q0全列读，均相隔2；Top q6→next q0为3。
            // 其他同行/跨行同址访问更远；G>1同group间隔只会增加。
            // 此证明依赖下方行读门控与现有q调度，改动它们必须重新审核。
            const bool row_end = tile == last_tile;
            const bool last_row = y == last_y;
            const unsigned middle_bank = y.to_uint() & 1u, top_bank = 1u - middle_bank;
            const strip_address_t address = (pong.to_uint() << STRIP_PONG_SHIFT) | group.to_uint();
            const strip_address_t next_address = ((1u - pong.to_uint()) << STRIP_PONG_SHIFT) | group.to_uint();
            const strip_address_t destination = (pong.to_uint() << STRIP_PONG_SHIFT) | bottom_group.to_uint();
            const bool load_bottom = !last_row && kernel_pos < 6 && bottom_x < bottom_end;
            // 每迭代独立写命令；无效Bank的data为零，address仍由下方译码。
            // 不保留旧data，也不强制下一迭代提前读取本迭代的RAM返回。
            WriteBeat next = {};
#pragma HLS ARRAY_PARTITION variable=next.enable complete dim=0
#pragma HLS ARRAY_PARTITION variable=next.address complete dim=0
#pragma HLS ARRAY_PARTITION variable=next.data complete dim=0
            word_t banks[3][6];
#pragma HLS ARRAY_PARTITION variable=banks complete dim=0
            for (int row = 0; row < 3; ++row) {
#pragma HLS UNROLL
                for (int c = 0; c < 6; ++c) {
#pragma HLS UNROLL
                    // 控制独立于copy/prefetch/input数据分支；每Bank一个显式
                    // 布尔译码，不由嵌套数据分支的phi传播写使能。
                    if (row < 2) {
                        const bool overlap = !row_end && c < 2 &&
                            kernel_pos == (row == 0 ? 3 : 6);
                        const bool prefetch = !row_end && kernel_pos == 4 && c >= 2;
                        const bool restart = row_end && !last_row &&
                            ((kernel_pos == 6 && c <= 4) || (kernel_pos == 7 && c == 5));
                        next.enable[row][c] = overlap || prefetch || restart;
                        next.address[row][c] = next_address;
                    } else {
                        const bool load = load_bottom && c == bottom_column;
                        const bool prepare = kernel_pos == 8 && (!row_end || !last_row);
                        next.enable[row][c] = load || prepare;
                        next.address[row][c] = load ? destination : next_address;
                    }
                    // 每Bank只读一次；未使用行不发请求，避免Bottom准备期间
                    // 无用的RAM读让写使能也陷入跨迭代memory反馈。
                    const bool use_row = row == 0 ? kernel_pos <= 3
                        : row == 1 ? (kernel_pos >= 3 && kernel_pos <= 6)
                                   : kernel_pos >= 6;
                    banks[row][c] = use_row ? strip[row][c][address.to_uint()] : word_t(0);
                }
            }
            word_t words[4];
#pragma HLS ARRAY_PARTITION variable=words complete
            output_words(banks, kernel_pos, words);
            activation_pack_t output;
            output.spatial_valid = row_end ? geometry.tail_spatial_valid_mask : 15;
            for (int s = 0; s < 4; ++s) {
#pragma HLS UNROLL
                output.pixel[s] = unpack_activation_lane(output.spatial_valid[s] ? words[s] : word_t(0));
            }
            output_stream.write(output);
#ifndef __SYNTHESIS__
            ++writes;
#endif
            // 输入使用独立x/group计数，保持先列后组的stream顺序。
            // 首Tile最多读5G拍，其余最多4G拍；W13末Tile的bottom_x与
            // end_x同为13，不再读新输入，列11/12已通过重叠保留。
            word_t input_word = 0;
            if (load_bottom) {
                input_word = normalized_word(input_stream.read(), bottom_group, geometry.cin);
                // 覆盖前旧Top已进入current Strip：前两列来自上一Tile重叠，
                // 其余列来自上一Tile的q4预取，绝不回读已覆盖的旧Top。
                const line_address_t bottom_address = ((bottom_x >> 2) << group_shift) + bottom_group;
                // 固定物理Bank的写译码，不让ap索引通过前端展开成
                // 通用动态数组访问；每Bank仍最多一次写，不复制存储。
                for (int physical_row = 0; physical_row < 2; ++physical_row) {
#pragma HLS UNROLL
                    for (int bank = 0; bank < 4; ++bank) {
#pragma HLS UNROLL
                        if (unsigned(physical_row) == top_bank &&
                            (bottom_x.to_uint() & 3u) == unsigned(bank))
                            line[physical_row][bank][bottom_address.to_uint()] = input_word;
                    }
                }
#ifndef __SYNTHESIS__
                ++reads;
#endif
            }
            // 三种预取事件共用每个物理Line Bank的唯一读入口。
            // 目标Strip列c的Line Bank恒为(c-1)&3；q6/q7只改变请求地址。
            word_t physical[2][4];
#pragma HLS ARRAY_PARTITION variable=physical complete dim=0
            const bool normal_request = !row_end && kernel_pos == 4;
            const bool restart_first = row_end && !last_row && kernel_pos == 6;
            const bool restart_last = row_end && !last_row && kernel_pos == 7;
            for (int bank = 0; bank < 4; ++bank) {
#pragma HLS UNROLL
                const coordinate_t normal_x = base + (bank == 0 ? 8 : 4 + bank);
                const bool normal_real = normal_request && normal_x < image_width;
                const bool restart_request = restart_first || (restart_last && bank == 0);
                const bool request = normal_real || restart_request;
                const line_address_t normal_address = ((normal_x >> 2) << group_shift) + group;
                const line_address_t restart_address = restart_last ? line_address_t(groups + group) : line_address_t(group);
                const line_address_t request_address =
                    (normal_address & line_address_t(normal_real ? ~line_address_t(0) : line_address_t(0))) |
                    (restart_address & line_address_t(restart_request ? ~line_address_t(0) : line_address_t(0)));
                for (int physical_row = 0; physical_row < 2; ++physical_row) {
#pragma HLS UNROLL
                    // 首输出行普通预取不可读未初始化旧Top；换行则两行均已就绪。
                    const bool initialized = row_end || y != 0 || unsigned(physical_row) == middle_bank;
                    physical[physical_row][bank] = request && initialized
                        ? line[physical_row][bank][request_address.to_uint()] : word_t(0);
                }
            }
            const unsigned next_top_row = row_end ? middle_bank : top_bank;
            word_t routed[2][4];
#pragma HLS ARRAY_PARTITION variable=routed complete dim=0
            for (int row = 0; row < 2; ++row) {
#pragma HLS UNROLL
                const bool take_zero = row == 0 ? next_top_row == 0 : next_top_row != 0;
                for (int bank = 0; bank < 4; ++bank) {
#pragma HLS UNROLL
                    routed[row][bank] =
                        (physical[0][bank] & word_t(take_zero ? ~word_t(0) : word_t(0))) |
                        (physical[1][bank] & word_t(!take_zero ? ~word_t(0) : word_t(0)));
                }
            }
            for (int row = 0; row < 2; ++row) {
#pragma HLS UNROLL
                const bool copy = !row_end && kernel_pos == (row == 0 ? 3 : 6);
                for (int c = 0; c < 6; ++c) {
#pragma HLS UNROLL
                    word_t overlap_word = 0, line_word = 0;
                    if (c < 2)
                        overlap_word = banks[row][4 + c] & word_t(copy ? ~word_t(0) : word_t(0));
                    if (c >= 1) {
                        const bool use_line = (normal_request && c >= 2) ||
                            (restart_first && c <= 4) || (restart_last && c == 5);
                        line_word = routed[row][(c - 1) & 3] &
                            word_t(use_line ? ~word_t(0) : word_t(0));
                    }
                    // 每个目的Bank一次data赋值，互斥来源用onehot OR合并。
                    next.data[row][c] = overlap_word | line_word;
                }
            }
            for (int c = 0; c < 6; ++c) {
#pragma HLS UNROLL
                const bool take_input = load_bottom && c == bottom_column;
                word_t copied = 0;
                if (c < 2) {
                    const bool copy = !row_end && kernel_pos == 8;
                    copied = banks[2][4 + c] & word_t(copy ? ~word_t(0) : word_t(0));
                }
                // q8保留原始Bank4/5，不穿输出选择器，也不复制masked Payload。
                next.data[2][c] = copied |
                    (input_word & word_t(take_input ? ~word_t(0) : word_t(0)));
            }
            // 先收集全部读返回和目的标签，再每Bank仅一次写入。提交位于
            // 本迭代末尾；真实RAM读延迟与返回寄存器由HLS流水实现。
            commit_strip_writes(strip, next);
            if (load_bottom) {
                if (bottom_group == last_group) {
                    bottom_group = 0; ++bottom_x; ++bottom_column;
                } else ++bottom_group;
            }
            if (group == last_group) {
                group = 0;
                if (kernel_pos == 8) {
                    kernel_pos = 0; pong = 1 - pong;
                    if (row_end) {
                        tile = 0; ++y; base = 0; bottom_x = 0;
                        bottom_end = first_end; bottom_column = 1;
                    } else {
                        ++tile; base += 4; bottom_x = base + 1; bottom_column = 2;
                        // 仅Tile切换更新终点。输入握手路径只比较两个9-bit寄存器，
                        // 不再每拍串联base+5、min(width)、bottom_x比较。
                        const coordinate_t candidate = base + 5;
                        bottom_end = candidate < image_width ? candidate : image_width;
                    }
                    bottom_group = 0;
                } else ++kernel_pos;
            } else ++group;
        }
    }
#ifndef __SYNTHESIS__
    assert(reads == geometry.input_transactions);
    assert(writes == geometry.im2row_tokens);
#endif
}
