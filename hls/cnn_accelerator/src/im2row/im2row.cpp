#include "im2row.h"
#include "../channel_pack.h"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace {
// 内部64-bit字始终只含一个位置的一组8通道；G保留在RAM地址中。
using word_t = packed_activation_lane_t;
using coordinate_t = ap_uint<bits_for_max(IM2ROW_MAX_W + 8)>;  // 源列坐标需容纳最大W及预取偏移。
using tile_t = ap_uint<bits_for_max(IM2ROW_MAX_BLOCKS_PER_ROW)>;  // Tile计数含逐行末Tile边界。
using group_t = ap_uint<bits_for_max(IM2ROW_MAX_GROUPS)>;  // Cin组计数需覆盖最大G及初始化终点。
using strip_address_t = ap_uint<bits_for_max(IM2ROW_STRIP_BANK_DEPTH - 1)>;  // Strip地址覆盖两个槽，每槽MAX_G组。
using line_address_t = ap_uint<bits_for_max(IM2ROW_LINE_BANK_DEPTH)>;  // Line地址额外容纳预装块的右开终点。
using input_count_t = ap_uint<bits_for_max(yolo_max_field(&ConvGeometry::input_transactions))>;  // 输入事务计数含循环右开终点。
using output_count_t = ap_uint<bits_for_max(yolo_max_field(&ConvGeometry::im2row_tokens))>;  // 输出事务计数含循环右开终点。
constexpr int STRIP_PONG_SHIFT = bits_for_max(IM2ROW_MAX_GROUPS - 1);  // 槽号左移位数：低位保留给Cin组。
static_assert((IM2ROW_MAX_GROUPS & (IM2ROW_MAX_GROUPS - 1)) == 0,
              "Strip pong concatenation requires a power-of-two group capacity");
word_t normalized_word(const channel_pack_t& source, int cin_group, int cin) {
#pragma HLS INLINE
    channel_pack_t normalized;
    for (int cin_lane = 0; cin_lane < FEATURE_PACK_CHANNELS; ++cin_lane) {
#pragma HLS UNROLL
        normalized.channel[cin_lane] = cin_group * FEATURE_PACK_CHANNELS + cin_lane < cin
            ? source.channel[cin_lane] : activation_t(0);
    }
    return pack_activation_lane(normalized);
}

// UNROLL固定空间Bank选择，onehot掩码避免优先级选择链；
// 只有槽号/Cin组进入RAM地址，不把通道组展开成寄存器阵列。
void output_words(const word_t strip_read_data[3][6], ap_uint<4> kernel_pos, word_t window_words[4]) {
#pragma HLS INLINE
    // 独立one-hot掩码后做OR，不生成9级有优先级的switch选择链。
    for (int spatial_lane = 0; spatial_lane < 4; ++spatial_lane) {
#pragma HLS UNROLL
        word_t row_word[3];
#pragma HLS ARRAY_PARTITION variable=row_word complete
        for (int strip_row = 0; strip_row < 3; ++strip_row) {
#pragma HLS UNROLL
            const word_t kw0_word =
                strip_read_data[strip_row][spatial_lane] & word_t(kernel_pos == strip_row * 3 ? ~word_t(0) : word_t(0));
            const word_t kw1_word =
                strip_read_data[strip_row][spatial_lane + 1] & word_t(kernel_pos == strip_row * 3 + 1 ? ~word_t(0) : word_t(0));
            const word_t kw2_word =
                strip_read_data[strip_row][spatial_lane + 2] & word_t(kernel_pos == strip_row * 3 + 2 ? ~word_t(0) : word_t(0));
            row_word[strip_row] = kw0_word | kw1_word | kw2_word;
        }
        window_words[spatial_lane] = row_word[0] | row_word[1] | row_word[2];
    }
}

// 有限的当前事务写命令，不按G扩展。copy/prefetch/写零/输入写入
// 均先收集目的地址和数据，再从每Bank唯一入口提交。BRAM实际返回寄存器
// 与标签对齐由HLS按latency=1及真实RAW排程，不人为建立跨迭代提前消费。
struct StripWriteCommand {
    bool enable[3][6];  // 各逻辑行列Bank的本事务写使能。
    strip_address_t address[3][6];  // 携带目的槽及Cin组，角色切换不改变标签。
    word_t data[3][6];  // 每Bank唯一写入数据，无效Bank默认零。
};
void commit_strip_writes(word_t window_strip[3][6][IM2ROW_STRIP_BANK_DEPTH],
                         const StripWriteCommand& command) {
#pragma HLS INLINE
    for (int strip_row = 0; strip_row < 3; ++strip_row) {
#pragma HLS UNROLL
        for (int strip_column = 0; strip_column < 6; ++strip_column) {
#pragma HLS UNROLL
            if (command.enable[strip_row][strip_column])
                window_strip[strip_row][strip_column][command.address[strip_row][strip_column].to_uint()] =
                    command.data[strip_row][strip_column];
        }
    }
}
} // 匿名命名空间

void im2row(hls::stream<channel_pack_t>& input_stream,
            hls::stream<activation_pack_t>& output_stream,
            const ConvGeometry& geometry) {
    // 调用方只提供固定层表配置；不逐字段检查或构造兜底配置。
    // 两行循环×四列Bank，地址=(x/4)*G+cin_group，总数据容量32KiB。
    // Strip物理18Bank，地址=current_strip_slot*128+cin_group，总数据容量36KiB。
    // 每Bank只允许一读一写；Group和pong维不作complete partition。
    word_t line_buffer[2][4][IM2ROW_LINE_BANK_DEPTH];
    word_t window_strip[3][6][IM2ROW_STRIP_BANK_DEPTH];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=1
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=bram latency=1
#pragma HLS ARRAY_PARTITION variable=window_strip complete dim=1
#pragma HLS ARRAY_PARTITION variable=window_strip complete dim=2
#pragma HLS BIND_STORAGE variable=window_strip type=ram_s2p impl=bram latency=1
    const int cin_groups = geometry.cin_groups;  // G：每个真实空间位置包含的8通道组数。
    const int width = geometry.w;  // 真实输入/输出行宽，不含Padding。
    const int height = geometry.h;  // 真实输入/输出行数。
    const ap_uint<3> cin_group_shift = geometry.cin_group_shift;  // 固定层表已预计算log2(G)。
    const group_t last_cin_group = cin_groups - 1;  // 当前调用最后一个有效Cin组编号。
    const coordinate_t image_width = width;  // 窄位宽的真实列右开终点。
    const coordinate_t last_output_y = height - 1;  // 最后一个输出行坐标。
    const tile_t last_tile_x = geometry.last_tile_x;  // 每行最后一个空间Tile编号。
    const coordinate_t first_tile_input_end = IM2ROW_FIRST_TILE_INPUT_COLUMNS;  // 表内W均>=5，首Tile固定读取五列。
    const input_count_t input_transactions = geometry.input_transactions;  // 一次调用必须读取的真实输入Pack总数。
    const output_count_t output_transactions = geometry.im2row_tokens;  // 一次调用必须输出的窗口Token总数。
#ifndef __SYNTHESIS__
    int reads = 0, writes = 0;
#endif
    if (geometry.kernel == 1) {
        // 1×1收集：前r-1个位置复用Strip的三个Bank，最后位置到来时直接输出。
        // 满Tile不另花G拍输出；尾Tile若r=1则完全不读缓存。
        tile_t tile_x = 0;  // 1×1按行收集Tile，不跨行拼接空间Lane。
        ap_uint<2> gather_lane = 0;  // 当前输入在Tile内的位置，不是kernel位置。
        group_t cin_group = 0;  // 当前输入的8通道组。
        const ap_uint<2> tail_last_gather_lane = YOLO_TINY_PREDICTION_TAIL_LAST_LANE;  // 唯一1x1层W13，尾Tile只有Lane0。
        // 跨位置、Tile和行的单连续流水；不逐Tile排空/重新启动。
        for (input_count_t transaction = 0; transaction < input_transactions; ++transaction) {
#pragma HLS PIPELINE II=1
            // 此位置输入直接参与输出。
            const ap_uint<2> last_gather_lane = tile_x == last_tile_x ? tail_last_gather_lane : ap_uint<2>(3);
            const word_t input_lane_word = normalized_word(input_stream.read(), cin_group, geometry.cin);
#ifndef __SYNTHESIS__
            ++reads;
#endif
            if (gather_lane != last_gather_lane) {
                for (int column_bank = 0; column_bank < 3; ++column_bank) {
#pragma HLS UNROLL
                    if (gather_lane == column_bank) window_strip[0][column_bank][cin_group] = input_lane_word;
                }
            } else {
                // 直接输入、缓存返回与mask同属一次阻塞事务，由HLS随RAM延迟对齐；
                // 输入有气泡时，不保证输出仍能连续。
                activation_pack_t output;
                output.spatial_valid = tile_x == last_tile_x ? geometry.tail_spatial_valid_mask : 15;
                for (int spatial_lane = 0; spatial_lane < 4; ++spatial_lane) {
#pragma HLS UNROLL
                    word_t word = 0;
                    if (spatial_lane == last_gather_lane) word = input_lane_word;
                    else if (spatial_lane < last_gather_lane) word =
                        window_strip[0][spatial_lane][cin_group];
                    output.pixel[spatial_lane] = unpack_activation_lane(word);
                }
                output_stream.write(output);
#ifndef __SYNTHESIS__
                ++writes;
#endif
            }
            if (cin_group == last_cin_group) {
                cin_group = 0;
                if (gather_lane == last_gather_lane) {
                    gather_lane = 0;
                    tile_x = tile_x == last_tile_x ? tile_t(0) : tile_t(tile_x + 1);
                } else ++gather_lane;
            } else ++cin_group;
        }
    } else {
        // 仅初始化首个活跃窗口；后续Padding在预取/复制前在线准备为零，
        // 不能只在最终输出选择器上抹零而允许复制未初始化RAM内容。
        for (group_t cin_group = 0; cin_group <= last_cin_group; ++cin_group) {
#pragma HLS PIPELINE II=1
            for (int strip_row = 0; strip_row < 3; ++strip_row) {
#pragma HLS UNROLL
                for (int strip_column = 0; strip_column < 6; ++strip_column) {
#pragma HLS UNROLL
                    window_strip[strip_row][strip_column][cin_group] = 0;
                }
            }
        }
        coordinate_t preload_x = 0;  // 预装真实首行的当前源列。
        group_t preload_group = 0;  // 预装源列内的Cin组。
        line_address_t preload_block = 0;  // 当前四列块的Line基地址，每跨四列增加G。
        // 单输入事务循环：不让flatten用W*G重建边界乘法。
        while (preload_x < image_width) {
#pragma HLS PIPELINE II=1
            const word_t word = normalized_word(input_stream.read(), preload_group, geometry.cin);
            const line_address_t preload_address = preload_block + preload_group;  // 四列块基地址加当前Cin组。
            for (int column_bank = 0; column_bank < 4; ++column_bank) {
#pragma HLS UNROLL
                if ((preload_x.to_uint() & 3u) == unsigned(column_bank))
                    line_buffer[0][column_bank][preload_address.to_uint()] = word;
            }
            for (int strip_column = 1; strip_column < 6; ++strip_column) {
#pragma HLS UNROLL
                if (preload_x == strip_column - 1) window_strip[1][strip_column][preload_group] = word;
            }
#ifndef __SYNTHESIS__
            ++reads;
#endif
            if (preload_group == last_cin_group) {
                preload_group = 0;
                if ((preload_x & 3) == 3) preload_block += cin_groups;
                ++preload_x;
            } else ++preload_group;
        }
        // 3×3主扫描：输出游标与Bottom输入游标独立，保持输入x→group顺序。
        coordinate_t output_y = 0;  // 当前输出行。
        coordinate_t tile_base_x = 0;  // 当前Tile的第一个输出列。
        coordinate_t bottom_input_x = 0;  // 下一次读取的真实Bottom源列。
        coordinate_t bottom_input_end = first_tile_input_end;  // 本Tile新Bottom输入的右开终点。
        tile_t tile_x = 0;  // 当前行内Tile编号。
        ap_uint<4> kernel_pos = 0;  // kh=kernel_pos/3，kw=kernel_pos%3。
        group_t cin_group = 0;  // 输出与后台预取使用的通道组。
        group_t bottom_input_group = 0;  // 独立推进的Bottom输入通道组。
        ap_uint<3> bottom_strip_column = 1;  // 首Tile列0为左Padding；后续从列2开始填，前两列保留重叠。
        ap_uint<1> current_strip_slot = 0;  // 当前Strip的Ping-Pong槽，另一个槽准备下一Tile。
        // 3×3连续主循环：Tile及行之间不额外排空；II=1是目标，不等同于RTL吞吐。
        // 所有真实RAW交由HLS结合latency=1排程，不用跨迭代手工寄存器强制提前消费。
        for (output_count_t emitted = 0; emitted < output_transactions; ++emitted) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=window_strip inter RAW distance=2 true
            // 保留真实RAW：G=1时Bottom q4写→q6读、换行Top列5 q7写→下Tile q0读，最短均为2。
            // Top换行q6写→下Tile q0为3，其余访问更远；G>1同组距离只增加。
            // 证明依赖固定层表、下方行读门控与q调度，改动必须重新审核。
            const bool is_last_tile = tile_x == last_tile_x;
            const bool is_last_output_row = output_y == last_output_y;
            const unsigned middle_physical_row = output_y.to_uint() & 1u;  // 保存输入行output_y的Line物理行。
            const unsigned top_physical_row = 1u - middle_physical_row;  // 旧Top逐位置覆盖为新Bottom。
            // 高位槽号、低位Cin组。
            const strip_address_t current_strip_address =
                (current_strip_slot.to_uint() << STRIP_PONG_SHIFT) | cin_group.to_uint();
            // 相反槽，同一Cin组。
            const strip_address_t next_strip_address =
                ((1u - current_strip_slot.to_uint()) << STRIP_PONG_SHIFT) | cin_group.to_uint();
            // 当前槽，独立Bottom输入组。
            const strip_address_t bottom_write_address =
                (current_strip_slot.to_uint() << STRIP_PONG_SHIFT) | bottom_input_group.to_uint();
            const bool load_bottom = !is_last_output_row && kernel_pos < 6 && bottom_input_x < bottom_input_end;
            // 每迭代独立写命令；无效Bank的data为零，address仍由下方译码。
            // 不保留旧data，也不强制下一迭代提前读取本迭代的RAM返回。
            StripWriteCommand strip_write = {};  // 写控制：每Bank唯一写入口，地址标签不随后续状态切换。
#pragma HLS ARRAY_PARTITION variable=strip_write.enable complete dim=0
#pragma HLS ARRAY_PARTITION variable=strip_write.address complete dim=0
#pragma HLS ARRAY_PARTITION variable=strip_write.data complete dim=0
            word_t strip_read_data[3][6];  // Strip读返回同时供输出及重叠复制，避免重复读口。
#pragma HLS ARRAY_PARTITION variable=strip_read_data complete dim=0
            for (int strip_row = 0; strip_row < 3; ++strip_row) {
#pragma HLS UNROLL
                for (int strip_column = 0; strip_column < 6; ++strip_column) {
#pragma HLS UNROLL
                    // 控制独立于copy/prefetch/input数据分支；每Bank一个显式
                    // 布尔译码，不由嵌套数据分支的phi传播写使能。
                    if (strip_row < 2) {
                        const bool overlap = !is_last_tile && strip_column < 2 &&  // 同行q3/q6保留Top/Middle最后两列。
                            kernel_pos == (strip_row == 0 ? 3 : 6);
                        // 同行q4写下一Strip的四个新列。
                        const bool prefetch = !is_last_tile && kernel_pos == 4 && strip_column >= 2;
                        const bool restart = is_last_tile && !is_last_output_row &&  // 换行q6/q7建立下一行首Strip。
                            ((kernel_pos == 6 && strip_column <= 4) || (kernel_pos == 7 && strip_column == 5));
                        strip_write.enable[strip_row][strip_column] = overlap || prefetch || restart;
                        strip_write.address[strip_row][strip_column] = next_strip_address;
                    } else {
                        const bool load = load_bottom && strip_column == bottom_strip_column;  // 当前Bottom新输入只写对应Strip列。
                        // q8准备下一Strip Bottom，零值也占写口。
                        const bool prepare = kernel_pos == 8 && (!is_last_tile || !is_last_output_row);
                        strip_write.enable[strip_row][strip_column] = load || prepare;
                        strip_write.address[strip_row][strip_column] = load ? bottom_write_address : next_strip_address;
                    }
                    // 每Bank只读一次；未使用行不发请求，避免Bottom准备期间
                    // 无用的RAM读让写使能也陷入跨迭代memory反馈。
                    // 按q门控实际需要的Strip行，避免无用读造成反馈。
                    const bool use_row = strip_row == 0 ? kernel_pos <= 3
                        : strip_row == 1 ? (kernel_pos >= 3 && kernel_pos <= 6)
                                   : kernel_pos >= 6;
                    strip_read_data[strip_row][strip_column] =
                        use_row ? window_strip[strip_row][strip_column][current_strip_address.to_uint()] : word_t(0);
                }
            }
            // Strip输出：spatial_valid只标记输出位置是否存在，窗口Padding不清除它。
            word_t window_words[4];
#pragma HLS ARRAY_PARTITION variable=window_words complete
            output_words(strip_read_data, kernel_pos, window_words);
            activation_pack_t output;
            output.spatial_valid = is_last_tile ? geometry.tail_spatial_valid_mask : 15;
            for (int spatial_lane = 0; spatial_lane < 4; ++spatial_lane) {
#pragma HLS UNROLL
                output.pixel[spatial_lane] =
                    unpack_activation_lane(output.spatial_valid[spatial_lane] ? window_words[spatial_lane] : word_t(0));
            }
            output_stream.write(output);
#ifndef __SYNTHESIS__
            ++writes;
#endif
            // 输入使用独立x/group计数，保持先列后组的stream顺序。
            // 首Tile最多读5G拍，其余最多4G拍；W13末Tile的bottom_input_x与
            // bottom_input_end同为13，不再读新输入，列11/12已通过重叠保留。
            word_t input_word = 0;
            if (load_bottom) {
                input_word = normalized_word(input_stream.read(), bottom_input_group, geometry.cin);
                // 覆盖前旧Top已进入current Strip：前两列来自上一Tile重叠，
                // 其余列来自上一Tile的q4预取，绝不回读已覆盖的旧Top。
                // 新Bottom源列所在四列块的基地址，加独立输入Cin组。
                const line_address_t bottom_address = ((bottom_input_x >> 2) << cin_group_shift) + bottom_input_group;
                // 固定物理Bank的写译码，不让ap索引通过前端展开成
                // 通用动态数组访问；每Bank仍最多一次写，不复制存储。
                for (int physical_row = 0; physical_row < 2; ++physical_row) {
#pragma HLS UNROLL
                    for (int column_bank = 0; column_bank < 4; ++column_bank) {
#pragma HLS UNROLL
                        if (unsigned(physical_row) == top_physical_row &&
                            (bottom_input_x.to_uint() & 3u) == unsigned(column_bank))
                            line_buffer[physical_row][column_bank][bottom_address.to_uint()] = input_word;
                    }
                }
#ifndef __SYNTHESIS__
                ++reads;
#endif
            }
            // 三种预取事件共用每个物理Line Bank的唯一读入口。
            // 目标Strip列c的Line Bank恒为(strip_column-1)&3；q6/q7只改变请求地址。
            word_t line_read_data[2][4];  // Line预取：按物理行与列Bank保存读返回。
#pragma HLS ARRAY_PARTITION variable=line_read_data complete dim=0
            const bool prefetch_next_tile = !is_last_tile && kernel_pos == 4;  // 同行下一Tile：请求4个新列。
            // 换行：先请求真实列0～3。
            const bool prefetch_next_row_prefix = is_last_tile && !is_last_output_row && kernel_pos == 6;
            // 换行：Bank0再请求真实列4。
            const bool prefetch_next_row_last_column = is_last_tile && !is_last_output_row && kernel_pos == 7;
            for (int column_bank = 0; column_bank < 4; ++column_bank) {
#pragma HLS UNROLL
                // 普通预取列按物理Bank轮转排列，覆盖base+5至base+8。
                const coordinate_t normal_x = tile_base_x + (column_bank == 0 ? 8 : 4 + column_bank);
                const bool normal_real = prefetch_next_tile && normal_x < image_width;  // 普通预取只读取真实图像范围。
                // 换行前四列共用四Bank，最后列仅使用Bank0。
                const bool restart_request =
                    prefetch_next_row_prefix || (prefetch_next_row_last_column && column_bank == 0);
                const bool request = normal_real || restart_request;  // 合并互斥预取事件，每物理Bank只发一个请求。
                const line_address_t normal_address = ((normal_x >> 2) << cin_group_shift) + cin_group;
                const line_address_t restart_address =
                    prefetch_next_row_last_column ? line_address_t(cin_groups + cin_group) : line_address_t(cin_group);
                // 互斥预取事件以one-hot掩码合成唯一Line读地址。
                const line_address_t request_address =
                    (normal_address & line_address_t(normal_real ? ~line_address_t(0) : line_address_t(0))) |
                    (restart_address & line_address_t(restart_request ? ~line_address_t(0) : line_address_t(0)));
                for (int physical_row = 0; physical_row < 2; ++physical_row) {
#pragma HLS UNROLL
                    // 首输出行普通预取不可读未初始化旧Top；换行则两行均已就绪。
                    const bool initialized =
                        is_last_tile || output_y != 0 || unsigned(physical_row) == middle_physical_row;
                    line_read_data[physical_row][column_bank] = request && initialized
                        ? line_buffer[physical_row][column_bank][request_address.to_uint()] : word_t(0);
                }
            }
            // 换行后旧Middle成为新Top。
            const unsigned next_top_physical_row = is_last_tile ? middle_physical_row : top_physical_row;
            word_t next_strip_row_data[2][4];  // 路由后按下一Strip的逻辑Top/Middle索引。
#pragma HLS ARRAY_PARTITION variable=next_strip_row_data complete dim=0
            for (int strip_row = 0; strip_row < 2; ++strip_row) {
#pragma HLS UNROLL
                // 选择物理行0的数据，不是选择零值。
                const bool select_physical_row0 =
                    strip_row == 0 ? next_top_physical_row == 0 : next_top_physical_row != 0;
                for (int column_bank = 0; column_bank < 4; ++column_bank) {
#pragma HLS UNROLL
                    next_strip_row_data[strip_row][column_bank] =
                        (line_read_data[0][column_bank] & word_t(select_physical_row0 ? ~word_t(0) : word_t(0))) |
                        (line_read_data[1][column_bank] & word_t(!select_physical_row0 ? ~word_t(0) : word_t(0)));
                }
            }
            for (int strip_row = 0; strip_row < 2; ++strip_row) {
#pragma HLS UNROLL
                const bool copy = !is_last_tile && kernel_pos == (strip_row == 0 ? 3 : 6);  // 仅在本逻辑行的复制时隙保留重叠列。
                for (int strip_column = 0; strip_column < 6; ++strip_column) {
#pragma HLS UNROLL
                    word_t overlap_word = 0;  // 同行重叠来源：当前Strip列4/5。
                    word_t line_word = 0;  // 预取来源：越界或左Padding时保持零。
                    if (strip_column < 2)
                        overlap_word =
                            strip_read_data[strip_row][4 + strip_column] & word_t(copy ? ~word_t(0) : word_t(0));
                    if (strip_column >= 1) {
                        // 只接收本次预取事件对应的目标列。
                        const bool use_line = (prefetch_next_tile && strip_column >= 2) ||
                            (prefetch_next_row_prefix && strip_column <= 4) || (prefetch_next_row_last_column && strip_column == 5);
                        line_word = next_strip_row_data[strip_row][(strip_column - 1) & 3] &
                            word_t(use_line ? ~word_t(0) : word_t(0));
                    }
                    // 每个目的Bank一次data赋值，互斥来源用onehot OR合并。
                    strip_write.data[strip_row][strip_column] = overlap_word | line_word;
                }
            }
            for (int strip_column = 0; strip_column < 6; ++strip_column) {
#pragma HLS UNROLL
                const bool take_input = load_bottom && strip_column == bottom_strip_column;
                word_t copied = 0;
                if (strip_column < 2) {
                    const bool copy = !is_last_tile && kernel_pos == 8;  // 仅在本逻辑行的复制时隙保留重叠列。
                    copied = strip_read_data[2][4 + strip_column] & word_t(copy ? ~word_t(0) : word_t(0));
                }
                // q8保留原始Bank4/5，不穿输出选择器，也不复制masked Payload。
                strip_write.data[2][strip_column] = copied |
                    (input_word & word_t(take_input ? ~word_t(0) : word_t(0)));
            }
            // 先收集全部读返回和目的标签，再每Bank仅一次写入。提交位于
            // 本迭代末尾；真实RAM读延迟与返回寄存器由HLS流水实现。
            commit_strip_writes(window_strip, strip_write);
            // 状态推进：读写命令及其目的地址已收集完，再推进独立输入/输出游标。
            if (load_bottom) {
                if (bottom_input_group == last_cin_group) {
                    bottom_input_group = 0; ++bottom_input_x; ++bottom_strip_column;
                } else ++bottom_input_group;
            }
            if (cin_group == last_cin_group) {
                cin_group = 0;
                if (kernel_pos == 8) {
                    kernel_pos = 0; current_strip_slot = 1 - current_strip_slot;
                    if (is_last_tile) {
                        tile_x = 0; ++output_y; tile_base_x = 0; bottom_input_x = 0;
                        bottom_input_end = first_tile_input_end; bottom_strip_column = 1;
                    } else {
                        ++tile_x; tile_base_x += 4; bottom_input_x = tile_base_x + 1; bottom_strip_column = 2;
                        // 仅Tile切换更新终点。输入握手路径只比较两个9-bit寄存器，
                        // 表内仅末Tile需要截到真实行尾，不做表外尺寸的通用min比较。
                        const coordinate_t candidate = tile_base_x + IM2ROW_FIRST_TILE_INPUT_COLUMNS;  // 下一Tile新Bottom输入的候选右开终点。
                        bottom_input_end = tile_x == last_tile_x ? image_width : candidate;
                    }
                    bottom_input_group = 0;
                } else ++kernel_pos;
            } else ++cin_group;
        }
    }
#ifndef __SYNTHESIS__
    assert(reads == geometry.input_transactions);
    assert(writes == geometry.im2row_tokens);
#endif
}
