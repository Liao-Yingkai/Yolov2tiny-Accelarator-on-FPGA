#include "pool.h"

#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace {

// 每个输入事务有4个 spatial lane 和4个 Cout lane，每个元素为INT8，故一个
// Line Buffer 字为128 bit。固定布局为 lane-major / cout-lane-minor：
// word[(spatial_lane * 4 + cout_lane) * 8 +: 8] = value[spatial_lane][cout_lane]。
// 写入和读取都复制INT8二进制补码位型；解包后再恢复为 activation_t 做有符号比较。
static const int POOL_LINE_WORD_BITS =
    SYSTOLIC_SPATIAL_PAR * SYSTOLIC_COUT_PAR * ACTIVATION_BITS;
typedef ap_uint<POOL_LINE_WORD_BITS> pool_line_word_t;

static const int POOL_MAX_TILES_PER_ROW = 7;
static const int POOL_MAX_COUT_TILES = 4;

pool_line_word_t pack_line_word(const activation_tile_t& input) {
#pragma HLS INLINE
    pool_line_word_t packed = 0;
    for (int spatial_lane = 0; spatial_lane < SYSTOLIC_SPATIAL_PAR;
         ++spatial_lane) {
#pragma HLS UNROLL
        for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
             ++cout_lane) {
#pragma HLS UNROLL
            const int low_bit =
                (spatial_lane * SYSTOLIC_COUT_PAR + cout_lane) *
                ACTIVATION_BITS;
            const int high_bit = low_bit + ACTIVATION_BITS - 1;
            packed.range(high_bit, low_bit) =
                input.value[spatial_lane][cout_lane].range(
                    ACTIVATION_BITS - 1, 0);
        }
    }
    return packed;
}

activation_t unpack_line_value(const pool_line_word_t& packed,
                               int spatial_lane,
                               int cout_lane) {
#pragma HLS INLINE
    const int low_bit =
        (spatial_lane * SYSTOLIC_COUT_PAR + cout_lane) * ACTIVATION_BITS;
    const int high_bit = low_bit + ACTIVATION_BITS - 1;
    activation_t value;
    value.range(ACTIVATION_BITS - 1, 0) = packed.range(high_bit, low_bit);
    return value;
}

activation_t signed_max2(activation_t left, activation_t right) {
#pragma HLS INLINE
    return (left > right) ? left : right;
}

// 以平衡树完成四个INT8值的有符号最大值；不能比较原始 ap_uint 位型。
activation_t signed_max4(activation_t top_left,
                         activation_t top_right,
                         activation_t bottom_left,
                         activation_t bottom_right) {
#pragma HLS INLINE
    const activation_t top_max = signed_max2(top_left, top_right);
    const activation_t bottom_max = signed_max2(bottom_left, bottom_right);
    return signed_max2(top_max, bottom_max);
}

ap_uint<32> make_pooled_half(const pool_line_word_t& top_word,
                             const activation_tile_t& bottom_tile,
                             int first_spatial_lane) {
#pragma HLS INLINE
    ap_uint<32> half = 0;
    for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
        const activation_t pooled = signed_max4(
            unpack_line_value(top_word, first_spatial_lane, cout_lane),
            unpack_line_value(top_word, first_spatial_lane + 1, cout_lane),
            bottom_tile.value[first_spatial_lane][cout_lane],
            bottom_tile.value[first_spatial_lane + 1][cout_lane]);
        const int low_bit = cout_lane * ACTIVATION_BITS;
        const int high_bit = low_bit + ACTIVATION_BITS - 1;
        half.range(high_bit, low_bit) = pooled.range(ACTIVATION_BITS - 1, 0);
    }
    return half;
}

channel_pack_t make_channel_pack(ap_uint<32> low_half, ap_uint<32> high_half) {
#pragma HLS INLINE
    channel_pack_t pack;
    for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
#pragma HLS UNROLL
        const ap_uint<32> source = (channel < SYSTOLIC_COUT_PAR) ?
            low_half : high_half;
        const int local_channel =
            (channel < SYSTOLIC_COUT_PAR) ? channel :
                                            channel - SYSTOLIC_COUT_PAR;
        const int low_bit = local_channel * ACTIVATION_BITS;
        const int high_bit = low_bit + ACTIVATION_BITS - 1;
        pack.channel[channel].range(ACTIVATION_BITS - 1, 0) =
            source.range(high_bit, low_bit);
    }
    return pack;
}

// 每个输入空间 tile 产生两个横向的2x2窗口。Pool2末tile仅lane0/1有效，
// 因而只产生position0。
ap_uint<2> make_pool_valid(ap_uint<SYSTOLIC_SPATIAL_PAR> top_valid,
                           ap_uint<SYSTOLIC_SPATIAL_PAR> bottom_valid) {
#pragma HLS INLINE
    ap_uint<2> valid = 0;
    valid[0] = top_valid[0] && top_valid[1] &&
               bottom_valid[0] && bottom_valid[1];
    valid[1] = top_valid[2] && top_valid[3] &&
               bottom_valid[2] && bottom_valid[3];
    return valid;
}

}  // namespace

void maxpool(hls::stream<activation_tile_t>& input_stream,
             hls::stream<channel_pack_t>& output_stream,
             int H,
             int W,
             int Cout) {
    const bool is_pool2 = (Cout == CNN_CONV2_COUT);
#ifndef __SYNTHESIS__
    assert((H == CNN_CONV1_H && W == CNN_CONV1_W &&
            Cout == CNN_CONV1_COUT) ||
           (H == CNN_CONV2_H && W == CNN_CONV2_W &&
            Cout == CNN_CONV2_COUT));
#endif
    // 综合态只保留Cout到固定Pool模式的一位选择；H/W不进入核心调度。
    const int feature_height = is_pool2 ? CNN_CONV2_H : CNN_CONV1_H;
    const int tiles_per_row = is_pool2 ? 4 : 7;
    const int cout_tiles = is_pool2 ? 4 : 2;

    // 地址按每行输入事务顺序递增。Pool1使用地址0..13，Pool2使用
    // 地址0..15；偶数行只写、随后的奇数行只读，因此
    // 单端口BRAM在一个控制分支内不会有并发读写需求。
    pool_line_word_t line_buffer[16];
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_1p impl=bram

    // line_valid 只保存上一偶数行各空间tile的4-bit valid；同一tile所有
    // Cout transaction 的mask必须一致，非综合态立即检查这一流协议。
    ap_uint<SYSTOLIC_SPATIAL_PAR> line_valid[POOL_MAX_TILES_PER_ROW];
#pragma HLS ARRAY_PARTITION variable=line_valid complete dim=1

    // [pooled position][Cout tile]，每个half含该Cout tile的4个pool结果。
    // 完全分区后每拍可独立写两个half，避免64-bit连续read-modify-write依赖。
    ap_uint<32> pooled_half[2][POOL_MAX_COUT_TILES];
#pragma HLS ARRAY_PARTITION variable=pooled_half complete dim=0

#ifndef __SYNTHESIS__
    int input_read_count = 0;
    int output_write_count = 0;
#endif

    for (int row = 0; row < feature_height; ++row) {
        if ((row & 1) == 0) {
            // 偶数行只消费输入并写上一行缓存，严格不读取该single-port BRAM。
            int line_address = 0;
            for (int tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
                for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
#pragma HLS PIPELINE II=1
                    const activation_tile_t input = input_stream.read();
#ifndef __SYNTHESIS__
                    ++input_read_count;
#endif
                    if (cout_tile == 0) {
                        line_valid[tile_x] = input.spatial_valid;
                    } else {
#ifndef __SYNTHESIS__
                        assert(input.spatial_valid == line_valid[tile_x]);
#endif
                    }
                    line_buffer[line_address] = pack_line_word(input);
                    ++line_address;
                }
            }
        } else {
            // 奇数行只读取上一偶数行缓存。每一个Cout transaction产生两个
            // 32-bit half；全部Cout tile完成后才按position -> pack输出。
            int line_address = 0;
            for (int tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
                ap_uint<SYSTOLIC_SPATIAL_PAR> bottom_valid = 0;
                ap_uint<2> pool_valid = 0;
                for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
#pragma HLS PIPELINE II=1
                    const activation_tile_t bottom = input_stream.read();
                    const pool_line_word_t top = line_buffer[line_address];
#ifndef __SYNTHESIS__
                    ++input_read_count;
#endif
                    if (cout_tile == 0) {
                        bottom_valid = bottom.spatial_valid;
#ifndef __SYNTHESIS__
                        assert(bottom_valid == line_valid[tile_x]);
#endif
                        pool_valid = make_pool_valid(line_valid[tile_x],
                                                     bottom_valid);
                    } else {
#ifndef __SYNTHESIS__
                        assert(bottom.spatial_valid == bottom_valid);
#endif
                    }
                    pooled_half[0][cout_tile] = make_pooled_half(top, bottom, 0);
                    pooled_half[1][cout_tile] = make_pooled_half(top, bottom, 2);
                    ++line_address;
                }

                // Cout%8是pack完整性的实际固定约束。Pool1有一个pack，Pool2
                // 有两个；对每个有效pooled位置，pack顺序总是低Cout组在前。
                const int channel_packs = is_pool2 ? 2 : 1;
                for (int pooled_position = 0; pooled_position < 2;
                     ++pooled_position) {
                    if (pool_valid[pooled_position]) {
                        for (int pack_index = 0; pack_index < channel_packs;
                             ++pack_index) {
                            const int low_cout_tile = pack_index * 2;
                            const channel_pack_t output = make_channel_pack(
                                pooled_half[pooled_position][low_cout_tile],
                                pooled_half[pooled_position]
                                           [low_cout_tile + 1]);
                            output_stream.write(output);
#ifndef __SYNTHESIS__
                            ++output_write_count;
#endif
                        }
                    }
                }
            }
        }
    }

#ifndef __SYNTHESIS__
    const int expected_input_tiles = is_pool2 ? 224 : 392;
    const int expected_output_packs = is_pool2 ? 98 : 196;
    assert(input_read_count == expected_input_tiles);
    assert(output_write_count == expected_output_packs);
#endif
}
