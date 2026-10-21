#include "fc.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace {

using fc_activation_word_t = ap_uint<FC_ACTIVATION_WORD_BITS>;

// 保留 activation 的二进制补码位型；Pool2 channel_pack_t 的 channel[0..7]
// 就是当前 NHWC token 的连续8个通道。
fc_activation_word_t pack_activation(const channel_pack_t& input) {
#pragma HLS INLINE
    fc_activation_word_t packed = 0;
    for (int cin_lane = 0; cin_lane < FC_CIN_PAR; ++cin_lane) {
#pragma HLS UNROLL
        const int low_bit = cin_lane * ACTIVATION_BITS;
        packed.range(low_bit + ACTIVATION_BITS - 1, low_bit) =
            input.channel[cin_lane].range(ACTIVATION_BITS - 1, 0);
    }
    return packed;
}

activation_t unpack_activation(const fc_activation_word_t& packed,
                               int cin_lane) {
#pragma HLS INLINE
    activation_t result;
    const int low_bit = cin_lane * ACTIVATION_BITS;
    result.range(ACTIVATION_BITS - 1, 0) =
        packed.range(low_bit + ACTIVATION_BITS - 1, low_bit);
    return result;
}

// 固定8项INT8 dot的4层平衡加法树。中间位宽均从PRODUCT_BITS派生，避免
// 工具将链式加法改写为不必要的长进位路径。
ap_int<FC_DOT_BITS> dot8(const fc_activation_word_t& activation_word,
                         const fc_weight_pack_t& weights,
                         int cout_lane) {
#pragma HLS INLINE
    ap_int<PRODUCT_BITS> product[FC_CIN_PAR];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
    for (int cin_lane = 0; cin_lane < FC_CIN_PAR; ++cin_lane) {
#pragma HLS UNROLL
        product[cin_lane] = unpack_activation(activation_word, cin_lane) *
                            weights.value[cout_lane][cin_lane];
    }

    const ap_int<FC_SUM2_BITS> sum2_0 =
        (ap_int<FC_SUM2_BITS>)product[0] + product[1];
    const ap_int<FC_SUM2_BITS> sum2_1 =
        (ap_int<FC_SUM2_BITS>)product[2] + product[3];
    const ap_int<FC_SUM2_BITS> sum2_2 =
        (ap_int<FC_SUM2_BITS>)product[4] + product[5];
    const ap_int<FC_SUM2_BITS> sum2_3 =
        (ap_int<FC_SUM2_BITS>)product[6] + product[7];
    const ap_int<FC_SUM4_BITS> sum4_0 =
        (ap_int<FC_SUM4_BITS>)sum2_0 + sum2_1;
    const ap_int<FC_SUM4_BITS> sum4_1 =
        (ap_int<FC_SUM4_BITS>)sum2_2 + sum2_3;
    return (ap_int<FC_DOT_BITS>)sum4_0 + sum4_1;
}

}  // namespace

void fully_connected(hls::stream<channel_pack_t>& input_stream,
                     hls::stream<fc_weight_pack_t>& weight_stream,
                     hls::stream<fc_acc_pack_t>& output_stream) {
    // FC_ACTIVATION_BUFFER_DEPTH x FC_ACTIVATION_WORD_BITS bit。只在第一个out tile写，第二个out tile
    // 只读，因此一个 ram_1p BRAM 足以承担该回放缓存。
    fc_activation_word_t activation_buffer[FC_ACTIVATION_BUFFER_DEPTH];
#pragma HLS BIND_STORAGE variable=activation_buffer type=ram_1p impl=bram

#ifndef __SYNTHESIS__
    int input_read_count = 0;
    int weight_read_count = 0;
    int output_write_count = 0;
#endif

    // tile0: logit0-4, tile1: logit5-9
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        acc_t accumulator[FC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1

        for (int token = 0; token < FC_INPUT_PACKS; ++token) {
#pragma HLS PIPELINE II=1
            fc_activation_word_t activation_word;
            if (out_tile == 0) {
                // 第一Tile直接使用当前stream token计算，并显式写入一个完整
                // FC activation word；不能要求上游为第二Tile重放输入事务。
                const channel_pack_t input = input_stream.read();
                activation_word = pack_activation(input);
                activation_buffer[token] = activation_word;
#ifndef __SYNTHESIS__
                ++input_read_count;
#endif
            } else {
                activation_word = activation_buffer[token];
            }

            const fc_weight_pack_t weights = weight_stream.read();
#pragma HLS ARRAY_PARTITION variable=weights.value complete dim=0
#ifndef __SYNTHESIS__
            ++weight_read_count;
#endif
            for (int cout_lane = 0; cout_lane < FC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
                const ap_int<FC_DOT_BITS> dot = dot8(activation_word, weights,
                                                     cout_lane);
                // token0 覆盖旧值，避免跨 out tile 或跨函数调用残留累加器状态。
                if (token == 0) {
                    accumulator[cout_lane] = dot;
                } else {
                    accumulator[cout_lane] += dot;
                }
            }
        }

        fc_acc_pack_t output;
#pragma HLS ARRAY_PARTITION variable=output.value complete dim=1


        for (int cout_lane = 0; cout_lane < FC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
            output.value[cout_lane] = accumulator[cout_lane];
        }
        output_stream.write(output);
#ifndef __SYNTHESIS__
        ++output_write_count;
#endif
    }

#ifndef __SYNTHESIS__
    assert(input_read_count == FC_INPUT_PACKS);
    assert(weight_read_count == FC_OUT_TILES * FC_INPUT_PACKS);
    assert(output_write_count == FC_OUT_TILES);
#endif
}
