#include "fc.h"

#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace {

typedef ap_uint<FC_CIN_PAR * ACTIVATION_BITS> fc_activation_word_t;

// 保留 activation 的二进制补码位型；Pool2 channel_pack_t 的 channel[0..7]
// 就是当前 NHWC token 的连续8个通道。
static fc_activation_word_t pack_activation(const channel_pack_t& input) {
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

static activation_t unpack_activation(const fc_activation_word_t& packed,
                                      int cin_lane) {
#pragma HLS INLINE
    activation_t result;
    const int low_bit = cin_lane * ACTIVATION_BITS;
    result.range(ACTIVATION_BITS - 1, 0) =
        packed.range(low_bit + ACTIVATION_BITS - 1, low_bit);
    return result;
}

// 固定8项 INT8 dot 的4层平衡加法树。明确的16/17/18/19位中间值避免
// 工具将链式加法改写为不必要的长进位路径。
static ap_int<19> dot8(const fc_activation_word_t& activation_word,
                       const fc_weight_pack_t& weights,
                       int cout_lane) {
#pragma HLS INLINE
    ap_int<16> product[FC_CIN_PAR];
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
    for (int cin_lane = 0; cin_lane < FC_CIN_PAR; ++cin_lane) {
#pragma HLS UNROLL
        product[cin_lane] = unpack_activation(activation_word, cin_lane) *
                            weights.value[cout_lane][cin_lane];
    }

    const ap_int<17> sum17_0 = (ap_int<17>)product[0] + product[1];
    const ap_int<17> sum17_1 = (ap_int<17>)product[2] + product[3];
    const ap_int<17> sum17_2 = (ap_int<17>)product[4] + product[5];
    const ap_int<17> sum17_3 = (ap_int<17>)product[6] + product[7];
    const ap_int<18> sum18_0 = (ap_int<18>)sum17_0 + sum17_1;
    const ap_int<18> sum18_1 = (ap_int<18>)sum17_2 + sum17_3;
    return (ap_int<19>)sum18_0 + sum18_1;
}

static ap_int<64> arithmetic_right_shift(ap_int<64> value,
                                         ap_int<32> shift) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
    assert(shift >= 0);
#endif
    // NumPy signed integer right_shift 对 >=64 的效果是符号扩展，而不是将
    // 移位量截断为6 bit。
    if (shift >= 64) {
        return (value < 0) ? (ap_int<64>)-1 : (ap_int<64>)0;
    }
    return value >> (ap_uint<6>)shift;
}

static ap_int<32> checked_bias_add(acc_t accumulator, ap_int<32> bias) {
#pragma HLS INLINE
    const ap_int<33> biased_wide = (ap_int<33>)accumulator +
                                   (ap_int<33>)bias;
#ifndef __SYNTHESIS__
    assert(biased_wide >= (ap_int<33>)-2147483648LL);
    assert(biased_wide <= (ap_int<33>)2147483647LL);
#endif
    return (ap_int<32>)biased_wide;
}

}  // namespace

void fully_connected(hls::stream<channel_pack_t>& input_stream,
                     hls::stream<fc_weight_pack_t>& weight_stream,
                     hls::stream<fc_acc_pack_t>& output_stream) {
    // 98 x 64 bit = 784 bytes。只在第一个 out tile 写，第二个 out tile
    // 只读，因此一个 ram_1p BRAM 足以承担该回放缓存。
    ap_uint<64> activation_buffer[FC_INPUT_PACKS];
#pragma HLS BIND_STORAGE variable=activation_buffer type=ram_1p impl=bram

#ifndef __SYNTHESIS__
    int input_read_count = 0;
    int weight_read_count = 0;
    int output_write_count = 0;
#endif

    //tile0: logit0-4, tile1: logit 5-8
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        acc_t accumulator[FC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=1

        for (int token = 0; token < FC_INPUT_PACKS; ++token) {
#pragma HLS PIPELINE II=1
            fc_activation_word_t activation_word;
            if (out_tile == 0) {
                // 第一 Tile 直接使用当前 stream token 计算，并显式写入64-bit
                // Buffer 字；不能要求上游为第二 Tile 重放98个输入事务。
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
                const ap_int<19> dot = dot8(activation_word, weights,
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

logit_t fc_requantize(acc_t accumulator,
                      ap_int<32> bias,
                      ap_int<32> multiplier,
                      ap_int<32> shift) {
#ifndef __SYNTHESIS__
    // Exporter 的固定量化参数契约：multiplier 和 shift 都非负。
    assert(multiplier >= 0);
    assert(shift >= 0);
#endif
    const ap_int<32> biased = checked_bias_add(accumulator, bias);
    // 不做 ReLU：分类 logit 必须保留符号。操作数保持32位，使乘法器是
    // signed 32x32 -> signed 64，而非由显式扩展意外推导为64x64。
    const ap_int<64> product = biased * multiplier;
    const ap_int<64> scaled = arithmetic_right_shift(product, shift);
    if (scaled > 127) {
        return (logit_t)127;
    }
    if (scaled < -128) {
        return (logit_t)-128;
    }
    return (logit_t)scaled;
}

void fc_postprocess(hls::stream<fc_acc_pack_t>& input_stream,
                    hls::stream<fc_param_pack_t>& param_stream,
                    hls::stream<logits_pack_t>& output_stream) {
    logits_pack_t logits;
#pragma HLS ARRAY_PARTITION variable=logits.value complete dim=1

#ifndef __SYNTHESIS__
    int input_read_count = 0;
    int param_read_count = 0;
    int output_write_count = 0;
#endif

    // 故意不 UNROLL 该 lane loop，也不对 fc_requantize 使用 INLINE pragma：
    // 每次迭代只有一次32x32乘法，表达单套 requant datapath 的时间复用。
    // 实际 DSP 个数与该循环的II仍须以后续 HLS 综合报告为准。
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        const fc_acc_pack_t accumulator = input_stream.read();
        const fc_param_pack_t param = param_stream.read();
#ifndef __SYNTHESIS__
        ++input_read_count;
        ++param_read_count;
#endif
        for (int cout_lane = 0; cout_lane < FC_COUT_PAR; ++cout_lane) {
#pragma HLS PIPELINE II=1
            logits.value[out_tile * FC_COUT_PAR + cout_lane] =
                fc_requantize(accumulator.value[cout_lane],
                              param.bias[cout_lane],
                              param.multiplier[cout_lane],
                              param.shift[cout_lane]);
        }
    }

    output_stream.write(logits);
#ifndef __SYNTHESIS__
    ++output_write_count;
    assert(input_read_count == FC_OUT_TILES);
    assert(param_read_count == FC_OUT_TILES);
    assert(output_write_count == 1);
#endif
}
