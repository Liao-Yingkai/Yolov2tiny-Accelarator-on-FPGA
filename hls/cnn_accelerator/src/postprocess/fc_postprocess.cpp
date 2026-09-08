#include "fc_postprocess.h"

#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace {

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
