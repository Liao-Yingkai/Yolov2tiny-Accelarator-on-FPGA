#include "softmax.h"
#include <ap_fixed.h>
#include <hls_math.h>
#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace {
using exp_t = ap_fixed<28, 8>;
using exp_raw_t = ap_uint<SOFTMAX_EXP_BITS>;
const unsigned EXP_ONE = 1U << SOFTMAX_SCALE_FRAC_BITS;
static_assert(FC_OUTPUTS == 10 && ACTIVATION_BITS == 8 &&
              SOFTMAX_SCALE_FRAC_BITS == 20 && SOFTMAX_EXP_BITS == 21 &&
              SOFTMAX_SUM_BITS == 24 && SOFTMAX_PROBABILITY_SCALE == 10000,
              "Softmax arithmetic widths require the frozen ten-class Q20 geometry");

struct maximum_candidate_t {
    logit_t value;
    class_index_t index;
};

maximum_candidate_t choose_maximum(maximum_candidate_t left, maximum_candidate_t right) {
#pragma HLS INLINE
    // 左子树索引连续且全部小于右子树，等值选左即可保留first-max。
    return left.value >= right.value ? left : right;
}

// 一个共享的定点指数调用点；实际实例数、II和资源必须以后续综合报告确认。
exp_raw_t exponential(ap_uint<8> delta, ap_uint<32> scale_q20) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    if (delta == 0) return EXP_ONE;
    const ap_uint<40> product = delta * scale_q20;
    if (product >= (16U << SOFTMAX_SCALE_FRAC_BITS)) return 0;
    const ap_int<28> magnitude = product;
    const ap_int<28> negative_bits = -magnitude;
    exp_t x;
    // product 是Q20原始位码，不能作为整数值赋给ap_fixed。
    x.range(27, 0) = negative_bits.range(27, 0);
    const exp_t value = hls::exp(x);
    if (value <= 0) return 0;
    if (value >= 1) return EXP_ONE;
    return exp_raw_t(value.range(20, 0));
}

probability_t normalize(exp_raw_t value, ap_uint<SOFTMAX_SUM_BITS> sum) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    // 21x14乘积最大2^20*10000，小于2^34；保留必要宽度。
    const ap_uint<34> numerator = value * ap_uint<14>(SOFTMAX_PROBABILITY_SCALE);
    const ap_uint<14> quotient = numerator / sum;
    const ap_uint<38> used = quotient * sum;
    // 数学余数严格小于24位sum；翻倍比较需要25位。
    const ap_uint<24> remainder = numerator - used;
    const ap_uint<25> twice_remainder = ap_uint<25>(remainder) << 1;
    ap_uint<15> rounded = quotient;
    // 最近偶数舍入；余数由乘减计算，避免额外的取模除法器。
    if (twice_remainder > sum ||
        (twice_remainder == sum && quotient[0])) ++rounded;
    if (rounded > SOFTMAX_PROBABILITY_SCALE) rounded = SOFTMAX_PROBABILITY_SCALE;
    return probability_t(rounded);
}
} // namespace

void softmax(hls::stream<logits_pack_t>& input,
             hls::stream<inference_result_t>& output,
             ap_uint<32> scale_q20) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    // 有效输入契约；不为硬件新增错误端口或控制分支。
    assert(scale_q20 > 0);
#endif
    const logits_pack_t logits = input.read();
#pragma HLS ARRAY_PARTITION variable=logits.value complete
    inference_result_t result;
#pragma HLS ARRAY_PARTITION variable=result.probability complete
    maximum_candidate_t maximum0[10], maximum1[5], maximum2[3], maximum3[2];
#pragma HLS ARRAY_PARTITION variable=maximum0 complete
#pragma HLS ARRAY_PARTITION variable=maximum1 complete
#pragma HLS ARRAY_PARTITION variable=maximum2 complete
#pragma HLS ARRAY_PARTITION variable=maximum3 complete
    // 显式10→5→3→2→1树；仅展开便宜的比较，不展开指数或除法。
    for (int i = 0; i < 10; ++i) {
#pragma HLS UNROLL
        maximum0[i].value = logits.value[i];
        maximum0[i].index = i;
    }
    for (int i = 0; i < 5; ++i) {
#pragma HLS UNROLL
        maximum1[i] = choose_maximum(maximum0[2 * i], maximum0[2 * i + 1]);
    }
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        maximum2[i] = choose_maximum(maximum1[2 * i], maximum1[2 * i + 1]);
    }
    maximum2[2] = maximum1[4];
    maximum3[0] = choose_maximum(maximum2[0], maximum2[1]);
    maximum3[1] = maximum2[2];
    const maximum_candidate_t maximum = choose_maximum(maximum3[0], maximum3[1]);
    result.class_index = maximum.index;
    exp_raw_t values[FC_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=values complete
    // 指数先独立写数组，移除sum的循环携带依赖；单一helper调用位置不展开。
    for (int i = 0; i < FC_OUTPUTS; ++i) {
#pragma HLS PIPELINE II=1
        const ap_int<9> signed_delta = ap_int<9>(maximum.value) - ap_int<9>(logits.value[i]);
        values[i] = exponential(ap_uint<8>(signed_delta), scale_q20);
    }
    ap_uint<22> sum2[5];
    ap_uint<23> sum4[2];
#pragma HLS ARRAY_PARTITION variable=sum2 complete
#pragma HLS ARRAY_PARTITION variable=sum4 complete
    for (int i = 0; i < 5; ++i) {
#pragma HLS UNROLL
        sum2[i] = ap_uint<22>(values[2 * i]) + ap_uint<22>(values[2 * i + 1]);
    }
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        sum4[i] = ap_uint<23>(sum2[2 * i]) + ap_uint<23>(sum2[2 * i + 1]);
    }
    const ap_uint<24> sum8 = ap_uint<24>(sum4[0]) + ap_uint<24>(sum4[1]);
    const ap_uint<SOFTMAX_SUM_BITS> sum = sum8 + ap_uint<24>(sum2[4]);
    // 至少一个delta为零，因此sum始终>=2^20，除数不会为零。
    for (int i = 0; i < FC_OUTPUTS; ++i) {
#pragma HLS PIPELINE II=1
        result.probability[i] = normalize(values[i], sum);
    }
    output.write(result);
}
