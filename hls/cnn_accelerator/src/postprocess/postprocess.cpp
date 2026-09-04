#include "postprocess.h"

#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace {

// 与 NumPy 对 signed integer 的 right_shift 语义一致。当前卷积的 ReLU
// 使输入 product 非负；仍保留负值的符号扩展分支，便于函数数值语义完整。
static ap_int<64> arithmetic_right_shift(ap_int<64> value,
                                         ap_int<32> shift) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
    assert(shift >= 0);
#endif
    if (shift >= 64) {
        return (value < 0) ? (ap_int<64>)-1 : (ap_int<64>)0;
    }

    // shift 已由 exporter 保证非负；截成6 bit只用于硬件移位器控制，不会
    // 改变 0..63 范围内的数值。这里没有 rounding offset。
    const ap_uint<6> shift_amount = (ap_uint<6>)shift;
    return value >> shift_amount;
}

static ap_int<32> checked_bias_add(acc_t accumulator, ap_int<32> bias) {
#pragma HLS INLINE
    const ap_int<33> biased_wide = (ap_int<33>)accumulator +
                                   (ap_int<33>)bias;
#ifndef __SYNTHESIS__
    // 网络参数契约要求 bias add 的结果仍可表示为 INT32。硬件不在这里做
    // 饱和；若违反契约，转换为 INT32 前必须在 C-sim 立即暴露。
    assert(biased_wide >= (ap_int<33>)-2147483648LL);
    assert(biased_wide <= (ap_int<33>)2147483647LL);
#endif
    return (ap_int<32>)biased_wide;
}

}  // namespace

activation_t postprocess_requantize(acc_t accumulator,
                                    ap_int<32> bias,
                                    ap_int<32> multiplier,
                                    ap_int<32> shift) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
    assert(multiplier >= 0);
    assert(shift >= 0);
#endif
    const ap_int<32> biased = checked_bias_add(accumulator, bias);
    const ap_int<32> relu_value = (biased > 0) ? biased : (ap_int<32>)0;
    // ap_int<32> x ap_int<32> 的乘法表达式结果为 ap_int<64>。不要先把
    // 操作数扩展到64 bit，否则会推导为不必要的64x64乘法器。
    const ap_int<64> product = relu_value * multiplier;
    const ap_int<64> scaled = arithmetic_right_shift(product, shift);

    // ReLU 和 non-negative multiplier 保证 scaled 不小于0。仅保留上限
    // 饱和，严格匹配当前量化网络的 INT8 activation 规则。
    return (scaled > 127) ? (activation_t)127 : (activation_t)scaled;
}

void postprocess(hls::stream<acc_tile_t>& input_stream,
                 hls::stream<postprocess_param_pack_t>& param_stream,
                 hls::stream<activation_tile_t>& output_stream,
                 int Cout) {
    const bool is_conv2 = (Cout == CNN_CONV2_COUT);
#ifndef __SYNTHESIS__
    assert(Cout == CNN_CONV1_COUT || Cout == CNN_CONV2_COUT);
#endif
    const int spatial_tiles = is_conv2 ? 56 : 196;
    const int cout_tiles = is_conv2 ? 4 : 2;

    // 参数仅在本层第一个 spatial tile 从 stream 装载。两个维度 complete
    // partition 后为寄存器阵列，后续 Tile 每拍可同时取得4条 Cout Lane。
    ap_int<32> bias_cache[SYSTOLIC_MAX_COUT_TILES][SYSTOLIC_COUT_PAR];
    ap_int<32> multiplier_cache[SYSTOLIC_MAX_COUT_TILES]
                                [SYSTOLIC_COUT_PAR];
    ap_int<32> shift_cache[SYSTOLIC_MAX_COUT_TILES][SYSTOLIC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=bias_cache complete dim=0
#pragma HLS ARRAY_PARTITION variable=multiplier_cache complete dim=0
#pragma HLS ARRAY_PARTITION variable=shift_cache complete dim=0

#ifndef __SYNTHESIS__
    int input_read_count = 0;
    int param_read_count = 0;
    int output_write_count = 0;
#endif

    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const bool first_spatial_tile = (spatial_tile == 0);
        for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
            acc_tile_t input = input_stream.read();
#ifndef __SYNTHESIS__
            ++input_read_count;
#endif

            // 首个 spatial tile 顺序读取每个 Cout tile 的参数，并同时写入
            // 寄存器缓存；后续 tile 不再消耗参数流。
            if (first_spatial_tile) {
                const postprocess_param_pack_t stream_param =
                    param_stream.read();
#ifndef __SYNTHESIS__
                ++param_read_count;
#endif
                for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                     ++cout_lane) {
#pragma HLS UNROLL
#ifndef __SYNTHESIS__
                    assert(stream_param.multiplier[cout_lane] >= 0);
                    assert(stream_param.shift[cout_lane] >= 0);
#endif
                    bias_cache[cout_tile][cout_lane] =
                        stream_param.bias[cout_lane];
                    multiplier_cache[cout_tile][cout_lane] =
                        stream_param.multiplier[cout_lane];
                    shift_cache[cout_tile][cout_lane] =
                        stream_param.shift[cout_lane];
                }
            }

            activation_tile_t output;
#pragma HLS ARRAY_PARTITION variable=output.value complete dim=0
            output.spatial_valid = input.spatial_valid;

            // 每拍完成一个 spatial lane 的4条 Cout lane。cout lane 完全展开，
            // 对应4套并行 INT32xINT32 requant datapath；乘法器由 HLS 根据
            // 目标器件优先映射 DSP，避免绑定到特定工具版本的实现 pragma。
            for (int spatial_lane = 0;
                 spatial_lane < SYSTOLIC_SPATIAL_PAR; ++spatial_lane) {
#pragma HLS PIPELINE II=1
                for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                     ++cout_lane) {
#pragma HLS UNROLL
                    if (input.spatial_valid[spatial_lane]) {
                        output.value[spatial_lane][cout_lane] =
                            postprocess_requantize(
                                input.value[spatial_lane][cout_lane],
                                bias_cache[cout_tile][cout_lane],
                                multiplier_cache[cout_tile][cout_lane],
                                shift_cache[cout_tile][cout_lane]);
                    } else {
                        // invalid lane 不能经过 bias 或 ReLU。
                        output.value[spatial_lane][cout_lane] = 0;
                    }
                }
            }

            output_stream.write(output);
#ifndef __SYNTHESIS__
            ++output_write_count;
#endif
        }
    }

#ifndef __SYNTHESIS__
    assert(input_read_count == spatial_tiles * cout_tiles);
    assert(param_read_count == cout_tiles);
    assert(output_write_count == spatial_tiles * cout_tiles);
#endif
}
