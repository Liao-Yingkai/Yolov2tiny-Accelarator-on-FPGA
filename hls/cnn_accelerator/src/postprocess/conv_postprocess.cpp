#include "conv_postprocess.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace {

// 约定与 Python export.py 的 integer_postprocess 一致，alpha 不占参数 BIN 字段。
constexpr int LEAKY_ALPHA_MULTIPLIER = 1717986918;
constexpr int LEAKY_ALPHA_SHIFT = 34;

struct ChannelParams {
    acc_t bias;
    ap_int<32> multiplier_positive;
    ap_int<32> multiplier_negative;
    ap_int<32> shift;
};

ap_int<32> leaky_negative_multiplier(ap_int<32> multiplier) {
#pragma HLS INLINE
    const ap_int<64> product = multiplier * ap_int<32>(LEAKY_ALPHA_MULTIPLIER);
    // 只在准备非负系数时 half-up；最终逐元素右移没有这个舍入项。
    const ap_int<64> rounded = product + (1LL << (LEAKY_ALPHA_SHIFT - 1));
    return ap_int<32>(rounded >> LEAKY_ALPHA_SHIFT);
}

ap_int<64> arithmetic_right_shift(ap_int<64> value, ap_int<32> shift) {
#pragma HLS INLINE
    // 先处理大 shift，再缩窄控制位，避免例如 S=64 回绕成 0。
    if (shift >= 64) return value < 0 ? ap_int<64>(-1) : ap_int<64>(0);
    return value >> ap_uint<6>(shift);
}

acc_t checked_bias_add(acc_t accumulator, acc_t bias) {
#pragma HLS INLINE
    const ap_int<33> wide = ap_int<33>(accumulator) + ap_int<33>(bias);
#ifndef __SYNTHESIS__
    // Exporter/Golden 要求相加结果可表示为 INT32，不在这里做中间饱和。
    assert(wide >= ap_int<33>(-2147483648LL));
    assert(wide <= ap_int<33>(2147483647LL));
#endif
    return acc_t(wide);
}

}  // namespace

activation_t conv_postprocess_requantize(acc_t accumulator,
                                        acc_t bias,
                                        ap_int<32> multiplier_positive,
                                        ap_int<32> multiplier_negative,
                                        ap_int<32> shift) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
    assert(multiplier_positive >= 0 && multiplier_negative >= 0 && shift >= 0);
#endif
    const acc_t biased = checked_bias_add(accumulator, bias);
    // 必须按加 bias 后的符号选系数，且在乘法前选，避免正负支各生成一个乘法器。
    const ap_int<32> multiplier = biased < 0 ? multiplier_negative : multiplier_positive;
    const ap_int<64> product = biased * multiplier;  // signed 32×32，保留完整乘积。
    const ap_int<64> scaled = arithmetic_right_shift(product, shift);
    if (scaled > 127) return activation_t(127);
    if (scaled < -128) return activation_t(-128);
    return activation_t(scaled);
}

void conv_postprocess(hls::stream<yolo_acc_tile_t>& input_stream,
                      hls::stream<postprocess_param_pack_t>& param_stream,
                      hls::stream<yolo_activation_tile_t>& output_stream,
                      const ConvGeometry& geometry) {
    const int block_capacity = geometry.resident_cout_tiles;  // 最多 8 个 4-Cout Tile。
    const int layer_cout_tiles = geometry.cout_tiles;
    const int spatial_tiles = geometry.spatial_tiles;
    ChannelParams param_cache[YOLO_TINY_MAX_RESIDENT_COUT_TILES][SYSTOLIC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=param_cache complete dim=2

#ifndef __SYNTHESIS__
    assert(block_capacity > 0 && block_capacity <= YOLO_TINY_MAX_RESIDENT_COUT_TILES);
    int param_reads = 0;
    int acc_reads = 0;
    int result_writes = 0;
#endif

    // 一次调用消费完整一层。用与 Systolic 相同的块尺寸、空间计数划分流，
    // 无需 Top 每块重调本模块，也不需要在每个 Tile 携带坐标或块编号。
    for (int first_cout_tile = 0; first_cout_tile < layer_cout_tiles;
         first_cout_tile += block_capacity) {
        const int block_tiles = config_min(block_capacity, layer_cout_tiles - first_cout_tile);

        // 当前块独占参数缓存，不预取下一块。
        for (int local_tile = 0; local_tile < block_tiles; ++local_tile) {
            const postprocess_param_pack_t pack = param_stream.read();
            const int first_channel = (first_cout_tile + local_tile) * SYSTOLIC_COUT_PAR;
#ifndef __SYNTHESIS__
            ++param_reads;
#endif
            for (int lane = 0; lane < SYSTOLIC_COUT_PAR; ++lane) {
#pragma HLS PIPELINE II=1
                // 准备阶段按 lane 串行，以复用一条常量乘移路径。Prediction
                // 不计算 Leaky 系数：正负支都缓存原始 M，元素路径无需再判断层模式。
                ChannelParams params{0, 0, 0, 0};
                if (first_channel + lane < geometry.cout) {
                    params.bias = pack.bias[lane];
                    params.multiplier_positive = pack.multiplier[lane];
                    params.shift = pack.shift[lane];
#ifndef __SYNTHESIS__
                    assert(params.multiplier_positive >= 0 && params.shift >= 0);
#endif
                    params.multiplier_negative = geometry.use_leaky_relu
                        ? leaky_negative_multiplier(params.multiplier_positive)
                        : params.multiplier_positive;
                }
                param_cache[local_tile][lane] = params;
            }
        }

        for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
            for (int local_tile = 0; local_tile < block_tiles; ++local_tile) {
                const yolo_acc_tile_t input = input_stream.read();
                yolo_activation_tile_t output;
#pragma HLS ARRAY_PARTITION variable=output.value complete dim=0
                output.spatial_valid = input.spatial_valid;
                output.cout_valid = input.cout_valid;
#ifndef __SYNTHESIS__
                ++acc_reads;
#endif

                // 每个 Tile 读取一次缓存，四个空间 lane 共用这组参数。
                ChannelParams tile_params[SYSTOLIC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=tile_params complete
                for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
#pragma HLS UNROLL
                    tile_params[j] = input.cout_valid[j] ? param_cache[local_tile][j]
                                                         : ChannelParams{0, 0, 0, 0};
                }

                // 一步处理一个空间 lane 的四个 Cout，排满流水后的目标 II=1。
                for (int i = 0; i < SYSTOLIC_SPATIAL_PAR; ++i) {
#pragma HLS PIPELINE II=1
                    for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
#pragma HLS UNROLL
                        if (input.spatial_valid[i] && input.cout_valid[j]) {
                            const ChannelParams& params = tile_params[j];
                            output.value[i][j] = conv_postprocess_requantize(
                                input.value[i][j], params.bias, params.multiplier_positive,
                                params.multiplier_negative, params.shift);
                        } else {
                            // 不让 bias 把无效 lane 的零变成有效的非零输出。
                            output.value[i][j] = 0;
                        }
                    }
                }
                output_stream.write(output);
#ifndef __SYNTHESIS__
                ++result_writes;
#endif
            }
        }
        // 本块最后一个结果提交到流后，下一轮才会读新参数并覆盖缓存。
    }

#ifndef __SYNTHESIS__
    assert(param_reads == layer_cout_tiles);
    assert(acc_reads == spatial_tiles * layer_cout_tiles);
    assert(result_writes == acc_reads);
#endif
}
