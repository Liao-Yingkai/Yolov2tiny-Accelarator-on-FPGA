#include "systolic.h"
#include "../channel_pack.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace {

using product_t = ap_int<PRODUCT_BITS>;
using sum2_t = ap_int<SYSTOLIC_SUM2_BITS>;
using sum4_t = ap_int<SYSTOLIC_SUM4_BITS>;
using dot_t = ap_int<SYSTOLIC_DOT_BITS>;

// 使用显式平衡树，避免依赖累加表达式的默认位宽或综合器将其解释为不平衡链。
// 8 个 signed INT8 乘积的精确和需要至少19位有符号数。
dot_t dot_product(const pixel_t& activation,
                  const weight_lane_t& weight, int logical_cin) {
#pragma HLS INLINE
    product_t product0 =
        (product_t)(activation.channel[0] * weight.channel[0]);
    // 逻辑 Cin 由计算模块自身保证：Conv1 无效通道即使双侧非零也不能贡献。
    product_t product1 =
        (1 < logical_cin) ? (product_t)(activation.channel[1] * weight.channel[1]) : product_t(0);
    product_t product2 =
        (2 < logical_cin) ? (product_t)(activation.channel[2] * weight.channel[2]) : product_t(0);
    product_t product3 =
        (3 < logical_cin) ? (product_t)(activation.channel[3] * weight.channel[3]) : product_t(0);
    product_t product4 =
        (4 < logical_cin) ? (product_t)(activation.channel[4] * weight.channel[4]) : product_t(0);
    product_t product5 =
        (5 < logical_cin) ? (product_t)(activation.channel[5] * weight.channel[5]) : product_t(0);
    product_t product6 =
        (6 < logical_cin) ? (product_t)(activation.channel[6] * weight.channel[6]) : product_t(0);
    product_t product7 =
        (7 < logical_cin) ? (product_t)(activation.channel[7] * weight.channel[7]) : product_t(0);

    sum2_t sum01 = (sum2_t)product0 + (sum2_t)product1;
    sum2_t sum23 = (sum2_t)product2 + (sum2_t)product3;
    sum2_t sum45 = (sum2_t)product4 + (sum2_t)product5;
    sum2_t sum67 = (sum2_t)product6 + (sum2_t)product7;
    sum4_t sum0123 = (sum4_t)sum01 + (sum4_t)sum23;
    sum4_t sum4567 = (sum4_t)sum45 + (sum4_t)sum67;
    return (dot_t)((dot_t)sum0123 + (dot_t)sum4567);
}

}  // namespace

void systolic(hls::stream<activation_pack_t>& activation_stream,
             hls::stream<weight_pack_t>& weight_stream,
             hls::stream<acc_tile_t>& output_stream,
             layer_mode_t layer_mode) {
    const ConvGeometry geometry = conv_geometry(layer_mode);
    // 同一个物理PE阵列由一位调用级模式在Conv1、Conv2之间分时复用。
    const int spatial_tiles = geometry.spatial_tiles;
    const int cout_tiles = geometry.cout_tiles;
    const int activation_replay_used = geometry.reduction_steps;

    // 四个独立64-bit Cout Lane Bank。首个 spatial tile 只写入，后续 Tile
    // 只读取；ram_1p 因此足够。综合时必须确认这些 Bank 映射到 BRAM。
    packed_weight_lane_t
        weight_bank[SYSTOLIC_COUT_PAR][SYSTOLIC_WEIGHT_BUFFER_DEPTH];
#pragma HLS ARRAY_PARTITION variable=weight_bank complete dim=1
#pragma HLS BIND_STORAGE variable=weight_bank type=ram_1p impl=bram

    // 四个独立64-bit spatial Lane Bank。每字是一个 pixel_t 的8个通道；首个
    // Cout Tile 只写入，后续 Cout Tile 只回放。lane 维完整分区确保每拍可从
    // 4个物理 Bank 同时取得一个完整 activation token（256 bit/cycle）。
    // spatial_valid 在同一 Tile 的9个 token 内不变，单独保存在
    // tile_spatial_valid，不进入 replay BRAM。
    packed_activation_lane_t
        activation_bank[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_ACTIVATION_REPLAY_CAPACITY];
#pragma HLS ARRAY_PARTITION variable=activation_bank complete dim=1
#pragma HLS BIND_STORAGE variable=activation_bank type=ram_1p impl=bram

    acc_t accumulator[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=0

    // 4条 Cout Lane 在同一周期广播给全部 spatial PE 行，必须完整分区。
    weight_lane_t active_weight[SYSTOLIC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=active_weight complete dim=1

#ifndef __SYNTHESIS__
    int activation_read_count = 0;
    int weight_read_count = 0;
    int result_write_count = 0;
#endif

    // spatial_tile 与 Im2Row 的 row -> tile_x 顺序完全等价：Conv1 共有196个，
    // Conv2 共有56个。合并计数器不改变 activation 或 result 的事务顺序。
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const bool first_spatial_tile = (spatial_tile == 0);
        ap_uint<SYSTOLIC_SPATIAL_PAR> tile_spatial_valid = 0;
        ap_uint<SYSTOLIC_WEIGHT_ADDRESS_BITS> weight_base = 0;

        for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
            for (int token = 0; token < activation_replay_used;
                 ++token) {
#pragma HLS PIPELINE II=1
                activation_pack_t activation;
                const ap_uint<SYSTOLIC_WEIGHT_ADDRESS_BITS> weight_address = weight_base + token;

                // 每个 spatial Tile 仅在第一个 Cout Tile 消费9个新的
                // Im2Row token。其余 Cout Tile 通过本地缓冲回放。
                if (cout_tile == 0) {
                    activation = activation_stream.read();
#ifndef __SYNTHESIS__
                    ++activation_read_count;
#endif
                    for (int spatial_lane = 0;
                         spatial_lane < SYSTOLIC_SPATIAL_PAR;
                         ++spatial_lane) {
#pragma HLS UNROLL
                        activation_bank[spatial_lane][token] =
                            pack_activation_lane(
                                activation.pixel[spatial_lane]);
                    }

                    if (token == 0) {
                        tile_spatial_valid = activation.spatial_valid;
                    } else {
#ifndef __SYNTHESIS__
                        assert(activation.spatial_valid ==
                               tile_spatial_valid);
#endif
                    }
                } else {
                    activation.spatial_valid = tile_spatial_valid;
                    for (int spatial_lane = 0;
                         spatial_lane < SYSTOLIC_SPATIAL_PAR;
                         ++spatial_lane) {
#pragma HLS UNROLL
                        activation.pixel[spatial_lane] =
                            unpack_activation_lane(
                                activation_bank[spatial_lane][token]);
                    }
                }

                // 整层只在第一个 spatial Tile 读取权重流，计算本拍的
                // 同时写入4个64-bit BRAM Bank；其余 Tile 从 Bank 回放。
                if (first_spatial_tile) {
                    const weight_pack_t stream_weight = weight_stream.read();
#ifndef __SYNTHESIS__
                    ++weight_read_count;
#endif
                    for (int cout_lane = 0;
                         cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
                        active_weight[cout_lane] =
                            stream_weight.lane[cout_lane];
                        weight_bank[cout_lane][weight_address] =
                            pack_weight_lane(stream_weight.lane[cout_lane]);
                    }
                } else {
                    for (int cout_lane = 0;
                         cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
                        active_weight[cout_lane] = unpack_weight_lane(
                            weight_bank[cout_lane][weight_address]);
                    }
                }

                // 4x4 PE Array：物理8Cin，Conv1仅c0贡献，Conv2全8路。token0以
                // dot覆盖旧 accumulator，因此没有额外清零周期。
                for (int spatial_lane = 0;
                     spatial_lane < SYSTOLIC_SPATIAL_PAR;
                     ++spatial_lane) {
#pragma HLS UNROLL
                    for (int cout_lane = 0;
                         cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
                        const dot_t dot = dot_product(
                            activation.pixel[spatial_lane],
                            active_weight[cout_lane], geometry.cin);
                        if (token == 0) {
                            accumulator[spatial_lane][cout_lane] =
                                (acc_t)dot;
                        } else {
                            accumulator[spatial_lane][cout_lane] =
                                accumulator[spatial_lane][cout_lane] +
                                (acc_t)dot;
                        }
                    }
                }
            }

            // token8 的累加已经写回 accumulator；本次 stream write 若阻塞，
            // 外层控制器不会推进到下一 Cout Tile 或消费下一 activation。
            acc_tile_t result;
            result.spatial_valid = tile_spatial_valid;
            for (int spatial_lane = 0;
                 spatial_lane < SYSTOLIC_SPATIAL_PAR; ++spatial_lane) {
#pragma HLS UNROLL
                for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                     ++cout_lane) {
#pragma HLS UNROLL
                    result.value[spatial_lane][cout_lane] =
                        accumulator[spatial_lane][cout_lane];
                }
            }
            output_stream.write(result);
#ifndef __SYNTHESIS__
            ++result_write_count;
#endif
            weight_base += SYSTOLIC_KERNEL_TOKENS;
        }
    }

#ifndef __SYNTHESIS__
    assert(activation_read_count ==
           spatial_tiles * SYSTOLIC_KERNEL_TOKENS);
    assert(weight_read_count == cout_tiles * SYSTOLIC_KERNEL_TOKENS);
    assert(result_write_count == spatial_tiles * cout_tiles);
#endif
}
