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
                  const weight_lane_t& weight) {
#pragma HLS INLINE
    product_t product0 =
        (product_t)(activation.channel[0] * weight.channel[0]);
    // 不在 PE 内判断 Cin：权重 BIN 与 Im2Row 已按约定对 Cin 尾通道补零。
    product_t product1 =
        (product_t)(activation.channel[1] * weight.channel[1]);
    product_t product2 =
        (product_t)(activation.channel[2] * weight.channel[2]);
    product_t product3 =
        (product_t)(activation.channel[3] * weight.channel[3]);
    product_t product4 =
        (product_t)(activation.channel[4] * weight.channel[4]);
    product_t product5 =
        (product_t)(activation.channel[5] * weight.channel[5]);
    product_t product6 =
        (product_t)(activation.channel[6] * weight.channel[6]);
    product_t product7 =
        (product_t)(activation.channel[7] * weight.channel[7]);

    sum2_t sum01 = (sum2_t)product0 + (sum2_t)product1;
    sum2_t sum23 = (sum2_t)product2 + (sum2_t)product3;
    sum2_t sum45 = (sum2_t)product4 + (sum2_t)product5;
    sum2_t sum67 = (sum2_t)product6 + (sum2_t)product7;
    sum4_t sum0123 = (sum4_t)sum01 + (sum4_t)sum23;
    sum4_t sum4567 = (sum4_t)sum45 + (sum4_t)sum67;
    return (dot_t)((dot_t)sum0123 + (dot_t)sum4567);
}

const pixel_with_valid_pack_t zero_activation = {{{0, 0, 0, 0, 0, 0, 0, 0}}, 0};
const weight_with_valid_pack_t zero_weight = {{{0, 0, 0, 0, 0, 0, 0, 0}}, 0};

void clear_activation_trans(
    pixel_with_valid_pack_t (&activation_trans)[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR]) {
#pragma HLS INLINE
    for (int spatial_lane = 0; spatial_lane < SYSTOLIC_SPATIAL_PAR; ++spatial_lane) {
#pragma HLS UNROLL
        for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
            activation_trans[spatial_lane][cout_lane] = zero_activation;
        }
    }
}

void clear_weight_trans(
    weight_with_valid_pack_t (&weight_trans)[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR]) {
#pragma HLS INLINE
    for (int spatial_lane = 0; spatial_lane < SYSTOLIC_SPATIAL_PAR; ++spatial_lane) {
#pragma HLS UNROLL
        for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
            weight_trans[spatial_lane][cout_lane] = zero_weight;
        }
    }
}

void clear_accumulator(
    acc_t (&accumulator)[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR]) {
#pragma HLS INLINE
    for (int spatial_lane = 0; spatial_lane < SYSTOLIC_SPATIAL_PAR; ++spatial_lane) {
#pragma HLS UNROLL
        for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
            accumulator[spatial_lane][cout_lane] = (acc_t) 0;
        }
    }
}

using spatial_mask_t = ap_uint<SYSTOLIC_SPATIAL_PAR>;
using cout_mask_t = ap_uint<SYSTOLIC_COUT_PAR>;
using yolo_weight_address_t = ap_uint<YOLO_TINY_WEIGHT_ADDRESS_BITS>;
using yolo_step_t = ap_uint<bits_for_max(YOLO_TINY_ACTIVATION_REPLAY_CAPACITY +
                                        SYSTOLIC_WAVEFRONT_DRAIN_STEPS)>;

void advance_os_array(
    const pixel_with_valid_pack_t (&activation_edge)[SYSTOLIC_SPATIAL_PAR],
    const weight_with_valid_pack_t (&weight_edge)[SYSTOLIC_COUT_PAR],
    pixel_with_valid_pack_t (&activation_trans)[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR],
    weight_with_valid_pack_t (&weight_trans)[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR],
    acc_t (&accumulator)[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR],
    const spatial_mask_t& spatial_valid) {
#pragma HLS INLINE
    // 从右下往左上更新，保证读取的是相邻 PE 上一步的操作数。
    for (int i = SYSTOLIC_SPATIAL_PAR - 1; i >= 0; --i) {
#pragma HLS UNROLL
        for (int j = SYSTOLIC_COUT_PAR - 1; j >= 0; --j) {
#pragma HLS UNROLL
            weight_trans[i][j] = i == 0 ? weight_edge[j] : weight_trans[i - 1][j];
            activation_trans[i][j] = j == 0 ? activation_edge[i] : activation_trans[i][j - 1];
            if (activation_trans[i][j].valid && weight_trans[i][j].valid && spatial_valid[i]) {
                const dot_t dot = dot_product(activation_trans[i][j].pixel, weight_trans[i][j].weight);
                accumulator[i][j] += acc_t(dot);
            }
        }
    }
}

template <typename Operand, int Lanes>
void clear_os_skew(Operand (&delay)[Lanes][Lanes - 1], const Operand& zero) {
#pragma HLS INLINE
    // Lane i 只使用前 i 级，Lane 0 直通。未使用的单元不读、不写。
    for (int lane = 1; lane < Lanes; ++lane) {
#pragma HLS UNROLL
        for (int stage = 0; stage < Lanes - 1; ++stage) {
#pragma HLS UNROLL
            if (stage < lane) delay[lane][stage] = zero;
        }
    }
}

template <typename Operand, int Lanes>
void skew_os_input(const Operand (&input)[Lanes],
                   Operand (&delay)[Lanes][Lanes - 1],
                   Operand (&edge)[Lanes]) {
#pragma HLS INLINE
    edge[0] = input[0];
    for (int lane = 1; lane < Lanes; ++lane) {
#pragma HLS UNROLL
        edge[lane] = delay[lane][lane - 1];
        for (int stage = Lanes - 2; stage > 0; --stage) {
#pragma HLS UNROLL
            if (stage < lane) delay[lane][stage] = delay[lane][stage - 1];
        }
        delay[lane][0] = input[lane];  // 最后写新 token，payload 与 valid 同步延迟。
    }
}

}  // namespace

void systolic_OS(hls::stream<activation_pack_t>& activation_stream,
                 hls::stream<weight_pack_t>& weight_stream,
                 hls::stream<yolo_acc_tile_t>& output_stream,
                 const ConvGeometry& geometry,
                 int first_cout_tile,
                 int block_cout_tiles) {
#ifndef __SYNTHESIS__
    // 层几何由固定表提供；只校验调用方传入的块范围与缓存边界。
    assert(geometry.reduction_steps > 0 &&
           geometry.reduction_steps <= YOLO_TINY_ACTIVATION_REPLAY_CAPACITY);
    assert(block_cout_tiles > 0 && block_cout_tiles <= geometry.resident_cout_tiles &&
           block_cout_tiles <= YOLO_TINY_MAX_RESIDENT_COUT_TILES &&
           first_cout_tile >= 0 && first_cout_tile <= geometry.cout_tiles - block_cout_tiles);
    assert(block_cout_tiles * geometry.reduction_steps <= YOLO_TINY_WEIGHT_BUFFER_DEPTH);
#endif
    const yolo_step_t reduction_steps = geometry.reduction_steps;  // R：包括全部 kernel 位置和 Cin Group。
    const yolo_step_t wavefront_steps = reduction_steps + SYSTOLIC_WAVEFRONT_DRAIN_STEPS;
    const ap_uint<bits_for_max(YOLO_TINY_MAX_RESIDENT_COUT_TILES)> resident_tiles = block_cout_tiles;
    const yolo_weight_address_t weight_packs = resident_tiles * reduction_steps;
    const int last_global_tile_local = geometry.cout_tiles - 1 - first_cout_tile;
    const bool replay_needed = block_cout_tiles > 1;

    // 只分区 lane 维：每拍四个 64-bit 字，local_cout_tile 和 token 共享深度。
    packed_weight_lane_t weight_bank[SYSTOLIC_COUT_PAR][YOLO_TINY_WEIGHT_BUFFER_DEPTH];
#pragma HLS ARRAY_PARTITION variable=weight_bank complete dim=1
#pragma HLS BIND_STORAGE variable=weight_bank type=ram_1p impl=bram
    packed_activation_lane_t activation_bank[SYSTOLIC_SPATIAL_PAR][YOLO_TINY_ACTIVATION_REPLAY_CAPACITY];
#pragma HLS ARRAY_PARTITION variable=activation_bank complete dim=1
#pragma HLS BIND_STORAGE variable=activation_bank type=ram_1p impl=bram

    // 权重 stream 已按 local_cout_tile -> token 排列，按地址顺序装入即可。
    // 首版不做预取；直到本块权重全部写完，才消费第一笔 Im2Row activation。
    for (yolo_weight_address_t address = 0; address < weight_packs; ++address) {
#pragma HLS PIPELINE II=1
        const weight_pack_t weight = weight_stream.read();
        for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
#pragma HLS UNROLL
            weight_bank[j][address.to_uint()] = pack_weight_lane(weight.lane[j]);
        }
    }

    // Im2Row 的 y -> tile_x 顺序可用一个空间 Tile 游标表示，不跨块保存状态。
    for (int spatial_tile = 0; spatial_tile < geometry.spatial_tiles; ++spatial_tile) {
        spatial_mask_t spatial_valid = 0;
        yolo_weight_address_t weight_base = 0;
        for (int local_cout_tile = 0; local_cout_tile < block_cout_tiles; ++local_cout_tile) {
            const cout_mask_t cout_valid = local_cout_tile == last_global_tile_local
                ? cout_mask_t(geometry.last_cout_valid_mask)
                : cout_mask_t((1u << SYSTOLIC_COUT_PAR) - 1u);
            const bool from_stream = local_cout_tile == 0;

            acc_t accumulator[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
            pixel_with_valid_pack_t activation_trans[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
            weight_with_valid_pack_t weight_trans[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
            pixel_with_valid_pack_t activation_delay[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_SPATIAL_PAR - 1];
            weight_with_valid_pack_t weight_delay[SYSTOLIC_COUT_PAR][SYSTOLIC_COUT_PAR - 1];
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=0
#pragma HLS ARRAY_PARTITION variable=activation_trans complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight_trans complete dim=0
#pragma HLS ARRAY_PARTITION variable=activation_delay complete dim=0
#pragma HLS ARRAY_PARTITION variable=weight_delay complete dim=0
            clear_accumulator(accumulator);
            clear_activation_trans(activation_trans);
            clear_weight_trans(weight_trans);
            clear_os_skew(activation_delay, zero_activation);
            clear_os_skew(weight_delay, zero_weight);

            // 一个 Tile 内不区分 3×3 / 1×1：只在全部 R 个 token 结束后排空。
            for (yolo_step_t step = 0; step < wavefront_steps; ++step) {
#pragma HLS PIPELINE II=1
                pixel_with_valid_pack_t activation_in[SYSTOLIC_SPATIAL_PAR];
                weight_with_valid_pack_t weight_in[SYSTOLIC_COUT_PAR];
                pixel_with_valid_pack_t activation_edge[SYSTOLIC_SPATIAL_PAR];
                weight_with_valid_pack_t weight_edge[SYSTOLIC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=activation_in complete
#pragma HLS ARRAY_PARTITION variable=weight_in complete
#pragma HLS ARRAY_PARTITION variable=activation_edge complete
#pragma HLS ARRAY_PARTITION variable=weight_edge complete
                if (step < reduction_steps) {
                    const unsigned token = step.to_uint();
                    if (from_stream) {
                        const activation_pack_t activation = activation_stream.read();
                        if (step == 0) spatial_valid = activation.spatial_valid;
                        for (int i = 0; i < SYSTOLIC_SPATIAL_PAR; ++i) {
#pragma HLS UNROLL
                            activation_in[i].pixel = activation.pixel[i];
                            activation_in[i].valid = 1;
                            if (replay_needed) {
                                activation_bank[i][token] = pack_activation_lane(activation.pixel[i]);
                            }
                        }
                    } else {
                        for (int i = 0; i < SYSTOLIC_SPATIAL_PAR; ++i) {
#pragma HLS UNROLL
                            activation_in[i].pixel = unpack_activation_lane(activation_bank[i][token]);
                            activation_in[i].valid = 1;
                        }
                    }
                    const yolo_weight_address_t weight_address = weight_base + step;
                    for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
#pragma HLS UNROLL
                        weight_in[j].weight = unpack_weight_lane(weight_bank[j][weight_address.to_uint()]);
                        weight_in[j].valid = cout_valid[j];  // 随权重错拍和传播；Cin 补零由 BIN 保证。
                    }
                } else {
                    // 排空只停止注入；已进入错拍线或阵列的有效 token 继续推进。
                    for (int i = 0; i < SYSTOLIC_SPATIAL_PAR; ++i) {
#pragma HLS UNROLL
                        activation_in[i] = zero_activation;
                    }
                    for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
#pragma HLS UNROLL
                        weight_in[j] = zero_weight;
                    }
                }
                skew_os_input(activation_in, activation_delay, activation_edge);
                skew_os_input(weight_in, weight_delay, weight_edge);
                advance_os_array(activation_edge, weight_edge, activation_trans,
                                 weight_trans, accumulator, spatial_valid);
            }

            // 无效 lane 从未更新，其值仍为清零后的 0；下游依然需要两种 mask。
            yolo_acc_tile_t result;
            result.spatial_valid = spatial_valid;
            result.cout_valid = cout_valid;
            for (int i = 0; i < SYSTOLIC_SPATIAL_PAR; ++i) {
#pragma HLS UNROLL
                for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
#pragma HLS UNROLL
                    result.value[i][j] = accumulator[i][j];
                }
            }
            output_stream.write(result);
            weight_base += reduction_steps;  // 下一本地计算 Tile 的连续权重段。
        }
    }
}
