#include "systolic.h"

namespace {

typedef ap_int<PRODUCT_BITS> product_t;
typedef ap_int<SYSTOLIC_SUM2_BITS> sum2_t;
typedef ap_int<SYSTOLIC_SUM4_BITS> sum4_t;
typedef ap_int<SYSTOLIC_DOT_BITS> dot_t;
// Activation replay 的一个物理 Bank 字对应一个 spatial lane 的8个 INT8通道。
// 显式打包避免 HLS 将 activation_pack_t 拆成32个窄存储器。
typedef ap_uint<SYSTOLIC_ACTIVATION_LANE_BITS> packed_activation_lane_t;
typedef ap_uint<SYSTOLIC_WEIGHT_LANE_BITS> packed_weight_lane_t;

// 使用显式平衡树，避免依赖累加表达式的默认位宽或综合器将其解释为不平衡链。
// 8 个 signed INT8 乘积的精确和需要至少19位有符号数。
static dot_t dot_product(const pixel_t& activation,
                         const weight_lane_t& weight, bool is_conv2) {
#pragma HLS INLINE
    product_t product0 =
        (product_t)(activation.channel[0] * weight.channel[0]);
    // 逻辑 Cin 由计算模块自身保证：Conv1 无效通道即使双侧非零也不能贡献。
    product_t product1 =
        is_conv2 ? (product_t)(activation.channel[1] * weight.channel[1]) : product_t(0);
    product_t product2 =
        is_conv2 ? (product_t)(activation.channel[2] * weight.channel[2]) : product_t(0);
    product_t product3 =
        is_conv2 ? (product_t)(activation.channel[3] * weight.channel[3]) : product_t(0);
    product_t product4 =
        is_conv2 ? (product_t)(activation.channel[4] * weight.channel[4]) : product_t(0);
    product_t product5 =
        is_conv2 ? (product_t)(activation.channel[5] * weight.channel[5]) : product_t(0);
    product_t product6 =
        is_conv2 ? (product_t)(activation.channel[6] * weight.channel[6]) : product_t(0);
    product_t product7 =
        is_conv2 ? (product_t)(activation.channel[7] * weight.channel[7]) : product_t(0);

    sum2_t sum01 = (sum2_t)product0 + (sum2_t)product1;
    sum2_t sum23 = (sum2_t)product2 + (sum2_t)product3;
    sum2_t sum45 = (sum2_t)product4 + (sum2_t)product5;
    sum2_t sum67 = (sum2_t)product6 + (sum2_t)product7;
    sum4_t sum0123 = (sum4_t)sum01 + (sum4_t)sum23;
    sum4_t sum4567 = (sum4_t)sum45 + (sum4_t)sum67;
    return (dot_t)((dot_t)sum0123 + (dot_t)sum4567);
}

// Weight Buffer 的一个物理 Bank 字对应一个 Cout Lane 的8个 INT8权重。
static packed_weight_lane_t pack_weight_lane(const weight_lane_t& lane) {
#pragma HLS INLINE
    packed_weight_lane_t packed = 0;
    for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
#pragma HLS UNROLL
        packed.range((channel + 1) * ACTIVATION_BITS - 1,
                     channel * ACTIVATION_BITS) =
            (ap_uint<ACTIVATION_BITS>)lane.channel[channel];
    }
    return packed;
}

static weight_lane_t unpack_weight_lane(packed_weight_lane_t packed) {
#pragma HLS INLINE
    weight_lane_t lane;
    for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
#pragma HLS UNROLL
        lane.channel[channel] = (weight_t)packed.range(
            (channel + 1) * ACTIVATION_BITS - 1,
            channel * ACTIVATION_BITS);
    }
    return lane;
}

// activation 的 bit 顺序与 weight bank 一致：channel 0 位于最低8 bit，
// channel 7 位于最高8 bit。转换为 ap_uint 只保留 INT8 的二进制补码位型，
// unpack 时转换回 ap_int<8> 恢复有符号语义。
static packed_activation_lane_t pack_activation_lane(const pixel_t& pixel) {
#pragma HLS INLINE
    packed_activation_lane_t packed = 0;
    for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
#pragma HLS UNROLL
        packed.range((channel + 1) * ACTIVATION_BITS - 1,
                     channel * ACTIVATION_BITS) =
            (ap_uint<ACTIVATION_BITS>)pixel.channel[channel];
    }
    return packed;
}

static pixel_t unpack_activation_lane(packed_activation_lane_t packed) {
#pragma HLS INLINE
    pixel_t pixel;
    for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
#pragma HLS UNROLL
        pixel.channel[channel] = (activation_t)packed.range(
            (channel + 1) * ACTIVATION_BITS - 1,
            channel * ACTIVATION_BITS);
    }
    return pixel;
}

static pixel_with_valid_pack_t zero_activation = {{{0, 0, 0, 0, 0, 0, 0, 0}}, 0};
static weight_with_valid_pack_t zero_weight = {{{0, 0, 0, 0, 0, 0, 0, 0}}, 0};

static void clear_activation_trans(pixel_with_valid_pack_t (&activation_trans)[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR]) {
#pragma HLS INLINE
    for (int spatial_lane = 0; spatial_lane < SYSTOLIC_SPATIAL_PAR; ++spatial_lane){
#pragma HLS UNROLL
        for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane){
#pragma HLS UNROLL
            activation_trans[spatial_lane][cout_lane] = zero_activation;
        }
    }
};
static void clear_weight_trans(weight_with_valid_pack_t (&weight_trans)[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR]){
    #pragma HLS INLINE
    for (int spatial_lane = 0; spatial_lane < SYSTOLIC_SPATIAL_PAR; ++spatial_lane){
#pragma HLS UNROLL
        for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane){
#pragma HLS UNROLL
            weight_trans[spatial_lane][cout_lane] = zero_weight;
        }
    }
};
static void clear_accumualtor(acc_t (&accumulator)[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR]){
#pragma HLS INLINE
    for (int spatial_lane = 0; spatial_lane < SYSTOLIC_SPATIAL_PAR; ++spatial_lane){
#pragma HLS UNROLL
        for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane){
#pragma HLS UNROLL
            accumulator[spatial_lane][cout_lane] = (acc_t) 0;
        }
    }
};

}  // namespace

void systolic_OS(hls::stream<activation_pack_t>& activation_stream,
              hls::stream<weight_pack_t>& weight_stream,
              hls::stream<acc_tile_t>& output_stream,
              layer_mode_t layer_mode) {
    const bool is_conv2 = (layer_mode == LAYER2_MODE);
    // 同一个物理PE阵列由一位调用级模式在Conv1、Conv2之间分时复用。
    const int spatial_tiles = is_conv2 ? CNN_CONV2_SPATIAL_TILES :
                                         CNN_CONV1_SPATIAL_TILES;
    const int cout_tiles = is_conv2 ? CNN_CONV2_COUT_TILES :
                                      CNN_CONV1_COUT_TILES;

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
        activation_bank[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_KERNEL_TOKENS];
#pragma HLS ARRAY_PARTITION variable=activation_bank complete dim=1
#pragma HLS BIND_STORAGE variable=activation_bank type=ram_1p impl=bram

    acc_t accumulator[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=accumulator complete dim=0

//activation传播寄存器
    pixel_with_valid_pack_t activation_trans[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=activation_trans complete dim=0

//weight传播寄存器
    weight_with_valid_pack_t weight_trans[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
#pragma HLS ARRAY_PARTITION variable=weight_trans complete dim=0

//activation pe边界输入寄存器
    pixel_with_valid_pack_t activation_edge[SYSTOLIC_SPATIAL_PAR] = {zero_activation,zero_activation,zero_activation,zero_activation};
#pragma HLS ARRAY_PARTITION variable=activation_edge complete dim=0

//weight pe边界输入寄存器
    weight_with_valid_pack_t weight_edge[SYSTOLIC_COUT_PAR] = {zero_weight,zero_weight,zero_weight,zero_weight};
#pragma HLS ARRAY_PARTITION variable=weight_edge complete dim=0

//activation错拍寄存器
    pixel_with_valid_pack_t A1_delay[1] = {zero_activation,};
    pixel_with_valid_pack_t A2_delay[2] = {zero_activation,zero_activation,};
    pixel_with_valid_pack_t A3_delay[3] = {zero_activation,zero_activation,zero_activation,};
#pragma HLS ARRAY_PARTITION variable=A1_delay complete
#pragma HLS ARRAY_PARTITION variable=A2_delay complete
#pragma HLS ARRAY_PARTITION variable=A3_delay complete

//weight错拍寄存器
    weight_with_valid_pack_t W1_delay[1] = {zero_weight,};
    weight_with_valid_pack_t W2_delay[2] = {zero_weight,zero_weight,};
    weight_with_valid_pack_t W3_delay[3] = {zero_weight,zero_weight,zero_weight,};
#pragma HLS ARRAY_PARTITION variable=W1_delay complete
#pragma HLS ARRAY_PARTITION variable=W2_delay complete
#pragma HLS ARRAY_PARTITION variable=W3_delay complete

    // spatial_tile 与 Im2Row 的 row -> tile_x 顺序完全等价：Conv1 共有196个，
    // Conv2 共有56个。合并计数器不改变 activation 或 result 的事务顺序。
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
            const bool first_spatial_tile = (spatial_tile == 0);
            ap_uint<SYSTOLIC_SPATIAL_PAR> tile_spatial_valid = 0;
            ap_uint<6> weight_base = 0;


            for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
                clear_accumualtor(accumulator);
                clear_activation_trans(activation_trans);
                clear_weight_trans(weight_trans);
                for (int cycle = 0; cycle < SYSTOLIC_EXCU_TIME; ++cycle) {
#pragma HLS PIPELINE II=1

                    //边界填充
                    activation_edge[1] = A1_delay[0];
                    activation_edge[2] = A2_delay[1];
                    activation_edge[3] = A3_delay[2];
                    weight_edge[1] = W1_delay[0];
                    weight_edge[2] = W2_delay[1];
                    weight_edge[3] = W3_delay[2];

                    //滞后寄存器
                    A3_delay[2] = A3_delay[1];
                    A3_delay[1] = A3_delay[0];
                    A2_delay[1] = A2_delay[0];

                    W3_delay[2] = W3_delay[1];
                    W3_delay[1] = W3_delay[0];
                    W2_delay[1] = W2_delay[0];

                    activation_pack_t activation_raw;
                    weight_pack_t weight_raw;
                    if(cycle < SYSTOLIC_KERNEL_TOKENS) {
                        const ap_uint<6> weight_address = weight_base + cycle;

                        if (cout_tile == 0) {
                            activation_raw = activation_stream.read();

                            if (cycle == 0) {
                                tile_spatial_valid = activation_raw.spatial_valid;
                            }

                            //填充activation
                            activation_edge[0].pixel = activation_raw.pixel[0];
                            A1_delay[0].pixel = activation_raw.pixel[1];
                            A2_delay[0].pixel = activation_raw.pixel[2];
                            A3_delay[0].pixel = activation_raw.pixel[3];

                            //保存activation
                            for (int spatial_lane = 0; spatial_lane < SYSTOLIC_SPATIAL_PAR;
                             ++spatial_lane) {
#pragma HLS UNROLL
                                activation_bank[spatial_lane][cycle] =
                                    pack_activation_lane(activation_raw.pixel[spatial_lane]);
                            }
                        }
                        else {
                            //填充activation
                            activation_edge[0].pixel = unpack_activation_lane(activation_bank[0][cycle]);
                            A1_delay[0].pixel = unpack_activation_lane(activation_bank[1][cycle]);
                            A2_delay[0].pixel = unpack_activation_lane(activation_bank[2][cycle]);
                            A3_delay[0].pixel = unpack_activation_lane(activation_bank[3][cycle]);
                        }
                            activation_edge[0].valid = (ap_uint<1>) 1;
                            A1_delay[0].valid = (ap_uint<1>) 1;
                            A2_delay[0].valid = (ap_uint<1>) 1;
                            A3_delay[0].valid = (ap_uint<1>) 1;

                        if(first_spatial_tile) {
                            weight_raw = weight_stream.read();

                            weight_edge[0].weight = weight_raw.lane[0];
                            W1_delay[0].weight = weight_raw.lane[1];
                            W2_delay[0].weight = weight_raw.lane[2];
                            W3_delay[0].weight = weight_raw.lane[3];

                            //保存weight
                            for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
                            weight_bank[cout_lane][weight_address] =
                                pack_weight_lane(weight_raw.lane[cout_lane]);
                            }
                        }
                        else {
                            weight_edge[0].weight = unpack_weight_lane(weight_bank[0][weight_address]);
                            W1_delay[0].weight = unpack_weight_lane(weight_bank[1][weight_address]);
                            W2_delay[0].weight = unpack_weight_lane(weight_bank[2][weight_address]);
                            W3_delay[0].weight = unpack_weight_lane(weight_bank[3][weight_address]);
                        }
                            weight_edge[0].valid = (ap_uint<1>) 1;
                            W1_delay[0].valid = (ap_uint<1>) 1;
                            W2_delay[0].valid = (ap_uint<1>) 1;
                            W3_delay[0].valid = (ap_uint<1>) 1;
                        }
                    else {
                        //接收tokrn完毕
                            activation_edge[0] = zero_activation;
                            A1_delay[0] = zero_activation;
                            A2_delay[0] = zero_activation;
                            A3_delay[0] = zero_activation;
                            weight_edge[0] = zero_weight;
                            W1_delay[0] = zero_weight;
                            W2_delay[0] = zero_weight;
                            W3_delay[0] = zero_weight;
                    }

                    //Output-Stational脉动阵列
                    for (int spatial_lane = SYSTOLIC_SPATIAL_PAR-1; spatial_lane >= 0; --spatial_lane) {
#pragma HLS UNROLL
                        for (int cout_lane = SYSTOLIC_COUT_PAR-1; cout_lane >= 0; --cout_lane) {
#pragma HLS UNROLL
                            //activation与weight array脉动赋值
                            weight_trans[spatial_lane][cout_lane] = (spatial_lane== 0) ?
                                weight_edge[cout_lane] : weight_trans[spatial_lane-1][cout_lane];
                            activation_trans[spatial_lane][cout_lane] = (cout_lane== 0) ?
                                activation_edge[spatial_lane] : activation_trans[spatial_lane][cout_lane-1];

                            if (activation_trans[spatial_lane][cout_lane].valid
                                && weight_trans[spatial_lane][cout_lane].valid && tile_spatial_valid[spatial_lane]) {
                                    const dot_t dot = dot_product(activation_trans[spatial_lane][cout_lane].pixel,weight_trans[spatial_lane][cout_lane].weight,is_conv2);
                                    accumulator[spatial_lane][cout_lane] += (acc_t) dot;
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
                weight_base += SYSTOLIC_KERNEL_TOKENS;
            }
    }

}
