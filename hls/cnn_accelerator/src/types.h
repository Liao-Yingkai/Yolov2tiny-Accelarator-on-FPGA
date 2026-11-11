#ifndef CNN_ACCELERATOR_TYPES_H_
#define CNN_ACCELERATOR_TYPES_H_

#include <ap_int.h>
#include <hls_stream.h>
#include "config.h"

// 对称八位有符号量化激活，数值0既是量化零点，也是卷积 Padding 的填充值。
using activation_t = ap_int<ACTIVATION_BITS>;

// Systolic 使用 8 位有符号权重和 32 位有符号累加器。虽然当前
// IM2ROW_P 与输出通道并行度都为4，但二者在概念上独立，避免后续将空间
// 并行度变化无意传递为 Cout 并行度变化。
using weight_t = ap_int<WEIGHT_BITS>;
using acc_t = ap_int<ACCUMULATOR_BITS>;

// 一个空间位置上的连续8通道组；不承诺包含完整逻辑 pixel 的全部通道。
// channel c 对应逻辑通道 group*8+c；输入生产端将逻辑 Cin 之外的尾槽置零。
struct channel_pack_t {
    activation_t channel[FEATURE_PACK_CHANNELS];
};
using pixel_t = channel_pack_t;  // 一个物理像素通道组；逻辑 Cin 大于8时由多个组表示。

// activation_pack_t 是 Im2Row 到 Systolic 的基本流事务：P 个像素属于同一输出行、
// 同一 Tile 和同一个 (kh,kw,cin_group)。
// spatial_valid 在同一 Tile 的全部 R 个事务内保持不变；卷积 Padding 不清除
// valid，行尾不存在的 Lane 则 valid=0 且其 pixel 全零。
struct activation_pack_t {
    pixel_t pixel[IM2ROW_P];
    ap_uint<IM2ROW_P> spatial_valid;
};

// 一个 weight_lane_t 对应一个 Cout Lane 的同一组 8 个物理 Cin 权重。
// 权重 BIN 由 exporter 按组补零，物理尾槽不贡献点积。
struct weight_lane_t {
    weight_t channel[SYSTOLIC_CIN_PAR];
};

// 一个事务是同一 (cout_tile, kh, kw, cin_group) 的 4 Cout × 8 Cin 权重。
// 块内按 local_cout_tile -> kh -> kw -> cin_group 发送。
struct weight_pack_t {
    weight_lane_t lane[SYSTOLIC_COUT_PAR];
};

// Systolic 输出4个相邻空间位置 × 4条输出通道的 INT32 累加结果。
// 空间与 Cout 无效 lane 的结果为零；下游访问 bias/M/S 前仍须检查 cout_valid。
struct yolo_acc_tile_t {
    acc_t value[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
    ap_uint<SYSTOLIC_SPATIAL_PAR> spatial_valid;
    ap_uint<SYSTOLIC_COUT_PAR> cout_valid;
};

// YOLO PostProcess 到 Pool/Bypass：保留 Cout 块内的 y -> tile_x -> local_cout_tile
// 顺序和两个 mask；无效 lane 为零，但不视作有效的 MaxPool 输入。
struct yolo_activation_tile_t {
    activation_t value[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
    ap_uint<SYSTOLIC_SPATIAL_PAR> spatial_valid;
    ap_uint<SYSTOLIC_COUT_PAR> cout_valid;
};

//Systolic使用的activation数据+当前pe核中的activation是否valid
struct pixel_with_valid_pack_t {
    pixel_t pixel;
    ap_uint<1> valid;
};

//Systolic使用的weight数据+当前pe核中的weight是否valid
struct weight_with_valid_pack_t {
    weight_lane_t weight;
    ap_uint<1> valid;
};

// 一个参数 Pack 对应一个 Cout tile 的4条输出通道。所有字段都是已经由
// exporter 离线生成的有符号 INT32：bias 位于 accumulator 域，multiplier 与
// shift 共同表示 real_multiplier ~= multiplier / 2^shift。
struct postprocess_param_pack_t {
    ap_int<32> bias[SYSTOLIC_COUT_PAR];
    ap_int<32> multiplier[SYSTOLIC_COUT_PAR];
    ap_int<32> shift[SYSTOLIC_COUT_PAR];
};

#endif  // CNN_ACCELERATOR_TYPES_H_
