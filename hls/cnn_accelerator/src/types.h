#ifndef CNN_ACCELERATOR_TYPES_H_
#define CNN_ACCELERATOR_TYPES_H_

#include <ap_int.h>
#include <hls_stream.h>
#include "config.h"

// 对称八位有符号量化激活，数值0既是量化零点，也是卷积 Padding 的填充值。
using activation_t = ap_int<ACTIVATION_BITS>;

// Systolic 第一版固定使用 8 位有符号权重和 32 位有符号累加器。虽然当前
// IM2ROW_P 与输出通道并行度都为4，但二者在概念上独立，避免后续将空间
// 并行度变化无意传递为 Cout 并行度变化。
using weight_t = ap_int<WEIGHT_BITS>;
using acc_t = ap_int<ACCUMULATOR_BITS>;

// 一个 FC 权重事务对应一个 out_tile 和一个8通道输入 token，逻辑位宽为
// 5 x 8 x 8 = 320 bit。流顺序固定为 out_tile -> token。
struct fc_weight_pack_t {
    weight_t value[FC_COUT_PAR][FC_CIN_PAR];
};

// 一个 FC accumulator 事务对应一个 out_tile 的5个输出通道。
struct fc_acc_pack_t {
    acc_t value[FC_COUT_PAR];
};

// FC 后处理参数同样以5个输出通道为一组；字段均为 exporter 导出的
// signed INT32，bias 位于 accumulator 域。
struct fc_param_pack_t {
    ap_int<32> bias[FC_COUT_PAR];
    ap_int<32> multiplier[FC_COUT_PAR];
    ap_int<32> shift[FC_COUT_PAR];
};

using logit_t = ap_int<ACTIVATION_BITS>;
// MNIST 的类别索引范围为0..9，4 bit足以表达全部类别。
using class_index_t = ap_uint<4>;

// FC 最终只写出一个完整的10类 signed INT8 logit Pack。
struct logits_pack_t {
    logit_t value[FC_OUTPUTS];
};

// 一个 pixel_t 表示一个空间位置的全部8个物理输入通道。
// Im2Row 在输入读取时完成规范化：Conv1 仅 channel0 有效，Conv2 使用全部通道。
struct pixel_t {
    activation_t channel[IM2ROW_MAX_CHANNELS];
};

// Pool 与后续层之间也以一个空间位置的连续8个物理通道为基本事务。该别名
// 明确表达其通道 pack 语义，同时保证 Pool1 的输出可直接作为 Conv2 Im2Row
// 所需的 pixel_t 输入。
using channel_pack_t = pixel_t;

// activation_pack_t 是 Im2Row 到 Systolic 的基本流事务：P 个像素属于同一输出行、
// 同一 Tile 和同一个 (kh,kw)。Systolic 的接口目标是在每个成功接收周期消费一个 Pack。
// spatial_valid 只标记空间 Lane 是否真实存在，在同一 Tile 的9拍内保持不变；
// 卷积 Padding 对应的 Lane 仍有效，而行尾不存在的 Lane 为0且其 pixel 在9拍内均为全零。
struct activation_pack_t {
    pixel_t pixel[IM2ROW_P];
    ap_uint<IM2ROW_P> spatial_valid;
};

// 一个 weight_lane_t 对应当前 Cout Tile 中的一条输出通道，并固定打包8个
// 物理输入通道。Conv1 的有效 Cin 由 Systolic 保证；stream 保持8个物理 Cin，
// DDR 权重仍使用紧凑的原始逻辑 Cin 布局。
struct weight_lane_t {
    weight_t channel[SYSTOLIC_CIN_PAR];
};

// Weight stream 的一个事务对应相同 (cout_tile, kh, kw) 下的全部4条 Cout
// Lane。其 stream 顺序为 cout_tile -> token(kh*3+kw)。
struct weight_pack_t {
    weight_lane_t lane[SYSTOLIC_COUT_PAR];
};

// Systolic 输出的一个事务包含4个相邻空间位置和当前4条输出通道的 INT32
// 累加结果；空间尾 Lane 的有效性不由数值本身表达，必须继续携带该掩码。
struct acc_tile_t {
    acc_t value[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
    ap_uint<SYSTOLIC_SPATIAL_PAR> spatial_valid;
};

// ConvPostProcess 输出保持与 acc_tile_t 相同的 spatial_tile -> cout_tile 事务
// 粒度，只将每个 INT32 累加结果 requantize 为 INT8 activation。尾部无效
// spatial lane 的 value 必须为零，不能由 bias 生成伪有效数据。
struct activation_tile_t {
    activation_t value[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
    ap_uint<SYSTOLIC_SPATIAL_PAR> spatial_valid;
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
