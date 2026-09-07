#ifndef CNN_ACCELERATOR_FC_FC_H_
#define CNN_ACCELERATOR_FC_FC_H_

#include "../types.h"

// 固定 MNIST 独立 FC。input_stream 是 Pool2 紧密 NHWC 输出，顺序为
// pooled_y -> pooled_x -> channel_pack，恰好98个 channel_pack_t。权重流为
// out_tile -> token，恰好2x98个320-bit逻辑 Pack；结果输出两个5-logit
// accumulator Pack。第一 out tile 直接消费输入并写入内部回放 BRAM，第二
// out tile 从该 BRAM 回放，因而输入流仅被读取一次。
void fully_connected(hls::stream<channel_pack_t>& input_stream,
                     hls::stream<fc_weight_pack_t>& weight_stream,
                     hls::stream<fc_acc_pack_t>& output_stream);

// 单个 FC 输出的 signed requant 规则：INT32 bias add（以33位检查）、
// signed INT32xINT32 -> signed INT64、直接算术右移、双向饱和[-128,127]。
// 该 FC 路径没有 ReLU，也不加入 rounding offset。
logit_t fc_requantize(acc_t accumulator,
                      ap_int<32> bias,
                      ap_int<32> multiplier,
                      ap_int<32> shift);

// 读取两个 FC accumulator Pack 和两个参数 Pack，串行复用一套32x32 signed
// requant datapath 处理每组的5个 lane，最后只输出一次完整10路 logits_pack_t。
void fc_postprocess(hls::stream<fc_acc_pack_t>& input_stream,
                    hls::stream<fc_param_pack_t>& param_stream,
                    hls::stream<logits_pack_t>& output_stream);

#endif  // CNN_ACCELERATOR_FC_FC_H_
