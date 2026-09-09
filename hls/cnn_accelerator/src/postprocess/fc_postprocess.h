#ifndef CNN_ACCELERATOR_POSTPROCESS_FC_POSTPROCESS_H_
#define CNN_ACCELERATOR_POSTPROCESS_FC_POSTPROCESS_H_

#include "../types.h"

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

#endif  // CNN_ACCELERATOR_POSTPROCESS_FC_POSTPROCESS_H_
