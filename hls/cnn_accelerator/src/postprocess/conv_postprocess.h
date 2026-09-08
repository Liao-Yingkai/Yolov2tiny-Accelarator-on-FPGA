#ifndef CNN_ACCELERATOR_POSTPROCESS_CONV_POSTPROCESS_H_
#define CNN_ACCELERATOR_POSTPROCESS_CONV_POSTPROCESS_H_

#include "../types.h"

// 对单个卷积 accumulator 执行 ConvPostProcess 数值规则：INT32 bias add、ReLU、
// INT64 multiply、直接算术右移和上限127饱和。此函数不加入任何 rounding
// offset，供独立数值测试和 conv_postprocess 主循环共同使用。
activation_t conv_postprocess_requantize(acc_t accumulator,
                                    ap_int<32> bias,
                                    ap_int<32> multiplier,
                                    ap_int<32> shift);

// 一次调用处理完整 Conv1 或 Conv2 层的 Systolic 结果。
//
// input_stream：spatial_tile(row -> tile_x) -> cout_tile 的 acc_tile_t。
// param_stream：只在第一个 spatial tile 消费，顺序为 cout_tile。Conv1
//               消费2个 Pack，Conv2消费4个 Pack；其余 spatial tile 由本地
//               参数缓存回放。
// output_stream：与 input_stream 完全相同的事务顺序，元素为 requantized
//                activation_tile_t。
// Cout：仅允许8（Conv1）或16（Conv2）；非法值仅在非综合态断言。
void conv_postprocess(hls::stream<acc_tile_t>& input_stream,
                      hls::stream<postprocess_param_pack_t>& param_stream,
                      hls::stream<activation_tile_t>& output_stream,
                      int Cout);

#endif  // CNN_ACCELERATOR_POSTPROCESS_CONV_POSTPROCESS_H_
