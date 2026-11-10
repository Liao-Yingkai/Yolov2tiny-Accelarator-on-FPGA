#ifndef CNN_ACCELERATOR_POSTPROCESS_CONV_POSTPROCESS_H_
#define CNN_ACCELERATOR_POSTPROCESS_CONV_POSTPROCESS_H_

#include "../types.h"

// 加 INT32 bias 后按符号选择已准备的 multiplier；一次 signed 乘移和双向 INT8 饱和。
// 禁用 LeakyReLU 时传入相同的正/负支 multiplier。逐元素乘积不加 rounding offset。
activation_t conv_postprocess_requantize(acc_t accumulator,
                                        acc_t bias,
                                        ap_int<32> multiplier_positive,
                                        ap_int<32> multiplier_negative,
                                        ap_int<32> shift);

// 一次调用处理固定层表中的完整 YOLO 层，内部切换 Cout 驻留块。
// 每块先读 B 个参数 Pack，再读/写 spatial_tiles*B 个 Tile。
// 输入输出顺序：Cout 块 -> y -> tile_x -> local_cout_tile；两个 mask 均透传。
// 参数按全局 4-Cout Tile 递增，最后不足 4 路的无效 lane 由 Loader 填零。
void conv_postprocess(hls::stream<yolo_acc_tile_t>& input_stream,
                      hls::stream<postprocess_param_pack_t>& param_stream,
                      hls::stream<yolo_activation_tile_t>& output_stream,
                      const ConvGeometry& geometry);

#endif  // CNN_ACCELERATOR_POSTPROCESS_CONV_POSTPROCESS_H_
