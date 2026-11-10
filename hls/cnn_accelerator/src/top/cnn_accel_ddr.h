#ifndef CNN_ACCELERATOR_CNN_ACCEL_DDR_H_
#define CNN_ACCELERATOR_CNN_ACCEL_DDR_H_

#include "../types.h"
#include "../axi/ddr_layout.h"

// 每次调用处理一张 YOLOv2-tiny 输入图。feature 为连续 A/B 两区；
// 紧凑 RGB 放 A，Prediction 结果在 B（13×13×125 逻辑通道，物理步长 128）。
// weights 和 params 各按固定层表顺序拼接，段之间不插入额外填充。
void cnn_accelerator_ddr(ap_uint<64>* feature,
                         const ap_uint<64>* weights,
                         const ap_int<32>* params);

#endif  // CNN_ACCELERATOR_CNN_ACCEL_DDR_H_
