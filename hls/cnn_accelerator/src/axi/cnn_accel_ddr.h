#ifndef CNN_ACCELERATOR_CNN_ACCEL_DDR_H_
#define CNN_ACCELERATOR_CNN_ACCEL_DDR_H_
#include "../types.h"
#include "ddr_layout.h"

// 单次 ap_ctrl_hs 调用：四个指针均为 DDR 基地址，结果保留 logits 和类别。
void cnn_accelerator_ddr(const ap_uint<64>* image,
                         const ap_uint<64>* weights,
                         const ap_int<32>* params,
                         ap_uint<64>* result);
#endif
