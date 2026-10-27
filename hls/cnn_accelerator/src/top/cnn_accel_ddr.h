#ifndef CNN_ACCELERATOR_CNN_ACCEL_DDR_H_
#define CNN_ACCELERATOR_CNN_ACCEL_DDR_H_
#include "../types.h"
#include "../axi/ddr_layout.h"

// 单次 ap_ctrl_hs 调用：四个DDR基地址；scale在本次调用期间稳定。
void cnn_accelerator_ddr(const ap_uint<64>* image,
                         const ap_uint<64>* weights,
                         const ap_int<32>* params,
                         ap_uint<64>* result,
                         ap_uint<32> softmax_scale_q20);
#endif
