#ifndef CNN_ACCELERATOR_AXI_DDR_IO_H_
#define CNN_ACCELERATOR_AXI_DDR_IO_H_

#include "../types.h"
#include "ddr_layout.h"

// DDR 与内部流之间的独立数据搬运进程。
namespace ddr_io {
void image_loader(const ap_uint<64>* image, hls::stream<pixel_t>& output);
void weight_loader(const ap_uint<64>* weights,
                   hls::stream<weight_pack_t>& conv1,
                   hls::stream<weight_pack_t>& conv2,
                   hls::stream<fc_weight_pack_t>& fc);
void param_loader(const ap_int<32>* params,
                  hls::stream<postprocess_param_pack_t>& conv1,
                  hls::stream<postprocess_param_pack_t>& conv2,
                  hls::stream<fc_param_pack_t>& fc);
void result_writer(hls::stream<inference_result_t>& input, ap_uint<64>* result);
}  // 命名空间 ddr_io

#endif
