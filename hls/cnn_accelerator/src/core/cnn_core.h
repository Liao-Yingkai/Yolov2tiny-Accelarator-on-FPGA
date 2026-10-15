#ifndef CNN_ACCELERATOR_CORE_CNN_CORE_H_
#define CNN_ACCELERATOR_CORE_CNN_CORE_H_

#include "../types.h"

// 一次调用完成一张固定28x28 MNIST图像的完整推理。第一版保持普通
// hls::stream接口和默认ap_ctrl_hs控制，不在此层引入AXI、DDR或DMA。
//
// 外部流的固定事务数：
// - image_stream: 784 pixel_t；
// - conv1_weight_stream / conv2_weight_stream: 18 / 36 weight_pack_t；
// - conv1_param_stream / conv2_param_stream: 2 / 4 postprocess_param_pack_t；
// - fc_weight_stream / fc_param_stream: 196 / 2个FC Pack；
// - logits_stream: 1个完整10路 logits_pack_t。
void cnn_accelerator(
    hls::stream<pixel_t>& image_stream,
    hls::stream<weight_pack_t>& conv1_weight_stream,
    hls::stream<postprocess_param_pack_t>& conv1_param_stream,
    hls::stream<weight_pack_t>& conv2_weight_stream,
    hls::stream<postprocess_param_pack_t>& conv2_param_stream,
    hls::stream<fc_weight_pack_t>& fc_weight_stream,
    hls::stream<fc_param_pack_t>& fc_param_stream,
    hls::stream<logits_pack_t>& logits_stream);

#endif  // 核心接口头文件
