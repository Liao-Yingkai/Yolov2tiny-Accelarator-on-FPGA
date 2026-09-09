#ifndef CNN_ACCELERATOR_POOL_POOL_H_
#define CNN_ACCELERATOR_POOL_POOL_H_

#include "../types.h"

// 固定2x2、stride=2 MaxPool。输入事务必须保持
// row -> tile_x -> cout_tile 顺序，其中每个 activation_tile_t 包含4个连续
// 空间 Lane 和4条连续输出通道。输出为紧密的 NHWC 通道 Pack 流：
// pooled_H -> pooled_W -> channel_pack。
//
// layer_mode在一次完整层调用中保持稳定：LAYER1_MODE选择Pool1，LAYER2_MODE选择Pool2。
void maxpool(hls::stream<activation_tile_t>& input_stream,
             hls::stream<channel_pack_t>& output_stream,
             layer_mode_t layer_mode);

#endif  // CNN_ACCELERATOR_POOL_POOL_H_
