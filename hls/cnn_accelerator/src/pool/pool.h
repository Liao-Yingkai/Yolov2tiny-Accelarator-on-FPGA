#ifndef CNN_ACCELERATOR_POOL_POOL_H_
#define CNN_ACCELERATOR_POOL_POOL_H_

#include "../types.h"

// 固定 YOLO 层表：一次调用处理整层，每个 Cout 块独立建立行缓存状态。
// 支持 2x2 stride=2、Conv6 的 stride=1 右/下补边，以及无 Pool 的 Bypass。
// 输入与输出均为 Cout 块 -> y -> tile_x -> local_cout_tile；每个输出 Tile
// 含同一行最多 4 个连续输出位置和 4 个 Cout，不在这里组装 8 通道 Pack。
void maxpool(hls::stream<yolo_activation_tile_t>& input_stream,
             hls::stream<yolo_activation_tile_t>& output_stream,
             const ConvGeometry& geometry);

#endif  // CNN_ACCELERATOR_POOL_POOL_H_
