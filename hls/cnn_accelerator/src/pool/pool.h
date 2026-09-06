#ifndef CNN_ACCELERATOR_POOL_POOL_H_
#define CNN_ACCELERATOR_POOL_POOL_H_

#include "../types.h"

// 固定2x2、stride=2 MaxPool。输入事务必须保持
// row -> tile_x -> cout_tile 顺序，其中每个 activation_tile_t 包含4个连续
// 空间 Lane 和4条连续输出通道。输出为紧密的 NHWC 通道 Pack 流：
// pooled_H -> pooled_W -> channel_pack。
//
// 只允许 Pool1 (28, 28, 8) 与 Pool2 (14, 14, 16)。综合态仅以 Cout 选择
// 固定模式，H/W 的完整组合由顶层调用契约保证；非综合态会断言完整三元组。
void maxpool(hls::stream<activation_tile_t>& input_stream,
             hls::stream<channel_pack_t>& output_stream,
             int H,
             int W,
             int Cout);

#endif  // CNN_ACCELERATOR_POOL_POOL_H_
