#ifndef CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_
#define CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_

#include "../types.h"

// 一次调用处理完整的一层3x3卷积，不含 bias、requant 或 ReLU。
//
// activation_stream：严格消费 Im2Row 的 row -> tile_x -> token 顺序。
// weight_stream：仅在本层第一个 spatial tile 消费，顺序为
//                cout_tile -> token(kh*3+kw)。
// output_stream：保持 Im2Row 等价的 spatial_tile(row -> tile_x) -> cout_tile 顺序输出 INT32 累加 Tile。
//
// layer_mode在一次完整层调用中保持稳定：LAYER1_MODE选择Conv1，LAYER2_MODE选择Conv2。
void systolic(hls::stream<activation_pack_t>& activation_stream,
              hls::stream<weight_pack_t>& weight_stream,
              hls::stream<acc_tile_t>& output_stream,
              layer_mode_t layer_mode);

#endif  // CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_
