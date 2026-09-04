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
// 接口仅接受 Conv1=(28,28,1,8) 或 Conv2=(14,14,8,16)。H/W/Cin/Cout 保留以供
// 单物理模块分时复用；综合态由 Cin 选择固定调度，非法组合仅在非综合态断言。
void systolic(hls::stream<activation_pack_t>& activation_stream,
              hls::stream<weight_pack_t>& weight_stream,
              hls::stream<acc_tile_t>& output_stream,
              int H,
              int W,
              int Cin,
              int Cout);

#endif  // CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_
