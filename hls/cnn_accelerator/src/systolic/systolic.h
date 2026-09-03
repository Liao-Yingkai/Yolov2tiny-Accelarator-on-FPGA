#ifndef CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_
#define CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_

#include "../types.h"

// 一次调用处理完整的一层3x3卷积，不含 bias、requant 或 ReLU。
//
// activation_stream：严格消费 Im2Row 的 row -> tile_x -> token 顺序。
// weight_stream：仅在本层第一个 spatial tile 消费，顺序为
//                cout_tile -> token(kh*3+kw)。
// output_stream：按 row -> tile_x -> cout_tile 输出 INT32 累加 Tile。
//
// H/W/Cin/Cout 必须位于 types.h 定义的编译期范围内，且 Cout 必须是
// SYSTOLIC_COUT_PAR 的整数倍。非法参数在综合逻辑中直接返回。
void systolic(hls::stream<activation_pack_t>& activation_stream,
              hls::stream<weight_pack_t>& weight_stream,
              hls::stream<acc_tile_t>& output_stream,
              int H,
              int W,
              int Cin,
              int Cout);

#endif  // CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_
