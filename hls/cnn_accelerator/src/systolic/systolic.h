#ifndef CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_
#define CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_

#include "../types.h"

// 旧 MNIST 广播式实现：一次调用处理完整的一层3x3卷积，不含 bias、requant 或 ReLU。
//
// activation_stream：严格消费 Im2Row 的 row -> tile_x -> token 顺序。
// weight_stream：仅在本层第一个 spatial tile 消费，顺序为
//                cout_tile -> token(kh*3+kw)。
// output_stream：保持 Im2Row 等价的 spatial_tile(row -> tile_x) -> cout_tile 顺序输出 INT32 累加 Tile。
//
// layer_mode在一次完整层调用中保持稳定：LAYER1_MODE选择Conv1，LAYER2_MODE选择Conv2。
// Conv1仅channel0参与点积，无效Cin允许非零；Conv2全部8路参与。
// 物理stream宽度与replay格式不变，spatial_valid仍按既有契约透传。
void systolic(hls::stream<activation_pack_t>& activation_stream,
              hls::stream<weight_pack_t>& weight_stream,
              hls::stream<acc_tile_t>& output_stream,
              layer_mode_t layer_mode);

// YOLO：geometry 为固定层表中的一层，调用期间配置不变。
// 一次处理从 first_cout_tile 开始的 block_cout_tiles 个连续 4-Cout Tile。
// 先读 B*R 个权重 Pack（local_cout_tile -> kh -> kw -> group），再消费
// 一次完整 Im2Row 扫描 S*R 个 Pack。结果按 y -> tile_x -> local_cout_tile
// 写出 S*B 个 INT32 Tile；不含 bias/量化，空间和 Cout mask 均随结果传递。
void systolic_OS(hls::stream<activation_pack_t>& activation_stream,
                 hls::stream<weight_pack_t>& weight_stream,
                 hls::stream<yolo_acc_tile_t>& output_stream,
                 const ConvGeometry& geometry,
                 int first_cout_tile,
                 int block_cout_tiles);

#endif  // CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_
