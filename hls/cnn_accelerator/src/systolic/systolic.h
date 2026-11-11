#ifndef CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_
#define CNN_ACCELERATOR_SYSTOLIC_SYSTOLIC_H_

#include "../types.h"

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
