#ifndef CNN_ACCELERATOR_IM2ROW_H_
#define CNN_ACCELERATOR_IM2ROW_H_
#include "../types.h"

// 只支持yolov2_tiny_geometry()固定层表，整个调用期间geometry不可变化。
//
// 输入顺序：y -> x -> cin_group，恰好H*W*G个真实64-bit通道Pack；
// 输出顺序：y -> tile_x -> kernel_pos -> cin_group，恰好
// H*ceil(W/4)*Kh*Kw*G个4空间×8通道Pack。Tile不能跨图像行。
//
// 输入的通道尾槽由生产端置零。Im2Row 只处理空间边界：不存在的空间 Lane
// 输出为零，卷积 Padding 为零但不清除真实输出位置的 spatial_valid。
// stream阻塞时保留完整事务与目的标签，RAM返回和直接输入由调度器对齐。
//
// 调用方只提供固定层表配置，不逐字段检查或构造兜底配置。
// 非综合构建仅保留输入/输出事务守恒断言。
// Top 对每个 Cout 驻留块重新调用本模块，从预填充开始完整扫描一次输入。
void im2row(hls::stream<channel_pack_t>& input_stream,
            hls::stream<activation_pack_t>& output_stream,
            const ConvGeometry& geometry);
#endif
