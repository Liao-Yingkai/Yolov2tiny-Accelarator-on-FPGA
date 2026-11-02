#ifndef CNN_ACCELERATOR_IM2ROW_H_
#define CNN_ACCELERATOR_IM2ROW_H_
#include "../types.h"

// 只支持yolov2_tiny_geometry()固定层表，整个调用期间geometry不可变化。
// 输入顺序：y -> x -> cin_group，恰好H*W*G个真实64-bit通道Pack；
// 输出顺序：y -> tile_x -> kernel_pos -> cin_group，恰好
// H*ceil(W/4)*Kh*Kw*G个4空间×8通道Pack。Tile不能跨图像行。
// 不存在的空间Lane与通道尾部清零；卷积Padding为零但不清除spatial_valid。
// stream阻塞时保留完整事务与目的标签，RAM返回和直接输入由调度器对齐。
// 非综合构建严格断言层表契约；硬件不为非法配置生成静默拒绝返回逻辑。
// 不提供旧bool接口；Core/Top暂不迁移，待所有模块重构完成后统一接入新接口。
void im2row(hls::stream<channel_pack_t>& input_stream,
            hls::stream<activation_pack_t>& output_stream,
            const ConvGeometry& geometry);
#endif
