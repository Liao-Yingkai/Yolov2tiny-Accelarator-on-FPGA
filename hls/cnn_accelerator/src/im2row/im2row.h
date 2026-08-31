#ifndef CNN_ACCELERATOR_IM2ROW_H_
#define CNN_ACCELERATOR_IM2ROW_H_

#include "../types.h"

// Im2Row 流式接口契约：
// - input_stream 按行优先顺序提供恰好 H×W 个真实 pixel_t，不携带卷积 Padding 或行尾补齐；
// - output_stream 产生恰好 H×ceil(W/P)×9 个 activation_pack_t，每个 Pack 是同一 (kh,kw) 下的 P 个空间 Lane；
// - 每行独立划分 Tile，尾部无效 Lane 的9拍均为零且 spatial_valid=0，边界 Padding 为零但 Lane 仍有效；
// - 输入读取和输出写入均采用阻塞语义，下游阻塞时内部 Token、Strip 和输入前沿不得推进；
// - 启动后连续输出以 II=1 为综合目标，实际 II 和存储端口行为仍需综合确认。
void im2row(
    hls::stream<pixel_t>& input_stream,
    hls::stream<activation_pack_t>& output_stream,
    int H,
    int W,
    int Cin
);

#endif  // CNN_ACCELERATOR_IM2ROW_H_
