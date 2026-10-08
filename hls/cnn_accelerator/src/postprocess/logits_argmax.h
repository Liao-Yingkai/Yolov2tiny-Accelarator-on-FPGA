#ifndef CNN_ACCELERATOR_POSTPROCESS_LOGITS_ARGMAX_H_
#define CNN_ACCELERATOR_POSTPROCESS_LOGITS_ARGMAX_H_

#include "../types.h"

// 独立10类 argmax：每次调用读取一个 logits Pack，并恰好输出一个类别索引。
// 相同最大值时选择较小的类别索引，与 scalar reference 的 first-maximum
// 规则一致。该模块可以在后续接到 FCPostProcess 之后；当前不改动 Top 接口。
void logits_argmax(hls::stream<logits_pack_t>& input_stream,
                   hls::stream<class_index_t>& output_stream);

#endif  // CNN_ACCELERATOR_POSTPROCESS_LOGITS_ARGMAX_H_
