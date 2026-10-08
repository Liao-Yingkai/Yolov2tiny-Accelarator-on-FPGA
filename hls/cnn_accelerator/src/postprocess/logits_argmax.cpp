#include "logits_argmax.h"

static_assert(FC_OUTPUTS == 10,
              "logits_argmax tree is fixed to exactly 10 classes");

namespace {

struct argmax_candidate_t {
    logit_t value;
    class_index_t index;
};

static argmax_candidate_t choose_better(argmax_candidate_t left,
                                        argmax_candidate_t right) {
#pragma HLS INLINE
    // 固定平衡树中，每次调用的 left/right 分别来自严格递增且不重叠的
    // 类别索引区间，left 区间始终较小。因此 >= 在相等时保留 first index，
    // 每个节点只需一次 signed INT8 比较。
    return (left.value >= right.value) ? left : right;
}

}  // namespace

void logits_argmax(hls::stream<logits_pack_t>& input_stream,
                   hls::stream<class_index_t>& output_stream) {
    const logits_pack_t logits = input_stream.read();
    argmax_candidate_t level0[FC_OUTPUTS];
    argmax_candidate_t level1[5];
    argmax_candidate_t level2[3];
    argmax_candidate_t level3[2];
#pragma HLS ARRAY_PARTITION variable=level0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=level1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=level2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=level3 complete dim=1

    // 第一层：10 -> 5，五个比较器并行工作。
    for (int i = 0; i < FC_OUTPUTS; ++i) {
#pragma HLS UNROLL
        level0[i].value = logits.value[i];
        level0[i].index = (class_index_t)i;
    }
    for (int i = 0; i < 5; ++i) {
#pragma HLS UNROLL
        level1[i] = choose_better(level0[2 * i], level0[2 * i + 1]);
    }

    // 后续平衡树：5 -> 3 -> 2 -> 1。共9次 signed INT8 比较，比较
    // 深度为4层；不构造沿10个元素逐项累积的串行选择链。
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        level2[i] = choose_better(level1[2 * i], level1[2 * i + 1]);
    }
    level2[2] = level1[4];

    level3[0] = choose_better(level2[0], level2[1]);
    level3[1] = level2[2];

    const argmax_candidate_t winner = choose_better(level3[0], level3[1]);
    output_stream.write(winner.index);
}
