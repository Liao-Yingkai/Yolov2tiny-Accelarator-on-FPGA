#ifndef CNN_ACCELERATOR_DDR_LAYOUT_H_
#define CNN_ACCELERATOR_DDR_LAYOUT_H_

#include "../config.h"

namespace ddr_layout {

constexpr int WORD_BITS = 64;
constexpr int WORD_BYTES = WORD_BITS / 8;
constexpr int PARAM_WORD_BYTES = 32 / 8;
constexpr int PARAM_FIELDS = 3;
constexpr int WEIGHT_WORDS_PER_PACK = CONV_COUT_PAR;  // 每个 Cout lane 占一个 64-bit 字。
constexpr int FEATURE_BANK_COUNT = 2;
constexpr int FEATURE_ALIGNMENT_BYTES = 64;

constexpr int feature_input_bytes(const ConvGeometry& geometry) {
    return geometry.h * geometry.w * geometry.cin;  // RGB 首图在 DDR 中仍是紧凑的 3 通道。
}

constexpr int weight_layer_words(const ConvGeometry& geometry) {
    return geometry.weight_transactions * WEIGHT_WORDS_PER_PACK;
}

constexpr int param_layer_words(const ConvGeometry& geometry) {
    return PARAM_FIELDS * geometry.cout;  // [全部 bias][全部 multiplier][全部 shift]。
}

// 按固定层表累加。前 layer_count 层的长度也就是下一层的段偏移。
constexpr int weight_prefix_words(int layer_count) {
    return layer_count == 0 ? 0 :
        weight_prefix_words(layer_count - 1) + weight_layer_words(YOLO_TINY_GEOMETRIES[layer_count - 1]);
}

constexpr int param_prefix_words(int layer_count) {
    return layer_count == 0 ? 0 :
        param_prefix_words(layer_count - 1) + param_layer_words(YOLO_TINY_GEOMETRIES[layer_count - 1]);
}

constexpr int IMAGE_BYTES = feature_input_bytes(yolov2_tiny_geometry(YoloTinyLayer::Conv1));
constexpr int IMAGE_WORDS = IMAGE_BYTES / WORD_BYTES;
constexpr int FEATURE_BANK_BYTES = FEATURE_ALIGNMENT_BYTES * ceil_div(
    config_max(IMAGE_BYTES, WORD_BYTES * yolo_max_field(&ConvGeometry::output_transactions)),
    FEATURE_ALIGNMENT_BYTES);
constexpr int FEATURE_BANK_WORDS = FEATURE_BANK_BYTES / WORD_BYTES;
constexpr int FEATURE_WORDS = FEATURE_BANK_COUNT * FEATURE_BANK_WORDS;
constexpr int FEATURE_BYTES = FEATURE_WORDS * WORD_BYTES;
constexpr int WEIGHT_WORDS = weight_prefix_words(YOLO_TINY_LAYER_COUNT);
constexpr int WEIGHT_BYTES = WEIGHT_WORDS * WORD_BYTES;
constexpr int PARAM_WORDS = param_prefix_words(YOLO_TINY_LAYER_COUNT);
constexpr int PARAM_BYTES = PARAM_WORDS * PARAM_WORD_BYTES;
constexpr int RESULT_WORDS = yolov2_tiny_geometry(YoloTinyLayer::Prediction).output_transactions;
// 从 A 读入，奇数个层依次交换角色，最终结果在 B。
constexpr int RESULT_WORD_OFFSET = (YOLO_TINY_LAYER_COUNT % FEATURE_BANK_COUNT) * FEATURE_BANK_WORDS;
constexpr int RESULT_BYTES = RESULT_WORDS * WORD_BYTES;

static_assert(FEATURE_PACK_CHANNELS * ACTIVATION_BITS == WORD_BITS &&
              CONV_CIN_PAR * WEIGHT_BITS == WORD_BITS,
              "One 64-bit word must contain one feature pack or one Cout lane of a weight pack");
static_assert(IMAGE_BYTES % WORD_BYTES == 0 && FEATURE_BANK_BYTES % WORD_BYTES == 0,
              "RGB input and Feature bank boundaries must be aligned to 64-bit words");
static_assert(RESULT_WORD_OFFSET == FEATURE_BANK_WORDS,
              "The fixed nine-layer accelerator must write its final output to bank B");

}  // namespace ddr_layout

#endif  // CNN_ACCELERATOR_DDR_LAYOUT_H_
