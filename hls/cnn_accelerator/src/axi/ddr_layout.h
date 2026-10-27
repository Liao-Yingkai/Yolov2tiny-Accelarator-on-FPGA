#ifndef CNN_ACCELERATOR_DDR_LAYOUT_H_
#define CNN_ACCELERATOR_DDR_LAYOUT_H_

#include "../config.h"

// PL侧word布局；PS另行维护常量。仅使用内建整数，不依赖ap_int/hls_stream。
namespace ddr_layout {
constexpr int WORD_BITS = 64;
constexpr int WORD_BYTES = WORD_BITS / 8;
constexpr int PARAM_WORD_BYTES = 32 / 8;
constexpr int IMAGE_WORDS = CNN_CONV1_INPUT_TRANSACTIONS / WORD_BYTES;
constexpr int CONV2_WEIGHT_WORDS_PER_PACK = SYSTOLIC_COUT_PAR;
constexpr int FC_WEIGHT_WORDS_PER_PACK = FC_COUT_PAR;
constexpr int CONV1_WEIGHT_OFFSET = 0;
// 权重头布局 v2：直接紧密拼接原始字节；Conv1 不保存七个无效 Cin。
constexpr int WEIGHT_LAYOUT_VERSION = 2;
constexpr int CONV1_WEIGHT_WORDS = CNN_CONV1_COUT * SYSTOLIC_KERNEL_TOKENS * CNN_CONV1_CIN / WORD_BYTES;
constexpr int CONV2_WEIGHT_OFFSET = CONV1_WEIGHT_OFFSET + CONV1_WEIGHT_WORDS;
constexpr int CONV2_WEIGHT_WORDS = CNN_CONV2_WEIGHT_TRANSACTIONS * CONV2_WEIGHT_WORDS_PER_PACK;
constexpr int FC_WEIGHT_OFFSET = CONV2_WEIGHT_OFFSET + CONV2_WEIGHT_WORDS;
constexpr int FC_WEIGHT_PACKS = FC_OUT_TILES * FC_INPUT_PACKS;
constexpr int FC_WEIGHT_WORDS = FC_WEIGHT_PACKS * FC_WEIGHT_WORDS_PER_PACK;
constexpr int WEIGHT_WORDS = FC_WEIGHT_OFFSET + FC_WEIGHT_WORDS;
constexpr int PARAM_FIELDS = 3;
// 逻辑 stream pack 的大小，不表示 DDR 中连续的参数块。
constexpr int CONV_PARAM_WORDS_PER_PACK = PARAM_FIELDS * SYSTOLIC_COUT_PAR;
constexpr int FC_PARAM_WORDS_PER_PACK = PARAM_FIELDS * FC_COUT_PAR;
constexpr int CONV1_PARAM_FIELD_STRIDE = CNN_CONV1_COUT;
constexpr int CONV2_PARAM_FIELD_STRIDE = CNN_CONV2_COUT;
constexpr int FC_PARAM_FIELD_STRIDE = FC_OUTPUTS;
constexpr int CONV1_PARAM_OFFSET = 0;
constexpr int CONV1_PARAM_WORDS = CNN_CONV1_PARAM_TRANSACTIONS * CONV_PARAM_WORDS_PER_PACK;
constexpr int CONV2_PARAM_OFFSET = CONV1_PARAM_OFFSET + CONV1_PARAM_WORDS;
constexpr int CONV2_PARAM_WORDS = CNN_CONV2_PARAM_TRANSACTIONS * CONV_PARAM_WORDS_PER_PACK;
constexpr int FC_PARAM_OFFSET = CONV2_PARAM_OFFSET + CONV2_PARAM_WORDS;
constexpr int FC_PARAM_WORDS = FC_OUT_TILES * FC_PARAM_WORDS_PER_PACK;
constexpr int PARAM_WORDS = FC_PARAM_OFFSET + FC_PARAM_WORDS;
constexpr int RESULT_PROBABILITY_BYTES = 2;
constexpr int RESULT_CLASS_BYTE_OFFSET = FC_OUTPUTS * RESULT_PROBABILITY_BYTES;
constexpr int CLASS_WORD_OFFSET = RESULT_CLASS_BYTE_OFFSET / WORD_BYTES;
constexpr int CLASS_BIT_OFFSET = (RESULT_CLASS_BYTE_OFFSET % WORD_BYTES) * 8;
constexpr int RESULT_BYTES = RESULT_CLASS_BYTE_OFFSET + 4;
constexpr int RESULT_WORDS = (RESULT_BYTES + WORD_BYTES - 1) / WORD_BYTES;
constexpr int IMAGE_BYTES = IMAGE_WORDS * WORD_BYTES;
constexpr int WEIGHT_BYTES = WEIGHT_WORDS * WORD_BYTES;
constexpr int PARAM_BYTES = PARAM_WORDS * PARAM_WORD_BYTES;
static_assert(ACTIVATION_BITS == 8 && WEIGHT_BITS == 8 && ACCUMULATOR_BITS == 32,
              "DDR format requires signed INT8 and INT32");
static_assert(CNN_CONV1_INPUT_TRANSACTIONS % WORD_BYTES == 0 && CNN_CONV1_CIN == 1,
              "Image must fill raster words with one logical channel");
static_assert(SYSTOLIC_CIN_PAR == WORD_BYTES && FC_CIN_PAR == WORD_BYTES,
              "One weight word must hold one complete Cin lane");
static_assert(FC_OUTPUTS == 10 && RESULT_WORDS == 3 &&
              RESULT_BYTES == 24 && CLASS_WORD_OFFSET == 2 && CLASS_BIT_OFFSET == 32,
              "Result: ten uint16 probabilities and uint32 class");
static_assert(IMAGE_WORDS == 98 && CONV1_WEIGHT_WORDS == 9 &&
              CONV2_WEIGHT_OFFSET == 9 && FC_WEIGHT_OFFSET == 153 &&
              WEIGHT_WORDS == 1133 && PARAM_WORDS == 102,
              "DDR wrapper is fixed to the selected MNIST network");
}  // namespace ddr_layout
#endif
