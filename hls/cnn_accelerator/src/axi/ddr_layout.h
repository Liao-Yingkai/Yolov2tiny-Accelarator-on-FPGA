#ifndef CNN_ACCELERATOR_DDR_LAYOUT_H_
#define CNN_ACCELERATOR_DDR_LAYOUT_H_

#include "../config.h"

// PS 与 HLS 共用的 word 布局；仅使用内建整数，不依赖 ap_int/hls_stream。
namespace ddr_layout {
static const int WORD_BITS = 64;
static const int WORD_BYTES = WORD_BITS / 8;
static const int PARAM_WORD_BYTES = 32 / 8;
static const int IMAGE_WORDS = CNN_CONV1_INPUT_TRANSACTIONS / WORD_BYTES;
static const int CONV_WEIGHT_WORDS_PER_PACK = SYSTOLIC_COUT_PAR;
static const int FC_WEIGHT_WORDS_PER_PACK = FC_COUT_PAR;
static const int CONV1_WEIGHT_OFFSET = 0;
static const int CONV1_WEIGHT_WORDS = CNN_CONV1_WEIGHT_TRANSACTIONS * CONV_WEIGHT_WORDS_PER_PACK;
static const int CONV2_WEIGHT_OFFSET = CONV1_WEIGHT_OFFSET + CONV1_WEIGHT_WORDS;
static const int CONV2_WEIGHT_WORDS = CNN_CONV2_WEIGHT_TRANSACTIONS * CONV_WEIGHT_WORDS_PER_PACK;
static const int FC_WEIGHT_OFFSET = CONV2_WEIGHT_OFFSET + CONV2_WEIGHT_WORDS;
static const int FC_WEIGHT_PACKS = FC_OUT_TILES * FC_INPUT_PACKS;
static const int FC_WEIGHT_WORDS = FC_WEIGHT_PACKS * FC_WEIGHT_WORDS_PER_PACK;
static const int WEIGHT_WORDS = FC_WEIGHT_OFFSET + FC_WEIGHT_WORDS;
static const int PARAM_FIELDS = 3;
// 逻辑 stream pack 的大小，不表示 DDR 中连续的参数块。
static const int CONV_PARAM_WORDS_PER_PACK = PARAM_FIELDS * SYSTOLIC_COUT_PAR;
static const int FC_PARAM_WORDS_PER_PACK = PARAM_FIELDS * FC_COUT_PAR;
static const int CONV1_PARAM_FIELD_STRIDE = CNN_CONV1_COUT;
static const int CONV2_PARAM_FIELD_STRIDE = CNN_CONV2_COUT;
static const int FC_PARAM_FIELD_STRIDE = FC_OUTPUTS;
static const int CONV1_PARAM_OFFSET = 0;
static const int CONV1_PARAM_WORDS = CNN_CONV1_PARAM_TRANSACTIONS * CONV_PARAM_WORDS_PER_PACK;
static const int CONV2_PARAM_OFFSET = CONV1_PARAM_OFFSET + CONV1_PARAM_WORDS;
static const int CONV2_PARAM_WORDS = CNN_CONV2_PARAM_TRANSACTIONS * CONV_PARAM_WORDS_PER_PACK;
static const int FC_PARAM_OFFSET = CONV2_PARAM_OFFSET + CONV2_PARAM_WORDS;
static const int FC_PARAM_WORDS = FC_OUT_TILES * FC_PARAM_WORDS_PER_PACK;
static const int PARAM_WORDS = FC_PARAM_OFFSET + FC_PARAM_WORDS;
static const int LOGIT_WORDS = (FC_OUTPUTS + WORD_BYTES - 1) / WORD_BYTES;
static const int RESULT_CLASS_BYTE_OFFSET = 12;
static const int CLASS_WORD_OFFSET = RESULT_CLASS_BYTE_OFFSET / WORD_BYTES;
static const int CLASS_BIT_OFFSET = (RESULT_CLASS_BYTE_OFFSET % WORD_BYTES) * 8;
static const int RESULT_WORDS = LOGIT_WORDS;
static const int IMAGE_BYTES = IMAGE_WORDS * WORD_BYTES;
static const int WEIGHT_BYTES = WEIGHT_WORDS * WORD_BYTES;
static const int PARAM_BYTES = PARAM_WORDS * PARAM_WORD_BYTES;
static const int RESULT_BYTES = RESULT_WORDS * WORD_BYTES;
static_assert(ACTIVATION_BITS == 8 && WEIGHT_BITS == 8 && ACCUMULATOR_BITS == 32,
              "DDR format requires signed INT8 and INT32");
static_assert(CNN_CONV1_INPUT_TRANSACTIONS % WORD_BYTES == 0 && CNN_CONV1_CIN == 1,
              "Image must fill raster words with one logical channel");
static_assert(SYSTOLIC_CIN_PAR == WORD_BYTES && FC_CIN_PAR == WORD_BYTES,
              "One weight word must hold one complete Cin lane");
static_assert(FC_OUTPUTS == 10 && LOGIT_WORDS == 2 && RESULT_WORDS == 2 &&
              RESULT_BYTES == 16 && CLASS_WORD_OFFSET == 1 && CLASS_BIT_OFFSET == 32,
              "Result format is fixed to ten logits, two reserved bytes and uint32 class");
static_assert(IMAGE_WORDS == 98 && WEIGHT_WORDS == 1196 && PARAM_WORDS == 102,
              "DDR wrapper is fixed to the selected MNIST network");
}  // namespace ddr_layout
#endif
