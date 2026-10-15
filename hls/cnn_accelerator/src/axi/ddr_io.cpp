#include "ddr_io.h"
#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace ddr_io {
// 辅助函数仅内联到所属加载器，保持原有读取顺序和流水约束。
namespace {
void load_conv_weights(const ap_uint<64>* weights, int offset, int count,
                       hls::stream<weight_pack_t>& output) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
    int reads = 0, writes = 0;
#endif
    for (int pack = 0; pack < count; ++pack) {
        weight_pack_t value;
#pragma HLS ARRAY_PARTITION variable=value.lane complete
        for (int lane = 0; lane < SYSTOLIC_COUT_PAR; ++lane) {
#pragma HLS PIPELINE II=1
            const ap_uint<64> word = weights[offset + pack * SYSTOLIC_COUT_PAR + lane];
            for (int c = 0; c < SYSTOLIC_CIN_PAR; ++c) {
#pragma HLS UNROLL
                value.lane[lane].channel[c] = (ap_int<8>)word.range(c * 8 + 7, c * 8);
            }
#ifndef __SYNTHESIS__
            ++reads;
#endif
        }
        output.write(value);
#ifndef __SYNTHESIS__
        ++writes;
#endif
    }
#ifndef __SYNTHESIS__
    assert(reads == count * SYSTOLIC_COUT_PAR && writes == count);
#endif
}

template <typename Pack, int Lanes>
void load_params(const ap_int<32>* params, int offset, int count,
                 hls::stream<Pack>& output) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
    int reads = 0, writes = 0;
#endif
    for (int pack = 0; pack < count; ++pack) {
        Pack value;
#pragma HLS ARRAY_PARTITION variable=value.bias complete
#pragma HLS ARRAY_PARTITION variable=value.multiplier complete
#pragma HLS ARRAY_PARTITION variable=value.shift complete
        // DDR 每层按完整 bias/multiplier/shift 数组排列；仅缓存当前 pack。
        for (int field = 0; field < ddr_layout::PARAM_FIELDS; ++field) {
            for (int lane = 0; lane < Lanes; ++lane) {
#pragma HLS PIPELINE II=1
                const ap_int<32> word = params[offset + field * (count * Lanes) + pack * Lanes + lane];
                if (field == 0) value.bias[lane] = word;
                else if (field == 1) value.multiplier[lane] = word;
                else value.shift[lane] = word;
#ifndef __SYNTHESIS__
                ++reads;
#endif
            }
        }
        output.write(value);
#ifndef __SYNTHESIS__
        ++writes;
#endif
    }
#ifndef __SYNTHESIS__
    assert(reads == count * ddr_layout::PARAM_FIELDS * Lanes && writes == count);
#endif
}
}  // 匿名命名空间

void image_loader(const ap_uint<64>* image, hls::stream<pixel_t>& output) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    int reads = 0, writes = 0;
#endif
    for (int word = 0; word < ddr_layout::IMAGE_WORDS; ++word) {
        const ap_uint<64> packed = image[word];
#ifndef __SYNTHESIS__
        ++reads;
#endif
        // 一个 DDR word 只读一次，随后用固定小计数逐拍送出八个像素。
        for (int byte = 0; byte < ddr_layout::WORD_BYTES; ++byte) {
#pragma HLS PIPELINE II=1
            pixel_t pixel;
#pragma HLS ARRAY_PARTITION variable=pixel.channel complete
            for (int c = 0; c < IM2ROW_MAX_CHANNELS; ++c) {
#pragma HLS UNROLL
                pixel.channel[c] = 0;
            }
            pixel.channel[0] = (ap_int<8>)packed.range(byte * 8 + 7, byte * 8);
            output.write(pixel);
#ifndef __SYNTHESIS__
            ++writes;
#endif
        }
    }
#ifndef __SYNTHESIS__
    assert(reads == ddr_layout::IMAGE_WORDS);
    assert(writes == CNN_CONV1_INPUT_TRANSACTIONS);
#endif
}

void weight_loader(const ap_uint<64>* weights,
                   hls::stream<weight_pack_t>& conv1,
                   hls::stream<weight_pack_t>& conv2,
                   hls::stream<fc_weight_pack_t>& fc) {
#pragma HLS INLINE off
    load_conv_weights(weights, ddr_layout::CONV1_WEIGHT_OFFSET,
                      CNN_CONV1_WEIGHT_TRANSACTIONS, conv1);
    load_conv_weights(weights, ddr_layout::CONV2_WEIGHT_OFFSET,
                      CNN_CONV2_WEIGHT_TRANSACTIONS, conv2);
#ifndef __SYNTHESIS__
    int reads = 0, writes = 0;
#endif
    for (int pack = 0; pack < ddr_layout::FC_WEIGHT_PACKS; ++pack) {
        fc_weight_pack_t value;
#pragma HLS ARRAY_PARTITION variable=value.value complete dim=0
        for (int lane = 0; lane < FC_COUT_PAR; ++lane) {
#pragma HLS PIPELINE II=1
            const ap_uint<64> word = weights[ddr_layout::FC_WEIGHT_OFFSET + pack * FC_COUT_PAR + lane];
            for (int c = 0; c < FC_CIN_PAR; ++c) {
#pragma HLS UNROLL
                value.value[lane][c] = (ap_int<8>)word.range(c * 8 + 7, c * 8);
            }
#ifndef __SYNTHESIS__
            ++reads;
#endif
        }
        fc.write(value);
#ifndef __SYNTHESIS__
        ++writes;
#endif
    }
#ifndef __SYNTHESIS__
    assert(reads == ddr_layout::FC_WEIGHT_WORDS && writes == ddr_layout::FC_WEIGHT_PACKS);
#endif
}

void param_loader(const ap_int<32>* params,
                  hls::stream<postprocess_param_pack_t>& conv1,
                  hls::stream<postprocess_param_pack_t>& conv2,
                  hls::stream<fc_param_pack_t>& fc) {
#pragma HLS INLINE off
    load_params<postprocess_param_pack_t, SYSTOLIC_COUT_PAR>(params, ddr_layout::CONV1_PARAM_OFFSET, CNN_CONV1_PARAM_TRANSACTIONS, conv1);
    load_params<postprocess_param_pack_t, SYSTOLIC_COUT_PAR>(params, ddr_layout::CONV2_PARAM_OFFSET, CNN_CONV2_PARAM_TRANSACTIONS, conv2);
    load_params<fc_param_pack_t, FC_COUT_PAR>(params, ddr_layout::FC_PARAM_OFFSET, FC_OUT_TILES, fc);
}

void logits_split(hls::stream<logits_pack_t>& input,
                  hls::stream<logits_pack_t>& writer,
                  hls::stream<logits_pack_t>& argmax) {
#pragma HLS INLINE off
    const logits_pack_t value = input.read();
    writer.write(value);
    argmax.write(value);
}

void result_writer(hls::stream<logits_pack_t>& logits,
                   hls::stream<class_index_t>& classes, ap_uint<64>* result) {
#pragma HLS INLINE off
    const logits_pack_t value = logits.read();
    const class_index_t predicted = classes.read();
    ap_uint<64> words[ddr_layout::RESULT_WORDS];
#pragma HLS ARRAY_PARTITION variable=words complete
    for (int i = 0; i < ddr_layout::RESULT_WORDS; ++i) {
#pragma HLS UNROLL
        words[i] = 0;
    }
    for (int i = 0; i < FC_OUTPUTS; ++i) {
#pragma HLS UNROLL
        // 显式转 unsigned byte 保留 signed INT8 的二补码，不做符号扩展。
        words[i / ddr_layout::WORD_BYTES].range((i % ddr_layout::WORD_BYTES) * 8 + 7, (i % ddr_layout::WORD_BYTES) * 8) = (ap_uint<8>)value.value[i];
    }
    words[ddr_layout::CLASS_WORD_OFFSET].range(ddr_layout::CLASS_BIT_OFFSET + 31, ddr_layout::CLASS_BIT_OFFSET) = (ap_uint<32>)predicted;
    for (int i = 0; i < ddr_layout::RESULT_WORDS; ++i) {
#pragma HLS PIPELINE II=1
        result[i] = words[i];
    }
}
}  // 命名空间 ddr_io
