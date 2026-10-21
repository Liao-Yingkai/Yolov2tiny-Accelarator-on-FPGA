#include "ddr_io.h"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace ddr_io {
// 辅助函数仅内联到所属加载器；卷积共用一套按字节消费的固定网络加载逻辑。
namespace {
void load_conv_weights(const ap_uint<64>* weights,
                        hls::stream<weight_pack_t>& conv1,
                        hls::stream<weight_pack_t>& conv2,
                        layer_mode_t layer_mode) {
#pragma HLS INLINE
    const bool is_conv2 = (layer_mode == LAYER2_MODE);
    const int count = is_conv2 ? CNN_CONV2_WEIGHT_TRANSACTIONS : CNN_CONV1_WEIGHT_TRANSACTIONS;
    int word_address = is_conv2 ? ddr_layout::CONV2_WEIGHT_OFFSET : ddr_layout::CONV1_WEIGHT_OFFSET;
    ap_uint<64> cached = 0;
    ap_uint<3> byte_offset = 0;
#ifndef __SYNTHESIS__
    int reads = 0, writes = 0;
#endif
    // 游标和 word 缓存在整层内连续，不在 pack/Cout tile 边界重置。
    // Conv1 每 lane 消费1字节，pack8/9共享word；Conv2每lane消费完整8字节。
conv_pack_loop:
    for (int pack = 0; pack < count; ++pack) {
        weight_pack_t value;
#pragma HLS ARRAY_PARTITION variable=value.lane complete
conv_lane_loop:
        for (int lane = 0; lane < SYSTOLIC_COUT_PAR; ++lane) {
#pragma HLS PIPELINE II=1
            if (byte_offset == 0) {
                cached = weights[word_address++];
#ifndef __SYNTHESIS__
                ++reads;
#endif
            }
            // 每个字段都来自已读word，完整初始化。Conv1的c1..7只是
            // 确定don't-care，不补零；有效Cin由Systolic选择性点积保证。
            for (int c = 0; c < SYSTOLIC_CIN_PAR; ++c) {
#pragma HLS UNROLL
                value.lane[lane].channel[c] = (ap_int<8>)cached.range(c * 8 + 7, c * 8);
            }
            if (!is_conv2) {
                const int bit = byte_offset.to_int() * 8;
                value.lane[lane].channel[0] = (ap_int<8>)cached.range(bit + 7, bit);
                byte_offset = byte_offset + 1; // 3bit自然绕回0，下次才读新word。
            }
        }
        if (is_conv2) conv2.write(value);
        else conv1.write(value);
#ifndef __SYNTHESIS__
        ++writes;
#endif
    }
#ifndef __SYNTHESIS__
    assert(reads == (is_conv2 ? ddr_layout::CONV2_WEIGHT_WORDS : ddr_layout::CONV1_WEIGHT_WORDS));
    assert(writes == count);
    assert(byte_offset == 0);
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
                // 当前 word 的字节作为确定 don't-care，不额外读或补零。
                // Im2Row 保持既有 Conv1 通道规范化，丢弃 channel1..7。
                pixel.channel[c] = (ap_int<8>)packed.range(c * 8 + 7, c * 8);
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
    // 非展开、非流水的两阶段控制，只有一个helper源码调用点。
    // 统一lane循环后Conv1每pack消费4个lane周期，不再使用原并行快路径。
conv_layer_loop:
    for (int layer = 0; layer < 2; ++layer) {
        const layer_mode_t mode = (layer == 0) ? LAYER1_MODE : LAYER2_MODE;
        load_conv_weights(weights, conv1, conv2, mode);
    }
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
