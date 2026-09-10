#include "cnn_accel.h"

#include "fc/fc.h"
#include "im2row/im2row.h"
#include "pool/pool.h"
#include "postprocess/conv_postprocess.h"
#include "postprocess/fc_postprocess.h"
#include "systolic/systolic.h"

#ifndef __SYNTHESIS__
#include <assert.h>
#endif

namespace {

typedef ap_uint<TOP_POOL1_FEATURE_BUFFER_WORD_BITS> pool1_feature_word_t;

// channel[c]固定映射bits[8*c+7:8*c]；位段复制保留INT8补码，不依赖
// channel_pack_t或pixel_t的C++对象布局。
pool1_feature_word_t pack_pool1_channel_pack(const channel_pack_t& input) {
#pragma HLS INLINE
    pool1_feature_word_t word = 0;
    for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
#pragma HLS UNROLL
        const int low_bit = channel * ACTIVATION_BITS;
        word.range(low_bit + ACTIVATION_BITS - 1, low_bit) =
            input.channel[channel].range(ACTIVATION_BITS - 1, 0);
    }
    return word;
}

pixel_t unpack_pool1_pixel(const pool1_feature_word_t& word) {
#pragma HLS INLINE
    pixel_t output;
    for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
#pragma HLS UNROLL
        const int low_bit = channel * ACTIVATION_BITS;
        output.channel[channel].range(ACTIVATION_BITS - 1, 0) =
            word.range(low_bit + ACTIVATION_BITS - 1, low_bit);
    }
    return output;
}

// 唯一的阶段间Feature Buffer进程。数组不作为DATAFLOW进程之间的参数传递：
// Layer1只从buffer_write_stream写入，Layer2只向buffer_read_stream读出。
void feature_buffer_process(
    hls::stream<channel_pack_t>& buffer_write_stream,
    hls::stream<pixel_t>& buffer_read_stream,
    layer_mode_t layer_mode
) {
    static pool1_feature_word_t
        pool1_feature_buffer[TOP_POOL1_FEATURE_BUFFER_DEPTH];
#pragma HLS BIND_STORAGE variable=pool1_feature_buffer type=ram_1p impl=bram

#ifndef __SYNTHESIS__
    int write_stream_read_count = 0;
    int buffer_write_count = 0;
    int buffer_read_count = 0;
    int read_stream_write_count = 0;
#endif

    if (layer_mode == LAYER1_MODE) {
        for (int index = 0; index < CNN_POOL1_OUTPUT_TRANSACTIONS; ++index) {
#pragma HLS PIPELINE II=1
            pool1_feature_buffer[index] =
                pack_pool1_channel_pack(buffer_write_stream.read());
#ifndef __SYNTHESIS__
            ++write_stream_read_count;
            ++buffer_write_count;
#endif
        }
    } else {
        for (int index = 0; index < CNN_CONV2_INPUT_TRANSACTIONS; ++index) {
#pragma HLS PIPELINE II=1
            buffer_read_stream.write(
                unpack_pool1_pixel(pool1_feature_buffer[index]));
#ifndef __SYNTHESIS__
            ++buffer_read_count;
            ++read_stream_write_count;
#endif
        }
    }

#ifndef __SYNTHESIS__
    if (layer_mode == LAYER1_MODE) {
        assert(write_stream_read_count == CNN_POOL1_OUTPUT_TRANSACTIONS);
        assert(buffer_write_count == CNN_POOL1_OUTPUT_TRANSACTIONS);
        assert(buffer_read_count == 0);
        assert(read_stream_write_count == 0);
    } else {
        assert(write_stream_read_count == 0);
        assert(buffer_write_count == 0);
        assert(buffer_read_count == CNN_CONV2_INPUT_TRANSACTIONS);
        assert(read_stream_write_count == CNN_CONV2_INPUT_TRANSACTIONS);
    }
#endif
}

// 唯一输入Router。Layer1使用外部图像；Layer2只使用Feature Buffer输出。
void input_router(
    hls::stream<pixel_t>& image_stream,
    hls::stream<pixel_t>& buffer_read_stream,
    hls::stream<pixel_t>& conv_input_stream,
    layer_mode_t layer_mode
) {
#ifndef __SYNTHESIS__
    int input_read_count = 0;
    int conv_write_count = 0;
#endif

    const int transaction_count = (layer_mode == LAYER1_MODE)
        ? CNN_CONV1_INPUT_TRANSACTIONS : CNN_CONV2_INPUT_TRANSACTIONS;
    for (int index = 0; index < transaction_count; ++index) {
#pragma HLS PIPELINE II=1
        const pixel_t input = (layer_mode == LAYER1_MODE)
            ? image_stream.read() : buffer_read_stream.read();
        conv_input_stream.write(input);
#ifndef __SYNTHESIS__
        ++input_read_count;
        ++conv_write_count;
#endif
    }

#ifndef __SYNTHESIS__
    assert(input_read_count == transaction_count);
    assert(conv_write_count == transaction_count);
#endif
}

// 唯一卷积权重Router，仅消费当前layer所属的外部权重流。
void conv_weight_router(
    hls::stream<weight_pack_t>& conv1_weight_stream,
    hls::stream<weight_pack_t>& conv2_weight_stream,
    hls::stream<weight_pack_t>& conv_weight_stream,
    layer_mode_t layer_mode
) {
#ifndef __SYNTHESIS__
    int input_read_count = 0;
    int output_write_count = 0;
#endif

    const int transaction_count = (layer_mode == LAYER1_MODE)
        ? CNN_CONV1_WEIGHT_TRANSACTIONS : CNN_CONV2_WEIGHT_TRANSACTIONS;
    for (int index = 0; index < transaction_count; ++index) {
#pragma HLS PIPELINE II=1
        const weight_pack_t weight = (layer_mode == LAYER1_MODE)
            ? conv1_weight_stream.read() : conv2_weight_stream.read();
        conv_weight_stream.write(weight);
#ifndef __SYNTHESIS__
        ++input_read_count;
        ++output_write_count;
#endif
    }

#ifndef __SYNTHESIS__
    assert(input_read_count == transaction_count);
    assert(output_write_count == transaction_count);
#endif
}

// 唯一卷积后处理参数Router，仅消费当前layer所属的外部参数流。
void conv_param_router(
    hls::stream<postprocess_param_pack_t>& conv1_param_stream,
    hls::stream<postprocess_param_pack_t>& conv2_param_stream,
    hls::stream<postprocess_param_pack_t>& conv_param_stream,
    layer_mode_t layer_mode
) {
#ifndef __SYNTHESIS__
    int input_read_count = 0;
    int output_write_count = 0;
#endif

    const int transaction_count = (layer_mode == LAYER1_MODE)
        ? CNN_CONV1_PARAM_TRANSACTIONS : CNN_CONV2_PARAM_TRANSACTIONS;
    for (int index = 0; index < transaction_count; ++index) {
#pragma HLS PIPELINE II=1
        const postprocess_param_pack_t param = (layer_mode == LAYER1_MODE)
            ? conv1_param_stream.read() : conv2_param_stream.read();
        conv_param_stream.write(param);
#ifndef __SYNTHESIS__
        ++input_read_count;
        ++output_write_count;
#endif
    }

#ifndef __SYNTHESIS__
    assert(input_read_count == transaction_count);
    assert(output_write_count == transaction_count);
#endif
}

// 唯一输出Router。Layer1只写入Feature Buffer路径；Layer2只送入FC。
void output_router(
    hls::stream<channel_pack_t>& pool_output_stream,
    hls::stream<channel_pack_t>& buffer_write_stream,
    hls::stream<channel_pack_t>& fc_input_stream,
    layer_mode_t layer_mode
) {
#ifndef __SYNTHESIS__
    int pool_read_count = 0;
    int buffer_write_count = 0;
    int fc_write_count = 0;
#endif

    const int transaction_count = (layer_mode == LAYER1_MODE)
        ? CNN_POOL1_OUTPUT_TRANSACTIONS : CNN_POOL2_OUTPUT_TRANSACTIONS;
    for (int index = 0; index < transaction_count; ++index) {
#pragma HLS PIPELINE II=1
        const channel_pack_t output = pool_output_stream.read();
        if (layer_mode == LAYER1_MODE) {
            buffer_write_stream.write(output);
#ifndef __SYNTHESIS__
            ++buffer_write_count;
#endif
        } else {
            fc_input_stream.write(output);
#ifndef __SYNTHESIS__
            ++fc_write_count;
#endif
        }
#ifndef __SYNTHESIS__
        ++pool_read_count;
#endif
    }

#ifndef __SYNTHESIS__
    assert(pool_read_count == transaction_count);
    if (layer_mode == LAYER1_MODE) {
        assert(buffer_write_count == CNN_POOL1_OUTPUT_TRANSACTIONS);
        assert(fc_write_count == 0);
    } else {
        assert(buffer_write_count == 0);
        assert(fc_write_count == CNN_POOL2_OUTPUT_TRANSACTIONS);
    }
#endif
}

// FC进程在Layer1不读写任何stream；Layer2复用现有独立FC模块。
void fully_connected_guard(
    hls::stream<channel_pack_t>& fc_input_stream,
    hls::stream<fc_weight_pack_t>& fc_weight_stream,
    hls::stream<fc_acc_pack_t>& fc_acc_stream,
    layer_mode_t layer_mode
) {
    if (layer_mode == LAYER2_MODE) {
        fully_connected(fc_input_stream, fc_weight_stream, fc_acc_stream);
    }
}

// FC后处理进程在Layer1不读写任何stream；Layer2只输出一次10-logit Pack。
void fc_postprocess_guard(
    hls::stream<fc_acc_pack_t>& fc_acc_stream,
    hls::stream<fc_param_pack_t>& fc_param_stream,
    hls::stream<logits_pack_t>& logits_stream,
    layer_mode_t layer_mode
) {
    if (layer_mode == LAYER2_MODE) {
        fc_postprocess(fc_acc_stream, fc_param_stream, logits_stream);
    }
}

// 卷积流水及FC tail的唯一DATAFLOW区域。所有process始终有一个调用点；
// mode仅决定每个Router/Guard的固定事务数量，不能改变进程拓扑。
void run_layer_phase(
    hls::stream<pixel_t>& image_stream,
    hls::stream<weight_pack_t>& conv1_weight_stream,
    hls::stream<postprocess_param_pack_t>& conv1_param_stream,
    hls::stream<weight_pack_t>& conv2_weight_stream,
    hls::stream<postprocess_param_pack_t>& conv2_param_stream,
    hls::stream<fc_weight_pack_t>& fc_weight_stream,
    hls::stream<fc_param_pack_t>& fc_param_stream,
    hls::stream<logits_pack_t>& logits_stream,
    layer_mode_t layer_mode
) {
#pragma HLS INLINE off
    hls::stream<pixel_t> buffer_read_stream("buffer_read_stream");
    hls::stream<pixel_t> conv_input_stream("conv_input_stream");
    hls::stream<activation_pack_t> activation_stream("activation_stream");
    hls::stream<weight_pack_t> conv_weight_stream("conv_weight_stream");
    hls::stream<acc_tile_t> acc_stream("acc_stream");
    hls::stream<postprocess_param_pack_t> conv_param_stream(
        "conv_param_stream");
    hls::stream<activation_tile_t> post_output_stream("post_output_stream");
    hls::stream<channel_pack_t> pool_output_stream("pool_output_stream");
    hls::stream<channel_pack_t> buffer_write_stream("buffer_write_stream");
    hls::stream<channel_pack_t> fc_input_stream("fc_input_stream");
    hls::stream<fc_acc_pack_t> fc_acc_stream("fc_acc_stream");

#pragma HLS STREAM variable=buffer_read_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=conv_input_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=activation_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=conv_weight_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=acc_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=conv_param_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=post_output_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=pool_output_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=buffer_write_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=fc_input_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=fc_acc_stream depth=TOP_STREAM_DEPTH

#pragma HLS DATAFLOW
    feature_buffer_process(buffer_write_stream, buffer_read_stream, layer_mode);
    input_router(image_stream, buffer_read_stream, conv_input_stream,
                 layer_mode);
    conv_weight_router(conv1_weight_stream, conv2_weight_stream,
                       conv_weight_stream, layer_mode);
    conv_param_router(conv1_param_stream, conv2_param_stream,
                      conv_param_stream, layer_mode);
    im2row(conv_input_stream, activation_stream, layer_mode);
    systolic(activation_stream, conv_weight_stream, acc_stream, layer_mode);
    conv_postprocess(acc_stream, conv_param_stream, post_output_stream,
                     layer_mode);
    maxpool(post_output_stream, pool_output_stream, layer_mode);
    output_router(pool_output_stream, buffer_write_stream, fc_input_stream,
                  layer_mode);
    fully_connected_guard(fc_input_stream, fc_weight_stream, fc_acc_stream,
                          layer_mode);
    fc_postprocess_guard(fc_acc_stream, fc_param_stream, logits_stream,
                         layer_mode);
}

}  // namespace

void cnn_accelerator(
    hls::stream<pixel_t>& image_stream,
    hls::stream<weight_pack_t>& conv1_weight_stream,
    hls::stream<postprocess_param_pack_t>& conv1_param_stream,
    hls::stream<weight_pack_t>& conv2_weight_stream,
    hls::stream<postprocess_param_pack_t>& conv2_param_stream,
    hls::stream<fc_weight_pack_t>& fc_weight_stream,
    hls::stream<fc_param_pack_t>& fc_param_stream,
    hls::stream<logits_pack_t>& logits_stream
) {
    // 顶层没有DATAFLOW、PIPELINE或UNROLL：两次顺序调用以函数返回形成完整
    // layer barrier，并在源码中只保留一个run_layer_phase调用位置。
    for (int phase = 0; phase < TOP_PHASE_COUNT; ++phase) {
        const layer_mode_t layer_mode =
            (phase == 0) ? LAYER1_MODE : LAYER2_MODE;
        run_layer_phase(image_stream, conv1_weight_stream, conv1_param_stream,
                        conv2_weight_stream, conv2_param_stream,
                        fc_weight_stream, fc_param_stream, logits_stream,
                        layer_mode);
    }
}
