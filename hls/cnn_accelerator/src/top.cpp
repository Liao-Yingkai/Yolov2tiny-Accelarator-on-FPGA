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

// 唯一输入Router。Layer1使用外部图像；Layer2直接消费跨阶段Pool1 FIFO。
void input_router(
    hls::stream<pixel_t>& image_stream,
    hls::stream<channel_pack_t>& pool1_feature_fifo,
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
            ? image_stream.read() : pool1_feature_fifo.read();
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

// 唯一输出Router。Layer1直接写入跨阶段Pool1 FIFO；Layer2只送入FC。
void output_router(
    hls::stream<channel_pack_t>& pool_output_stream,
    hls::stream<channel_pack_t>& pool1_feature_fifo,
    hls::stream<channel_pack_t>& fc_input_stream,
    layer_mode_t layer_mode
) {
#ifndef __SYNTHESIS__
    int pool_read_count = 0;
    int feature_fifo_write_count = 0;
    int fc_write_count = 0;
#endif

    const int transaction_count = (layer_mode == LAYER1_MODE)
        ? CNN_POOL1_OUTPUT_TRANSACTIONS : CNN_POOL2_OUTPUT_TRANSACTIONS;
    for (int index = 0; index < transaction_count; ++index) {
#pragma HLS PIPELINE II=1
        const channel_pack_t output = pool_output_stream.read();
        if (layer_mode == LAYER1_MODE) {
            pool1_feature_fifo.write(output);
#ifndef __SYNTHESIS__
            ++feature_fifo_write_count;
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
        assert(feature_fifo_write_count == CNN_POOL1_OUTPUT_TRANSACTIONS);
        assert(fc_write_count == 0);
    } else {
        assert(feature_fifo_write_count == 0);
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
    // static 保证Pool1写入的196个pack跨phase调用保留；Layer2会按原顺序
    // 全部读出，因此连续Top调用的FIFO初始状态仍为空。
    static hls::stream<channel_pack_t> pool1_feature_fifo(
        "pool1_feature_fifo");
    hls::stream<pixel_t> conv_input_stream("conv_input_stream");
    hls::stream<activation_pack_t> activation_stream("activation_stream");
    hls::stream<weight_pack_t> conv_weight_stream("conv_weight_stream");
    hls::stream<acc_tile_t> acc_stream("acc_stream");
    hls::stream<postprocess_param_pack_t> conv_param_stream(
        "conv_param_stream");
    hls::stream<activation_tile_t> post_output_stream("post_output_stream");
    hls::stream<channel_pack_t> pool_output_stream("pool_output_stream");
    hls::stream<channel_pack_t> fc_input_stream("fc_input_stream");
    hls::stream<fc_acc_pack_t> fc_acc_stream("fc_acc_stream");

#pragma HLS STREAM variable=pool1_feature_fifo depth=TOP_POOL1_FEATURE_FIFO_DEPTH
#pragma HLS BIND_STORAGE variable=pool1_feature_fifo type=fifo impl=bram
#pragma HLS STREAM variable=conv_input_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=activation_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=conv_weight_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=acc_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=conv_param_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=post_output_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=pool_output_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=fc_input_stream depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=fc_acc_stream depth=TOP_STREAM_DEPTH

#pragma HLS DATAFLOW
    input_router(image_stream, pool1_feature_fifo, conv_input_stream,
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
    output_router(pool_output_stream, pool1_feature_fifo, fc_input_stream,
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
