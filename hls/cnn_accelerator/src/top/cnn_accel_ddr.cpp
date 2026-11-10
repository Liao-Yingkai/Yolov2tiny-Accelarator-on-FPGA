#include "cnn_accel_ddr.h"

#include "../axi/ddr_io.h"
#include "../im2row/im2row.h"
#include "../pool/pool.h"
#include "../postprocess/conv_postprocess.h"
#include "../systolic/systolic.h"

namespace {

constexpr int FIFO_DEPTH = 2;

// 一次只扫描并计算一个 Cout 驻留块。新调用重新开始 Im2Row 的滑窗状态。
void run_conv_block(const ap_uint<64>* feature_src,
                    hls::stream<weight_pack_t>& weight_stream,
                    hls::stream<yolo_acc_tile_t>& acc_stream,
                    const ConvGeometry& geometry,
                    int first_cout_tile, int block_tiles) {
#pragma HLS INLINE off
    hls::stream<channel_pack_t> feature_stream("feature_stream");
    hls::stream<activation_pack_t> window_stream("window_stream");
#pragma HLS STREAM variable=feature_stream depth=FIFO_DEPTH
#pragma HLS STREAM variable=window_stream depth=FIFO_DEPTH
#pragma HLS DATAFLOW
    ddr_io::feature_loader(feature_src, feature_stream, geometry);
    im2row(feature_stream, window_stream, geometry);
    systolic_OS(window_stream, weight_stream, acc_stream, geometry,
                first_cout_tile, block_tiles);
}

// 层内的卷积生产端。重复调用只改变 Cout 范围，Feature 始终从同一基址读。
void run_conv_blocks(const ap_uint<64>* feature_src,
                     hls::stream<weight_pack_t>& weight_stream,
                     hls::stream<yolo_acc_tile_t>& acc_stream,
                     const ConvGeometry& geometry) {
#pragma HLS INLINE off
    int first_cout_tile = 0;
cout_block_loop:
    for (int block = 0; block < geometry.cache_blocks; ++block) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
        const int block_tiles = block == geometry.cache_blocks - 1
            ? geometry.last_cache_tiles : geometry.resident_cout_tiles;
        run_conv_block(feature_src, weight_stream, acc_stream, geometry,
                       first_cout_tile, block_tiles);
        first_cout_tile += block_tiles;
    }
}

// 权重、参数和结果端各处理整层；它们通过阻塞流和 Cout 块生产端配速。
void run_layer(const ap_uint<64>* feature_src,
               ap_uint<64>* feature_dst,
               const ap_uint<64>* weights,
               const ap_int<32>* params,
               const ConvGeometry& geometry) {
#pragma HLS INLINE off
    hls::stream<weight_pack_t> weight_stream("weight_stream");
    hls::stream<postprocess_param_pack_t> param_stream("param_stream");
    hls::stream<yolo_acc_tile_t> acc_stream("acc_stream");
    hls::stream<yolo_activation_tile_t> post_stream("post_stream");
    hls::stream<yolo_activation_tile_t> pool_stream("pool_stream");
#pragma HLS STREAM variable=weight_stream depth=FIFO_DEPTH
#pragma HLS STREAM variable=param_stream depth=FIFO_DEPTH
#pragma HLS STREAM variable=acc_stream depth=FIFO_DEPTH
#pragma HLS STREAM variable=post_stream depth=FIFO_DEPTH
#pragma HLS STREAM variable=pool_stream depth=FIFO_DEPTH
#pragma HLS DATAFLOW
    ddr_io::weight_loader(weights, weight_stream, geometry);
    ddr_io::param_loader(params, param_stream, geometry);
    run_conv_blocks(feature_src, weight_stream, acc_stream, geometry);
    conv_postprocess(acc_stream, param_stream, post_stream, geometry);
    maxpool(post_stream, pool_stream, geometry);
    ddr_io::feature_writer(pool_stream, feature_dst, geometry);
}

}  // namespace

void cnn_accelerator_ddr(ap_uint<64>* feature,
                         const ap_uint<64>* weights,
                         const ap_int<32>* params) {
#pragma HLS INTERFACE m_axi port=feature offset=slave bundle=gmem_feature depth=ddr_layout::FEATURE_WORDS max_widen_bitwidth=64
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem_weights depth=ddr_layout::WEIGHT_WORDS max_widen_bitwidth=64
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem_params depth=ddr_layout::PARAM_WORDS max_widen_bitwidth=32
#pragma HLS INTERFACE s_axilite port=feature bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int weight_offset = 0;  // 以 64-bit 权重字为单位。
    int param_offset = 0;   // 以 INT32 参数字为单位。
layer_loop:
    for (int layer = 0; layer < YOLO_TINY_LAYER_COUNT; ++layer) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL factor=1
        const ConvGeometry geometry = YOLO_TINY_GEOMETRIES[layer];
        const bool read_bank_b = (layer & 1) != 0;
        const int src_offset = read_bank_b ? ddr_layout::FEATURE_BANK_WORDS : 0;
        const int dst_offset = read_bank_b ? 0 : ddr_layout::FEATURE_BANK_WORDS;
        // 唯一层调用点：等待本层全部进程（包括 Writer）结束，下一迭代才交换 A/B。
        run_layer(feature + src_offset, feature + dst_offset,
                  weights + weight_offset, params + param_offset, geometry);
        weight_offset += ddr_layout::weight_layer_words(geometry);
        param_offset += ddr_layout::param_layer_words(geometry);
    }
}
