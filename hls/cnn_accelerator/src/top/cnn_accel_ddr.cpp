#include "cnn_accel_ddr.h"
#include "../core/cnn_core.h"
#include "../axi/ddr_io.h"
#include "../postprocess/logits_argmax.h"
#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace {
void compute_runner(hls::stream<pixel_t>& image,
                    hls::stream<weight_pack_t>& w1,
                    hls::stream<postprocess_param_pack_t>& p1,
                    hls::stream<weight_pack_t>& w2,
                    hls::stream<postprocess_param_pack_t>& p2,
                    hls::stream<fc_weight_pack_t>& wf,
                    hls::stream<fc_param_pack_t>& pf,
                    hls::stream<logits_pack_t>& logits) {
#pragma HLS INLINE off
    // 计算 core 的顺序 phase barrier 保留在本 actor 内，不暴露给外层 DATAFLOW。
    cnn_accelerator(image, w1, p1, w2, p2, wf, pf, logits);
}

void argmax_runner(hls::stream<logits_pack_t>& input,
                   hls::stream<class_index_t>& output) {
#pragma HLS INLINE off
    logits_argmax(input, output);
}
}  // 匿名命名空间

void cnn_accelerator_ddr(const ap_uint<64>* image, const ap_uint<64>* weights,
                         const ap_int<32>* params, ap_uint<64>* result) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=gmem_io depth=ddr_layout::IMAGE_WORDS max_widen_bitwidth=64
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem_weights depth=ddr_layout::WEIGHT_WORDS max_widen_bitwidth=64
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=gmem_params depth=ddr_layout::PARAM_WORDS max_widen_bitwidth=32
#pragma HLS INTERFACE m_axi port=result offset=slave bundle=gmem_io depth=ddr_layout::RESULT_WORDS max_widen_bitwidth=64
#pragma HLS INTERFACE s_axilite port=image bundle=control
#pragma HLS INTERFACE s_axilite port=weights bundle=control
#pragma HLS INTERFACE s_axilite port=params bundle=control
#pragma HLS INTERFACE s_axilite port=result bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    hls::stream<pixel_t> pixels("ddr_pixels");
    hls::stream<weight_pack_t> w1("ddr_w1"), w2("ddr_w2");
    hls::stream<postprocess_param_pack_t> p1("ddr_p1"), p2("ddr_p2");
    hls::stream<fc_weight_pack_t> wf("ddr_wf");
    hls::stream<fc_param_pack_t> pf("ddr_pf");
    hls::stream<logits_pack_t> computed("ddr_computed"), write_logits("ddr_write_logits"), argmax_logits("ddr_argmax_logits");
    hls::stream<class_index_t> classes("ddr_classes");
#pragma HLS STREAM variable=pixels depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=w1 depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=w2 depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=p1 depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=p2 depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=wf depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=pf depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=computed depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=write_logits depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=argmax_logits depth=TOP_STREAM_DEPTH
#pragma HLS STREAM variable=classes depth=TOP_STREAM_DEPTH
#pragma HLS DATAFLOW
    ddr_io::image_loader(image, pixels);
    ddr_io::weight_loader(weights, w1, w2, wf);
    ddr_io::param_loader(params, p1, p2, pf);
    compute_runner(pixels, w1, p1, w2, p2, wf, pf, computed);
    ddr_io::logits_split(computed, write_logits, argmax_logits);
    argmax_runner(argmax_logits, classes);
    ddr_io::result_writer(write_logits, classes, result);
#ifndef __SYNTHESIS__
    assert(pixels.empty() && w1.empty() && w2.empty() && wf.empty());
    assert(p1.empty() && p2.empty() && pf.empty());
    assert(computed.empty() && write_logits.empty() && argmax_logits.empty() && classes.empty());
#endif
}
