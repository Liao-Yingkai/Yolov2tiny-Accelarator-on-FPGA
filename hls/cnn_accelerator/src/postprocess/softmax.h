#ifndef CNN_ACCELERATOR_SOFTMAX_H_
#define CNN_ACCELERATOR_SOFTMAX_H_
#include "../types.h"
void softmax(hls::stream<logits_pack_t>& input,
             hls::stream<inference_result_t>& output,
             ap_uint<32> scale_q20);
#endif
