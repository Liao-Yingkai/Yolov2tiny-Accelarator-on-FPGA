#ifndef CNN_ACCELERATOR_CHANNEL_PACK_H_
#define CNN_ACCELERATOR_CHANNEL_PACK_H_
#include "types.h"

// channel 0 位于最低位；range 赋值保留补码，恢复为 signed 标量。
using packed_activation_lane_t = ap_uint<FEATURE_PACK_CHANNELS * ACTIVATION_BITS>;
using packed_weight_lane_t = ap_uint<CONV_CIN_PAR * WEIGHT_BITS>;

inline packed_activation_lane_t pack_activation_lane(const channel_pack_t& input) {
#pragma HLS INLINE
    packed_activation_lane_t word = 0;
    for (int c = 0; c < FEATURE_PACK_CHANNELS; ++c) {
#pragma HLS UNROLL
        word.range((c + 1) * ACTIVATION_BITS - 1, c * ACTIVATION_BITS) =
            input.channel[c].range(ACTIVATION_BITS - 1, 0);
    }
    return word;
}
inline activation_t unpack_activation_channel(const packed_activation_lane_t& word, int c) {
#pragma HLS INLINE
    activation_t value;
    value.range(ACTIVATION_BITS - 1, 0) =
        word.range((c + 1) * ACTIVATION_BITS - 1, c * ACTIVATION_BITS);
    return value;
}
inline channel_pack_t unpack_activation_lane(const packed_activation_lane_t& word) {
#pragma HLS INLINE
    channel_pack_t output;
    for (int c = 0; c < FEATURE_PACK_CHANNELS; ++c) {
#pragma HLS UNROLL
        output.channel[c] = unpack_activation_channel(word, c);
    }
    return output;
}
inline packed_weight_lane_t pack_weight_lane(const weight_lane_t& input) {
#pragma HLS INLINE
    packed_weight_lane_t word = 0;
    for (int c = 0; c < CONV_CIN_PAR; ++c) {
#pragma HLS UNROLL
        word.range((c + 1) * WEIGHT_BITS - 1, c * WEIGHT_BITS) =
            input.channel[c].range(WEIGHT_BITS - 1, 0);
    }
    return word;
}
inline weight_lane_t unpack_weight_lane(const packed_weight_lane_t& word) {
#pragma HLS INLINE
    weight_lane_t output;
    for (int c = 0; c < CONV_CIN_PAR; ++c) {
#pragma HLS UNROLL
        output.channel[c].range(WEIGHT_BITS - 1, 0) =
            word.range((c + 1) * WEIGHT_BITS - 1, c * WEIGHT_BITS);
    }
    return output;
}
#endif

