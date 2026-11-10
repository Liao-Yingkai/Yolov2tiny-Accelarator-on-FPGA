#include "ddr_io.h"

#include "../channel_pack.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace ddr_io {
namespace {

constexpr int DDR_WORD_BITS = 64;
constexpr int RGB_CHANNELS = 3;
constexpr int RGB_PIXEL_BITS = RGB_CHANNELS * ACTIVATION_BITS;
constexpr int RGB_CHUNK_PIXELS = DDR_WORD_BITS / ACTIVATION_BITS;
constexpr int RGB_CHUNK_WORDS = RGB_CHANNELS;
constexpr int RGB_CHUNK_BITS = RGB_CHUNK_WORDS * DDR_WORD_BITS;
constexpr int HALF_WORD_BITS = CONV_COUT_PAR * ACTIVATION_BITS;
constexpr int COUT_TILES_PER_WORD = FEATURE_PACK_CHANNELS / CONV_COUT_PAR;
constexpr int MAX_BLOCK_GROUPS = YOLO_TINY_COUT_CACHE_TILES / COUT_TILES_PER_WORD;
constexpr int PARAM_FIELDS = 3;

static_assert(FEATURE_PACK_CHANNELS * ACTIVATION_BITS == DDR_WORD_BITS &&
                  CONV_CIN_PAR * WEIGHT_BITS == DDR_WORD_BITS &&
                  COUT_TILES_PER_WORD == 2 && 2 * HALF_WORD_BITS == DDR_WORD_BITS,
              "DDR word must contain one 8-channel pack or two 4-Cout halves");
static_assert(RGB_CHUNK_PIXELS * RGB_PIXEL_BITS == RGB_CHUNK_BITS,
              "Each RGB chunk must end on both pixel and DDR word boundaries");

ap_uint<HALF_WORD_BITS> pack_output_half(const yolo_activation_tile_t& tile,
                                         int spatial_lane) {
#pragma HLS INLINE
    ap_uint<HALF_WORD_BITS> half = 0;
    for (int cout_lane = 0; cout_lane < CONV_COUT_PAR; ++cout_lane) {
#pragma HLS UNROLL
        if (tile.cout_valid[cout_lane]) {
            half.range((cout_lane + 1) * ACTIVATION_BITS - 1, cout_lane * ACTIVATION_BITS) =
                tile.value[spatial_lane][cout_lane].range(ACTIVATION_BITS - 1, 0);
        }
    }
    return half;
}

}  // namespace

void feature_loader(const ap_uint<64>* feature,
                    hls::stream<channel_pack_t>& output,
                    const ConvGeometry& geometry) {
#pragma HLS INLINE off
    if (geometry.cin == RGB_CHANNELS) {
        // 紧凑 RGB 无 DDR 尾槽：连续 3 字包含 8 个像素，像素可能跨 DDR 字。
#ifndef __SYNTHESIS__
        assert(geometry.w % RGB_CHUNK_PIXELS == 0 && geometry.cin_groups == 1);
#endif
        const int chunk_count = geometry.input_transactions / RGB_CHUNK_PIXELS;
        int word_address = 0;
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            ap_uint<DDR_WORD_BITS> words[RGB_CHUNK_WORDS];
#pragma HLS ARRAY_PARTITION variable=words complete
            for (int i = 0; i < RGB_CHUNK_WORDS; ++i) {
#pragma HLS PIPELINE II=1
                words[i] = feature[word_address++];
            }

            // 最早字节在低位。固定步长移位，不让跨字的 RGB 像素丢失。
            ap_uint<RGB_CHUNK_BITS> pixels;
            for (int i = 0; i < RGB_CHUNK_WORDS; ++i) {
#pragma HLS UNROLL
                pixels.range((i + 1) * DDR_WORD_BITS - 1, i * DDR_WORD_BITS) = words[i];
            }
            for (int i = 0; i < RGB_CHUNK_PIXELS; ++i) {
#pragma HLS PIPELINE II=1
                packed_activation_lane_t packed = 0;  // 只在发流时补齐 C3..C7。
                packed.range(RGB_PIXEL_BITS - 1, 0) = pixels.range(RGB_PIXEL_BITS - 1, 0);
                output.write(unpack_activation_lane(packed));
                pixels >>= RGB_PIXEL_BITS;
            }
        }
    } else {
        // 其余固定网络层的 Cin 已整除 8，DDR 字与输入 Pack 一一对应。
#ifndef __SYNTHESIS__
        assert(geometry.cin == geometry.cin_groups * FEATURE_PACK_CHANNELS);
#endif
        for (int word = 0; word < geometry.input_transactions; ++word) {
#pragma HLS PIPELINE II=1
            output.write(unpack_activation_lane(feature[word]));
        }
    }
}

void weight_loader(const ap_uint<64>* weights,
                   hls::stream<weight_pack_t>& output,
                   const ConvGeometry& geometry) {
#pragma HLS INLINE off
    weight_pack_t pack;
#pragma HLS ARRAY_PARTITION variable=pack.lane complete
    const int word_count = geometry.weight_transactions * CONV_COUT_PAR;
    // 已导出的顺序就是 Systolic 的消费顺序；每四字组包，跨 Cout 块不重置游标。
    for (int word = 0; word < word_count; ++word) {
#pragma HLS PIPELINE II=1
        const int cout_lane = word % CONV_COUT_PAR;
        pack.lane[cout_lane] = unpack_weight_lane(weights[word]);
        if (cout_lane == CONV_COUT_PAR - 1) {
            output.write(pack);  // 消费端装完当前块后，依靠阻塞写传递反压。
        }
    }
}

void param_loader(const ap_int<32>* params,
                  hls::stream<postprocess_param_pack_t>& output,
                  const ConvGeometry& geometry) {
#pragma HLS INLINE off
    const int cout = geometry.cout;
    for (int tile = 0; tile < geometry.cout_tiles; ++tile) {
        postprocess_param_pack_t pack;
#pragma HLS ARRAY_PARTITION variable=pack.bias complete
#pragma HLS ARRAY_PARTITION variable=pack.multiplier complete
#pragma HLS ARRAY_PARTITION variable=pack.shift complete

        // 按字段取当前 Tile 的四个值。单个内层迭代最多读一次 AXI。
        for (int field = 0; field < PARAM_FIELDS; ++field) {
            const int field_base = field * cout;
            for (int lane = 0; lane < CONV_COUT_PAR; ++lane) {
#pragma HLS PIPELINE II=1
                const int oc = tile * CONV_COUT_PAR + lane;
                ap_int<32> value = 0;
                if (oc < cout) {
                    value = params[field_base + oc];  // 先判界，末 Tile 不读无效通道。
                }
                if (field == 0) {
                    pack.bias[lane] = value;
                } else if (field == 1) {
                    pack.multiplier[lane] = value;
                } else {
                    pack.shift[lane] = value;
                }
            }
        }
        output.write(pack);
    }
}

void feature_writer(hls::stream<yolo_activation_tile_t>& input,
                    ap_uint<64>* feature,
                    const ConvGeometry& geometry) {
#pragma HLS INLINE off
    const int height = geometry.output_h;
    const int width = geometry.output_w;
    const int tile_count_x = ceil_div(width, CONV_SPATIAL_PAR);
    const int output_groups = geometry.output_channel_groups;
    const int row_words = width * output_groups;
    ap_uint<DDR_WORD_BITS> scratch[CONV_SPATIAL_PAR][MAX_BLOCK_GROUPS];
#pragma HLS ARRAY_PARTITION variable=scratch complete dim=0
#ifndef __SYNTHESIS__
    int reads = 0;
    int writes = 0;
#endif

    int first_group = 0;
    for (int block = 0; block < geometry.cache_blocks; ++block) {
        const int block_tiles = block == geometry.cache_blocks - 1
            ? geometry.last_cache_tiles : geometry.resident_cout_tiles;
#ifndef __SYNTHESIS__
        assert(block_tiles > 0 && block_tiles % COUT_TILES_PER_WORD == 0);
#endif
        const int block_groups = block_tiles / COUT_TILES_PER_WORD;
        int row_base = first_group;
        for (int y = 0; y < height; ++y, row_base += row_words) {
            int tile_base = row_base;
            for (int tile_x = 0; tile_x < tile_count_x;
                 ++tile_x, tile_base += CONV_SPATIAL_PAR * output_groups) {
                ap_uint<CONV_SPATIAL_PAR> spatial_valid = 0;
                // 同一空间 Tile 每次收两个 4-Cout 事务，合成四个像素各自的 8-Cout 字。
                for (int group = 0; group < block_groups; ++group) {
                    const yolo_activation_tile_t low = input.read();
                    const yolo_activation_tile_t high = input.read();
                    if (group == 0) {
                        spatial_valid = low.spatial_valid;
                    }
#ifndef __SYNTHESIS__
                    assert(low.spatial_valid == spatial_valid && high.spatial_valid == spatial_valid);
                    const int valid_lanes = config_min(CONV_SPATIAL_PAR,
                                                       width - tile_x * CONV_SPATIAL_PAR);
                    assert(spatial_valid == (1u << valid_lanes) - 1u);
                    reads += COUT_TILES_PER_WORD;
#endif
                    for (int lane = 0; lane < CONV_SPATIAL_PAR; ++lane) {
#pragma HLS UNROLL
                        ap_uint<DDR_WORD_BITS> word;
                        word.range(HALF_WORD_BITS - 1, 0) = pack_output_half(low, lane);
                        word.range(DDR_WORD_BITS - 1, HALF_WORD_BITS) = pack_output_half(high, lane);
                        scratch[lane][group] = word;
                    }
                }

                // 当前块内，一个像素的各组地址相邻；像素之间按完整 Cout 步长跳转。
                // 写完才收下一空间 Tile；scratch 只属于本 Writer，不需要 ping-pong。
                int pixel_base = tile_base;
                for (int lane = 0; lane < CONV_SPATIAL_PAR; ++lane, pixel_base += output_groups) {
                    if (spatial_valid[lane]) {
                        for (int group = 0; group < block_groups; ++group) {
#pragma HLS PIPELINE II=1
                            feature[pixel_base + group] = scratch[lane][group];
#ifndef __SYNTHESIS__
                            ++writes;
#endif
                        }
                    }
                }
            }
        }
        first_group += block_groups;
    }
#ifndef __SYNTHESIS__
    assert(first_group == output_groups);
    assert(reads == height * tile_count_x * geometry.cout_tiles);
    assert(writes == geometry.output_transactions);
#endif
}

}  // namespace ddr_io
