#include "pool.h"

#ifndef __SYNTHESIS__
#include <cassert>
#endif

namespace {

constexpr int CHANNEL_TILE_BITS = CONV_COUT_PAR * ACTIVATION_BITS;
constexpr int LINE_WORD_BITS = CONV_SPATIAL_PAR * CHANNEL_TILE_BITS;
constexpr int PAIR_BITS = 2 * CHANNEL_TILE_BITS;
constexpr unsigned FULL_SPATIAL_MASK = (1u << CONV_SPATIAL_PAR) - 1;
constexpr unsigned FULL_COUT_MASK = (1u << CONV_COUT_PAR) - 1;
constexpr int POOL_PAD_VALUE = -(1 << (ACTIVATION_BITS - 1));  // signed INT8 的最小值。
using line_word_t = ap_uint<LINE_WORD_BITS>;       // 4 个位置 × 4 个 Cout，保存原始值。
using channel_tile_t = ap_uint<CHANNEL_TILE_BITS>;  // 单个位置的 4 个 Cout。
using pair_word_t = ap_uint<PAIR_BITS>;            // stride=2 产生的两个池化位置。

static_assert(CONV_SPATIAL_PAR == 4, "Pool 每个空间 Tile 固定为四个位置");
constexpr ConvGeometry STRIDE1_GEOMETRY =
    YOLO_TINY_GEOMETRIES[static_cast<int>(YoloTinyLayer::Conv6)];
static_assert(STRIDE1_GEOMETRY.pool_stride == 1 && STRIDE1_GEOMETRY.tail_spatial_lanes == 1 &&
                  STRIDE1_GEOMETRY.pool_padding_right == 1 &&
                  STRIDE1_GEOMETRY.pool_padding_bottom == 1,
              "stride=1 调度要求右/下补一格，且末 Tile 只有一个真实位置");

activation_t unpack_value(const line_word_t& word, int lane, int oc) {
#pragma HLS INLINE
    const int lo = lane * CHANNEL_TILE_BITS + oc * ACTIVATION_BITS;
    activation_t value;
    value.range(ACTIVATION_BITS - 1, 0) = word.range(lo + ACTIVATION_BITS - 1, lo);
    return value;
}

line_word_t pack_input(const yolo_activation_tile_t& input) {
#pragma HLS INLINE
    line_word_t word = 0;
    for (int lane = 0; lane < CONV_SPATIAL_PAR; ++lane) {
#pragma HLS UNROLL
        for (int oc = 0; oc < CONV_COUT_PAR; ++oc) {
#pragma HLS UNROLL
            const int lo = lane * CHANNEL_TILE_BITS + oc * ACTIVATION_BITS;
            word.range(lo + ACTIVATION_BITS - 1, lo) =
                input.value[lane][oc].range(ACTIVATION_BITS - 1, 0);
        }
    }
    return word;
}

line_word_t read_input(hls::stream<yolo_activation_tile_t>& input_stream,
                       const ConvGeometry& g, int tile_x) {
#pragma HLS INLINE
    const yolo_activation_tile_t input = input_stream.read();
#ifndef __SYNTHESIS__
    const unsigned spatial_mask = tile_x == g.last_tile_x
        ? g.tail_spatial_valid_mask : FULL_SPATIAL_MASK;
    // 只有 Bypass 的 Prediction 有 Cout 尾组。参与 Pool 的层通道 Tile 均完整。
    assert(input.spatial_valid == spatial_mask && input.cout_valid == FULL_COUT_MASK);
#endif
    return pack_input(input);
}

activation_t signed_max2(activation_t a, activation_t b) {
#pragma HLS INLINE
    return a > b ? a : b;
}

// 分别解包为 signed INT8 后再比较，不比较 ap_uint 的原始位型。
activation_t signed_max4(activation_t a, activation_t b, activation_t c, activation_t d) {
#pragma HLS INLINE
    return signed_max2(signed_max2(a, b), signed_max2(c, d));
}

// 一个空间窗口、四个独立 Cout。左右可以来自同一 Tile，也可以来自相邻 Tile。
channel_tile_t pool_window(const line_word_t& top_left, int left_lane,
                           const line_word_t& bottom_left,
                           const line_word_t& top_right, int right_lane,
                           const line_word_t& bottom_right, bool right_valid) {
#pragma HLS INLINE
    channel_tile_t result = 0;
    for (int oc = 0; oc < CONV_COUT_PAR; ++oc) {
#pragma HLS UNROLL
        const activation_t value = signed_max4(
            unpack_value(top_left, left_lane, oc),
            right_valid ? unpack_value(top_right, right_lane, oc) : activation_t(POOL_PAD_VALUE),
            unpack_value(bottom_left, left_lane, oc),
            right_valid ? unpack_value(bottom_right, right_lane, oc) : activation_t(POOL_PAD_VALUE));
        const int lo = oc * ACTIVATION_BITS;
        result.range(lo + ACTIVATION_BITS - 1, lo) = value.range(ACTIVATION_BITS - 1, 0);
    }
    return result;
}

void write_output(hls::stream<yolo_activation_tile_t>& output_stream,
                  const line_word_t& word, unsigned spatial_mask) {
#pragma HLS INLINE
    yolo_activation_tile_t output;
    output.spatial_valid = spatial_mask;
    output.cout_valid = FULL_COUT_MASK;
    for (int lane = 0; lane < CONV_SPATIAL_PAR; ++lane) {
#pragma HLS UNROLL
        for (int oc = 0; oc < CONV_COUT_PAR; ++oc) {
#pragma HLS UNROLL
            output.value[lane][oc] = (spatial_mask & (1u << lane))
                ? unpack_value(word, lane, oc) : activation_t(0);
        }
    }
    output_stream.write(output);
}

void store_raw_row(hls::stream<yolo_activation_tile_t>& input_stream,
                   line_word_t line_buffer[YOLO_TINY_POOL_LINE_WORDS],
                   const ConvGeometry& g, int block_tiles) {
#pragma HLS INLINE
    int address = 0;  // 行内地址顺序：tile_x -> local_cout_tile。
    for (int tile_x = 0; tile_x < g.tiles_per_row; ++tile_x) {
        for (int ct = 0; ct < block_tiles; ++ct) {
#pragma HLS PIPELINE II=1
            line_buffer[address++] = read_input(input_stream, g, tile_x);
        }
    }
}

void pool_stride2(hls::stream<yolo_activation_tile_t>& input_stream,
                  hls::stream<yolo_activation_tile_t>& output_stream,
                  line_word_t line_buffer[YOLO_TINY_POOL_LINE_WORDS],
                  const ConvGeometry& g, int block_tiles) {
#pragma HLS INLINE
    pair_word_t pending_pair[YOLO_TINY_MAX_RESIDENT_COUT_TILES];  // 前一个空间 Tile 的两个结果。
#pragma HLS ARRAY_PARTITION variable=pending_pair complete
    const int output_tail_lanes = (g.output_w - 1) % CONV_SPATIAL_PAR + 1;
    const unsigned output_tail_mask = (1u << output_tail_lanes) - 1;

    for (int y = 0; y < g.h; ++y) {
        if ((y & 1) == 0) {
            store_raw_row(input_stream, line_buffer, g, block_tiles);
            continue;
        }

        int address = 0;
        for (int tile_x = 0; tile_x < g.tiles_per_row; ++tile_x) {
            const bool second_pair = (tile_x & 1) != 0;
            const bool last_tile = tile_x == g.last_tile_x;
            for (int ct = 0; ct < block_tiles; ++ct) {
#pragma HLS PIPELINE II=1
                const line_word_t top = line_buffer[address++];
                const line_word_t bottom = read_input(input_stream, g, tile_x);

                // 偶数行只写、奇数行只读，仍沿用旧的隔行缓存与两组 2x2 窗口。
                pair_word_t pair;
                pair.range(CHANNEL_TILE_BITS - 1, 0) =
                    pool_window(top, 0, bottom, top, 1, bottom, true);
                pair.range(PAIR_BITS - 1, CHANNEL_TILE_BITS) =
                    pool_window(top, 2, bottom, top, 3, bottom, true);

                if (second_pair || last_tile) {
                    line_word_t output_word = 0;
                    if (second_pair) {
                        output_word.range(PAIR_BITS - 1, 0) = pending_pair[ct];
                        output_word.range(LINE_WORD_BITS - 1, PAIR_BITS) = pair;
                    } else {
                        // Conv4/5 的最后一个输入 Tile 无右邻，直接发出未配对的结果。
                        output_word.range(PAIR_BITS - 1, 0) = pair;
                    }
                    write_output(output_stream, output_word,
                                 last_tile ? output_tail_mask : FULL_SPATIAL_MASK);
                } else {
                    pending_pair[ct] = pair;
                }
            }
        }
    }
}

void pool_stride1(hls::stream<yolo_activation_tile_t>& input_stream,
                  hls::stream<yolo_activation_tile_t>& output_stream,
                  line_word_t line_buffer[YOLO_TINY_POOL_LINE_WORDS],
                  const ConvGeometry& g, int block_tiles) {
#pragma HLS INLINE
    line_word_t pending_prefix[YOLO_TINY_MAX_RESIDENT_COUT_TILES];  // 上一 Tile 已算完的 lane0~2。
    channel_tile_t top_edge[YOLO_TINY_MAX_RESIDENT_COUT_TILES];     // 上一 Tile 顶行的原始 lane3。
    channel_tile_t bottom_edge[YOLO_TINY_MAX_RESIDENT_COUT_TILES];  // 上一 Tile 底行的原始 lane3。
#pragma HLS ARRAY_PARTITION variable=pending_prefix complete
#pragma HLS ARRAY_PARTITION variable=top_edge complete
#pragma HLS ARRAY_PARTITION variable=bottom_edge complete

    // 先存真实 row0。最后多走一行，用 INT8 最小值模拟下补边，不消费输入。
    store_raw_row(input_stream, line_buffer, g, block_tiles);
    line_word_t padded_bottom;
    for (int i = 0; i < CONV_SPATIAL_PAR * CONV_COUT_PAR; ++i) {
#pragma HLS UNROLL
        padded_bottom.range((i + 1) * ACTIVATION_BITS - 1, i * ACTIVATION_BITS) =
            activation_t(POOL_PAD_VALUE);
    }
    for (int bottom_y = 1; bottom_y <= g.h; ++bottom_y) {
        const bool pad_bottom = bottom_y == g.h;
        int address = 0;
        for (int tile_x = 0; tile_x < g.tiles_per_row; ++tile_x) {
            const bool last_tile = tile_x == g.last_tile_x;
            for (int ct = 0; ct < block_tiles; ++ct) {
#pragma HLS PIPELINE II=1
                const line_word_t top = line_buffer[address];  // 锁存上一行原值。
                line_word_t bottom = padded_bottom;
                if (!pad_bottom) {
                    bottom = read_input(input_stream, g, tile_x);
                    line_buffer[address] = bottom;  // 旧值已锁存，当前位置更新为当前行。
                }
                ++address;

                if (tile_x != 0) {
                    // 上一 Tile 的 lane3 与当前 Tile 的 lane0 成窗；不能跨 Cout Tile。
                    const line_word_t previous_top = top_edge[ct];
                    const line_word_t previous_bottom = bottom_edge[ct];
                    line_word_t output_word = pending_prefix[ct];
                    output_word.range(LINE_WORD_BITS - 1, 3 * CHANNEL_TILE_BITS) =
                        pool_window(previous_top, 0, previous_bottom, top, 0, bottom, true);
                    write_output(output_stream, output_word, FULL_SPATIAL_MASK);
                }

                line_word_t prefix = 0;
                // Conv6 末 Tile 只有 lane0；其右邻用 -128，lane1~3 不产生逻辑输出。
                for (int lane = 0; lane < 3; ++lane) {
#pragma HLS UNROLL
                    if (lane == 0 || !last_tile) {
                        prefix.range((lane + 1) * CHANNEL_TILE_BITS - 1, lane * CHANNEL_TILE_BITS) =
                            pool_window(top, lane, bottom, top, lane + 1, bottom, !last_tile);
                    }
                }
                pending_prefix[ct] = prefix;
                if (!last_tile) {
                    top_edge[ct] = top.range(LINE_WORD_BITS - 1, 3 * CHANNEL_TILE_BITS);
                    bottom_edge[ct] = bottom.range(LINE_WORD_BITS - 1, 3 * CHANNEL_TILE_BITS);
                }
            }
        }

        // 先发完倒数第二个空间 Tile 的全部 Cout，再按相同 Cout 顺序发末 Tile。
        for (int ct = 0; ct < block_tiles; ++ct) {
#pragma HLS PIPELINE II=1
            write_output(output_stream, pending_prefix[ct], g.tail_spatial_valid_mask);
        }
    }
}

}  // namespace

void maxpool(hls::stream<yolo_activation_tile_t>& input_stream,
             hls::stream<yolo_activation_tile_t>& output_stream,
             const ConvGeometry& geometry) {
    // 只缓存当前 Cout 块的上一行。Stride=1 需要读口与写口，stride=2 不同时读写。
    line_word_t line_buffer[YOLO_TINY_POOL_LINE_WORDS];
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=bram latency=1

    for (int block = 0; block < geometry.cache_blocks; ++block) {
        const int block_tiles = block == geometry.cache_blocks - 1
            ? geometry.last_cache_tiles : geometry.resident_cout_tiles;
        if (geometry.pool_stride == 2) {
            pool_stride2(input_stream, output_stream, line_buffer, geometry, block_tiles);
        } else if (geometry.pool_stride == 1) {
            pool_stride1(input_stream, output_stream, line_buffer, geometry, block_tiles);
        } else {
            // 无 Pool 层原样透传，含 Prediction 的空间尾 mask 和 Cout 尾 mask。
            const int block_transactions = geometry.spatial_tiles * block_tiles;
            for (int i = 0; i < block_transactions; ++i) {
#pragma HLS PIPELINE II=1
                output_stream.write(input_stream.read());
            }
        }
    }
}
