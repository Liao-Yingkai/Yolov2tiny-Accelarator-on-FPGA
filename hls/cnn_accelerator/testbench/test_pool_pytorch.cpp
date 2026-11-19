#include "../src/pool/pool.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::string trace_dir;
constexpr char trace_suffix[] = "_int8_hwc_0000.bin";
constexpr int MAX_PRINTED_ERRORS = 10;

bool read_bin(const std::string& path, std::size_t size, std::vector<std::int8_t>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(size)) {
        std::cerr << "BIN 不存在或长度错误: " << path << " 期望字节数=" << size << '\n';
        return false;
    }
    data.resize(size);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
        std::cerr << "BIN 读取失败: " << path << '\n';
        return false;
    }
    return true;
}

std::size_t hwc_index(int width, int channels, int y, int x, int oc) {
    return (std::size_t(y) * width + x) * channels + oc;
}

unsigned lane_mask(int first, int limit, int lanes) {
    return (1u << std::min(limit - first, lanes)) - 1u;
}

bool check_layer(const ConvGeometry& g, const std::string& name,
                 const std::vector<std::int8_t>& input, const std::vector<std::int8_t>& golden,
                 hls::stream<yolo_activation_tile_t>& input_stream,
                 hls::stream<yolo_activation_tile_t>& output_stream) {
    const int output_tiles_x = ceil_div(g.output_w, CONV_SPATIAL_PAR);
    const int expected_input = g.h * g.tiles_per_row * g.cout_tiles;
    const int expected_output = g.output_h * output_tiles_x * g.cout_tiles;
    int written = 0, read = 0, errors = 0;

    // BIN 是逻辑 HWC；硬件按 Cout 块 -> y -> tile_x -> 块内 Cout Tile 传输。
    // 每个事务只装四个位置和四个通道，不把 8 通道组装混到 Pool 的测试里。
    for (int first = 0; first < g.cout_tiles; first += g.resident_cout_tiles) {
        const int end = std::min(first + g.resident_cout_tiles, g.cout_tiles);
        for (int y = 0; y < g.h; ++y) {
            for (int x0 = 0; x0 < g.w; x0 += CONV_SPATIAL_PAR) {
                for (int ct = first; ct < end; ++ct) {
                    const int oc0 = ct * CONV_COUT_PAR;
                    yolo_activation_tile_t tile;
                    tile.spatial_valid = lane_mask(x0, g.w, CONV_SPATIAL_PAR);
                    tile.cout_valid = lane_mask(oc0, g.cout, CONV_COUT_PAR);
                    for (int lane = 0; lane < CONV_SPATIAL_PAR; ++lane) {
                        for (int oc_lane = 0; oc_lane < CONV_COUT_PAR; ++oc_lane) {
                            const int x = x0 + lane, oc = oc0 + oc_lane;
                            tile.value[lane][oc_lane] = x < g.w && oc < g.cout
                                ? int(input[hwc_index(g.w, g.cout, y, x, oc)]) : 0;
                        }
                    }
                    input_stream.write(tile);
                    ++written;
                }
            }
        }
    }

    maxpool(input_stream, output_stream, g);

    // 对每个输出直接用坐标索引 HWC golden。尤其检查下采样后的连续 Tile，
    // 而不是把每个输入 Tile 产生的半 Tile 当作一个输出事务。
    for (int first = 0; first < g.cout_tiles; first += g.resident_cout_tiles) {
        const int end = std::min(first + g.resident_cout_tiles, g.cout_tiles);
        for (int y = 0; y < g.output_h; ++y) {
            for (int x0 = 0; x0 < g.output_w; x0 += CONV_SPATIAL_PAR) {
                for (int ct = first; ct < end; ++ct) {
                    const int oc0 = ct * CONV_COUT_PAR;
                    if (output_stream.empty()) {
                        std::cerr << name << " 缺少输出: y=" << y << " x0=" << x0
                                  << " cout_tile=" << ct << '\n';
                        return false;
                    }
                    const yolo_activation_tile_t tile = output_stream.read();
                    ++read;
                    if (tile.spatial_valid != lane_mask(x0, g.output_w, CONV_SPATIAL_PAR) ||
                        tile.cout_valid != lane_mask(oc0, g.cout, CONV_COUT_PAR)) {
                        if (++errors <= MAX_PRINTED_ERRORS)
                            std::cerr << name << " valid 错误: y=" << y << " x0=" << x0
                                      << " cout_tile=" << ct << '\n';
                    }
                    for (int lane = 0; lane < CONV_SPATIAL_PAR; ++lane) {
                        for (int oc_lane = 0; oc_lane < CONV_COUT_PAR; ++oc_lane) {
                            const int x = x0 + lane, oc = oc0 + oc_lane;
                            const int expected = x < g.output_w && oc < g.cout
                                ? int(golden[hwc_index(g.output_w, g.cout, y, x, oc)]) : 0;
                            const int actual = tile.value[lane][oc_lane].to_int();
                            if (actual != expected && ++errors <= MAX_PRINTED_ERRORS)
                                std::cerr << name << " 结果错误: y=" << y << " x=" << x << " oc=" << oc
                                          << " 期望=" << expected << " 实际=" << actual << '\n';
                        }
                    }
                }
            }
        }
    }

    if (!input_stream.empty() || !output_stream.empty()) {
        std::cerr << name << " 事务数错误: 剩余输入=" << !input_stream.empty()
                  << " 多余输出=" << !output_stream.empty() << '\n';
        ++errors;
    }
    if (written != expected_input || read != expected_output) ++errors;
    std::cout << name << " 输入Tile=" << written << '/' << expected_input
              << " 输出Tile=" << read << '/' << expected_output
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS") << std::endl;
    return errors == 0;
}
}  // namespace

int main() {
    const char* configured_path = std::getenv("YOLO_DATA_DIR");
    if (!configured_path || !*configured_path) {
        std::cerr << "请设置环境变量 YOLO_DATA_DIR，指向 YOLO generated 根目录。\n";
        return 1;
    }
    trace_dir = std::string(configured_path) + "/trace/";
    const char* layer_names[] = {"conv1", "conv2", "conv3", "conv4", "conv5",
                                 "conv6", "conv7", "conv8", "prediction"};
    static_assert(sizeof(layer_names) / sizeof(layer_names[0]) == YOLO_TINY_LAYER_COUNT,
                  "BIN 层名称必须与固定层表一一对应");
    hls::stream<yolo_activation_tile_t> input_stream("pool_in");
    hls::stream<yolo_activation_tile_t> output_stream("pool_out");
    std::vector<std::int8_t> input, golden;
    for (int layer = 0; layer < YOLO_TINY_LAYER_COUNT; ++layer) {
        const ConvGeometry& g = YOLO_TINY_GEOMETRIES[layer];
        const std::string name = layer_names[layer];
        if (!read_bin(trace_dir + "trace_" + name + "_postprocess" + trace_suffix,
                      std::size_t(g.h) * g.w * g.cout, input)) return 1;
        if (g.pool_stride != 0 &&
            !read_bin(trace_dir + "trace_pool" + std::to_string(layer + 1) + trace_suffix,
                      std::size_t(g.output_h) * g.output_w * g.cout, golden)) return 1;
        const auto& expected = g.pool_stride == 0 ? input : golden;
        if (!check_layer(g, name, input, expected, input_stream, output_stream)) return 1;
    }

    // 全负数用例：边缘误用 0 padding 必定出错。reference 只按 HWC 坐标取
    // 2×2 原值，不复刻 DUT 的跨 Tile 暂存或行更新逻辑。
    const ConvGeometry& g = YOLO_TINY_GEOMETRIES[static_cast<int>(YoloTinyLayer::Conv6)];
    input.resize(std::size_t(g.h) * g.w * g.cout);
    golden.resize(input.size());
    for (int y = 0; y < g.h; ++y)
        for (int x = 0; x < g.w; ++x)
            for (int oc = 0; oc < g.cout; ++oc)
                input[hwc_index(g.w, g.cout, y, x, oc)] =
                    -128 + (y * 31 + x * 7 + oc * 13) % 127;
    for (int y = 0; y < g.h; ++y)
        for (int x = 0; x < g.w; ++x)
            for (int oc = 0; oc < g.cout; ++oc) {
                int maximum = -128;
                for (int dy = 0; dy < 2 && y + dy < g.h; ++dy)
                    for (int dx = 0; dx < 2 && x + dx < g.w; ++dx)
                        maximum = std::max(maximum,
                            int(input[hwc_index(g.w, g.cout, y + dy, x + dx, oc)]));
                golden[hwc_index(g.output_w, g.cout, y, x, oc)] = maximum;
            }
    if (!check_layer(g, "conv6 全负数", input, golden, input_stream, output_stream)) return 1;
    std::cout << "Pool PASS\n";
    return 0;
}
