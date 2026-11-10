#include "../src/postprocess/conv_postprocess.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/*
 * 独立验证 YOLO ConvPostProcess，不调用其他 DUT，也不生成数据文件。
 * 输入、期望结果分别来自 INT32 Systolic HWC 与 INT8 PostProcess HWC BIN；
 * 按 Cout 块 -> y -> tile_x -> local_cout_tile 组装输入并直接索引期望结果。
 * Bias BIN 为 [Cout]，quant BIN 为 [全部 M][全部 S]，均为 little-endian INT32。
 * 参数流按全局 4-Cout Tile 递增，每层只供给一次；一个 DUT 调用处理完整一层。
 */
namespace {
const std::string generated_directory =
    "/home/liaoyingkai/python_project/minist_train_fpga/YOLOV2_TINY/generated/";
constexpr char sample_suffix[] = "0000";
constexpr int INT32_BYTES = 4;
constexpr int MAX_PRINTED_ERRORS = 10;

bool read_binary(const std::string& path, std::size_t expected_bytes,
                 std::vector<std::int8_t>& bytes) {
    std::ifstream file(path.c_str(), std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(expected_bytes)) {
        std::cerr << "BIN不存在或长度错误: " << path
                  << " 期望字节数=" << expected_bytes << '\n';
        return false;
    }
    file.seekg(0);
    bytes.resize(expected_bytes);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(expected_bytes))) {
        std::cerr << "BIN读取失败: " << path << '\n';
        return false;
    }
    return true;
}

// 低三字节是无符号数据，高字节按 signed INT8 扩展。与宿主字节序无关。
std::int32_t read_i32_le(const std::vector<std::int8_t>& bytes, std::size_t index) {
    const std::int8_t* p = bytes.data() + index * INT32_BYTES;
    return std::uint8_t(p[0]) + (std::int32_t(std::uint8_t(p[1])) << 8) +
           (std::int32_t(std::uint8_t(p[2])) << 16) + std::int32_t(p[3]) * (1 << 24);
}

unsigned lane_mask(int first, int limit, int lanes) {
    unsigned mask = 0;
    for (int lane = 0; lane < lanes; ++lane)
        if (first + lane < limit) mask |= 1u << lane;
    return mask;
}

std::size_t hwc_index(const ConvGeometry& g, int y, int x, int oc) {
    return (std::size_t(y) * g.w + x) * g.cout + oc;
}

bool run_layer(const ConvGeometry& g, const char* name,
               hls::stream<yolo_acc_tile_t>& input_stream,
               hls::stream<postprocess_param_pack_t>& param_stream,
               hls::stream<yolo_activation_tile_t>& output_stream, int& total_errors) {
    const int tiles_per_row = ceil_div(g.w, SYSTOLIC_SPATIAL_PAR);
    const int spatial_tiles = g.h * tiles_per_row;
    const int cout_tiles = ceil_div(g.cout, SYSTOLIC_COUT_PAR);
    const int resident_tiles = std::min(cout_tiles, YOLO_TINY_COUT_CACHE_TILES);
    const int blocks = ceil_div(cout_tiles, resident_tiles);
    const int expected_tiles = spatial_tiles * cout_tiles;
    // 从逻辑尺寸独立算事务数，不照用 DUT 的派生字段。
    if (g.spatial_tiles != spatial_tiles || g.cout_tiles != cout_tiles ||
        g.resident_cout_tiles != resident_tiles || g.cache_blocks != blocks) {
        std::cerr << name << " 层配置与 BIN 事务契约不一致\n";
        return false;
    }
    if (!input_stream.empty() || !param_stream.empty() || !output_stream.empty()) {
        std::cerr << name << " 调用开始时 stream 非空\n";
        return false;
    }

    std::vector<std::int8_t> accumulators, biases, quant, golden;
    const std::string trace = generated_directory + "trace/trace_" + name;
    const std::string params = generated_directory + "bins/" + name;
    const std::string suffix = std::string("_") + sample_suffix + ".bin";
    const std::size_t elements = std::size_t(g.h) * g.w * g.cout;
    const std::size_t bias_bytes = std::size_t(g.cout) * INT32_BYTES;
    if (!read_binary(trace + "_systolic_int32_hwc" + suffix, elements * INT32_BYTES, accumulators) ||
        !read_binary(params + "_bias.bin", bias_bytes, biases) ||
        !read_binary(params + "_quant.bin", 2 * bias_bytes, quant) ||
        !read_binary(trace + "_postprocess_int8_hwc" + suffix, elements, golden)) return false;

    int params_written = 0, inputs_written = 0, outputs_read = 0, errors = 0;
    for (int ct = 0; ct < cout_tiles; ++ct) {
        postprocess_param_pack_t pack;
        for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
            const int oc = ct * SYSTOLIC_COUT_PAR + j;
            // Prediction 的文件只有 125 个逻辑 Cout，先判界再读取各字段。
            pack.bias[j] = oc < g.cout ? read_i32_le(biases, oc) : 0;
            pack.multiplier[j] = oc < g.cout ? read_i32_le(quant, oc) : 0;
            pack.shift[j] = oc < g.cout ? read_i32_le(quant, g.cout + oc) : 0;
        }
        param_stream.write(pack);
        ++params_written;
    }

    // 文件是紧凑 HWC，硬件是块优先；有效坐标按原值取样，文件中没有物理尾 lane。
    for (int first = 0; first < cout_tiles; first += resident_tiles) {
        const int end = std::min(first + resident_tiles, cout_tiles);
        for (int spatial = 0; spatial < spatial_tiles; ++spatial) {
            const int y = spatial / tiles_per_row;
            const int first_x = (spatial % tiles_per_row) * SYSTOLIC_SPATIAL_PAR;
            for (int ct = first; ct < end; ++ct) {
                const int first_oc = ct * SYSTOLIC_COUT_PAR;
                yolo_acc_tile_t input;
                input.spatial_valid = lane_mask(first_x, g.w, SYSTOLIC_SPATIAL_PAR);
                input.cout_valid = lane_mask(first_oc, g.cout, SYSTOLIC_COUT_PAR);
                for (int i = 0; i < SYSTOLIC_SPATIAL_PAR; ++i) {
                    for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
                        const int x = first_x + i, oc = first_oc + j;
                        input.value[i][j] = x < g.w && oc < g.cout
                            ? read_i32_le(accumulators, hwc_index(g, y, x, oc)) : 0;
                    }
                }
                input_stream.write(input);
                ++inputs_written;
            }
        }
    }

    conv_postprocess(input_stream, param_stream, output_stream, g);

    // 不重排整层输出；按预期事务顺序读流，从 y/x/全局 oc 直接索引 golden。
    for (int first = 0; first < cout_tiles; first += resident_tiles) {
        const int end = std::min(first + resident_tiles, cout_tiles);
        for (int y = 0; y < g.h; ++y) {
            for (int tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
                const int first_x = tile_x * SYSTOLIC_SPATIAL_PAR;
                const unsigned spatial_valid = lane_mask(first_x, g.w, SYSTOLIC_SPATIAL_PAR);
                for (int ct = first; ct < end; ++ct) {
                    if (output_stream.empty()) {
                        std::cerr << name << " 缺少输出: y=" << y << " tile_x=" << tile_x
                                  << " cout_tile=" << ct << '\n';
                        return false;
                    }
                    const yolo_activation_tile_t actual = output_stream.read();
                    ++outputs_read;
                    const int first_oc = ct * SYSTOLIC_COUT_PAR;
                    const unsigned cout_valid = lane_mask(first_oc, g.cout, SYSTOLIC_COUT_PAR);
                    if (actual.spatial_valid.to_uint() != spatial_valid ||
                        actual.cout_valid.to_uint() != cout_valid) {
                        if (++errors <= MAX_PRINTED_ERRORS)
                            std::cerr << name << " mask错误: y=" << y << " tile_x=" << tile_x
                                      << " cout_tile=" << ct
                                      << " spatial=" << actual.spatial_valid.to_uint() << '/' << spatial_valid
                                      << " cout=" << actual.cout_valid.to_uint() << '/' << cout_valid
                                      << " (实际/期望)\n";
                    }
                    for (int i = 0; i < SYSTOLIC_SPATIAL_PAR; ++i) {
                        for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
                            const int x = first_x + i, oc = first_oc + j;
                            // 先检查逻辑边界；无效 lane 期望 0，不使用 DUT mask 索引文件。
                            const int expected = x < g.w && oc < g.cout
                                ? int(golden[hwc_index(g, y, x, oc)]) : 0;
                            const int got = actual.value[i][j].to_int();
                            if (got != expected && ++errors <= MAX_PRINTED_ERRORS)
                                std::cerr << name << " INT8结果错误: y=" << y << " x=" << x
                                          << " oc=" << oc << " 实际=" << got << " 期望=" << expected << '\n';
                        }
                    }
                }
            }
        }
    }

    if (!input_stream.empty() || !param_stream.empty() || !output_stream.empty()) {
        std::cerr << name << " 事务数错误: 剩余输入=" << !input_stream.empty()
                  << " 剩余参数=" << !param_stream.empty()
                  << " 多余输出=" << !output_stream.empty() << '\n';
        return false;
    }
    if (params_written != cout_tiles || inputs_written != expected_tiles ||
        outputs_read != expected_tiles) ++errors;
    std::cout << name << " Cout块=" << blocks << " 参数Pack=" << params_written << '/' << cout_tiles
              << " 输入Tile=" << inputs_written << '/' << expected_tiles
              << " 输出Tile=" << outputs_read << '/' << expected_tiles
              << " 验证元素数=" << outputs_read * SYSTOLIC_SPATIAL_PAR * SYSTOLIC_COUT_PAR
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS") << std::endl;
    total_errors += errors;
    return true;
}
}  // namespace

int main() {
    const char* names[] = {"conv1", "conv2", "conv3", "conv4", "conv5",
                           "conv6", "conv7", "conv8", "prediction"};
    static_assert(sizeof(names) / sizeof(names[0]) == YOLO_TINY_LAYER_COUNT,
                  "BIN layer names must match the fixed YOLO layer table");
    hls::stream<yolo_acc_tile_t> input_stream("acc");
    hls::stream<postprocess_param_pack_t> param_stream("params");
    hls::stream<yolo_activation_tile_t> output_stream("result");
    int errors = 0;
    // 复用同一组流依次调用九层，检查层间及块间缓存切换。
    for (int i = 0; i < YOLO_TINY_LAYER_COUNT; ++i)
        if (!run_layer(YOLO_TINY_GEOMETRIES[i], names[i],
                       input_stream, param_stream, output_stream, errors)) return 1;
    std::cout << "ConvPostProcess " << (errors ? "FAIL" : "PASS")
              << " 错误数=" << errors << std::endl;
    return errors ? 1 : 0;
}
