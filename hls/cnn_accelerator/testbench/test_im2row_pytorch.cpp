#include "../src/im2row/im2row.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/*
 * YOLOv2-tiny 真实整数 golden 对照；不是浮点/QAT 中间激活。
 * 输入 y,x,group，每包8通道；输出 y,tile_x,kh,kw,group，每包4×8。
 * 输入 BIN 是紧凑 HWC：((y*W+x)*Cin+c)，尾通道由 TB 补零。
 * packed golden 为 [H,ceil(W/4),K*K*G,4,8]，沿 stream 顺序每次读32字节；
 * 第二参考 unfold 为 [H*W,K*K*Cin]，按逻辑坐标独立索引，不复刻缓存调度。
 * 输入/输出事务分别由 H*W*G、H*ceil(W/4)*K*K*G 推导。
 * 逐包检查 DUT 的 signed INT8 数据、packed/unfold 参考一致性与 valid，
 * 仅打印终端统计，不导出数据文件。C-sim 无法验证 RTL 的反压与吞吐行为。
 */
namespace {
constexpr int PACK_ELEMENTS = IM2ROW_P * FEATURE_PACK_CHANNELS;
const char* sample_suffix = "0000";
std::string trace_directory;

bool open_binary(std::ifstream& file, const std::string& path, std::size_t bytes) {
    file.open(path.c_str(), std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(bytes)) {
        std::cerr << "BIN不存在或字节长度错误: " << path << '\n';
        return false;
    }
    file.seekg(0);
    return bool(file);
}

bool read_binary(const std::string& path, std::size_t bytes,
                 std::vector<int8_t>& data) {
    std::ifstream file;
    if (!open_binary(file, path, bytes)) return false;
    data.resize(bytes);
    if (!file.read(reinterpret_cast<char*>(data.data()), bytes)) {
        std::cerr << "BIN读取失败: " << path << '\n';
        return false;
    }
    return true;
}

// 只输出前10条完整坐标；错误总数始终累计，不依赖 assert。
void report(int errors, const char* name, const char* kind, int y, int tile,
            int kh, int kw, int group, int lane, int channel, int expected, int actual) {
    if (errors <= 10)
        std::cerr << name << ' ' << kind << " y=" << y << " tile_x=" << tile
                  << " kh=" << kh << " kw=" << kw << " group=" << group
                  << " lane=" << lane << " channel=" << channel
                  << " expected=" << expected << " actual=" << actual << '\n';
}

// 返回 false 表示文件或 stream 状态错误，main 必须立即失败退出。
bool run_layer(const ConvGeometry& g, const char* name, const char* source,
               const char* case_name,
               hls::stream<channel_pack_t>& input,
               hls::stream<activation_pack_t>& output, int& total_errors) {
    if (!input.empty() || !output.empty()) {
        std::cerr << "调用开始时stream非空: " << case_name << '\n';
        return false;
    }
    const int groups = (g.cin + FEATURE_PACK_CHANNELS - 1) / FEATURE_PACK_CHANNELS;
    const int tiles = (g.w + IM2ROW_P - 1) / IM2ROW_P;
    const int expected_inputs = g.h * g.w * groups;
    const int expected_outputs = g.h * tiles * g.kernel * g.kernel * groups;
    const std::size_t input_bytes = std::size_t(g.h) * g.w * g.cin;
    const std::size_t unfold_bytes = input_bytes * g.kernel * g.kernel;
    std::vector<int8_t> image, unfold;
    std::ifstream packed;
    if (!read_binary(trace_directory + "trace_" + source + "_" + sample_suffix + ".bin", input_bytes, image) ||
        !read_binary(trace_directory + "trace_" + name + "_unfold_int8_" + sample_suffix + ".bin", unfold_bytes, unfold) ||
        !open_binary(packed, trace_directory + "trace_" + name + "_im2row_int8_" + sample_suffix + ".bin",
                     std::size_t(expected_outputs) * PACK_ELEMENTS)) return false;

    int errors = 0, written = 0, read = 0;
    std::size_t comparisons = 0;
    // 每层直接取真实上游 trace，绝不把未迁移的其它 HLS 模块串接进来。
    for (int y = 0; y < g.h; ++y)
        for (int x = 0; x < g.w; ++x)
            for (int group = 0; group < groups; ++group) {
                channel_pack_t pack;
                for (int cl = 0; cl < FEATURE_PACK_CHANNELS; ++cl) {
                    const int c = group * FEATURE_PACK_CHANNELS + cl;
                    const int value = c < g.cin ? int(image[(std::size_t(y) * g.w + x) * g.cin + c]) : 0;
                    pack.channel[cl] = value;
                }
                input.write(pack);
                ++written;
            }
    im2row(input, output, g);
    for (int y = 0; y < g.h; ++y)
        for (int tile = 0; tile < tiles; ++tile)
            for (int kh = 0; kh < g.kernel; ++kh)
                for (int kw = 0; kw < g.kernel; ++kw)
                    for (int group = 0; group < groups; ++group) {
                        int8_t expected[PACK_ELEMENTS];
                        if (!packed.read(reinterpret_cast<char*>(expected), sizeof(expected))) {
                            std::cerr << "packed BIN异常EOF: " << name << '\n';
                            return false;
                        }
                        // 缺包时不做阻塞 read；actual 用0，继续检查全部参考。
                        const bool present = !output.empty();
                        activation_pack_t actual;
                        if (present) { actual = output.read(); ++read; }
                        else {
                            actual.spatial_valid = 0;
                            for (int lane = 0; lane < IM2ROW_P; ++lane)
                                for (int cl = 0; cl < FEATURE_PACK_CHANNELS; ++cl)
                                    actual.pixel[lane].channel[cl] = 0;
                            ++errors;
                            report(errors, name, "missing_pack", y, tile, kh, kw, group, -1, -1, 1, 0);
                        }
                        unsigned mask = 0;
                        for (int lane = 0; lane < IM2ROW_P; ++lane)
                            if (tile * IM2ROW_P + lane < g.w) mask |= 1u << lane;
                        const unsigned actual_mask = actual.spatial_valid.to_uint();
                        if (present && mask != actual_mask) {
                            ++errors;
                            report(errors, name, "valid", y, tile, kh, kw, group, -1, -1, mask, actual_mask);
                        }
                        int got[PACK_ELEMENTS];
                        for (int lane = 0; lane < IM2ROW_P; ++lane)
                            for (int cl = 0; cl < FEATURE_PACK_CHANNELS; ++cl) {
                                const int c = group * FEATURE_PACK_CHANNELS + cl;
                                const int x = tile * IM2ROW_P + lane;
                                // unfold 无物理尾通道和不存在的位置；这些元素必须为零。
                                int logical = 0;
                                if (x < g.w && c < g.cin) {
                                    const std::size_t index =
                                        (std::size_t(y) * g.w + x) * g.kernel * g.kernel * g.cin +
                                        (kh * g.kernel + kw) * g.cin + c;
                                    logical = int(unfold[index]);
                                }
                                const int i = lane * FEATURE_PACK_CHANNELS + cl;
                                got[i] = actual.pixel[lane].channel[cl].to_int();
                                if (int(expected[i]) != logical) {
                                    ++errors;
                                    report(errors, name, "packed/unfold", y, tile, kh, kw, group, lane, c, logical, int(expected[i]));
                                }
                                if (present && got[i] != int(expected[i])) {
                                    ++errors;
                                    report(errors, name, "data", y, tile, kh, kw, group, lane, c, int(expected[i]), got[i]);
                                }
                                if (present) ++comparisons;
                            }
                    }
    // 精确长度已在打开时检查，这里再确认顺序读取恰好抵达EOF。
    if (packed.peek() != std::char_traits<char>::eof() || packed.bad()) {
        std::cerr << "packed BIN尾部读取异常: " << name << '\n';
        return false;
    }
    int remaining_inputs = 0, extra_outputs = 0;
    while (!input.empty()) { input.read(); ++remaining_inputs; }
    while (!output.empty()) { output.read(); ++extra_outputs; }
    errors += remaining_inputs + extra_outputs;
    if (written != expected_inputs || read != expected_outputs) ++errors;
    std::cout << case_name << " 输入事务=" << written << '/' << expected_inputs
              << " 输出事务=" << read << '/' << expected_outputs
              << " 剩余输入事务=" << remaining_inputs << " 多余输出事务=" << extra_outputs
              << " 验证元素数=" << comparisons << " 错误数=" << errors
              << ' ' << (errors ? "FAIL" : "PASS") << '\n';
    total_errors += errors;
    return true;
}
} // namespace

int main() {
    const char* configured_path = std::getenv("YOLO_DATA_DIR");
    if (!configured_path || !*configured_path) {
        std::cerr << "请设置环境变量 YOLO_DATA_DIR，指向 YOLO generated 根目录。\n";
        return 1;
    }
    trace_directory = std::string(configured_path) + "/trace/";
    const char* names[] = {"conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "prediction"};
    const char* sources[] = {"input_int8_hwc", "pool1_int8_hwc", "pool2_int8_hwc", "pool3_int8_hwc",
                             "pool4_int8_hwc", "pool5_int8_hwc", "pool6_int8_hwc",
                             "conv7_postprocess_int8_hwc", "conv8_postprocess_int8_hwc"};
    hls::stream<channel_pack_t> input;
    hls::stream<activation_pack_t> output;
    int errors = 0;
    for (int i = 0; i < YOLO_TINY_LAYER_COUNT; ++i)
        if (!run_layer(YOLO_TINY_GEOMETRIES[i], names[i], sources[i], names[i],
                       input, output, errors)) return 1;
    // 复用同一对stream，覆盖 Prediction G128/1×1 → Conv1 G1/3×3 的调用边界。
    if (!run_layer(YOLO_TINY_GEOMETRIES[0], names[0], sources[0], "conv1_repeat",
                   input, output, errors)) return 1;
    return errors ? 1 : 0;
}
