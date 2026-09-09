#include "../src/im2row/im2row.h"

// 临时使用绝对路径读取 PyTorch dump_trace.py 生成的单图 golden trace。
// 后续系统集成时可改为由构建脚本传入的相对 include 路径。
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data.h"

#include <errno.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * Im2Row 的 PyTorch 对照 testbench。
 *
 * dump_trace.py 把 PyTorch/NumPy golden model 的 unfold 矩阵按 row-major
 * 导出：Conv1 为 [784][9]，Conv2 为 [196][72]。本 TB 不重排 DUT 输出：
 * 直接按 Im2Row stream 的 row -> tile_x -> token -> lane -> physical channel
 * 顺序恢复逻辑 (y,x,kh,kw,c)，并索引对应 unfold 元素。
 *
 * Conv1 输入来自 trace_input_int8_hwc；其 physical c1..c7 特意填入非零值，
 * 用于确认 Im2Row 只保留逻辑 c0。Conv2 输入直接来自 pool1 trace，因为它
 * 是 Conv2 的实际上游 NHWC 输出。每层只调用一次 im2row，不与其他 HLS DUT
 * 串接，因此可作为独立 C-sim / co-sim top 使用。
 */
namespace {

static bool make_directory(const char* path) {
    return mkdir(path, 0777) == 0 || errno == EEXIST;
}

static bool open_csv(std::ofstream& file, const std::string& path,
                     const char* header) {
    file.open(path.c_str());
    if (!file) {
        std::cerr << "无法打开 CSV: " << path << std::endl;
        return false;
    }
    file << header << '\n';
    return true;
}

static int input_value(bool conv2, int y, int x, int channel) {
    if (!conv2) {
        // Conv1 实际只有 c0；非逻辑 physical channel 用 poison 验证规范化。
        return channel == 0 ? trace_input_int8_hwc[y * CNN_CONV1_W + x]
                            : 71 + channel;
    }
    return trace_pool1_int8_nhwc[(y * CNN_CONV2_W + x) * CNN_CONV2_CIN +
                                 channel];
}

static int unfold_value(bool conv2, int spatial, int token, int channel) {
    if (!conv2) {
        return channel == 0 ? trace_conv1_unfold_int8[
                                  spatial * SYSTOLIC_KERNEL_TOKENS + token]
                            : 0;
    }
    return trace_conv2_unfold_int8[
        spatial * (SYSTOLIC_KERNEL_TOKENS * CNN_CONV2_CIN) +
        token * CNN_CONV2_CIN + channel];
}

static int run_layer(layer_mode_t layer_mode, hls::stream<pixel_t>& input_stream,
                     hls::stream<activation_pack_t>& output_stream,
                     const char* name, const std::string& directory) {
    const bool conv2 = (layer_mode == LAYER2_MODE);
    const int H = conv2 ? CNN_CONV2_H : CNN_CONV1_H;
    const int W = conv2 ? CNN_CONV2_W : CNN_CONV1_W;
    const int tiles_per_row = conv2 ? CNN_CONV2_TILES_PER_ROW : CNN_CONV1_TILES_PER_ROW;
    const int expected_inputs = H * W;
    const int expected_outputs = H * tiles_per_row * SYSTOLIC_KERNEL_TOKENS;
    const std::string prefix = directory + "/" + name;
    std::ofstream input_csv, output_csv;
    if (!open_csv(input_csv, prefix + "_input.csv",
                  "y,x,physical_channel,value") ||
        !open_csv(output_csv, prefix + "_output.csv",
                  "transaction,y,tile_x,token,kh,kw,lane,physical_channel,"
                  "expected_valid,actual_valid,expected,actual,match")) {
        return 1;
    }

    int inputs_written = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            pixel_t pixel;
            for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
                const int value = input_value(conv2, y, x, channel);
                pixel.channel[channel] = value;
                input_csv << y << ',' << x << ',' << channel << ',' << value
                          << '\n';
            }
            input_stream.write(pixel);
            ++inputs_written;
        }
    }

    im2row(input_stream, output_stream, layer_mode);

    int errors = 0;
    int comparisons = 0;
    int outputs_read = 0;
    for (int y = 0; y < H; ++y) {
        for (int tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
            const int expected_valid = (conv2 && tile_x == CNN_CONV2_LAST_TILE_X)
                ? CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
            for (int token = 0; token < SYSTOLIC_KERNEL_TOKENS; ++token) {
                const activation_pack_t actual = output_stream.read();
                const int actual_valid = actual.spatial_valid.to_uint();
                if (actual_valid != expected_valid) {
                    ++errors;
                    if (errors <= 10) {
                        std::cerr << name << " valid错误: y=" << y
                                  << " tile=" << tile_x << " token=" << token
                                  << " 期望=" << expected_valid
                                  << " 实际=" << actual_valid << std::endl;
                    }
                }
                for (int lane = 0; lane < IM2ROW_P; ++lane) {
                    const int x = tile_x * IM2ROW_P + lane;
                    const bool valid = (expected_valid & (1 << lane)) != 0;
                    for (int channel = 0; channel < IM2ROW_MAX_CHANNELS;
                         ++channel) {
                        const int expected = valid ? unfold_value(
                            conv2, y * W + x, token, channel) : 0;
                        const int got = actual.pixel[lane].channel[channel].to_int();
                        const bool match = actual_valid == expected_valid &&
                                           got == expected;
                        output_csv << outputs_read << ',' << y << ',' << tile_x
                                   << ',' << token << ',' << token / IM2ROW_KERNEL << ','
                                   << token % IM2ROW_KERNEL << ',' << lane << ',' << channel
                                   << ',' << expected_valid << ',' << actual_valid
                                   << ',' << expected << ',' << got << ',' << match
                                   << '\n';
                        ++comparisons;
                        if (!match && errors < 10) {
                            std::cerr << name << " 数据错误: y=" << y
                                      << " tile=" << tile_x << " token=" << token
                                      << " lane=" << lane << " c=" << channel
                                      << " 期望=" << expected << " 实际=" << got
                                      << std::endl;
                        }
                        if (!match) ++errors;
                    }
                }
                ++outputs_read;
            }
        }
    }
    if (inputs_written != expected_inputs || outputs_read != expected_outputs ||
        !input_stream.empty() || !output_stream.empty()) {
        ++errors;
    }
    std::cout << "Im2Row PyTorch " << name << " 输入事务=" << inputs_written
              << '/' << expected_inputs << " 输出事务=" << outputs_read << '/'
              << expected_outputs << " 检查元素数=" << comparisons
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS")
              << std::endl;
    return errors;
}

}  // namespace

static_assert(sizeof(trace_input_int8_hwc) / sizeof(trace_input_int8_hwc[0]) ==
              CNN_CONV1_H * CNN_CONV1_W * CNN_CONV1_CIN,
              "trace input shape changed");
static_assert(sizeof(trace_conv1_unfold_int8) / sizeof(trace_conv1_unfold_int8[0]) ==
              CNN_CONV1_H * CNN_CONV1_W * SYSTOLIC_KERNEL_TOKENS,
              "conv1 unfold shape changed");
static_assert(sizeof(trace_pool1_int8_nhwc) / sizeof(trace_pool1_int8_nhwc[0]) ==
              CNN_CONV2_H * CNN_CONV2_W * CNN_CONV2_CIN,
              "pool1 trace shape changed");
static_assert(sizeof(trace_conv2_unfold_int8) / sizeof(trace_conv2_unfold_int8[0]) ==
              CNN_CONV2_H * CNN_CONV2_W * SYSTOLIC_KERNEL_TOKENS * CNN_CONV2_CIN,
              "conv2 unfold shape changed");

int main() {
    const std::string directory = "simulation_data/pytorch/im2row";
    if (!make_directory("simulation_data") ||
        !make_directory("simulation_data/pytorch") ||
        !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    hls::stream<pixel_t> input_stream;
    hls::stream<activation_pack_t> output_stream;
    int errors = 0;
    errors += run_layer(LAYER1_MODE, input_stream, output_stream, "conv1", directory);
    errors += run_layer(LAYER2_MODE, input_stream, output_stream, "conv2", directory);
    std::cout << "Im2Row PyTorch CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
