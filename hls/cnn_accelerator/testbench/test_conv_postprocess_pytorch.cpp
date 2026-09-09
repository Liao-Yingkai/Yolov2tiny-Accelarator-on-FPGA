#include "../src/postprocess/conv_postprocess.h"

#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv1_bias.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv1_quant.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv1_shift.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv2_bias.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv2_quant.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv2_shift.h"

#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * ConvPostProcess 的 PyTorch 对照 testbench。
 *
 * 输入直接由 trace_conv*_systolic_int32_nhwc 按 Systolic 输出流顺序打包，
 * 不依赖 HLS Systolic。每层参数仅在首 spatial tile 送入 Cout tile 顺序的
 * stream。DUT 输出逐事务直接对应 trace_conv*_requant_int8_nhwc。
 *
 * 在调用 DUT 前，本 TB 还独立检查 exporter 的边界：
 * max(trace_systolic + exported_bias, 0) 必须逐元素等于
 * trace_conv*_postprocess_int32_nhwc。该检查将 bias/ReLU 参数错误与
 * HLS requant 硬件错误分开报告。
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

static int systolic_value(bool conv2, int y, int x, int output) {
    return conv2 ? trace_conv2_systolic_int32_nhwc[
                       (y * CNN_CONV2_W + x) * CNN_CONV2_COUT + output]
                 : trace_conv1_systolic_int32_nhwc[
                       (y * CNN_CONV1_W + x) * CNN_CONV1_COUT + output];
}

static int postprocess_value(bool conv2, int y, int x, int output) {
    return conv2 ? trace_conv2_postprocess_int32_nhwc[
                       (y * CNN_CONV2_W + x) * CNN_CONV2_COUT + output]
                 : trace_conv1_postprocess_int32_nhwc[
                       (y * CNN_CONV1_W + x) * CNN_CONV1_COUT + output];
}

static int requant_value(bool conv2, int y, int x, int output) {
    return conv2 ? trace_conv2_requant_int8_nhwc[
                       (y * CNN_CONV2_W + x) * CNN_CONV2_COUT + output]
                 : trace_conv1_requant_int8_nhwc[
                       (y * CNN_CONV1_W + x) * CNN_CONV1_COUT + output];
}

static int bias_value(bool conv2, int output) {
    return conv2 ? conv2_bias[output] : conv1_bias[output];
}

static int multiplier_value(bool conv2, int output) {
    return conv2 ? conv2_multiplier[output] : conv1_multiplier[output];
}

static int shift_value(bool conv2, int output) {
    return conv2 ? conv2_shift[output] : conv1_shift[output];
}

static int bias_relu_value(bool conv2, int y, int x, int output) {
    const int64_t biased = (int64_t)systolic_value(conv2, y, x, output) +
                           bias_value(conv2, output);
    return biased > 0 ? (int)biased : 0;
}

static int run_layer(layer_mode_t layer_mode,
                     hls::stream<acc_tile_t>& input_stream,
                     hls::stream<postprocess_param_pack_t>& param_stream,
                     hls::stream<activation_tile_t>& output_stream,
                     const char* name, const std::string& directory) {
    const bool conv2 = (layer_mode == LAYER2_MODE);
    const int H = conv2 ? CNN_CONV2_H : CNN_CONV1_H;
    const int W = conv2 ? CNN_CONV2_W : CNN_CONV1_W;
    const int Cout = conv2 ? CNN_CONV2_COUT : CNN_CONV1_COUT;
    const int tiles_per_row = conv2 ? CNN_CONV2_TILES_PER_ROW : CNN_CONV1_TILES_PER_ROW;
    const int cout_tiles = conv2 ? CNN_CONV2_COUT_TILES : CNN_CONV1_COUT_TILES;
    const int spatial_tiles = conv2 ? CNN_CONV2_SPATIAL_TILES : CNN_CONV1_SPATIAL_TILES;
    const std::string prefix = directory + "/" + name;
    std::ofstream input_csv, param_csv, output_csv;
    if (!open_csv(input_csv, prefix + "_input.csv",
                  "transaction,y,tile_x,cout_tile,lane,cout_lane,output,valid,accumulator") ||
        !open_csv(param_csv, prefix + "_params.csv",
                  "transaction,cout_tile,cout_lane,output,bias,multiplier,shift") ||
        !open_csv(output_csv, prefix + "_output.csv",
                  "transaction,y,tile_x,cout_tile,lane,cout_lane,output,"
                  "expected_valid,actual_valid,postprocess_expected,postprocess_trace,"
                  "expected,actual,match")) {
        return 1;
    }

    int errors = 0;
    int inputs_written = 0;
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const int y = spatial_tile / tiles_per_row;
        const int tile_x = spatial_tile % tiles_per_row;
        const int valid_mask = (conv2 && tile_x == CNN_CONV2_LAST_TILE_X)
            ? CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
        for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
            acc_tile_t input;
            input.spatial_valid = valid_mask;
            for (int lane = 0; lane < SYSTOLIC_SPATIAL_PAR; ++lane) {
                const int x = tile_x * SYSTOLIC_SPATIAL_PAR + lane;
                for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                     ++cout_lane) {
                    const int output = cout_tile * SYSTOLIC_COUT_PAR + cout_lane;
                    const int value = (valid_mask & (1 << lane)) ?
                        systolic_value(conv2, y, x, output) : 0;
                    input.value[lane][cout_lane] = value;
                    input_csv << inputs_written << ',' << y << ',' << tile_x
                              << ',' << cout_tile << ',' << lane << ','
                              << cout_lane << ',' << output << ','
                              << ((valid_mask & (1 << lane)) != 0) << ','
                              << value << '\n';
                }
            }
            input_stream.write(input);
            ++inputs_written;
        }
    }

    int params_written = 0;
    for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
        postprocess_param_pack_t param;
        for (int lane = 0; lane < SYSTOLIC_COUT_PAR; ++lane) {
            const int output = cout_tile * SYSTOLIC_COUT_PAR + lane;
            param.bias[lane] = bias_value(conv2, output);
            param.multiplier[lane] = multiplier_value(conv2, output);
            param.shift[lane] = shift_value(conv2, output);
            param_csv << params_written << ',' << cout_tile << ',' << lane
                      << ',' << output << ',' << bias_value(conv2, output)
                      << ',' << multiplier_value(conv2, output) << ','
                      << shift_value(conv2, output) << '\n';
        }
        param_stream.write(param);
        ++params_written;
    }

    // 与 HLS stage 分离的 exporter 边界检查：这里仅有 INT32 bias add 与 ReLU。
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int output = 0; output < Cout; ++output) {
                const int expected = bias_relu_value(conv2, y, x, output);
                if (expected != postprocess_value(conv2, y, x, output)) {
                    ++errors;
                    if (errors <= 10) {
                        std::cerr << name << " golden postprocess错误: y=" << y
                                  << " x=" << x << " oc=" << output
                                  << " 期望=" << expected << " trace="
                                  << postprocess_value(conv2, y, x, output)
                                  << std::endl;
                    }
                }
            }
        }
    }

    conv_postprocess(input_stream, param_stream, output_stream, layer_mode);
    int comparisons = 0;
    int outputs_read = 0;
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const int y = spatial_tile / tiles_per_row;
        const int tile_x = spatial_tile % tiles_per_row;
        const int expected_valid = (conv2 && tile_x == CNN_CONV2_LAST_TILE_X)
            ? CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
        for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
            const activation_tile_t actual = output_stream.read();
            const int actual_valid = actual.spatial_valid.to_uint();
            for (int lane = 0; lane < SYSTOLIC_SPATIAL_PAR; ++lane) {
                const int x = tile_x * SYSTOLIC_SPATIAL_PAR + lane;
                for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                     ++cout_lane) {
                    const int output = cout_tile * SYSTOLIC_COUT_PAR + cout_lane;
                    const int expected_post = (expected_valid & (1 << lane)) ?
                        bias_relu_value(conv2, y, x, output) : 0;
                    const int expected = (expected_valid & (1 << lane)) ?
                        requant_value(conv2, y, x, output) : 0;
                    const int got = actual.value[lane][cout_lane].to_int();
                    const bool match = actual_valid == expected_valid && got == expected;
                    output_csv << outputs_read << ',' << y << ',' << tile_x
                               << ',' << cout_tile << ',' << lane << ','
                               << cout_lane << ',' << output << ','
                               << expected_valid << ',' << actual_valid << ','
                               << expected_post << ','
                               << ((expected_valid & (1 << lane)) ?
                                   postprocess_value(conv2, y, x, output) : 0)
                               << ',' << expected << ',' << got << ',' << match
                               << '\n';
                    ++comparisons;
                    if (!match) {
                        ++errors;
                        if (errors <= 10) {
                            std::cerr << name << " 错误: y=" << y << " tile="
                                      << tile_x << " oc=" << output
                                      << " 期望=" << expected << " 实际=" << got
                                      << " valid=" << actual_valid << std::endl;
                        }
                    }
                }
            }
            ++outputs_read;
        }
    }
    const int expected_transactions = spatial_tiles * cout_tiles;
    if (inputs_written != expected_transactions || params_written != cout_tiles ||
        outputs_read != expected_transactions || !input_stream.empty() ||
        !param_stream.empty() || !output_stream.empty()) {
        ++errors;
    }
    std::cout << "ConvPostProcess PyTorch " << name << " 输入事务="
              << inputs_written << '/' << expected_transactions << " 参数事务="
              << params_written << '/' << cout_tiles << " 输出事务="
              << outputs_read << '/' << expected_transactions << " 检查元素数="
              << comparisons << " 错误数=" << errors << ' '
              << (errors ? "FAIL" : "PASS") << std::endl;
    return errors;
}

}  // namespace

static_assert(sizeof(trace_conv1_systolic_int32_nhwc) /
              sizeof(trace_conv1_systolic_int32_nhwc[0]) == CNN_CONV1_H * CNN_CONV1_W * CNN_CONV1_COUT,
              "conv1 systolic trace shape changed");
static_assert(sizeof(trace_conv1_postprocess_int32_nhwc) /
              sizeof(trace_conv1_postprocess_int32_nhwc[0]) == CNN_CONV1_H * CNN_CONV1_W * CNN_CONV1_COUT,
              "conv1 postprocess trace shape changed");
static_assert(sizeof(trace_conv1_requant_int8_nhwc) /
              sizeof(trace_conv1_requant_int8_nhwc[0]) == CNN_CONV1_H * CNN_CONV1_W * CNN_CONV1_COUT,
              "conv1 requant trace shape changed");
static_assert(sizeof(trace_conv2_systolic_int32_nhwc) /
              sizeof(trace_conv2_systolic_int32_nhwc[0]) == CNN_CONV2_H * CNN_CONV2_W * CNN_CONV2_COUT,
              "conv2 systolic trace shape changed");
static_assert(sizeof(trace_conv2_postprocess_int32_nhwc) /
              sizeof(trace_conv2_postprocess_int32_nhwc[0]) == CNN_CONV2_H * CNN_CONV2_W * CNN_CONV2_COUT,
              "conv2 postprocess trace shape changed");
static_assert(sizeof(trace_conv2_requant_int8_nhwc) /
              sizeof(trace_conv2_requant_int8_nhwc[0]) == CNN_CONV2_H * CNN_CONV2_W * CNN_CONV2_COUT,
              "conv2 requant trace shape changed");
static_assert(sizeof(conv1_bias) / sizeof(conv1_bias[0]) == CNN_CONV1_COUT,
              "conv1 bias shape changed");
static_assert(sizeof(conv2_bias) / sizeof(conv2_bias[0]) == CNN_CONV2_COUT,
              "conv2 bias shape changed");
static_assert(sizeof(conv1_multiplier) / sizeof(conv1_multiplier[0]) == CNN_CONV1_COUT,
              "conv1 multiplier shape changed");
static_assert(sizeof(conv2_multiplier) / sizeof(conv2_multiplier[0]) == CNN_CONV2_COUT,
              "conv2 multiplier shape changed");
static_assert(sizeof(conv1_shift) / sizeof(conv1_shift[0]) == CNN_CONV1_COUT,
              "conv1 shift shape changed");
static_assert(sizeof(conv2_shift) / sizeof(conv2_shift[0]) == CNN_CONV2_COUT,
              "conv2 shift shape changed");

int main() {
    const std::string directory = "simulation_data/pytorch/conv_postprocess";
    if (!make_directory("simulation_data") ||
        !make_directory("simulation_data/pytorch") ||
        !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    hls::stream<acc_tile_t> input_stream;
    hls::stream<postprocess_param_pack_t> param_stream;
    hls::stream<activation_tile_t> output_stream;
    int errors = 0;
    errors += run_layer(LAYER1_MODE, input_stream, param_stream, output_stream,
                        "conv1", directory);
    errors += run_layer(LAYER2_MODE, input_stream, param_stream, output_stream,
                        "conv2", directory);
    std::cout << "ConvPostProcess PyTorch CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
