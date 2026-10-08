#include "../src/systolic/systolic.h"

#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data_0000.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv1_weight.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv2_weight.h"

#include <errno.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * Systolic 的 PyTorch 对照 testbench。
 *
 * activation 不由 HLS Im2Row 产生，而是直接由 PyTorch unfold trace 按
 * spatial_tile -> token -> lane 的硬件流顺序装入。weight 则直接读取 exporter
 * 的 [K, OC] header，并按 cout_tile -> token 装入。这使本 TB 仅定位 Systolic
 * 本身的 pack、activation replay、weight replay 与 MAC，而不掩盖上游错误。
 * 输出不做完整 feature map 重排；每个 output transaction 直接映射到 NHWC
 * trace_conv*_systolic_int32_nhwc 的 (y,x,oc) 元素。
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

static int unfold_value(bool conv2, int y, int x, int token, int channel) {
    if (!conv2) {
        return channel == 0 ? trace_conv1_unfold_int8[
            (y * CNN_CONV1_W + x) * SYSTOLIC_KERNEL_TOKENS + token] : 0;
    }
    return trace_conv2_unfold_int8[
        (y * CNN_CONV2_W + x) * (SYSTOLIC_KERNEL_TOKENS * CNN_CONV2_CIN) +
        token * CNN_CONV2_CIN + channel];
}

static int weight_value(bool conv2, int token, int channel, int output) {
    if (!conv2) {
        return channel == 0 ? conv1_weight[token * CNN_CONV1_COUT + output] : 0;
    }
    return conv2_weight[(token * CNN_CONV2_CIN + channel) * CNN_CONV2_COUT +
                        output];
}

static int golden_value(bool conv2, int y, int x, int output) {
    if (!conv2) {
        return trace_conv1_systolic_int32_nhwc[
            (y * CNN_CONV1_W + x) * CNN_CONV1_COUT + output];
    }
    return trace_conv2_systolic_int32_nhwc[
        (y * CNN_CONV2_W + x) * CNN_CONV2_COUT + output];
}

static int run_layer(layer_mode_t layer_mode,
                     hls::stream<activation_pack_t>& activation_stream,
                     hls::stream<weight_pack_t>& weight_stream,
                     hls::stream<acc_tile_t>& output_stream, const char* name,
                     const std::string& directory) {
    const bool conv2 = (layer_mode == LAYER2_MODE);
    const int tiles_per_row = conv2 ? CNN_CONV2_TILES_PER_ROW : CNN_CONV1_TILES_PER_ROW;
    const int spatial_tiles = conv2 ? CNN_CONV2_SPATIAL_TILES : CNN_CONV1_SPATIAL_TILES;
    const int cout_tiles = conv2 ? CNN_CONV2_COUT_TILES : CNN_CONV1_COUT_TILES;
    const std::string prefix = directory + "/" + name;
    std::ofstream activation_csv, weight_csv, output_csv;
    if (!open_csv(activation_csv, prefix + "_activations.csv",
                  "transaction,y,tile_x,token,lane,physical_channel,valid,value") ||
        !open_csv(weight_csv, prefix + "_weights.csv",
                  "transaction,cout_tile,token,cout_lane,physical_channel,output,value") ||
        !open_csv(output_csv, prefix + "_output.csv",
                  "transaction,y,tile_x,cout_tile,lane,cout_lane,output,"
                  "expected_valid,actual_valid,expected,actual,match")) {
        return 1;
    }

    int activations_written = 0;
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const int y = spatial_tile / tiles_per_row;
        const int tile_x = spatial_tile % tiles_per_row;
        const int valid_mask = (conv2 && tile_x == CNN_CONV2_LAST_TILE_X)
            ? CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
        for (int token = 0; token < SYSTOLIC_KERNEL_TOKENS; ++token) {
            activation_pack_t pack;
            pack.spatial_valid = valid_mask;
            for (int lane = 0; lane < SYSTOLIC_SPATIAL_PAR; ++lane) {
                const int x = tile_x * SYSTOLIC_SPATIAL_PAR + lane;
                for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
                    const int value = (valid_mask & (1 << lane)) ?
                        unfold_value(conv2, y, x, token, channel) : 0;
                    pack.pixel[lane].channel[channel] = value;
                    activation_csv << activations_written << ',' << y << ','
                                   << tile_x << ',' << token << ',' << lane << ','
                                   << channel << ','
                                   << ((valid_mask & (1 << lane)) != 0) << ','
                                   << value << '\n';
                }
            }
            activation_stream.write(pack);
            ++activations_written;
        }
    }

    int weights_written = 0;
    for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
        for (int token = 0; token < SYSTOLIC_KERNEL_TOKENS; ++token) {
            weight_pack_t pack;
            for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
                const int output = cout_tile * SYSTOLIC_COUT_PAR + cout_lane;
                for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
                    const int value = weight_value(conv2, token, channel, output);
                    pack.lane[cout_lane].channel[channel] = value;
                    weight_csv << weights_written << ',' << cout_tile << ','
                               << token << ',' << cout_lane << ',' << channel
                               << ',' << output << ',' << value << '\n';
                }
            }
            weight_stream.write(pack);
            ++weights_written;
        }
    }

    systolic(activation_stream, weight_stream, output_stream, layer_mode);

    int errors = 0;
    int comparisons = 0;
    int outputs_read = 0;
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const int y = spatial_tile / tiles_per_row;
        const int tile_x = spatial_tile % tiles_per_row;
        const int expected_valid = (conv2 && tile_x == CNN_CONV2_LAST_TILE_X)
            ? CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
        for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
            const acc_tile_t actual = output_stream.read();
            const int actual_valid = actual.spatial_valid.to_uint();
            for (int lane = 0; lane < SYSTOLIC_SPATIAL_PAR; ++lane) {
                const int x = tile_x * SYSTOLIC_SPATIAL_PAR + lane;
                for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                     ++cout_lane) {
                    const int output = cout_tile * SYSTOLIC_COUT_PAR + cout_lane;
                    const int expected = (expected_valid & (1 << lane)) ?
                        golden_value(conv2, y, x, output) : 0;
                    const int got = actual.value[lane][cout_lane].to_int();
                    const bool match = actual_valid == expected_valid && got == expected;
                    output_csv << outputs_read << ',' << y << ',' << tile_x << ','
                               << cout_tile << ',' << lane << ',' << cout_lane
                               << ',' << output << ',' << expected_valid << ','
                               << actual_valid << ',' << expected << ',' << got
                               << ',' << match << '\n';
                    ++comparisons;
                    if (!match) {
                        ++errors;
                        if (errors <= 10) {
                            std::cerr << name << " 错误: y=" << y << " tile="
                                      << tile_x << " cout_tile=" << cout_tile
                                      << " lane=" << lane << " oc=" << output
                                      << " 期望=" << expected << " 实际=" << got
                                      << " valid=" << actual_valid << std::endl;
                        }
                    }
                }
            }
            ++outputs_read;
        }
    }
    const int expected_activations = spatial_tiles * SYSTOLIC_KERNEL_TOKENS;
    const int expected_weights = cout_tiles * SYSTOLIC_KERNEL_TOKENS;
    const int expected_outputs = spatial_tiles * cout_tiles;
    if (activations_written != expected_activations ||
        weights_written != expected_weights || outputs_read != expected_outputs ||
        !activation_stream.empty() || !weight_stream.empty() ||
        !output_stream.empty()) {
        ++errors;
    }
    std::cout << "Systolic PyTorch " << name << " 输入事务="
              << activations_written << '/' << expected_activations
              << " 权重事务=" << weights_written << '/' << expected_weights
              << " 输出事务=" << outputs_read << '/' << expected_outputs
              << " 检查元素数=" << comparisons << " 错误数=" << errors << ' '
              << (errors ? "FAIL" : "PASS") << std::endl;
    return errors;
}

}  // namespace

static_assert(sizeof(trace_conv1_unfold_int8) / sizeof(trace_conv1_unfold_int8[0]) ==
              CNN_CONV1_H * CNN_CONV1_W * SYSTOLIC_KERNEL_TOKENS, "conv1 unfold shape changed");
static_assert(sizeof(trace_conv2_unfold_int8) / sizeof(trace_conv2_unfold_int8[0]) ==
              CNN_CONV2_H * CNN_CONV2_W * SYSTOLIC_KERNEL_TOKENS * CNN_CONV2_CIN, "conv2 unfold shape changed");
static_assert(sizeof(trace_conv1_systolic_int32_nhwc) /
              sizeof(trace_conv1_systolic_int32_nhwc[0]) == CNN_CONV1_H * CNN_CONV1_W * CNN_CONV1_COUT,
              "conv1 systolic trace shape changed");
static_assert(sizeof(trace_conv2_systolic_int32_nhwc) /
              sizeof(trace_conv2_systolic_int32_nhwc[0]) == CNN_CONV2_H * CNN_CONV2_W * CNN_CONV2_COUT,
              "conv2 systolic trace shape changed");
static_assert(sizeof(conv1_weight) / sizeof(conv1_weight[0]) == SYSTOLIC_KERNEL_TOKENS * CNN_CONV1_COUT,
              "conv1 weight shape changed");
static_assert(sizeof(conv2_weight) / sizeof(conv2_weight[0]) == SYSTOLIC_KERNEL_TOKENS * CNN_CONV2_CIN * CNN_CONV2_COUT,
              "conv2 weight shape changed");

int main() {
    const std::string directory = "simulation_data/pytorch/systolic";
    if (!make_directory("simulation_data") ||
        !make_directory("simulation_data/pytorch") ||
        !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    hls::stream<activation_pack_t> activation_stream;
    hls::stream<weight_pack_t> weight_stream;
    hls::stream<acc_tile_t> output_stream;
    int errors = 0;
    errors += run_layer(LAYER1_MODE, activation_stream, weight_stream, output_stream,
                        "conv1", directory);
    errors += run_layer(LAYER2_MODE, activation_stream, weight_stream, output_stream,
                        "conv2", directory);
    std::cout << "Systolic PyTorch CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
