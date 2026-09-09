#include "../src/pool/pool.h"

#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data.h"

#include <errno.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * MaxPool 的 PyTorch 对照 testbench。
 *
 * Pool 输入直接由 Conv requant NHWC trace 打包为硬件规定的
 * row -> tile_x -> cout_tile activation_tile_t 顺序，而不调用 ConvPostProcess。
 * 输出则按 Pool 的紧凑 NHWC stream：pooled_y -> pooled_x -> channel_pack
 * 逐项直接索引 trace_pool*_int8_nhwc。Pool2 输入行尾 tile 的 spatial_valid=0011
 * 也随输入写入，检查其不影响最后两个有效位置的 signed max。
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

static int input_value(bool pool2, int y, int x, int channel) {
    return pool2 ? trace_conv2_requant_int8_nhwc[
                       (y * CNN_CONV2_W + x) * CNN_CONV2_COUT + channel]
                 : trace_conv1_requant_int8_nhwc[
                       (y * CNN_CONV1_W + x) * CNN_CONV1_COUT + channel];
}

static int golden_value(bool pool2, int y, int x, int channel) {
    return pool2 ? trace_pool2_int8_nhwc[
                       (y * (CNN_CONV2_W / 2) + x) * CNN_CONV2_COUT + channel]
                 : trace_pool1_int8_nhwc[
                       (y * (CNN_CONV1_W / 2) + x) * CNN_CONV1_COUT + channel];
}

static int run_layer(bool pool2, hls::stream<activation_tile_t>& input_stream,
                     hls::stream<channel_pack_t>& output_stream,
                     const char* name, const std::string& directory) {
    const int H = pool2 ? CNN_CONV2_H : CNN_CONV1_H;
    const int W = pool2 ? CNN_CONV2_W : CNN_CONV1_W;
    const int Cout = pool2 ? CNN_CONV2_COUT : CNN_CONV1_COUT;
    const int tiles_per_row = (W + SYSTOLIC_SPATIAL_PAR - 1) /
                              SYSTOLIC_SPATIAL_PAR;
    const int cout_tiles = Cout / SYSTOLIC_COUT_PAR;
    const int pooled_h = H / 2;
    const int pooled_w = W / 2;
    const int channel_packs = Cout / IM2ROW_MAX_CHANNELS;
    const std::string prefix = directory + "/" + name;
    std::ofstream input_csv, output_csv;
    if (!open_csv(input_csv, prefix + "_input.csv",
                  "transaction,y,tile_x,cout_tile,lane,cout_lane,channel,valid,value") ||
        !open_csv(output_csv, prefix + "_output.csv",
                  "transaction,pooled_y,pooled_x,channel_pack,lane,channel,"
                  "expected,actual,match")) {
        return 1;
    }

    int inputs_written = 0;
    for (int y = 0; y < H; ++y) {
        for (int tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
            const int valid_mask = (pool2 && tile_x == tiles_per_row - 1) ? 0x3 : 0xf;
            for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
                activation_tile_t input;
                input.spatial_valid = valid_mask;
                for (int lane = 0; lane < SYSTOLIC_SPATIAL_PAR; ++lane) {
                    const int x = tile_x * SYSTOLIC_SPATIAL_PAR + lane;
                    for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                         ++cout_lane) {
                        const int channel = cout_tile * SYSTOLIC_COUT_PAR + cout_lane;
                        const int value = (valid_mask & (1 << lane)) ?
                            input_value(pool2, y, x, channel) : 0;
                        input.value[lane][cout_lane] = value;
                        input_csv << inputs_written << ',' << y << ',' << tile_x
                                  << ',' << cout_tile << ',' << lane << ','
                                  << cout_lane << ',' << channel << ','
                                  << ((valid_mask & (1 << lane)) != 0) << ','
                                  << value << '\n';
                    }
                }
                input_stream.write(input);
                ++inputs_written;
            }
        }
    }

    maxpool(input_stream, output_stream, H, W, Cout);

    int errors = 0;
    int comparisons = 0;
    int outputs_read = 0;
    for (int y = 0; y < pooled_h; ++y) {
        for (int x = 0; x < pooled_w; ++x) {
            for (int pack = 0; pack < channel_packs; ++pack) {
                const channel_pack_t actual = output_stream.read();
                for (int lane = 0; lane < IM2ROW_MAX_CHANNELS; ++lane) {
                    const int channel = pack * IM2ROW_MAX_CHANNELS + lane;
                    const int expected = golden_value(pool2, y, x, channel);
                    const int got = actual.channel[lane].to_int();
                    const bool match = got == expected;
                    output_csv << outputs_read << ',' << y << ',' << x << ','
                               << pack << ',' << lane << ',' << channel << ','
                               << expected << ',' << got << ',' << match << '\n';
                    ++comparisons;
                    if (!match) {
                        ++errors;
                        if (errors <= 10) {
                            std::cerr << name << " 错误: y=" << y << " x=" << x
                                      << " channel=" << channel << " 期望="
                                      << expected << " 实际=" << got << std::endl;
                        }
                    }
                }
                ++outputs_read;
            }
        }
    }
    const int expected_inputs = H * tiles_per_row * cout_tiles;
    const int expected_outputs = pooled_h * pooled_w * channel_packs;
    if (inputs_written != expected_inputs || outputs_read != expected_outputs ||
        !input_stream.empty() || !output_stream.empty()) {
        ++errors;
    }
    std::cout << "Pool PyTorch " << name << " 输入事务=" << inputs_written
              << '/' << expected_inputs << " 输出事务=" << outputs_read << '/'
              << expected_outputs << " 检查元素数=" << comparisons
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS")
              << std::endl;
    return errors;
}

}  // namespace

static_assert(sizeof(trace_conv1_requant_int8_nhwc) /
              sizeof(trace_conv1_requant_int8_nhwc[0]) == 28 * 28 * 8,
              "conv1 requant trace shape changed");
static_assert(sizeof(trace_pool1_int8_nhwc) / sizeof(trace_pool1_int8_nhwc[0]) ==
              14 * 14 * 8, "pool1 trace shape changed");
static_assert(sizeof(trace_conv2_requant_int8_nhwc) /
              sizeof(trace_conv2_requant_int8_nhwc[0]) == 14 * 14 * 16,
              "conv2 requant trace shape changed");
static_assert(sizeof(trace_pool2_int8_nhwc) / sizeof(trace_pool2_int8_nhwc[0]) ==
              7 * 7 * 16, "pool2 trace shape changed");

int main() {
    const std::string directory = "simulation_data/pytorch/pool";
    if (!make_directory("simulation_data") ||
        !make_directory("simulation_data/pytorch") ||
        !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    hls::stream<activation_tile_t> input_stream;
    hls::stream<channel_pack_t> output_stream;
    int errors = 0;
    errors += run_layer(false, input_stream, output_stream, "pool1", directory);
    errors += run_layer(true, input_stream, output_stream, "pool2", directory);
    std::cout << "Pool PyTorch CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
