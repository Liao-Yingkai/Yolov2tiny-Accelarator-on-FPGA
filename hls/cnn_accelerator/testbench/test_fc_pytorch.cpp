#include "../src/fc/fc.h"

#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data_0000.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/fc_weight.h"

#include <errno.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * 独立 FC MAC 的 PyTorch 对照 testbench。
 *
 * Pool2 trace 是 [7][7][16] NHWC。它被直接写成 98 个 channel_pack_t：每个
 * 空间位置先 c0..c7，后 c8..c15，因此 token * 8 + cin_lane 正是 exporter
 * 已离线转换的 HWC flatten 输入下标。v2 FC 权重 header 为 [2][98][5][8]，流顺序为
 * out_tile -> token。DUT 的两个 fc_acc_pack_t 直接对照十路 int32 systolic
 * trace；本 TB 不调用 Pool 或 FC PostProcess。
 */
namespace {
static_assert(FC_WEIGHT_LAYOUT_VERSION == 2, "requires blocked FC weight header v2");

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

static int run_fc(hls::stream<channel_pack_t>& input_stream,
                  hls::stream<fc_weight_pack_t>& weight_stream,
                  hls::stream<fc_acc_pack_t>& output_stream,
                  const std::string& directory) {
    std::ofstream input_csv, weight_csv, output_csv;
    if (!open_csv(input_csv, directory + "/input.csv",
                  "transaction,pooled_y,pooled_x,channel_pack,cin_lane,channel,value") ||
        !open_csv(weight_csv, directory + "/weights.csv",
                  "transaction,out_tile,token,cout_lane,cin_lane,output,input,value") ||
        !open_csv(output_csv, directory + "/output.csv",
                  "transaction,out_tile,cout_lane,output,expected,actual,match")) {
        return 1;
    }

    int inputs_written = 0;
    for (int y = 0; y < CNN_POOL2_H; ++y) {
        for (int x = 0; x < CNN_POOL2_W; ++x) {
            for (int pack = 0; pack < CNN_POOL2_CHANNEL_PACKS; ++pack) {
                channel_pack_t input;
                for (int lane = 0; lane < FC_CIN_PAR; ++lane) {
                    const int channel = pack * FC_CIN_PAR + lane;
                    const int value = trace_pool2_int8_nhwc[
                        (y * CNN_POOL2_W + x) * CNN_POOL2_C + channel];
                    input.channel[lane] = value;
                    input_csv << inputs_written << ',' << y << ',' << x << ','
                              << pack << ',' << lane << ',' << channel << ','
                              << value << '\n';
                }
                input_stream.write(input);
                ++inputs_written;
            }
        }
    }

    int weights_written = 0;
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        for (int token = 0; token < FC_INPUT_PACKS; ++token) {
            fc_weight_pack_t weight;
            for (int cout_lane = 0; cout_lane < FC_COUT_PAR; ++cout_lane) {
                const int output = out_tile * FC_COUT_PAR + cout_lane;
                for (int cin_lane = 0; cin_lane < FC_CIN_PAR; ++cin_lane) {
                    const int input = token * FC_CIN_PAR + cin_lane;
                    const int value = fc_weight[((out_tile * FC_INPUT_PACKS + token) * FC_COUT_PAR + cout_lane) * FC_CIN_PAR + cin_lane];
                    weight.value[cout_lane][cin_lane] = value;
                    weight_csv << weights_written << ',' << out_tile << ','
                               << token << ',' << cout_lane << ',' << cin_lane
                               << ',' << output << ',' << input << ',' << value
                               << '\n';
                }
            }
            weight_stream.write(weight);
            ++weights_written;
        }
    }

    fully_connected(input_stream, weight_stream, output_stream);

    int errors = 0;
    int comparisons = 0;
    int outputs_read = 0;
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        const fc_acc_pack_t actual = output_stream.read();
        for (int lane = 0; lane < FC_COUT_PAR; ++lane) {
            const int output = out_tile * FC_COUT_PAR + lane;
            const int expected = trace_fc_systolic_int32[output];
            const int got = actual.value[lane].to_int();
            const bool match = got == expected;
            output_csv << outputs_read << ',' << out_tile << ',' << lane << ','
                       << output << ',' << expected << ',' << got << ',' << match
                       << '\n';
            ++comparisons;
            if (!match) {
                ++errors;
                if (errors <= 10) {
                    std::cerr << "FC PyTorch错误: out_tile=" << out_tile
                              << " output=" << output << " 期望=" << expected
                              << " 实际=" << got << std::endl;
                }
            }
        }
        ++outputs_read;
    }
    if (inputs_written != FC_INPUT_PACKS ||
        weights_written != FC_OUT_TILES * FC_INPUT_PACKS ||
        outputs_read != FC_OUT_TILES || !input_stream.empty() ||
        !weight_stream.empty() || !output_stream.empty()) {
        ++errors;
    }
    std::cout << "FC PyTorch 输入事务=" << inputs_written << '/'
              << FC_INPUT_PACKS << " 权重事务=" << weights_written << '/'
              << FC_OUT_TILES * FC_INPUT_PACKS << " 输出事务=" << outputs_read
              << '/' << FC_OUT_TILES << " 检查元素数=" << comparisons
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS")
              << std::endl;
    return errors;
}

}  // namespace

static_assert(sizeof(trace_pool2_int8_nhwc) / sizeof(trace_pool2_int8_nhwc[0]) ==
              CNN_POOL2_H * CNN_POOL2_W * CNN_POOL2_C, "pool2 trace shape changed");
static_assert(sizeof(trace_fc_systolic_int32) / sizeof(trace_fc_systolic_int32[0]) ==
              FC_OUTPUTS, "fc systolic trace shape changed");
static_assert(sizeof(fc_weight) / sizeof(fc_weight[0]) ==
              FC_INPUT_PACKS * FC_CIN_PAR * FC_OUTPUTS,
              "fc weight shape changed");

int main() {
    const std::string directory = "simulation_data/pytorch/fc";
    if (!make_directory("simulation_data") ||
        !make_directory("simulation_data/pytorch") ||
        !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    hls::stream<channel_pack_t> input_stream;
    hls::stream<fc_weight_pack_t> weight_stream;
    hls::stream<fc_acc_pack_t> output_stream;
    const int errors = run_fc(input_stream, weight_stream, output_stream, directory);
    std::cout << "FC PyTorch CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
