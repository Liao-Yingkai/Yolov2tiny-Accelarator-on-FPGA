#include "../src/fc/fc.h"

#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * FC 完整定向 testbench。
 * - 每个 case 以 Pool2 NHWC 顺序 pooled_y -> pooled_x -> channel_pack 写98个
 *   输入 pack；权重严格为 out_tile -> token，共196个 pack。DUT 输出2个
 *   5-logit accumulator pack；连续两组固定数据检查 replay RAM/累加器不会
 *   将前一次调用的状态带入下一次。
 * - reference 对784个输入和10个输出做独立 signed INT32 MAC，不复用DUT的
 *   pack或回放实现。CSV以 case_id 和事务号完整保存输入、权重和结果，供
 *   波形与流顺序逐项对照。
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

static int activation_value(int case_id, int y, int x, int channel) {
    // 两个确定性模式必须不同，才能暴露跨调用残留；不使用随机数据。
    return case_id == 0 ?
        ((3 + y * 17 + x * 11 + channel * 7) % 31) - 15 :
        ((29 + y * 17 + x * 11 + channel * 7) % 31) - 15;
}

static int weight_value(int case_id, int output, int input) {
    return case_id == 0 ?
        ((3 + output * 13 + input * 5) % 15) - 7 :
        ((29 + output * 13 + input * 5) % 15) - 7;
}

static void write_case(hls::stream<channel_pack_t>& input_stream,
                       hls::stream<fc_weight_pack_t>& weight_stream,
                       int case_id, std::ofstream& input_csv,
                       std::ofstream& weight_csv, int& inputs_written,
                       int& weights_written) {
    inputs_written = 0;
    weights_written = 0;
    // NHWC 49像素各写两个连续8通道pack，恰好98个输入事务。
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 7; ++x) {
            for (int pack = 0; pack < 2; ++pack) {
                channel_pack_t input;
                for (int lane = 0; lane < FC_CIN_PAR; ++lane) {
                    const int channel = pack * FC_CIN_PAR + lane;
                    input.channel[lane] = activation_value(case_id, y, x, channel);
                    input_csv << case_id << ',' << inputs_written << ',' << y
                              << ',' << x << ',' << pack << ',' << lane << ','
                              << channel << ',' << input.channel[lane].to_int()
                              << '\n';
                }
                input_stream.write(input);
                ++inputs_written;
            }
        }
    }
    // 权重只随out_tile和token递增；每一pack含5x8条 signed 权重。
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        for (int token = 0; token < FC_INPUT_PACKS; ++token) {
            fc_weight_pack_t weights;
            for (int cout_lane = 0; cout_lane < FC_COUT_PAR; ++cout_lane) {
                for (int cin_lane = 0; cin_lane < FC_CIN_PAR; ++cin_lane) {
                    const int output = out_tile * FC_COUT_PAR + cout_lane;
                    const int input = token * FC_CIN_PAR + cin_lane;
                    weights.value[cout_lane][cin_lane] =
                        weight_value(case_id, output, input);
                    weight_csv << case_id << ',' << weights_written << ','
                               << out_tile << ',' << token << ',' << cout_lane
                               << ',' << cin_lane << ',' << output << ','
                               << input << ','
                               << weights.value[cout_lane][cin_lane].to_int()
                               << '\n';
                }
            }
            weight_stream.write(weights);
            ++weights_written;
        }
    }
}

static int check_mac(hls::stream<fc_acc_pack_t>& output_stream, int case_id,
                     std::ofstream& mac_csv, int& comparisons,
                     int& outputs_read) {
    int32_t reference[FC_OUTPUTS] = {0};
    // 明确的784x10参考公式，不复用DUT的pack/replay实现。
    for (int y = 0; y < 7; ++y) {
        for (int x = 0; x < 7; ++x) {
            for (int channel = 0; channel < 16; ++channel) {
                const int input = (y * 7 + x) * 16 + channel;
                const int activation = activation_value(case_id, y, x, channel);
                for (int output = 0; output < FC_OUTPUTS; ++output) {
                    reference[output] += activation *
                        weight_value(case_id, output, input);
                }
            }
        }
    }

    int errors = 0;
    outputs_read = 0;
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        const fc_acc_pack_t actual_pack = output_stream.read();
        for (int lane = 0; lane < FC_COUT_PAR; ++lane) {
            const int output = out_tile * FC_COUT_PAR + lane;
            const int actual = actual_pack.value[lane].to_int();
            const bool match = actual == reference[output];
            mac_csv << case_id << ',' << outputs_read << ',' << out_tile
                    << ",-1," << lane << ',' << output << ",1,"
                    << reference[output] << ',' << actual << ',' << match << '\n';
            ++comparisons;
            if (!match) {
                ++errors;
                if (errors <= 10) {
                    std::cerr << "FC MAC mismatch case=" << case_id
                              << " output=" << output << " expected="
                              << reference[output] << " actual=" << actual
                              << std::endl;
                }
            }
        }
        ++outputs_read;
    }
    return errors;
}

}  // namespace

int main() {
    const std::string directory = "simulation_data/fc";
    if (!make_directory("simulation_data") || !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    std::ofstream input_csv, weight_csv, mac_csv;
    if (!open_csv(input_csv, directory + "/input.csv",
                  "case_id,transaction,pooled_y,pooled_x,channel_pack,lane,channel,value") ||
        !open_csv(weight_csv, directory + "/weights.csv",
                  "case_id,transaction,out_tile,token,cout_lane,cin_lane,output,input,value") ||
        !open_csv(mac_csv, directory + "/mac_output.csv",
                  "case_id,transaction,out_tile,token_not_applicable,lane,channel,valid,expected,actual,match")) {
        return 1;
    }

    int errors = 0;
    hls::stream<channel_pack_t> input_stream;
    hls::stream<fc_weight_pack_t> weight_stream;
    hls::stream<fc_acc_pack_t> mac_stream;
    for (int case_id = 0; case_id < 2; ++case_id) {
        int inputs_written = 0;
        int weights_written = 0;
        int outputs_read = 0;
        int case_comparisons = 0;
        write_case(input_stream, weight_stream, case_id,
                   input_csv, weight_csv, inputs_written, weights_written);
        fully_connected(input_stream, weight_stream, mac_stream);
        int case_errors = check_mac(mac_stream, case_id, mac_csv,
                                    case_comparisons, outputs_read);
        if (inputs_written != FC_INPUT_PACKS ||
            weights_written != FC_OUT_TILES * FC_INPUT_PACKS ||
            outputs_read != FC_OUT_TILES) {
            ++case_errors;
        }
        if (!input_stream.empty() || !weight_stream.empty() || !mac_stream.empty()) {
            ++case_errors;
            std::cerr << "FC MAC stream 未完全消费 case=" << case_id << std::endl;
        }
        errors += case_errors;
        std::cout << "FC case_id=" << case_id << " 输入事务="
                  << inputs_written << '/' << FC_INPUT_PACKS
                  << " 权重事务=" << weights_written << '/'
                  << FC_OUT_TILES * FC_INPUT_PACKS << " 输出事务="
                  << outputs_read << '/' << FC_OUT_TILES
                  << " 检查元素数=" << case_comparisons
                  << " 错误数=" << case_errors << ' '
                  << (case_errors ? "FAIL" : "PASS") << std::endl;
    }
    std::cout << "FC CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
