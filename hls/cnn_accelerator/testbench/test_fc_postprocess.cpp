#include "../src/postprocess/fc_postprocess.h"

#include <cassert>
#include <cerrno>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * FC 后处理定向 testbench。
 * - 每个 case 向输入流和参数流各写两个5-lane Pack，并只从输出流读取一个
 *   10-lane logits Pack；连续执行两个 case，以检查调用之间没有残留状态。
 * - reference 完全独立采用 int64_t，覆盖 bias、负数算术右移、shift 40/41、
 *   上下 INT8 饱和；本 TB 只调用 fc_postprocess 这个 stage top。
 * - params/input/output CSV 均记录 case_id、事务及 lane，便于逐事务检查流顺序。
 */
namespace {

bool make_directory(const char* path) {
    return mkdir(path, 0777) == 0 || errno == EEXIST;
}

bool open_csv(std::ofstream& file, const std::string& path,
              const char* header) {
    file.open(path);
    if (!file) {
        std::cerr << "无法打开 CSV: " << path << std::endl;
        return false;
    }
    file << header << '\n';
    return true;
}

int8_t reference_requantize(int32_t accumulator, int32_t bias,
                            int32_t multiplier, int32_t shift) {
    assert(multiplier >= 0 && shift >= 0);
    const int64_t biased = (int64_t)accumulator + bias;
    assert(biased >= INT32_MIN && biased <= INT32_MAX);
    const int64_t product = biased * multiplier;
    const int64_t scaled = shift >= 64 ? (product < 0 ? -1 : 0) :
                           (product >> shift);
    return scaled > 127 ? 127 : (scaled < -128 ? -128 : (int8_t)scaled);
}

void make_case_value(int case_id, int output, int32_t& accumulator,
                     int32_t& bias, int32_t& multiplier,
                     int32_t& shift) {
    // 前五路显式覆盖负数 shift40/41、上饱和和下饱和；其余通道保持一般
    // signed bias/requant 组合。第二个 case 使用不同值检查没有上次调用残留。
    if (case_id == 0) {
        const int32_t accumulators[FC_OUTPUTS] =
            {-3072, -3072, 127, -127, -42, 48, 511, -511, 17, -17};
        const int32_t biases[FC_OUTPUTS] =
            {0, 0, 1, -2, 10, -64, 0, 0, -3, 3};
        const int32_t multipliers[FC_OUTPUTS] =
            {1 << 30, 1 << 30, 1, 1, 1, 1, 3, 3, 7, 7};
        const int32_t shifts[FC_OUTPUTS] = {40, 41, 0, 0, 0, 1, 3, 3, 2, 2};
        accumulator = accumulators[output];
        bias = biases[output];
        multiplier = multipliers[output];
        shift = shifts[output];
    } else {
        accumulator = (output - 4) * 777;
        bias = output * 19 - 73;
        multiplier = output == 7 ? 5 : (output + 1);
        shift = output % 4;
    }
}

int run_case(int case_id, hls::stream<fc_acc_pack_t>& input_stream,
             hls::stream<fc_param_pack_t>& param_stream,
             hls::stream<logits_pack_t>& output_stream,
             std::ofstream& params_csv, std::ofstream& input_csv,
             std::ofstream& output_csv, int& comparisons) {
    int32_t accumulator[FC_OUTPUTS];
    int32_t bias[FC_OUTPUTS];
    int32_t multiplier[FC_OUTPUTS];
    int32_t shift[FC_OUTPUTS];
    int inputs_written = 0;
    int params_written = 0;

    // 流顺序固定为 out_tile 0 后 out_tile 1；同号 transaction 的 accumulator
    // Pack 与 parameter Pack 一一对应。
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        fc_acc_pack_t input;
        fc_param_pack_t param;
        for (int lane = 0; lane < FC_COUT_PAR; ++lane) {
            const int output = out_tile * FC_COUT_PAR + lane;
            make_case_value(case_id, output, accumulator[output], bias[output],
                            multiplier[output], shift[output]);
            input.value[lane] = accumulator[output];
            param.bias[lane] = bias[output];
            param.multiplier[lane] = multiplier[output];
            param.shift[lane] = shift[output];
            input_csv << case_id << ',' << inputs_written << ',' << out_tile
                      << ',' << lane << ',' << output << ','
                      << accumulator[output] << '\n';
            params_csv << case_id << ',' << params_written << ',' << out_tile
                       << ',' << lane << ',' << output << ',' << bias[output]
                       << ',' << multiplier[output] << ',' << shift[output]
                       << '\n';
        }
        input_stream.write(input);
        param_stream.write(param);
        ++inputs_written;
        ++params_written;
    }

    fc_postprocess(input_stream, param_stream, output_stream);

    int errors = 0;
    const logits_pack_t actual_logits = output_stream.read();
    const int outputs_read = 1;
    for (int output = 0; output < FC_OUTPUTS; ++output) {
        const int expected = reference_requantize(accumulator[output],
                                                  bias[output],
                                                  multiplier[output],
                                                  shift[output]);
        const int actual = actual_logits.value[output].to_int();
        const bool match = actual == expected;
        output_csv << case_id << ",0," << output << ",1," << expected << ','
                   << actual << ',' << match << '\n';
        ++comparisons;
        if (!match) {
            ++errors;
            if (errors <= 10) {
                std::cerr << "FCPostProcess mismatch case=" << case_id
                          << " output=" << output << " expected=" << expected
                          << " actual=" << actual << std::endl;
            }
        }
    }
    if (inputs_written != FC_OUT_TILES || params_written != FC_OUT_TILES ||
        outputs_read != 1) {
        ++errors;
        std::cerr << "FCPostProcess transaction count mismatch case=" << case_id
                  << std::endl;
    }
    if (!input_stream.empty() || !param_stream.empty() || !output_stream.empty()) {
        ++errors;
        std::cerr << "FCPostProcess stream 未完全消费 case=" << case_id
                  << std::endl;
    }
    std::cout << "FCPostProcess case_id=" << case_id << " 输入事务="
              << inputs_written << '/' << FC_OUT_TILES << " 参数事务="
              << params_written << '/' << FC_OUT_TILES << " 输出事务="
              << outputs_read << "/1 检查元素数=" << comparisons
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS")
              << std::endl;
    return errors;
}

}  // namespace

int main() {
    const std::string directory = "simulation_data/fc_postprocess";
    if (!make_directory("simulation_data") || !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    std::ofstream params_csv, input_csv, output_csv;
    if (!open_csv(params_csv, directory + "/params.csv",
                  "case_id,transaction,out_tile,lane,output,bias,multiplier,shift") ||
        !open_csv(input_csv, directory + "/input.csv",
                  "case_id,transaction,out_tile,lane,output,accumulator") ||
        !open_csv(output_csv, directory + "/output.csv",
                  "case_id,transaction,output,valid,expected,actual,match")) {
        return 1;
    }

    int errors = 0;
    hls::stream<fc_acc_pack_t> input_stream;
    hls::stream<fc_param_pack_t> param_stream;
    hls::stream<logits_pack_t> output_stream;
    for (int case_id = 0; case_id < 2; ++case_id) {
        int case_comparisons = 0;
        const int case_errors = run_case(case_id, input_stream, param_stream,
                                         output_stream, params_csv, input_csv,
                                         output_csv, case_comparisons);
        errors += case_errors;
    }
    std::cout << "FCPostProcess CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
