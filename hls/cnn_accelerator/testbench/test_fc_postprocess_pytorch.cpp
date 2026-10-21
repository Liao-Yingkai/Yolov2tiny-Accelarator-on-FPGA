#include "../src/postprocess/fc_postprocess.h"

#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data_0000.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/fc_bias.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/fc_quant.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/fc_shift.h"

#include <cerrno>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * FC PostProcess 的 PyTorch 对照 testbench。
 *
 * 两个 fc_acc_pack_t 直接取自 trace_fc_systolic_int32，两个 parameter pack
 * 直接取 exporter headers，因而不依赖 HLS FC MAC。调用前先验证每个输出的
 * trace_fc_systolic_int32 + fc_bias 等于 trace_fc_postprocess_int32；随后单个
 * logits_pack_t 逐 lane 直接比较 trace_fc_requant_int8。FC 路径无 ReLU，负数
 * 必须保留给 signed multiply、算术右移和双向饱和。
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

int run_fc_postprocess(hls::stream<fc_acc_pack_t>& input_stream,
                       hls::stream<fc_param_pack_t>& param_stream,
                       hls::stream<logits_pack_t>& output_stream,
                       const std::string& directory) {
    std::ofstream input_csv, param_csv, output_csv;
    if (!open_csv(input_csv, directory + "/input.csv",
                  "transaction,out_tile,lane,output,accumulator") ||
        !open_csv(param_csv, directory + "/params.csv",
                  "transaction,out_tile,lane,output,bias,multiplier,shift") ||
        !open_csv(output_csv, directory + "/output.csv",
                  "transaction,output,postprocess_expected,postprocess_trace,"
                  "expected,actual,match")) {
        return 1;
    }

    int errors = 0;
    // 验证 Python trace 的 FC 模块边界；FC 后处理没有 ReLU。
    for (int output = 0; output < FC_OUTPUTS; ++output) {
        const int64_t expected = (int64_t)trace_fc_systolic_int32[output] +
                                 fc_bias[output];
        if (expected != trace_fc_postprocess_int32[output]) {
            ++errors;
            if (errors <= 10) {
                std::cerr << "FC golden postprocess错误: output=" << output
                          << " 期望=" << expected << " trace="
                          << trace_fc_postprocess_int32[output] << std::endl;
            }
        }
    }

    int inputs_written = 0;
    int params_written = 0;
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        fc_acc_pack_t input;
        fc_param_pack_t param;
        for (int lane = 0; lane < FC_COUT_PAR; ++lane) {
            const int output = out_tile * FC_COUT_PAR + lane;
            input.value[lane] = trace_fc_systolic_int32[output];
            param.bias[lane] = fc_bias[output];
            param.multiplier[lane] = fc_multiplier[output];
            param.shift[lane] = fc_shift[output];
            input_csv << inputs_written << ',' << out_tile << ',' << lane
                      << ',' << output << ',' << trace_fc_systolic_int32[output]
                      << '\n';
            param_csv << params_written << ',' << out_tile << ',' << lane
                      << ',' << output << ',' << fc_bias[output] << ','
                      << fc_multiplier[output] << ',' << fc_shift[output] << '\n';
        }
        input_stream.write(input);
        param_stream.write(param);
        ++inputs_written;
        ++params_written;
    }

    fc_postprocess(input_stream, param_stream, output_stream);

    const logits_pack_t actual = output_stream.read();
    int comparisons = 0;
    for (int output = 0; output < FC_OUTPUTS; ++output) {
        const int expected_post = trace_fc_systolic_int32[output] + fc_bias[output];
        const int expected = trace_fc_requant_int8[output];
        const int got = actual.value[output].to_int();
        const bool match = got == expected;
        output_csv << "0," << output << ',' << expected_post << ','
                   << trace_fc_postprocess_int32[output] << ',' << expected
                   << ',' << got << ',' << match << '\n';
        ++comparisons;
        if (!match) {
            ++errors;
            if (errors <= 10) {
                std::cerr << "FCPostProcess PyTorch错误: output=" << output
                          << " 期望=" << expected << " 实际=" << got << std::endl;
            }
        }
    }
    if (inputs_written != FC_OUT_TILES || params_written != FC_OUT_TILES ||
        !input_stream.empty() || !param_stream.empty() || !output_stream.empty()) {
        ++errors;
    }
    std::cout << "FCPostProcess PyTorch 输入事务=" << inputs_written << '/'
              << FC_OUT_TILES << " 参数事务=" << params_written << '/'
              << FC_OUT_TILES << " 输出事务=1/1 检查元素数=" << comparisons
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS")
              << std::endl;
    return errors;
}

}  // namespace

static_assert(sizeof(trace_fc_systolic_int32) / sizeof(trace_fc_systolic_int32[0]) ==
              FC_OUTPUTS, "fc systolic trace shape changed");
static_assert(sizeof(trace_fc_postprocess_int32) /
              sizeof(trace_fc_postprocess_int32[0]) == FC_OUTPUTS,
              "fc postprocess trace shape changed");
static_assert(sizeof(trace_fc_requant_int8) / sizeof(trace_fc_requant_int8[0]) ==
              FC_OUTPUTS, "fc requant trace shape changed");
static_assert(sizeof(fc_bias) / sizeof(fc_bias[0]) == FC_OUTPUTS,
              "fc bias shape changed");
static_assert(sizeof(fc_multiplier) / sizeof(fc_multiplier[0]) == FC_OUTPUTS,
              "fc multiplier shape changed");
static_assert(sizeof(fc_shift) / sizeof(fc_shift[0]) == FC_OUTPUTS,
              "fc shift shape changed");

int main() {
    const std::string directory = "simulation_data/pytorch/fc_postprocess";
    if (!make_directory("simulation_data") ||
        !make_directory("simulation_data/pytorch") ||
        !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    hls::stream<fc_acc_pack_t> input_stream;
    hls::stream<fc_param_pack_t> param_stream;
    hls::stream<logits_pack_t> output_stream;
    const int errors = run_fc_postprocess(input_stream, param_stream,
                                           output_stream, directory);
    std::cout << "FCPostProcess PyTorch CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
