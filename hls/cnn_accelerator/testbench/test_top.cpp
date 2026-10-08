#include <stdint.h>

#include "../src/cnn_accel.h"

// 生成 header 保持每个文件内部符号同名；先在全局引入 stdint.h，再将
// 三次 include 分别包进 namespace，避免同名 trace 数组互相冲突。
namespace trace_data_0000 {
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data_0000.h"
}  // namespace trace_data_0000
namespace trace_data_0001 {
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data_0001.h"
}  // namespace trace_data_0001
namespace trace_data_0002 {
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data_0002.h"
}  // namespace trace_data_0002
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv1_bias.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv1_quant.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv1_shift.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv1_weight.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv2_bias.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv2_quant.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv2_shift.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/conv2_weight.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/fc_bias.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/fc_quant.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/fc_shift.h"
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/headers/fc_weight.h"

#include <errno.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * cnn_accelerator 顶层的 PyTorch 对照 testbench。
 *
 * 输入图像依次取 dump_trace.py 导出的三个 HWC INT8 trace。三个 header 的
 * 数组使用同名符号，并由本文件的 include wrapper 分别放入
 * trace_data_0000/0001/0002 namespace，因而可以在同一个编译单元内同时
 * include。Conv 权重 header 为 [K, OC]，
 * 按 cout_tile -> token 事务顺序打包；FC 权重已经离线变换为
 * [HWC_flatten, OC]，按 out_tile -> token 事务顺序打包。每一次顶层调用前，
 * 都重新写入一张图像以及完整的 Conv/FC 权重和参数流。DUT 完整执行两层
 * 卷积、两次池化和 FC 后，将十路 signed INT8 logits pack 与对应 trace 的
 * trace_fc_requant_int8 逐 lane 精确比较。
 *
 *
 * Top 的Pool1阶段间缓存是持久的static FIFO：Layer1由Output Router写满，
 * Layer2由Input Router直接读空。Input Router在源码中早于Output Router，
 * 因而普通C++语义也遵循两phase先写后读；HLS仍须验证该模式互斥反馈通道的
 * DATAFLOW实现、死锁和跨调用FIFO状态。
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

struct trace_sample_t {
    int sample;
    const int8_t* input_int8_hwc;
    const int8_t* logits_int8;
};

static void write_image(const trace_sample_t& trace,
                        hls::stream<pixel_t>& image_stream,
                        std::ofstream& input_csv) {
    for (int y = 0; y < CNN_CONV1_H; ++y) {
        for (int x = 0; x < CNN_CONV1_W; ++x) {
            pixel_t pixel;
            for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
                // Conv1 的逻辑输入仅有 c0，未使用的物理 lane 必须清零。
                pixel.channel[channel] = channel == 0
                    ? trace.input_int8_hwc[y * CNN_CONV1_W + x] : 0;
            }
            // int8_t 常被 typedef 为 signed char；转成 int 才能稳定输出十进制数字。
            const int value = static_cast<int>(
                trace.input_int8_hwc[y * CNN_CONV1_W + x]);
            input_csv << trace.sample << ',' << y << ',' << x << ",0,"
                      << value << '\n';
            image_stream.write(pixel);
        }
    }
}

static int conv_weight_value(bool conv2, int token, int channel, int output) {
    if (!conv2) {
        return channel == 0
            ? conv1_weight[token * CNN_CONV1_COUT + output] : 0;
    }
    return conv2_weight[(token * CNN_CONV2_CIN + channel) * CNN_CONV2_COUT +
                        output];
}

static void write_conv_weights(bool conv2,
                               hls::stream<weight_pack_t>& weight_stream) {
    const int cout_tiles = conv2 ? CNN_CONV2_COUT_TILES : CNN_CONV1_COUT_TILES;
    for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
        for (int token = 0; token < SYSTOLIC_KERNEL_TOKENS; ++token) {
            weight_pack_t weight;
            for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                 ++cout_lane) {
                const int output = cout_tile * SYSTOLIC_COUT_PAR + cout_lane;
                for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
                    weight.lane[cout_lane].channel[channel] =
                        conv_weight_value(conv2, token, channel, output);
                }
            }
            weight_stream.write(weight);
        }
    }
}

static int conv_bias_value(bool conv2, int output) {
    return conv2 ? conv2_bias[output] : conv1_bias[output];
}

static int conv_multiplier_value(bool conv2, int output) {
    return conv2 ? conv2_multiplier[output] : conv1_multiplier[output];
}

static int conv_shift_value(bool conv2, int output) {
    return conv2 ? conv2_shift[output] : conv1_shift[output];
}

static void write_conv_params(bool conv2,
                              hls::stream<postprocess_param_pack_t>& param_stream) {
    const int cout_tiles = conv2 ? CNN_CONV2_COUT_TILES : CNN_CONV1_COUT_TILES;
    for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
        postprocess_param_pack_t param;
        for (int lane = 0; lane < SYSTOLIC_COUT_PAR; ++lane) {
            const int output = cout_tile * SYSTOLIC_COUT_PAR + lane;
            param.bias[lane] = conv_bias_value(conv2, output);
            param.multiplier[lane] = conv_multiplier_value(conv2, output);
            param.shift[lane] = conv_shift_value(conv2, output);
        }
        param_stream.write(param);
    }
}

static void write_fc_weights(hls::stream<fc_weight_pack_t>& weight_stream) {
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        for (int token = 0; token < FC_INPUT_PACKS; ++token) {
            fc_weight_pack_t weight;
            for (int cout_lane = 0; cout_lane < FC_COUT_PAR; ++cout_lane) {
                const int output = out_tile * FC_COUT_PAR + cout_lane;
                for (int cin_lane = 0; cin_lane < FC_CIN_PAR; ++cin_lane) {
                    const int input = token * FC_CIN_PAR + cin_lane;
                    weight.value[cout_lane][cin_lane] =
                        fc_weight[input * FC_OUTPUTS + output];
                }
            }
            weight_stream.write(weight);
        }
    }
}

static void write_fc_params(hls::stream<fc_param_pack_t>& param_stream) {
    for (int out_tile = 0; out_tile < FC_OUT_TILES; ++out_tile) {
        fc_param_pack_t param;
        for (int lane = 0; lane < FC_COUT_PAR; ++lane) {
            const int output = out_tile * FC_COUT_PAR + lane;
            param.bias[lane] = fc_bias[output];
            param.multiplier[lane] = fc_multiplier[output];
            param.shift[lane] = fc_shift[output];
        }
        param_stream.write(param);
    }
}

}  // namespace

#define ASSERT_TRACE_SHAPES(trace_namespace)                                    \
    static_assert(sizeof(trace_namespace::trace_input_int8_hwc) /              \
                      sizeof(trace_namespace::trace_input_int8_hwc[0]) ==      \
                  CNN_CONV1_INPUT_TRANSACTIONS * CNN_CONV1_CIN,                \
                  "trace input shape changed");                               \
    static_assert(sizeof(trace_namespace::trace_fc_requant_int8) /             \
                      sizeof(trace_namespace::trace_fc_requant_int8[0]) ==     \
                  FC_OUTPUTS,                                                   \
                  "trace fc logits shape changed")

// 三个 namespace 都要通过同一组断言，避免只检查 sample 0000 而漏掉其余 header.

ASSERT_TRACE_SHAPES(trace_data_0000);
ASSERT_TRACE_SHAPES(trace_data_0001);
ASSERT_TRACE_SHAPES(trace_data_0002);
#undef ASSERT_TRACE_SHAPES

static const trace_sample_t k_trace_samples[] = {
    {0, trace_data_0000::trace_input_int8_hwc,
     trace_data_0000::trace_fc_requant_int8},
    {1, trace_data_0001::trace_input_int8_hwc,
     trace_data_0001::trace_fc_requant_int8},
    {2, trace_data_0002::trace_input_int8_hwc,
     trace_data_0002::trace_fc_requant_int8},
};
static const int k_trace_sample_count =
    static_cast<int>(sizeof(k_trace_samples) / sizeof(k_trace_samples[0]));
static_assert(k_trace_sample_count == 3,
              "Top PyTorch testbench must contain exactly three trace samples");

static_assert(sizeof(conv1_weight) / sizeof(conv1_weight[0]) ==
              SYSTOLIC_KERNEL_TOKENS * CNN_CONV1_COUT,
              "conv1 weight shape changed");
static_assert(sizeof(conv2_weight) / sizeof(conv2_weight[0]) ==
              SYSTOLIC_KERNEL_TOKENS * CNN_CONV2_CIN * CNN_CONV2_COUT,
              "conv2 weight shape changed");
static_assert(sizeof(fc_weight) / sizeof(fc_weight[0]) ==
              FC_INPUT_PACKS * FC_CIN_PAR * FC_OUTPUTS,
              "fc weight shape changed");
static_assert(sizeof(conv1_bias) / sizeof(conv1_bias[0]) == CNN_CONV1_COUT &&
              sizeof(conv1_multiplier) / sizeof(conv1_multiplier[0]) == CNN_CONV1_COUT &&
              sizeof(conv1_shift) / sizeof(conv1_shift[0]) == CNN_CONV1_COUT,
              "conv1 parameter shape changed");
static_assert(sizeof(conv2_bias) / sizeof(conv2_bias[0]) == CNN_CONV2_COUT &&
              sizeof(conv2_multiplier) / sizeof(conv2_multiplier[0]) == CNN_CONV2_COUT &&
              sizeof(conv2_shift) / sizeof(conv2_shift[0]) == CNN_CONV2_COUT,
              "conv2 parameter shape changed");
static_assert(sizeof(fc_bias) / sizeof(fc_bias[0]) == FC_OUTPUTS &&
              sizeof(fc_multiplier) / sizeof(fc_multiplier[0]) == FC_OUTPUTS &&
              sizeof(fc_shift) / sizeof(fc_shift[0]) == FC_OUTPUTS,
              "fc parameter shape changed");

static int compare_logits(const trace_sample_t& trace,
                          const logits_pack_t& actual,
                          std::ofstream& output_csv) {
    int errors = 0;
    for (int output = 0; output < FC_OUTPUTS; ++output) {
        // trace 中的 INT8 先提升为 int，保证负数按十进制写入 CSV。
        const int expected = static_cast<int>(trace.logits_int8[output]);
        const int got = actual.value[output].to_int();
        const bool match = got == expected;
        output_csv << trace.sample << ',' << output << ',' << expected << ','
                   << got << ',' << match << '\n';
        if (!match) {
            ++errors;
            std::cerr << "Top PyTorch错误: sample=" << trace.sample
                      << " output=" << output << " 期望=" << expected
                      << " 实际=" << got << std::endl;
        }
    }
    return errors;
}

static int check_stream_conservation(
    int sample,
    hls::stream<pixel_t>& image_stream,
    hls::stream<weight_pack_t>& conv1_weight_stream,
    hls::stream<postprocess_param_pack_t>& conv1_param_stream,
    hls::stream<weight_pack_t>& conv2_weight_stream,
    hls::stream<postprocess_param_pack_t>& conv2_param_stream,
    hls::stream<fc_weight_pack_t>& fc_weight_stream,
    hls::stream<fc_param_pack_t>& fc_param_stream,
    hls::stream<logits_pack_t>& logits_stream) {
    int errors = 0;
    if (!image_stream.empty()) {
        ++errors;
        std::cerr << "Top PyTorch错误: sample=" << sample
                  << " image_stream存在未消费事务" << std::endl;
    }
    if (!conv1_weight_stream.empty()) {
        ++errors;
        std::cerr << "Top PyTorch错误: sample=" << sample
                  << " conv1_weight_stream存在未消费事务" << std::endl;
    }
    if (!conv1_param_stream.empty()) {
        ++errors;
        std::cerr << "Top PyTorch错误: sample=" << sample
                  << " conv1_param_stream存在未消费事务" << std::endl;
    }
    if (!conv2_weight_stream.empty()) {
        ++errors;
        std::cerr << "Top PyTorch错误: sample=" << sample
                  << " conv2_weight_stream存在未消费事务" << std::endl;
    }
    if (!conv2_param_stream.empty()) {
        ++errors;
        std::cerr << "Top PyTorch错误: sample=" << sample
                  << " conv2_param_stream存在未消费事务" << std::endl;
    }
    if (!fc_weight_stream.empty()) {
        ++errors;
        std::cerr << "Top PyTorch错误: sample=" << sample
                  << " fc_weight_stream存在未消费事务" << std::endl;
    }
    if (!fc_param_stream.empty()) {
        ++errors;
        std::cerr << "Top PyTorch错误: sample=" << sample
                  << " fc_param_stream存在未消费事务" << std::endl;
    }
    if (!logits_stream.empty()) {
        ++errors;
        std::cerr << "Top PyTorch错误: sample=" << sample
                  << " logits_stream存在未消费事务" << std::endl;
    }
    return errors;
}

int main() {
    const std::string directory = "simulation_data/pytorch/top";
    if (!make_directory("simulation_data") ||
        !make_directory("simulation_data/pytorch") ||
        !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    std::ofstream input_csv, output_csv;
    if (!open_csv(input_csv, directory + "/input.csv",
                  "sample,y,x,channel,value") ||
        !open_csv(output_csv, directory + "/output.csv",
                  "sample,output,expected,actual,match")) {
        return 1;
    }

    hls::stream<pixel_t> image_stream;
    hls::stream<weight_pack_t> conv1_weight_stream;
    hls::stream<postprocess_param_pack_t> conv1_param_stream;
    hls::stream<weight_pack_t> conv2_weight_stream;
    hls::stream<postprocess_param_pack_t> conv2_param_stream;
    hls::stream<fc_weight_pack_t> fc_weight_stream;
    hls::stream<fc_param_pack_t> fc_param_stream;
    hls::stream<logits_pack_t> logits_stream;

    int errors = 0;
    for (int sample_index = 0; sample_index < k_trace_sample_count;
         ++sample_index) {
        const trace_sample_t& trace = k_trace_samples[sample_index];

        // 每张图都重新写入完整输入和全部参数；stream 对象本身保持复用，
        // 让连续三次顶层调用直接暴露跨调用 FIFO 或残留事务问题。
        write_image(trace, image_stream, input_csv);
        write_conv_weights(false, conv1_weight_stream);
        write_conv_params(false, conv1_param_stream);
        write_conv_weights(true, conv2_weight_stream);
        write_conv_params(true, conv2_param_stream);
        write_fc_weights(fc_weight_stream);
        write_fc_params(fc_param_stream);

        cnn_accelerator(image_stream, conv1_weight_stream, conv1_param_stream,
                        conv2_weight_stream, conv2_param_stream,
                        fc_weight_stream, fc_param_stream, logits_stream);

        int sample_errors = 0;
        if (logits_stream.empty()) {
            ++sample_errors;
            std::cerr << "Top PyTorch错误: sample=" << trace.sample
                      << " 未产生 logits 事务" << std::endl;
        } else {
            const logits_pack_t actual = logits_stream.read();
            sample_errors += compare_logits(trace, actual, output_csv);
        }
        sample_errors += check_stream_conservation(
            trace.sample, image_stream, conv1_weight_stream, conv1_param_stream,
            conv2_weight_stream, conv2_param_stream, fc_weight_stream,
            fc_param_stream, logits_stream);
        errors += sample_errors;
        std::cout << "Top PyTorch sample=" << trace.sample
                  << " logits数量=" << FC_OUTPUTS
                  << " 错误数=" << sample_errors << ' '
                  << (sample_errors ? "FAIL" : "PASS") << std::endl;
    }

    std::cout << "Top PyTorch 连续样本数=" << k_trace_sample_count
              << " 输入事务/图=" << CNN_CONV1_INPUT_TRANSACTIONS
              << " Conv权重事务=" << CNN_CONV1_WEIGHT_TRANSACTIONS << '+'
              << CNN_CONV2_WEIGHT_TRANSACTIONS << " Conv参数事务="
              << CNN_CONV1_PARAM_TRANSACTIONS << '+' << CNN_CONV2_PARAM_TRANSACTIONS
              << " FC权重事务=" << FC_OUT_TILES * FC_INPUT_PACKS
              << " FC参数事务=" << FC_OUT_TILES << " logits检查="
              << k_trace_sample_count * FC_OUTPUTS << " 错误数=" << errors << ' '
              << (errors ? "FAIL" : "PASS") << std::endl;
    std::cout << "Top PyTorch CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
