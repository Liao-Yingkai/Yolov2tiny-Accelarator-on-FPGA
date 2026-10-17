#include <stdint.h>
#include "../src/top/cnn_accel_ddr.h"
#include <errno.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>

/*
 * DDR wrapper 独立功能对照：image/weights/result 为64-bit word，params为
 * signed32-bit word。低 byte 对应较早 INT8；模型布局为 Conv1 -> Conv2 -> FC，
 * 每层参数为完整 bias -> multiplier -> shift 数组，按输出通道递增。
 * 四区有效 word 数为98/1133/102/2；结果16字节含10 logits、2保留字节、uint32类别。
 * 三图固定地址连续调用，模型仅打包一次。
 * reference 来自三个 trace 的10个 signed logits；类别由软件 scalar first-max
 * 独立计算。输出 CSV 保存 sample/output/expected/actual/match，输入 CSV 保存
 * sample/pixel/value，可逐字节核对 PS 的紧密 raster 打包。守卫与快照检查越界
 * 写和只读区变化，但普通 C Simulation 不验证有限 FIFO、AXI 或缓存一致性。
 */
namespace trace_data_0000 {
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data_0000.h"
}
namespace trace_data_0001 {
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data_0001.h"
}
namespace trace_data_0002 {
#include "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/trace/trace_data_0002.h"
}
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

namespace {
static_assert(CONV1_WEIGHT_LAYOUT_VERSION == 2 && CONV2_WEIGHT_LAYOUT_VERSION == 2 &&
              FC_WEIGHT_LAYOUT_VERSION == 2, "requires blocked weight headers v2");
// 守卫紧邻有效区；输入和模型快照逐 word 比较，错误不依赖 assert/NDEBUG。
template <typename T, int N> struct guarded_buffer {
    T storage[N + 2];
    T* data() { return storage + 1; }
    void initialize(T guard) {
        storage[0] = guard; storage[N + 1] = guard;
        for (int i = 0; i < N; ++i) data()[i] = 0;
    }
    bool guards_ok(T guard) const { return storage[0] == guard && storage[N + 1] == guard; }
};
template <typename T, int N>
bool unchanged(guarded_buffer<T, N>& buffer, const T (&copy)[N]) {
    for (int i = 0; i < N; ++i) if (buffer.data()[i] != copy[i]) return false;
    return true;
}
void pack_conv_params(bool second, ap_int<32>* params) {
    const int outputs = second ? CNN_CONV2_COUT : CNN_CONV1_COUT;
    const int param_offset = second ? ddr_layout::CONV2_PARAM_OFFSET : ddr_layout::CONV1_PARAM_OFFSET;
    // 软件直接拷贝三个完整 header 数组，不复用 DUT 的 pack/field/lane 地址公式。
    for (int output = 0; output < outputs; ++output) {
        params[param_offset + output] = second ? conv2_bias[output] : conv1_bias[output];
        params[param_offset + outputs + output] = second ? conv2_multiplier[output] : conv1_multiplier[output];
        params[param_offset + 2 * outputs + output] = second ? conv2_shift[output] : conv1_shift[output];
    }
}
void pack_model(ap_uint<64>* weights, ap_int<32>* params) {
    pack_conv_params(false, params); pack_conv_params(true, params);
    // v2 头已经是硬件消费顺序。仅逐字节连续拼接，不补通道、不重排。
    const int8_t* sources[] = {conv1_weight, conv2_weight, fc_weight};
    const int sizes[] = {sizeof(conv1_weight), sizeof(conv2_weight), sizeof(fc_weight)};
    int byte_offset = 0;
    for (int layer = 0; layer < 3; ++layer)
        for (int byte = 0; byte < sizes[layer]; ++byte, ++byte_offset)
            weights[byte_offset / 8].range((byte_offset % 8) * 8 + 7, (byte_offset % 8) * 8) =
                (ap_uint<8>)sources[layer][byte];
    for (int output = 0; output < FC_OUTPUTS; ++output) {
        params[ddr_layout::FC_PARAM_OFFSET + output] = fc_bias[output];
        params[ddr_layout::FC_PARAM_OFFSET + FC_OUTPUTS + output] = fc_multiplier[output];
        params[ddr_layout::FC_PARAM_OFFSET + 2 * FC_OUTPUTS + output] = fc_shift[output];
    }
}
#define TRACE_SHAPE(ns) \
static_assert(sizeof(ns::trace_input_int8_hwc) / sizeof(ns::trace_input_int8_hwc[0]) == CNN_CONV1_INPUT_TRANSACTIONS, "trace image shape"); \
static_assert(sizeof(ns::trace_fc_requant_int8) / sizeof(ns::trace_fc_requant_int8[0]) == FC_OUTPUTS, "trace logits shape")
TRACE_SHAPE(trace_data_0000);
TRACE_SHAPE(trace_data_0001);
TRACE_SHAPE(trace_data_0002);
#undef TRACE_SHAPE
#define MODEL_SHAPE(array, count) static_assert(sizeof(array) / sizeof(array[0]) == count, "model shape")
MODEL_SHAPE(conv1_weight, SYSTOLIC_KERNEL_TOKENS * CNN_CONV1_CIN * CNN_CONV1_COUT);
MODEL_SHAPE(conv2_weight, SYSTOLIC_KERNEL_TOKENS * CNN_CONV2_CIN * CNN_CONV2_COUT);
MODEL_SHAPE(fc_weight, FC_INPUT_DIM * FC_OUTPUTS);
MODEL_SHAPE(conv1_bias, CNN_CONV1_COUT);
MODEL_SHAPE(conv1_multiplier, CNN_CONV1_COUT);
MODEL_SHAPE(conv1_shift, CNN_CONV1_COUT);
MODEL_SHAPE(conv2_bias, CNN_CONV2_COUT);
MODEL_SHAPE(conv2_multiplier, CNN_CONV2_COUT);
MODEL_SHAPE(conv2_shift, CNN_CONV2_COUT);
MODEL_SHAPE(fc_bias, FC_OUTPUTS);
MODEL_SHAPE(fc_multiplier, FC_OUTPUTS);
MODEL_SHAPE(fc_shift, FC_OUTPUTS);
#undef MODEL_SHAPE
}  // namespace

int main() {
    const int8_t* inputs[] = {trace_data_0000::trace_input_int8_hwc, trace_data_0001::trace_input_int8_hwc, trace_data_0002::trace_input_int8_hwc};
    const int8_t* expected[] = {trace_data_0000::trace_fc_requant_int8, trace_data_0001::trace_fc_requant_int8, trace_data_0002::trace_fc_requant_int8};
    const ap_uint<64> guard64 = 0xA55AC33C96F00F69ULL;
    const ap_int<32> guard32 = 0x5A69C33C;
    guarded_buffer<ap_uint<64>, ddr_layout::IMAGE_WORDS> image;
    guarded_buffer<ap_uint<64>, ddr_layout::WEIGHT_WORDS> weights;
    guarded_buffer<ap_int<32>, ddr_layout::PARAM_WORDS> params;
    guarded_buffer<ap_uint<64>, ddr_layout::RESULT_WORDS> result;
    image.initialize(guard64); weights.initialize(guard64);
    params.initialize(guard32); result.initialize(guard64);
    // 模型仅准备一次，三次调用保留同一组基地址，只覆盖图像数据。
    pack_model(weights.data(), params.data());
    ap_uint<64> weight_copy[ddr_layout::WEIGHT_WORDS];
    ap_int<32> param_copy[ddr_layout::PARAM_WORDS];
    for (int i = 0; i < ddr_layout::WEIGHT_WORDS; ++i) weight_copy[i] = weights.data()[i];
    for (int i = 0; i < ddr_layout::PARAM_WORDS; ++i) param_copy[i] = params.data()[i];
    const char* dirs[] = {"simulation_data", "simulation_data/pytorch", "simulation_data/pytorch/ddr_wrapper"};
    for (int i = 0; i < 3; ++i) {
        if (mkdir(dirs[i], 0777) != 0 && errno != EEXIST) { std::cerr << "无法创建 CSV 目录\n"; return 1; }
    }
    std::ofstream csv("simulation_data/pytorch/ddr_wrapper/output.csv");
    std::ofstream input_csv("simulation_data/pytorch/ddr_wrapper/input.csv");
    if (!csv || !input_csv) { std::cerr << "无法打开 CSV\n"; return 1; }
    csv << "sample,output,expected,actual,match\n";
    input_csv << "sample,pixel,value\n";
    int errors = 0;
    for (int sample = 0; sample < 3; ++sample) {
        const int errors_before = errors;
        ap_uint<64> image_copy[ddr_layout::IMAGE_WORDS];
        for (int word = 0; word < ddr_layout::IMAGE_WORDS; ++word) {
            ap_uint<64> value = 0;
            for (int byte = 0; byte < ddr_layout::WORD_BYTES; ++byte) {
                value.range(byte * 8 + 7, byte * 8) = (ap_uint<8>)inputs[sample][word * ddr_layout::WORD_BYTES + byte];
                input_csv << sample << ',' << word * ddr_layout::WORD_BYTES + byte << ','
                          << static_cast<int>(inputs[sample][word * ddr_layout::WORD_BYTES + byte]) << '\n';
            }
            image.data()[word] = value; image_copy[word] = value;
        }
        // 非零旧值检测 writer 是否覆盖整个结果，包括所有保留高位。
        for (int i = 0; i < ddr_layout::RESULT_WORDS; ++i) result.data()[i] = guard64;
        cnn_accelerator_ddr(image.data(), weights.data(), params.data(), result.data());
        int first_max = 0;
        for (int i = 0; i < FC_OUTPUTS; ++i) {
            const int wanted = static_cast<int>(expected[sample][i]);
            const int actual = ((ap_int<8>)result.data()[i / ddr_layout::WORD_BYTES].range((i % ddr_layout::WORD_BYTES) * 8 + 7, (i % ddr_layout::WORD_BYTES) * 8)).to_int();
            const bool match = wanted == actual;
            csv << sample << ',' << i << ',' << wanted << ',' << actual << ',' << match << '\n';
            if (!match) {
                if (errors - errors_before < 4)
                    std::cerr << "sample " << sample << " logit " << i << "：期望 " << wanted << "，实际 " << actual << '\n';
                ++errors;
            }
            // 软件 scalar first-maximum，只在严格更大时更新。
            if (expected[sample][i] > expected[sample][first_max]) first_max = i;
        }
        const unsigned actual_class = result.data()[ddr_layout::CLASS_WORD_OFFSET].range(ddr_layout::CLASS_BIT_OFFSET + 31, ddr_layout::CLASS_BIT_OFFSET).to_uint();
        csv << sample << ",pred_class," << first_max << ',' << actual_class << ',' << (actual_class == (unsigned)first_max) << '\n';
        if (actual_class != (unsigned)first_max) {
            std::cerr << "sample " << sample << " word1 高32位类别：期望 " << first_max << "，实际 " << actual_class << '\n';
            ++errors;
        }
        if (result.data()[1].range(31, 16) != 0) {
            std::cerr << "sample " << sample << " result word1 保留位非零\n"; ++errors;
        }
        if (!image.guards_ok(guard64) || !weights.guards_ok(guard64) || !params.guards_ok(guard32) || !result.guards_ok(guard64)) {
            std::cerr << "sample " << sample << " buffer 守卫被修改\n"; ++errors;
        }
        if (!unchanged(image, image_copy) || !unchanged(weights, weight_copy) || !unchanged(params, param_copy)) {
            std::cerr << "sample " << sample << " 只读 DDR 区发生变化\n"; ++errors;
        }
        std::cout << "sample " << sample << (errors == errors_before ? " PASS" : " FAIL") << '\n';
    }
    csv.flush(); input_csv.flush();
    if (!csv.good() || !input_csv.good()) ++errors;
    std::cout << "DDR wrapper：三图、30 logits 与类别检查，错误数 " << errors << '\n';
    return errors == 0 ? 0 : 1;
}

