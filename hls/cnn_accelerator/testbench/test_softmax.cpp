#include "../src/postprocess/softmax.h"
#include "model_softmax_params.h"
#include <cmath>
#include <iostream>

/*
 * 只准备测试源码，本轮未执行，不代表功能或精度已验证。
 * 协议为一次调用消费一个10类signed INT8 logits包，产出一个完整结果包；
 * 同包含10个0..10000的uint16概率和原logits的first-max类别（严格>才更新）。
 * scale为正Q20。独立double/std::exp稳定参考只在TB内使用，不复用DUT近似；
 * 10单位允差覆盖定点指数、截断、scale与输出舍入等近似，不是实测误差承诺。
 * 每例检查1入1出、调用后stream清空、概率范围及sum=10000±5；失败返回非零。
 */
// 浮点参考允差，仅供TB使用，不是已测得精度。
constexpr int PROBABILITY_ERROR_TOLERANCE = 10;
static int check(const char* name, const int (&q)[FC_OUTPUTS], unsigned scale, bool equal = false) {
    hls::stream<logits_pack_t> input;
    hls::stream<inference_result_t> output;
    logits_pack_t pack;
    int first_max = 0;
    for (int i = 0; i < FC_OUTPUTS; ++i) {
        pack.value[i] = q[i];
        if (q[i] > q[first_max]) first_max = i;
    }
    input.write(pack);
    softmax(input, output, scale);
    if (!input.empty() || output.empty()) {
        std::cerr << "case=" << name << " scale=" << scale << " 1入1出协议失败\n";
        return 1;
    }
    const inference_result_t result = output.read();
    int errors = 0;
    if (!output.empty()) {
        std::cerr << "case=" << name << " scale=" << scale << " 多余输出事务\n";
        ++errors;
    }
    if (result.class_index.to_uint() != unsigned(first_max)) {
        std::cerr << "case=" << name << " scale=" << scale << " class expected="
                  << first_max << " actual=" << result.class_index.to_uint() << '\n';
        ++errors;
    }
    double exp_values[FC_OUTPUTS], denominator = 0;
    for (int i = 0; i < FC_OUTPUTS; ++i) {
        exp_values[i] = std::exp((q[i] - q[first_max]) * (double(scale) / (1U << SOFTMAX_SCALE_FRAC_BITS)));
        denominator += exp_values[i];
    }
    int sum = 0;
    for (int i = 0; i < FC_OUTPUTS; ++i) {
        const unsigned actual = result.probability[i].to_uint();
        const double reference = SOFTMAX_PROBABILITY_SCALE * exp_values[i] / denominator;
        sum += actual;
        if (actual > SOFTMAX_PROBABILITY_SCALE ||
            std::fabs(actual - reference) > PROBABILITY_ERROR_TOLERANCE ||
            (equal && actual != 1000)) {
            if (errors < 4)
                std::cerr << "case=" << name << " scale=" << scale << " channel=" << i
                          << " expected=" << reference << " actual=" << actual << '\n';
            ++errors;
        }
    }
    if (std::abs(sum - SOFTMAX_PROBABILITY_SCALE) > 5) {
        if (errors < 4)
            std::cerr << "case=" << name << " scale=" << scale << " sum expected="
                      << SOFTMAX_PROBABILITY_SCALE << "+/-5 actual=" << sum << '\n';
        ++errors;
    }
    return errors;
}

int main() {
    static_assert(MODEL_SOFTMAX_SCALE_FRAC_BITS == SOFTMAX_SCALE_FRAC_BITS, "model scale format");
    const int same[FC_OUTPUTS] = {7,7,7,7,7,7,7,7,7,7};
    const int ties[FC_OUTPUTS] = {-128,127,127,0,-1,-12,42,126,-127,7};
    const int extremes[FC_OUTPUTS] = {127,-128,-128,-128,-128,-128,-128,-128,-128,-128};
    const int mixed[FC_OUTPUTS] = {-8,-3,0,2,9,8,7,-128,127,126};
    const unsigned scales[] = {MODEL_SOFTMAX_SCALE_Q20, 1U, 1048576U, 0xffffffffU};
    int errors = 0;
    const int all_negative_tail_max[FC_OUTPUTS] = {-110,-90,-72,-60,-45,-31,-25,-17,-7,-3};
    errors += check("all_negative_tail_max", all_negative_tail_max, MODEL_SOFTMAX_SCALE_Q20);
    // 连续调用相同函数、不同输入及scale，检查概率与类别同包且不残留。
    for (unsigned i = 0; i < sizeof(scales) / sizeof(scales[0]); ++i) {
        errors += check("all_equal", same, scales[i], true);
        errors += check("first_max_tie", ties, scales[i]);
        errors += check("delta_0_255", extremes, scales[i]);
        errors += check("mixed_signed", mixed, scales[i]);
    }
    // 每个叶子依次成为唯一最大，验证比较树所有分支与末尾两项。
    for (int winner = 0; winner < FC_OUTPUTS; ++winner) {
        int q[FC_OUTPUTS];
        for (int i = 0; i < FC_OUTPUTS; ++i) q[i] = -128;
        q[winner] = 127;
        errors += check("unique_max", q, MODEL_SOFTMAX_SCALE_Q20);
    }
    const int cross_ties[FC_OUTPUTS] = {-128,-128,-128,127,-128,-128,-128,-128,127,127};
    errors += check("cross_tree_ties", cross_ties, MODEL_SOFTMAX_SCALE_Q20);
    // delta=1：P位于16*2^20前一格、恰好阈值、后一格，覆盖clipping分支端点。
    const int clip[FC_OUTPUTS] = {0,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    errors += check("clip_below", clip, (16U << SOFTMAX_SCALE_FRAC_BITS) - 1U);
    errors += check("clip_exact", clip, 16U << SOFTMAX_SCALE_FRAC_BITS);
    errors += check("clip_above", clip, (16U << SOFTMAX_SCALE_FRAC_BITS) + 1U);
    std::cout << "Softmax检查错误数 " << errors << '\n';
    return errors == 0 ? 0 : 1;
}
