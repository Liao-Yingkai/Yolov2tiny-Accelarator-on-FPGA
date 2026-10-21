#include "../src/postprocess/logits_argmax.h"

#include <stdint.h>

#include <iostream>

namespace {

int scalar_reference(const logits_pack_t& logits) {
    int best_index = 0;
    int best_value = logits.value[0].to_int();
    for (int i = 1; i < FC_OUTPUTS; ++i) {
        const int value = logits.value[i].to_int();
        // 只在严格更大时更新，因此相同最大值保留最早（最小）的索引。
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    return best_index;
}

int run_case(const char* name, const logits_pack_t& logits,
             int expected_winner) {
    hls::stream<logits_pack_t> input_stream;
    hls::stream<class_index_t> output_stream;
    input_stream.write(logits);

    logits_argmax(input_stream, output_stream);

    if (output_stream.empty()) {
        std::cerr << "logits_argmax " << name
                  << " 未产生类别输出，FAIL" << std::endl;
        return 1;
    }
    const int actual = output_stream.read().to_uint();
    const int scalar = scalar_reference(logits);
    const bool pass = (actual == scalar) && (actual == expected_winner) &&
                      input_stream.empty() && output_stream.empty();
    std::cout << "logits_argmax " << name << " reference=" << scalar
              << " expected=" << expected_winner << " actual=" << actual
              << ' ' << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? 0 : 1;
}

}  // namespace

int main() {
    int errors = 0;
    logits_pack_t logits;

    // 所有值均为负数，最大值仍应被正确选出。
    const int all_negative[FC_OUTPUTS] =
        {-91, -18, -127, -63, -32, -99, -44, -120, -57, -73};
    for (int i = 0; i < FC_OUTPUTS; ++i) {
        logits.value[i] = (logit_t)all_negative[i];
    }
    errors += run_case("all_negative", logits, 1);

    // 覆盖 signed INT8 的两个端点以及正负比较。
    const int signed_extremes[FC_OUTPUTS] =
        {-128, 127, -1, 0, 126, -127, 12, -64, 100, -128};
    for (int i = 0; i < FC_OUTPUTS; ++i) {
        logits.value[i] = (logit_t)signed_extremes[i];
    }
    errors += run_case("int8_extremes", logits, 1);

    // 逐项让每个类别分别成为唯一最大值，数据模式确定且值均在 INT8 范围内。
    for (int winner = 0; winner < FC_OUTPUTS; ++winner) {
        for (int i = 0; i < FC_OUTPUTS; ++i) {
            logits.value[i] = (logit_t)(-120 + i);
        }
        logits.value[winner] = (logit_t)127;
        errors += run_case("each_class_wins", logits, winner);
    }

    // 非相邻类别并列最大，需选较早类别而非树上先遇到的任意候选。
    const int first_tie[FC_OUTPUTS] =
        {-30, 44, -5, 90, 0, -128, 90, 12, -64, -1};
    for (int i = 0; i < FC_OUTPUTS; ++i) {
        logits.value[i] = (logit_t)first_tie[i];
    }
    errors += run_case("tie_first", logits, 3);

    // 全部相等时类别0是 first maximum。
    for (int i = 0; i < FC_OUTPUTS; ++i) {
        logits.value[i] = (logit_t)-7;
    }
    errors += run_case("all_equal", logits, 0);

    std::cout << "logits_argmax errors=" << errors << ' '
              << (errors ? "FAIL" : "PASS") << std::endl;
    return errors ? 1 : 0;
}
