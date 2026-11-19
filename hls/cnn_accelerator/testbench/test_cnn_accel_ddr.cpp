#include "../src/top/cnn_accel_ddr.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// 单图端到端验证：PS 等价操作只负责复制 BIN，不在 TB 中补 RGB 尾槽。
// 模型区按层拼接，Feature A/B 在同一片连续模拟 DDR 中。仅打印结果，不导出文件。
namespace {

using word_t = ap_uint<64>;
constexpr int BYTES_PER_WORD = 8;
constexpr int BYTES_PER_PARAM = 4;
constexpr int MAX_PRINTED_ERRORS = 10;
const char* const layer_names[] = {"conv1", "conv2", "conv3", "conv4", "conv5",
                                   "conv6", "conv7", "conv8", "prediction"};
static_assert(sizeof(layer_names) / sizeof(layer_names[0]) == YOLO_TINY_LAYER_COUNT,
              "模型文件必须与固定层表匹配");

bool read_bin(const std::string& path, std::size_t length, std::vector<std::uint8_t>& bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(length)) {
        std::cerr << "BIN 不存在或长度错误: " << path << " 期望字节数=" << length << '\n';
        return false;
    }
    file.seekg(0);
    bytes.resize(length);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(length))) {
        std::cerr << "BIN 读取失败: " << path << '\n';
        return false;
    }
    return true;
}

// 仅恢复 little-endian BIN 位序，不重排通道或权重；也适用于 signed INT32 参数。
template <int WordBytes, typename Word>
void copy_words(const std::vector<std::uint8_t>& bytes, Word* output) {
    for (std::size_t i = 0; i < bytes.size() / WordBytes; ++i) {
        Word word = 0;
        for (int byte = 0; byte < WordBytes; ++byte)
            word.range(8 * byte + 7, 8 * byte) = bytes[i * WordBytes + byte];
        output[i] = word;
    }
}

bool load_model(const std::string& root, std::vector<word_t>& weights,
                std::vector<ap_int<32>>& params) {
    std::size_t weight_offset = 0, param_offset = 0;
    std::vector<std::uint8_t> bytes;
    for (int layer = 0; layer < YOLO_TINY_LAYER_COUNT; ++layer) {
        const ConvGeometry& g = YOLO_TINY_GEOMETRIES[layer];
        const std::string prefix = root + "/bins/" + layer_names[layer];
        const int weight_words = g.weight_transactions * CONV_COUT_PAR;
        if (!read_bin(prefix + "_weight.bin", weight_words * BYTES_PER_WORD, bytes)) return false;
        copy_words<BYTES_PER_WORD>(bytes, weights.data() + weight_offset);
        weight_offset += weight_words;
        if (!read_bin(prefix + "_bias.bin", g.cout * BYTES_PER_PARAM, bytes)) return false;
        copy_words<BYTES_PER_PARAM>(bytes, params.data() + param_offset);
        param_offset += g.cout;
        // quant 文件本来就是 [全部 M][全部 S]，紧随该层 bias 段。
        if (!read_bin(prefix + "_quant.bin", 2 * g.cout * BYTES_PER_PARAM, bytes)) return false;
        copy_words<BYTES_PER_PARAM>(bytes, params.data() + param_offset);
        param_offset += 2 * g.cout;
    }
    return weight_offset == weights.size() && param_offset == params.size();
}

}  // namespace

int main() {
    const char* configured_path = std::getenv("YOLO_DATA_DIR");
    if (!configured_path || !*configured_path) {
        std::cerr << "请设置环境变量 YOLO_DATA_DIR，指向 YOLO generated 根目录。\n";
        return 1;
    }
    const std::string root = configured_path;
    const auto& prediction = YOLO_TINY_GEOMETRIES[YOLO_TINY_LAYER_COUNT - 1];
    const int physical_channels = prediction.output_channel_groups * FEATURE_PACK_CHANNELS;
    std::vector<std::uint8_t> rgb, golden;
    if (!read_bin(root + "/trace/trace_input_int8_hwc_0000.bin",
                  ddr_layout::IMAGE_WORDS * BYTES_PER_WORD, rgb) ||
        !read_bin(root + "/trace/trace_prediction_postprocess_int8_hwc_0000.bin",
                  prediction.output_h * prediction.output_w * prediction.cout, golden)) return 1;

    std::vector<word_t> weights(ddr_layout::WEIGHT_WORDS);
    std::vector<ap_int<32>> params(ddr_layout::PARAM_WORDS);
    if (!load_model(root, weights, params)) return 1;

    const word_t sentinel = 0xA5A5A5A5A5A5A5A5ULL;
    std::vector<word_t> feature(ddr_layout::FEATURE_WORDS + 2, sentinel);
    word_t* const arena = feature.data() + 1;
    copy_words<BYTES_PER_WORD>(rgb, arena);  // A 区只复制紧凑 RGB 的原始字节。
    std::cout << "YOLOv2-tiny: 开始九层推理" << std::endl;
    cnn_accelerator_ddr(arena, weights.data(), params.data());

    // 最终 B 区采用物理 128 通道步长，golden 每位置仅有 125 通道。
    // 按坐标索引二者，额外核对末三个槽位为零，不能用紧凑步长读 DDR。
    int errors = 0;
    const word_t* const output = arena + ddr_layout::RESULT_WORD_OFFSET;
    for (int y = 0; y < prediction.output_h; ++y)
        for (int x = 0; x < prediction.output_w; ++x)
            for (int c = 0; c < physical_channels; ++c) {
                const int pixel = y * prediction.output_w + x;
                const int bit = (c % BYTES_PER_WORD) * 8;
                const ap_int<8> actual = output[pixel * prediction.output_channel_groups +
                                               c / BYTES_PER_WORD].range(bit + 7, bit);
                const ap_int<8> expected = c < prediction.cout
                    ? ap_int<8>(golden[pixel * prediction.cout + c]) : ap_int<8>(0);
                if (actual != expected && ++errors <= MAX_PRINTED_ERRORS)
                    std::cerr << "结果不匹配 y=" << y << " x=" << x << " c=" << c
                              << " 期望=" << expected.to_int() << " 实际=" << actual.to_int() << '\n';
            }
    if (feature.front() != sentinel || feature.back() != sentinel) {
        std::cerr << "Feature DDR 区外写入\n";
        ++errors;
    }
    std::cout << "Prediction 有效元素=" << golden.size()
              << " 物理尾槽=" << ddr_layout::RESULT_WORDS * BYTES_PER_WORD - golden.size()
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS") << '\n';
    return errors ? 1 : 0;
}

