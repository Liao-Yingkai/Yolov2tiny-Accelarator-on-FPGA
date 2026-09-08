#include "../src/im2row/im2row.h"

#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * Im2Row 完整层定向 testbench。
 * - 输入为行优先 NHWC pixel_t；输出严格为 row -> tile_x -> token(kh*3+kw)。
 *   Conv1 输入784、输出1764个activation pack；Conv2 输入196、输出504个
 *   pack，末 tile 的 spatial_valid 是0011。两层在同一对stream上连续调用。
 * - reference 独立对每个 row/tile/token/lane/channel 反查源图及padding；Conv1
 *   的物理c1..c7填入非零 poison，并验证其在输出中均被清零。
 * - input.csv 保留NHWC输入；output.csv逐元素保存逻辑坐标、padding、
 *   expected/actual value和expected/actual valid mask，方便波形对照。
 */
namespace {

static bool make_directory(const char* path) {
    return mkdir(path, 0777) == 0 || errno == EEXIST;
}

static int sample_value(int row, int col, int channel) {
    return ((row * 13 + col * 7 + channel * 5) % 31) - 15;
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

static int run_layer(hls::stream<pixel_t>& input_stream,
                     hls::stream<activation_pack_t>& output_stream,
                     int H, int W, int Cin, const char* name,
                     const std::string& directory) {
    const int tiles_per_row = (W + IM2ROW_P - 1) / IM2ROW_P;
    const int expected_inputs = H * W;
    const int expected_packs = H * tiles_per_row * SYSTOLIC_KERNEL_TOKENS;
    std::ofstream input_csv;
    std::ofstream output_csv;
    const std::string prefix = directory + "/" + name;
    if (!open_csv(input_csv, prefix + "_input.csv", "row,col,channel,value") ||
        !open_csv(output_csv, prefix + "_output.csv",
                  "transaction,row,tile_x,token,lane,channel,"
                  "expected_valid_mask,actual_valid_mask,valid_match,"
                  "padding,expected,actual,match")) {
        return 1;
    }

    // 输入始终是 NHWC pixel_t；Conv1 非逻辑通道的 poison 检查规范化路径。
    int inputs_written = 0;
    for (int row = 0; row < H; ++row) {
        for (int col = 0; col < W; ++col) {
            pixel_t pixel;
            for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
                const int value = (Cin == 1 && channel != 0) ?
                    (91 + channel) : sample_value(row, col, channel);
                pixel.channel[channel] = value;
                input_csv << row << ',' << col << ',' << channel << ','
                          << value << '\n';
            }
            input_stream.write(pixel);
            ++inputs_written;
        }
    }

    im2row(input_stream, output_stream, H, W, Cin);
    int errors = 0;
    int comparisons = 0;
    int transaction = 0;
    for (int row = 0; row < H; ++row) {
        for (int tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
            const int valid_mask = (Cin == CNN_CONV2_CIN && tile_x == 3) ?
                0x3 : 0xf;
            for (int token = 0; token < SYSTOLIC_KERNEL_TOKENS; ++token) {
                const activation_pack_t actual = output_stream.read();
                const int kh = token / 3;
                const int kw = token % 3;
                const int actual_valid = actual.spatial_valid.to_uint();
                const bool valid_match = actual_valid == valid_mask;
                if (!valid_match) {
                    ++errors;
                    if (errors <= 10) {
                        std::cerr << name << " valid mismatch row=" << row
                                  << " tile_x=" << tile_x << " token=" << token
                                  << " expected_mask=" << valid_mask
                                  << " actual_mask=" << actual_valid << std::endl;
                    }
                }
                for (int lane = 0; lane < IM2ROW_P; ++lane) {
                    for (int channel = 0; channel < IM2ROW_MAX_CHANNELS; ++channel) {
                        const int x = tile_x * IM2ROW_P + lane;
                        const int sy = row + kh - IM2ROW_PADDING;
                        const int sx = x + kw - IM2ROW_PADDING;
                        const bool valid = (valid_mask & (1 << lane)) != 0;
                        const bool padding = valid &&
                            (sy < 0 || sy >= H || sx < 0 || sx >= W);
                        int expected = 0;
                        if (valid && !padding && (Cin == 8 || channel == 0))
                            expected = sample_value(sy, sx, channel);
                        const int got = actual.pixel[lane].channel[channel].to_int();
                        const bool match = valid_match && got == expected;
                        output_csv << transaction << ',' << row << ',' << tile_x
                                   << ',' << token << ',' << lane << ',' << channel
                                   << ',' << valid_mask << ',' << actual_valid << ','
                                   << valid_match << ',' << padding << ',' << expected
                                   << ',' << got << ',' << match << '\n';
                        ++comparisons;
                        if (!match && got != expected) {
                            ++errors;
                            if (errors <= 10) std::cerr << name << " mismatch tx="
                                << transaction << " lane=" << lane << " ch=" << channel
                                << " expected=" << expected << " actual=" << got << std::endl;
                        }
                    }
                }
                ++transaction;
            }
        }
    }
    if (!input_stream.empty() || !output_stream.empty()) {
        ++errors;
        std::cerr << name << " stream 未完全消费" << std::endl;
    }
    if (inputs_written != expected_inputs || transaction != expected_packs) {
        ++errors;
    }
    std::cout << "Im2Row " << name << " 输入事务=" << inputs_written << '/'
              << expected_inputs << " 输出事务=" << transaction << '/'
              << expected_packs << " 检查元素数=" << comparisons
              << " 错误数=" << errors << ' '
              << (errors ? "FAIL" : "PASS") << std::endl;
    return errors;
}

}  // namespace

int main() {
    const std::string directory = "simulation_data/im2row";
    if (!make_directory("simulation_data") || !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    hls::stream<pixel_t> input_stream;
    hls::stream<activation_pack_t> output_stream;
    int errors = 0;
    errors += run_layer(input_stream, output_stream, 28, 28, 1, "conv1", directory);
    errors += run_layer(input_stream, output_stream, 14, 14, 8, "conv2", directory);
    std::cout << "Im2Row CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
