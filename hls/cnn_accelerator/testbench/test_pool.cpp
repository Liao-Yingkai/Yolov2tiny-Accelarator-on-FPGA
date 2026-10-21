#include "../src/pool/pool.h"

#include <cerrno>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/*
 * MaxPool 完整 Pool1/Pool2 定向 testbench。
 * - 输入事务严格为 row -> tile_x -> cout_tile。Pool1 发送392个 activation
 *   tile 并输出196个8通道 pack；Pool2 发送224个 tile，并由16通道输出98个
 *   pack。Pool2 行尾 tile 的 valid 为0011，越界 lane 故意填127。
 * - reference 独立对每个2x2窗口的四个 signed INT8 原值求最大值，再按输出
 *   pooled_y -> pooled_x -> channel_pack 顺序比较。
 * - input.csv 记录输入逻辑坐标与valid；output.csv记录每个输出通道的
 *   transaction、逻辑坐标、lane/channel、expected/actual/match。
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

int source_value(int y, int x, int channel) {
    return ((y * 19 + x * 11 + channel * 7) % 201) - 100;
}

int source_index(int y, int x, int channel, int width, int cout) {
    return (y * width + x) * cout + channel;
}

int run_layer(layer_mode_t layer_mode, const char* name,
              const std::string& directory) {
    const bool pool2 = (layer_mode == LAYER2_MODE);
    const int H = pool2 ? CNN_CONV2_H : CNN_CONV1_H;
    const int W = pool2 ? CNN_CONV2_W : CNN_CONV1_W;
    const int Cout = pool2 ? CNN_CONV2_COUT : CNN_CONV1_COUT;
    const int tiles_per_row = pool2 ? CNN_CONV2_TILES_PER_ROW : CNN_CONV1_TILES_PER_ROW;
    const int cout_tiles = pool2 ? CNN_CONV2_COUT_TILES : CNN_CONV1_COUT_TILES;
    const int pooled_h = pool2 ? CNN_POOL2_H : CNN_POOL1_H;
    const int pooled_w = pool2 ? CNN_POOL2_W : CNN_POOL1_W;
    const int channel_packs = pool2 ? CNN_POOL2_CHANNEL_PACKS : CNN_POOL1_CHANNEL_PACKS;
    const int expected_inputs = H * tiles_per_row * cout_tiles;
    const int expected_outputs = pooled_h * pooled_w * channel_packs;
    hls::stream<activation_tile_t> input_stream;
    hls::stream<channel_pack_t> output_stream;
    std::vector<int> source(H * W * Cout);
    std::ofstream input_csv, output_csv;
    const std::string prefix = directory + "/" + name;
    if (!open_csv(input_csv, prefix + "_input.csv",
                  "transaction,row,tile_x,cout_tile,lane,cout_lane,valid,value") ||
        !open_csv(output_csv, prefix + "_output.csv",
                  "transaction,pooled_y,pooled_x,channel_pack,"
                  "token_not_applicable,lane,channel,valid,expected,actual,"
                  "match")) {
        return 1;
    }

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int channel = 0; channel < Cout; ++channel) {
                source[source_index(y, x, channel, W, Cout)] =
                    source_value(y, x, channel);
            }
        }
    }

    int inputs_written = 0;
    for (int row = 0; row < H; ++row) {
        for (int tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
            const int valid_mask = (pool2 && tile_x == CNN_CONV2_LAST_TILE_X) ?
                CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
            for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
                activation_tile_t input;
                input.spatial_valid = valid_mask;
                for (int lane = 0; lane < SYSTOLIC_SPATIAL_PAR; ++lane) {
                    for (int cout_lane = 0;
                         cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
                        const int x = tile_x * SYSTOLIC_SPATIAL_PAR + lane;
                        const int channel = cout_tile * SYSTOLIC_COUT_PAR +
                                            cout_lane;
                        // 无效尾lane不应进入2x2池化；127可放大该路径的错误。
                        const int value = x < W ?
                            source[source_index(row, x, channel, W, Cout)] :
                            127;
                        input.value[lane][cout_lane] = value;
                        input_csv << inputs_written << ',' << row << ','
                                  << tile_x << ',' << cout_tile << ',' << lane
                                  << ',' << cout_lane << ','
                                  << ((valid_mask & (1 << lane)) != 0) << ','
                                  << value << '\n';
                    }
                }
                input_stream.write(input);
                ++inputs_written;
            }
        }
    }

    maxpool(input_stream, output_stream, layer_mode);
    int errors = 0;
    int comparisons = 0;
    int outputs_read = 0;
    for (int pooled_y = 0; pooled_y < pooled_h; ++pooled_y) {
        for (int pooled_x = 0; pooled_x < pooled_w; ++pooled_x) {
            for (int pack = 0; pack < channel_packs; ++pack) {
                const channel_pack_t actual_pack = output_stream.read();
                for (int lane = 0; lane < IM2ROW_MAX_CHANNELS; ++lane) {
                    const int channel = pack * IM2ROW_MAX_CHANNELS + lane;
                    int expected = source[source_index(pooled_y * 2,
                                                       pooled_x * 2, channel,
                                                       W, Cout)];
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            const int candidate = source[source_index(
                                pooled_y * 2 + dy, pooled_x * 2 + dx,
                                channel, W, Cout)];
                            if (candidate > expected) expected = candidate;
                        }
                    }
                    const int actual = actual_pack.channel[lane].to_int();
                    const bool match = actual == expected;
                    output_csv << outputs_read << ',' << pooled_y << ','
                               << pooled_x << ',' << pack << ",-1," << lane
                               << ',' << channel << ",1," << expected << ','
                               << actual << ',' << match << '\n';
                    ++comparisons;
                    if (!match) {
                        ++errors;
                        if (errors <= 10) {
                            std::cerr << name << " mismatch pooled_y="
                                      << pooled_y << " pooled_x=" << pooled_x
                                      << " channel=" << channel << " expected="
                                      << expected << " actual=" << actual << std::endl;
                        }
                    }
                }
                ++outputs_read;
            }
        }
    }
    if (inputs_written != expected_inputs || outputs_read != expected_outputs) {
        ++errors;
    }
    if (!input_stream.empty() || !output_stream.empty()) {
        ++errors;
        std::cerr << name << " stream 未完全消费" << std::endl;
    }
    std::cout << "Pool " << name << " 输入事务=" << inputs_written << '/'
              << expected_inputs << " 输出事务=" << outputs_read << '/'
              << expected_outputs << " 检查元素数=" << comparisons
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS")
              << std::endl;
    return errors;
}

}  // namespace

int main() {
    const std::string directory = "simulation_data/pool";
    if (!make_directory("simulation_data") || !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    int errors = 0;
    errors += run_layer(LAYER1_MODE, "pool1", directory);
    errors += run_layer(LAYER2_MODE, "pool2", directory);
    std::cout << "Pool CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
