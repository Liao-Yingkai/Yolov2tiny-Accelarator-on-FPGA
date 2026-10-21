#include "../src/systolic/systolic.h"

#include <cerrno>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>


/*
 * Systolic 完整层定向 testbench。
 * - activation流严格为 spatial_tile(row->tile_x) -> token；Conv1供给
 *   196*9个pack且仅c0非零，Conv2供给56*9个pack且8通道全使用，末tile
 *   valid为0011。weight流只供给首spatial tile所需 cout_tile->token：
 *   Conv1为18个pack、Conv2为36个pack；输出数分别为392和224个acc tile。
 * - reference 以独立int64累加9*8乘积并先检查INT32范围，再逐spatial/
 *   cout lane比较；无效space lane的结果预期为零。明确检查token0和token8
 *   对首个PE的dot贡献均非零，防止数据模式弱化token时序覆盖。
 * - activations/weights CSV保存投喂流；output CSV保存最终acc事务、逻辑
 *   坐标、无适用token标记、expected/actual valid mask和值比较。
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

int activation_value(int spatial_tile, int token, int lane,
                     int channel) {
    return ((spatial_tile * 3 + token * 5 + lane * 7 + channel * 11) % 15) - 7;
}

int weight_value(int cout_tile, int token, int cout_lane,
                 int channel) {
    return ((cout_tile * 5 + token * 3 + cout_lane * 7 + channel * 2) % 13) - 6;
}

int token_contribution(bool conv2, int token) {
    int sum = 0;
    for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
        const int activation = (conv2 || channel == 0) ?
            activation_value(0, token, 0, channel) : 0;
        sum += activation * weight_value(0, token, 0, channel);
    }
    return sum;
}

int run_layer(hls::stream<activation_pack_t>& activation_stream,
              hls::stream<weight_pack_t>& weight_stream,
              hls::stream<acc_tile_t>& output_stream,
              layer_mode_t layer_mode, const char* name,
              const std::string& directory) {
    const bool conv2 = (layer_mode == LAYER2_MODE);
    const int tiles_per_row = conv2 ? CNN_CONV2_TILES_PER_ROW : CNN_CONV1_TILES_PER_ROW;
    const int spatial_tiles = conv2 ? CNN_CONV2_SPATIAL_TILES : CNN_CONV1_SPATIAL_TILES;
    const int cout_tiles = conv2 ? CNN_CONV2_COUT_TILES : CNN_CONV1_COUT_TILES;
    const int expected_activations = spatial_tiles * SYSTOLIC_KERNEL_TOKENS;
    const int expected_weights = cout_tiles * SYSTOLIC_KERNEL_TOKENS;
    const int expected_outputs = spatial_tiles * cout_tiles;
    std::ofstream activation_csv, weight_csv, output_csv;
    const std::string prefix = directory + "/" + name;
    if (!open_csv(activation_csv, prefix + "_activations.csv",
                  "transaction,spatial_tile,row,tile_x,token,lane,channel,valid,value") ||
        !open_csv(weight_csv, prefix + "_weights.csv",
                  "transaction,cout_tile,token,cout_lane,channel,value") ||
        !open_csv(output_csv, prefix + "_output.csv",
                  "transaction,spatial_tile,row,tile_x,cout_tile,"
                  "token_not_applicable,spatial_lane,cout_lane,channel,"
                  "expected_valid_mask,actual_valid_mask,valid_match,"
                  "expected,actual,match")) {
        return 1;
    }

    int errors = 0;
    const int token0_dot = token_contribution(conv2, 0);
    const int token8_dot = token_contribution(conv2, SYSTOLIC_KERNEL_TOKENS - 1);
    if (token0_dot == 0 || token8_dot == 0) {
        ++errors;
        std::cerr << name << " token contribution is zero: token0="
                  << token0_dot << " token8=" << token8_dot << std::endl;
    }

    int activations_written = 0;
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const int row = spatial_tile / tiles_per_row;
        const int tile_x = spatial_tile % tiles_per_row;
        const int valid_mask = (conv2 && tile_x == CNN_CONV2_LAST_TILE_X) ?
            CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
        for (int token = 0; token < SYSTOLIC_KERNEL_TOKENS; ++token) {
            activation_pack_t pack;
            pack.spatial_valid = valid_mask;
            for (int lane = 0; lane < SYSTOLIC_SPATIAL_PAR; ++lane) {
                for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
                    // Conv1 无效 Cin 双侧非零 poison；有效 spatial lane 的
                    // direct/replay 都必须仅计算 c0。invalid spatial 仍保持零。
                    const int value = (valid_mask & (1 << lane)) ?
                        ((!conv2 && channel != 0) ? 37 :
                         activation_value(spatial_tile, token, lane, channel)) : 0;
                    pack.pixel[lane].channel[channel] = value;
                    activation_csv << activations_written << ',' << spatial_tile
                                   << ',' << row << ',' << tile_x << ',' << token
                                   << ',' << lane << ',' << channel << ','
                                   << ((valid_mask & (1 << lane)) != 0) << ','
                                   << value << '\n';
                }
            }
            activation_stream.write(pack);
            ++activations_written;
        }
    }

    int weights_written = 0;
    for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
        for (int token = 0; token < SYSTOLIC_KERNEL_TOKENS; ++token) {
            weight_pack_t pack;
            for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                 ++cout_lane) {
                for (int channel = 0; channel < SYSTOLIC_CIN_PAR; ++channel) {
                    const int value = (!conv2 && channel != 0) ? -29 :
                        weight_value(cout_tile, token, cout_lane, channel);
                    pack.lane[cout_lane].channel[channel] = value;
                    weight_csv << weights_written << ',' << cout_tile << ','
                               << token << ',' << cout_lane << ',' << channel
                               << ',' << value << '\n';
                }
            }
            weight_stream.write(pack);
            ++weights_written;
        }
    }

#if CNN_SYSTOLIC_USE_OS
    systolic_OS(activation_stream, weight_stream, output_stream, layer_mode);
#else
    systolic(activation_stream, weight_stream, output_stream, layer_mode);
#endif

    int comparisons = 0;
    int outputs_read = 0;
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const int row = spatial_tile / tiles_per_row;
        const int tile_x = spatial_tile % tiles_per_row;
        const int expected_valid = (conv2 && tile_x == CNN_CONV2_LAST_TILE_X) ?
            CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
        for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
            const acc_tile_t actual_tile = output_stream.read();
            const int actual_valid = actual_tile.spatial_valid.to_uint();
            const bool valid_match = actual_valid == expected_valid;
            if (!valid_match) {
                ++errors;
                if (errors <= 10) {
                    std::cerr << name << " valid mismatch spatial_tile="
                              << spatial_tile << " cout_tile=" << cout_tile
                              << " expected_mask=" << expected_valid
                              << " actual_mask=" << actual_valid << std::endl;
                }
            }
            for (int spatial_lane = 0;
                 spatial_lane < SYSTOLIC_SPATIAL_PAR; ++spatial_lane) {
                for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                     ++cout_lane) {
                    int64_t sum = 0;
                    if (expected_valid & (1 << spatial_lane)) {
                        for (int token = 0; token < SYSTOLIC_KERNEL_TOKENS;
                             ++token) {
                            // 独立 reference 按逻辑 Cin 累加，不读取 poison。
                            for (int channel = 0; channel < (conv2 ? SYSTOLIC_CIN_PAR : 1);
                                 ++channel) {
                                const int activation =
                                    activation_value(spatial_tile, token,
                                                     spatial_lane, channel);
                                sum += (int64_t)activation * weight_value(
                                    cout_tile, token, cout_lane, channel);
                            }
                        }
                    }
                    if (sum < INT32_MIN || sum > INT32_MAX) {
                        ++errors;
                        sum = 0;
                    }
                    const int actual =
                        actual_tile.value[spatial_lane][cout_lane].to_int();
                    const bool match = valid_match && actual == (int32_t)sum;
                    output_csv << outputs_read << ',' << spatial_tile << ','
                               << row << ',' << tile_x << ',' << cout_tile
                               << ",-1," << spatial_lane << ',' << cout_lane
                               << ',' << (cout_tile * SYSTOLIC_COUT_PAR + cout_lane) << ','
                               << expected_valid << ',' << actual_valid << ','
                               << valid_match << ',' << sum << ',' << actual
                               << ',' << match << '\n';
                    ++comparisons;
                    if (!match && actual != (int32_t)sum) {
                        ++errors;
                        if (errors <= 10) {
                            std::cerr << name << " mismatch spatial_tile="
                                      << spatial_tile << " cout_tile=" << cout_tile
                                      << " spatial_lane=" << spatial_lane
                                      << " cout_lane=" << cout_lane
                                      << " expected=" << sum << " actual="
                                      << actual << std::endl;
                        }
                    }
                }
            }
            ++outputs_read;
        }
    }
    if (activations_written != expected_activations ||
        weights_written != expected_weights || outputs_read != expected_outputs) {
        ++errors;
    }
    if (!activation_stream.empty() || !weight_stream.empty() ||
        !output_stream.empty()) {
        ++errors;
        std::cerr << name << " stream 未完全消费" << std::endl;
    }
    std::cout << "Systolic " << name << " 输入事务="
              << activations_written << '/' << expected_activations
              << " 权重事务=" << weights_written << '/' << expected_weights
              << " 输出事务=" << outputs_read << '/' << expected_outputs
              << " 检查元素数=" << comparisons << " 错误数=" << errors << ' '
              << (errors ? "FAIL" : "PASS") << std::endl;
    return errors;
}

}  // namespace

int main() {
    const std::string directory = "simulation_data/systolic";
    if (!make_directory("simulation_data") || !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    hls::stream<activation_pack_t> activation_stream;
    hls::stream<weight_pack_t> weight_stream;
    hls::stream<acc_tile_t> output_stream;
    int errors = 0;
    errors += run_layer(activation_stream, weight_stream, output_stream,
                        LAYER1_MODE, "conv1", directory);
    errors += run_layer(activation_stream, weight_stream, output_stream,
                        LAYER2_MODE, "conv2", directory);
    std::cout << "Systolic CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
