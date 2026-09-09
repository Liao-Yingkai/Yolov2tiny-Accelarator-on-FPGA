#include "../src/postprocess/conv_postprocess.h"

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <string>

/*
 * ConvPostProcess 完整层定向 testbench。
 * - 输入流顺序为 spatial_tile(row->tile_x) -> cout_tile；参数流仅在首个
 *   spatial tile 按 cout_tile 输入一次。Conv1 有196*2个输入事务，Conv2
 *   有56*4个输入事务，参数分别为2和4个 pack，输出事务数与输入相同。
 * - reference 独立执行 INT32 bias add、ReLU、乘法、直接右移和127饱和；
 *   经 stream 覆盖 bias 后负/零、低位截断、127/128、shift>=64，无效空间
 *   lane 必须直接输出零。本 TB 只调用 conv_postprocess 这个 stage top。
 * - input.csv 保存累加器，params.csv 保存每输出通道参数，output.csv 保存
 *   每个逻辑坐标的 expected/actual 及 expected/actual valid mask。
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

static int8_t reference_requantize(int32_t accumulator, int32_t bias,
                                   int32_t multiplier, int32_t shift) {
    assert(multiplier >= 0 && shift >= 0);
    const int64_t biased = (int64_t)accumulator + bias;
    assert(biased >= INT32_MIN && biased <= INT32_MAX);
    const int64_t relu = biased > 0 ? biased : 0;
    const int64_t scaled = shift >= 64 ? 0 : (relu * multiplier >> shift);
    return scaled > 127 ? 127 : (int8_t)scaled;
}

static int accumulator_value(int spatial_tile, int cout_tile,
                             int lane, int cout_lane) {
    if (spatial_tile == 0 && cout_tile == 0 && lane == 0) {
        // 与 make_param 的首个 Pack 对应：bias 后负、零、3>>1 的低位截断，
        // 以及恰好127的非饱和边界，均经完整 stream/DUT 路径覆盖。
        const int edge[] = {5, 0, 1, 127};
        return edge[cout_lane];
    }
    if (spatial_tile == 0 && cout_tile == 1 && lane == 0) {
        // 与第二个参数 Pack 对应：128上饱和和 shift>=64 归零。
        const int edge[] = {128, 123456, 0, 0};
        return edge[cout_lane];
    }
    return ((spatial_tile * 11 + cout_tile * 17 + lane * 23 +
             cout_lane * 29) % 241) - 120;
}

static postprocess_param_pack_t make_param(int cout_tile) {
    postprocess_param_pack_t param;
    for (int lane = 0; lane < SYSTOLIC_COUT_PAR; ++lane) {
        param.bias[lane] = -9 + cout_tile * 7 + lane * 3;
        param.multiplier[lane] = 1;
        param.shift[lane] = 0;
    }
    if (cout_tile == 0) {
        param.bias[0] = -10;  // accumulator=5，bias 后为负，ReLU 输出0。
        param.bias[1] = 0;    // accumulator=0，保持零。
        param.multiplier[2] = 3;
        param.shift[2] = 1;   // accumulator=1：3 >> 1 = 1，验证直接截断。
        // lane3 保持 multiplier=1、shift=0，accumulator=127。
    } else if (cout_tile == 1) {
        param.bias[0] = 0;    // accumulator=128，经上限饱和为127。
        param.bias[1] = 0;
        param.multiplier[1] = 1;
        param.shift[1] = 64;  // 正数 product 在 >=64 位右移后为0。
    }
    return param;
}

static int run_layer(layer_mode_t layer_mode, const char* name,
                     const std::string& directory) {
    const bool conv2 = (layer_mode == LAYER2_MODE);
    const int spatial_tiles = conv2 ? CNN_CONV2_SPATIAL_TILES : CNN_CONV1_SPATIAL_TILES;
    const int tiles_per_row = conv2 ? CNN_CONV2_TILES_PER_ROW : CNN_CONV1_TILES_PER_ROW;
    const int cout_tiles = conv2 ? CNN_CONV2_COUT_TILES : CNN_CONV1_COUT_TILES;
    const int expected_transactions = spatial_tiles * cout_tiles;
    hls::stream<acc_tile_t> input_stream;
    hls::stream<postprocess_param_pack_t> param_stream;
    hls::stream<activation_tile_t> output_stream;
    std::ofstream input_csv, params_csv, output_csv;
    const std::string prefix = directory + "/" + name;
    if (!open_csv(input_csv, prefix + "_input.csv",
                  "transaction,spatial_tile,cout_tile,lane,cout_lane,expected_valid_mask,accumulator") ||
        !open_csv(params_csv, prefix + "_params.csv",
                  "cout_tile,cout_lane,bias,multiplier,shift") ||
        !open_csv(output_csv, prefix + "_output.csv",
                  "transaction,spatial_tile,cout_tile,token_not_applicable,"
                  "lane,cout_lane,channel,expected_valid_mask,"
                  "actual_valid_mask,valid_match,expected,actual,match")) {
        return 1;
    }

    int params_written = 0;
    for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
        const postprocess_param_pack_t param = make_param(cout_tile);
        param_stream.write(param);
        ++params_written;
        for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR; ++cout_lane) {
            params_csv << cout_tile << ',' << cout_lane << ','
                       << param.bias[cout_lane].to_int() << ','
                       << param.multiplier[cout_lane].to_int() << ','
                       << param.shift[cout_lane].to_int() << '\n';
        }
    }

    int inputs_written = 0;
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const int valid_mask = (conv2 &&
                                spatial_tile % tiles_per_row == CNN_CONV2_LAST_TILE_X)
            ? CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
        for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
            acc_tile_t input;
            input.spatial_valid = valid_mask;
            for (int lane = 0; lane < SYSTOLIC_SPATIAL_PAR; ++lane) {
                for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                     ++cout_lane) {
                    input.value[lane][cout_lane] = (valid_mask & (1 << lane)) ?
                        accumulator_value(spatial_tile, cout_tile, lane,
                                          cout_lane) : 100000;
                    input_csv << inputs_written << ',' << spatial_tile << ','
                              << cout_tile << ',' << lane << ',' << cout_lane
                              << ',' << valid_mask << ','
                              << input.value[lane][cout_lane].to_int() << '\n';
                }
            }
            input_stream.write(input);
            ++inputs_written;
        }
    }

    conv_postprocess(input_stream, param_stream, output_stream, layer_mode);
    int errors = 0;
    int comparisons = 0;
    int outputs_read = 0;
    for (int spatial_tile = 0; spatial_tile < spatial_tiles; ++spatial_tile) {
        const int expected_valid = (conv2 &&
                                    spatial_tile % tiles_per_row == CNN_CONV2_LAST_TILE_X)
            ? CNN_CONV2_TAIL_SPATIAL_VALID_MASK : CNN_FULL_SPATIAL_VALID_MASK;
        for (int cout_tile = 0; cout_tile < cout_tiles; ++cout_tile) {
            const activation_tile_t actual_tile = output_stream.read();
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
            const postprocess_param_pack_t param = make_param(cout_tile);
            for (int lane = 0; lane < SYSTOLIC_SPATIAL_PAR; ++lane) {
                for (int cout_lane = 0; cout_lane < SYSTOLIC_COUT_PAR;
                     ++cout_lane) {
                    const int expected = (expected_valid & (1 << lane)) ?
                        reference_requantize(accumulator_value(spatial_tile,
                                             cout_tile, lane, cout_lane),
                                             param.bias[cout_lane].to_int(),
                                             param.multiplier[cout_lane].to_int(),
                                             param.shift[cout_lane].to_int()) : 0;
                    const int actual = actual_tile.value[lane][cout_lane].to_int();
                    const bool match = valid_match && actual == expected;
                    output_csv << outputs_read << ',' << spatial_tile << ','
                               << cout_tile << ",-1," << lane << ',' << cout_lane
                               << ',' << (cout_tile * SYSTOLIC_COUT_PAR + cout_lane) << ','
                               << expected_valid << ',' << actual_valid << ','
                               << valid_match << ',' << expected << ',' << actual
                               << ',' << match << '\n';
                    ++comparisons;
                    if (!match && actual == expected) continue;
                    if (!match) {
                        ++errors;
                        if (errors <= 10) {
                            std::cerr << name << " mismatch spatial_tile="
                                      << spatial_tile << " cout_tile=" << cout_tile
                                      << " lane=" << lane << " cout_lane="
                                      << cout_lane << " expected=" << expected
                                      << " actual=" << actual << std::endl;
                        }
                    }
                }
            }
            ++outputs_read;
        }
    }
    if (inputs_written != expected_transactions ||
        outputs_read != expected_transactions || params_written != cout_tiles) {
        ++errors;
    }
    if (!input_stream.empty() || !param_stream.empty() || !output_stream.empty()) {
        ++errors;
        std::cerr << name << " stream 未完全消费" << std::endl;
    }
    std::cout << "ConvPostProcess " << name << " 输入事务="
              << inputs_written << '/' << expected_transactions
              << " 参数事务=" << params_written << '/' << cout_tiles
              << " 输出事务=" << outputs_read << '/' << expected_transactions
              << " 检查元素数=" << comparisons << " 错误数=" << errors << ' '
              << (errors ? "FAIL" : "PASS") << std::endl;
    return errors;
}

}  // namespace

int main() {
    const std::string directory = "simulation_data/conv_postprocess";
    if (!make_directory("simulation_data") || !make_directory(directory.c_str())) {
        std::cerr << "无法创建 CSV 目录: " << directory << std::endl;
        return 1;
    }
    int errors = 0;
    errors += run_layer(LAYER1_MODE, "conv1", directory);
    errors += run_layer(LAYER2_MODE, "conv2", directory);
    std::cout << "ConvPostProcess CSV保存路径=" << directory << std::endl;
    return errors ? 1 : 0;
}
