#include "../src/systolic/systolic.h"
#include "yolo_data_path.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/*
 * YOLOv2-tiny Systolic OS 独立测试：输入和期望结果直接取同一套 Python 导出 BIN。
 * - Activation [H,T,R,4,8]：按 y -> tile_x -> token 读取，每个 Cout 块重送一次。
 * - Weight [ceil(Cout/4),R,4,8]：按全局 Cout Tile -> token 读取，每块只取自己的连续片段。
 * - token = (kh*Kw+kw)*G+group，G=ceil(Cin/8)，R=Kh*Kw*G。
 * - INT32 golden [H,W,Cout]：无 bias，little-endian；按真实 y/x/oc 随机索引。
 * - 结果按块 -> y -> tile_x -> local_cout_tile 接收；无效空间/Cout lane 期望为零，
 *   不访问 golden。分别比较两个 mask，不能用值为零代替 valid。
 * 逐包验证输入补零、输出 mask 与累加值，仅打印终端统计，不导出数据文件。
 */
namespace {
std::string generated_directory;
constexpr char sample_suffix[] = "0000";
constexpr int ACTIVATION_PACK_BYTES = SYSTOLIC_SPATIAL_PAR * SYSTOLIC_CIN_PAR;
constexpr int WEIGHT_PACK_BYTES = SYSTOLIC_COUT_PAR * SYSTOLIC_CIN_PAR;
constexpr int INT32_BYTES = 4;

bool read_binary(const std::string& path, std::size_t expected_bytes,
                 std::vector<std::int8_t>& bytes) {
    std::ifstream file(path.c_str(), std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(expected_bytes)) {
        std::cerr << "BIN不存在或长度错误: " << path << " 期望字节数=" << expected_bytes << '\n';
        return false;
    }
    file.seekg(0);
    bytes.resize(expected_bytes);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(expected_bytes))) {
        std::cerr << "BIN读取失败: " << path << '\n';
        return false;
    }
    return true;
}

// 前三字节按无符号解释，最高字节按有符号解释。不依赖宿主字节序或结构体布局。
std::int32_t read_le_int32(const std::int8_t* byte) {
    return std::uint8_t(byte[0]) + (std::int32_t(std::uint8_t(byte[1])) << 8) +
           (std::int32_t(std::uint8_t(byte[2])) << 16) + std::int32_t(byte[3]) * (1 << 24);
}

unsigned lane_mask(int first, int limit, int lanes) {
    unsigned mask = 0;
    for (int lane = 0; lane < lanes; ++lane)
        if (first + lane < limit) mask |= 1u << lane;
    return mask;
}

bool run_layer(const ConvGeometry& g, const char* name,
               hls::stream<activation_pack_t>& activation_stream,
               hls::stream<weight_pack_t>& weight_stream,
               hls::stream<yolo_acc_tile_t>& output_stream, int& total_errors) {
    // 用逻辑形状推导文件尺寸，避免照用 DUT 派生字段掩盖事务数错误。
    const int groups = ceil_div(g.cin, SYSTOLIC_CIN_PAR);
    const int reduction = g.kernel * g.kernel * groups;
    const int tiles_per_row = ceil_div(g.w, SYSTOLIC_SPATIAL_PAR);
    const int spatial_tiles = g.h * tiles_per_row;
    const int cout_tiles = ceil_div(g.cout, SYSTOLIC_COUT_PAR);
    const int resident_tiles = std::min(YOLO_TINY_COUT_CACHE_TILES, cout_tiles);
    const int blocks = ceil_div(cout_tiles, resident_tiles);
    const int activation_packs = spatial_tiles * reduction;
    const int weight_packs = cout_tiles * reduction;
    const int output_tiles = spatial_tiles * cout_tiles;
    if (g.reduction_steps != reduction || g.spatial_tiles != spatial_tiles ||
        g.cout_tiles != cout_tiles || g.resident_cout_tiles != resident_tiles ||
        g.cache_blocks != blocks || g.im2row_tokens != activation_packs ||
        g.weight_transactions != weight_packs) {
        std::cerr << name << " 层配置与 BIN 事务契约不一致\n";
        return false;
    }

    std::vector<std::int8_t> activation, weight, golden;
    const std::string layer = name;
    const std::string trace = generated_directory + "trace/trace_" + layer;
    const std::string suffix = std::string("_") + sample_suffix + ".bin";
    if (!read_binary(trace + "_im2row_int8" + suffix,
                     std::size_t(activation_packs) * ACTIVATION_PACK_BYTES, activation) ||
        !read_binary(generated_directory + "bins/" + layer + "_weight.bin",
                     std::size_t(weight_packs) * WEIGHT_PACK_BYTES, weight) ||
        !read_binary(trace + "_systolic_int32_hwc" + suffix,
                     std::size_t(g.h) * g.w * g.cout * INT32_BYTES, golden)) return false;

    int errors = 0, activations_written = 0, weights_written = 0, outputs_read = 0;
    for (int first = 0, block = 0; first < cout_tiles; first += resident_tiles, ++block) {
        const int count = std::min(resident_tiles, cout_tiles - first);
        if (!activation_stream.empty() || !weight_stream.empty() || !output_stream.empty()) {
            std::cerr << name << " block=" << block << " 调用开始时 stream 非空\n";
            return false;
        }

        // 每个全局 Cout Tile 在 BIN 中有连续 R 个 Pack。块边界只改变读取区间。
        for (int ct = first; ct < first + count; ++ct) {
            for (int token = 0; token < reduction; ++token) {
                const int index = ct * reduction + token;
                const std::int8_t* bytes = weight.data() + std::size_t(index) * WEIGHT_PACK_BYTES;
                weight_pack_t pack;
                for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
                    for (int c = 0; c < SYSTOLIC_CIN_PAR; ++c) {
                        const int value = bytes[j * SYSTOLIC_CIN_PAR + c];
                        const int ic = (token % groups) * SYSTOLIC_CIN_PAR + c;
                        const int oc = ct * SYSTOLIC_COUT_PAR + j;
                        if ((ic >= g.cin || oc >= g.cout) && value != 0) {
                            std::cerr << name << " 权重尾通道未补零: tile=" << ct
                                      << " token=" << token << " ic=" << ic << " oc=" << oc << '\n';
                            return false;
                        }
                        pack.lane[j].channel[c] = value;
                    }
                }
                weight_stream.write(pack);
                ++weights_written;
            }
        }

        // 原始 Activation BIN 不含 mask。按图像宽度重建，不修改有效数据，
        // 不为每个本地 Cout Tile 重发：块内 Replay 应由 DUT 自己完成。
        for (int spatial = 0; spatial < spatial_tiles; ++spatial) {
            const int y = spatial / tiles_per_row;
            const int tile_x = spatial % tiles_per_row;
            const unsigned spatial_valid = lane_mask(tile_x * SYSTOLIC_SPATIAL_PAR, g.w,
                                                     SYSTOLIC_SPATIAL_PAR);
            for (int token = 0; token < reduction; ++token) {
                const int index = spatial * reduction + token;
                const std::int8_t* bytes = activation.data() + std::size_t(index) * ACTIVATION_PACK_BYTES;
                activation_pack_t pack;
                pack.spatial_valid = spatial_valid;
                for (int i = 0; i < SYSTOLIC_SPATIAL_PAR; ++i) {
                    for (int c = 0; c < SYSTOLIC_CIN_PAR; ++c) {
                        const int value = bytes[i * SYSTOLIC_CIN_PAR + c];
                        pack.pixel[i].channel[c] = value;
                        if (block == 0) {
                            const int ic = (token % groups) * SYSTOLIC_CIN_PAR + c;
                            if (((spatial_valid & (1u << i)) == 0 || ic >= g.cin) && value != 0) {
                                std::cerr << name << " Activation 尾部未补零: y=" << y
                                          << " tile_x=" << tile_x << " token=" << token
                                          << " lane=" << i << " ic=" << ic << '\n';
                                return false;
                            }
                        }
                    }
                }
                activation_stream.write(pack);
                ++activations_written;
            }
        }

        systolic_OS(activation_stream, weight_stream, output_stream, g, first, count);

        // 不将结果重排成整图。由当前事务恢复 y、x、全局 oc，直接索引 HWC golden。
        for (int y = 0; y < g.h; ++y) {
            for (int tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
                const int first_x = tile_x * SYSTOLIC_SPATIAL_PAR;
                const unsigned spatial_valid = lane_mask(first_x, g.w, SYSTOLIC_SPATIAL_PAR);
                for (int ct = first; ct < first + count; ++ct) {
                    if (output_stream.empty()) {
                        std::cerr << name << " 缺少输出: block=" << block << " y=" << y
                                  << " tile_x=" << tile_x << " cout_tile=" << ct << '\n';
                        return false;
                    }
                    const yolo_acc_tile_t actual = output_stream.read();
                    const int first_oc = ct * SYSTOLIC_COUT_PAR;
                    const unsigned cout_valid = lane_mask(first_oc, g.cout, SYSTOLIC_COUT_PAR);
                    const unsigned actual_spatial_valid = actual.spatial_valid.to_uint();
                    const unsigned actual_cout_valid = actual.cout_valid.to_uint();
                    const bool match = actual_spatial_valid == spatial_valid && actual_cout_valid == cout_valid;
                    if (!match && ++errors <= 10)
                        std::cerr << name << " mask错误: block=" << block << " y=" << y
                                  << " tile_x=" << tile_x << " cout_tile=" << ct
                                  << " spatial=" << actual_spatial_valid << '/' << spatial_valid
                                  << " cout=" << actual_cout_valid << '/' << cout_valid
                                  << " (实际/期望)\n";

                    ++outputs_read;
                    for (int i = 0; i < SYSTOLIC_SPATIAL_PAR; ++i) {
                        const int x = first_x + i;
                        for (int j = 0; j < SYSTOLIC_COUT_PAR; ++j) {
                            const int oc = first_oc + j;
                            // HWC BIN 不保存物理尾部。必须先判界，不能用 DUT 的 mask 控制参考读取。
                            std::int32_t expected = 0;
                            if (x < g.w && oc < g.cout) {
                                const std::size_t index = (std::size_t(y) * g.w + x) * g.cout + oc;
                                expected = read_le_int32(golden.data() + index * INT32_BYTES);
                            }
                            const std::int32_t got = actual.value[i][j].to_int();
                            if (got != expected) {
                                if (++errors <= 10)
                                    std::cerr << name << " 累加值错误: block=" << block << " y=" << y
                                              << " x=" << x << " oc=" << oc << " actual=" << got
                                              << " expected=" << expected << '\n';
                            }
                        }
                    }
                }
            }
        }
        if (!activation_stream.empty() || !weight_stream.empty() || !output_stream.empty()) {
            std::cerr << name << " block=" << block << " 事务数错误: 剩余 activation="
                      << !activation_stream.empty() << " 剩余 weight=" << !weight_stream.empty()
                      << " 多余 output=" << !output_stream.empty() << '\n';
            return false;
        }
    }

    if (activations_written != blocks * activation_packs ||
        weights_written != weight_packs || outputs_read != output_tiles) ++errors;
    std::cout << name << " Cout块=" << blocks << " Activation事务=" << activations_written
              << " Weight事务=" << weights_written << " 输出事务=" << outputs_read
              << " 验证元素数=" << outputs_read * SYSTOLIC_SPATIAL_PAR * SYSTOLIC_COUT_PAR
              << " 错误数=" << errors << ' ' << (errors ? "FAIL" : "PASS") << '\n';
    total_errors += errors;
    return true;
}
}  // namespace

int main() {
    const std::string configured_path = yolo_test_data_dir();
    if (configured_path.empty()) return 1;
    generated_directory = std::string(configured_path) + "/";
    const char* names[] = {"conv1", "conv2", "conv3", "conv4", "conv5",
                           "conv6", "conv7", "conv8", "prediction"};
    static_assert(sizeof(names) / sizeof(names[0]) == YOLO_TINY_LAYER_COUNT,
                  "BIN layer names must match the fixed YOLO layer table");
    hls::stream<activation_pack_t> activation_stream("activation");
    hls::stream<weight_pack_t> weight_stream("weight");
    hls::stream<yolo_acc_tile_t> output_stream("result");
    int errors = 0;
    // 同一组流依次运行全部层、全部 Cout 块，覆盖 Conv8 跨权重段和 Prediction 尾 Cout。
    for (int i = 0; i < YOLO_TINY_LAYER_COUNT; ++i)
        if (!run_layer(YOLO_TINY_GEOMETRIES[i], names[i],
                       activation_stream, weight_stream, output_stream, errors)) return 1;
    std::cout << "Systolic OS " << (errors ? "FAIL" : "PASS") << " 错误数=" << errors << '\n';
    return errors ? 1 : 0;
}
