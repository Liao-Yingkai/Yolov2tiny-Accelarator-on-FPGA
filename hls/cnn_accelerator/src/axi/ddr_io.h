#ifndef CNN_ACCELERATOR_AXI_DDR_IO_H_
#define CNN_ACCELERATOR_AXI_DDR_IO_H_

#include "../types.h"

// DDR 指针已指向当前 Tensor 或模型层段基址，地址以各指针的元素为单位。
namespace ddr_io {
// 每次扫描完整输入一次；紧凑 RGB 解包并补 Cin 尾零，其余层直接解包 NHWC。
void feature_loader(const ap_uint<64>* feature,
                    hls::stream<channel_pack_t>& output,
                    const ConvGeometry& geometry);

// 整层连续供权重，不负责 Cout 块调度或驻留缓存。
void weight_loader(const ap_uint<64>* weights,
                   hls::stream<weight_pack_t>& output,
                   const ConvGeometry& geometry);

// 参数层段按 [全部 bias][全部 multiplier][全部 shift] 排列。
void param_loader(const ap_int<32>* params,
                  hls::stream<postprocess_param_pack_t>& output,
                  const ConvGeometry& geometry);

// 按 Cout 块消费整层 Pool/Bypass 结果，按池化后坐标写入 NHWC。
void feature_writer(hls::stream<yolo_activation_tile_t>& input,
                    ap_uint<64>* feature,
                    const ConvGeometry& geometry);
}  // namespace ddr_io

#endif
