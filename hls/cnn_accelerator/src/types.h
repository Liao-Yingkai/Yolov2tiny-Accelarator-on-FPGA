#ifndef CNN_ACCELERATOR_TYPES_H_
#define CNN_ACCELERATOR_TYPES_H_

#include <ap_int.h>
#include <hls_stream.h>

// 以下宏决定 Im2Row 及其 Systolic 接口的编译期物理规模，可在综合前通过编译选项覆盖。
// P 是每个输出 Pack 的空间 Lane 数，也是 Systolic 每次接收的激活路数。
#ifndef CNN_ACCEL_IM2ROW_P
#define CNN_ACCEL_IM2ROW_P 4
#endif

#ifndef CNN_ACCEL_IM2ROW_MAX_H
#define CNN_ACCEL_IM2ROW_MAX_H 28
#endif

#ifndef CNN_ACCEL_IM2ROW_MAX_W
#define CNN_ACCEL_IM2ROW_MAX_W 28
#endif

#ifndef CNN_ACCEL_IM2ROW_MAX_CHANNELS
#define CNN_ACCEL_IM2ROW_MAX_CHANNELS 8
#endif

// 运行时参数不能超过这些编译期上限；它们共同决定数组、接口和并行硬件规模。
static const int IM2ROW_P = CNN_ACCEL_IM2ROW_P;
static const int IM2ROW_MAX_H = CNN_ACCEL_IM2ROW_MAX_H;
static const int IM2ROW_MAX_W = CNN_ACCEL_IM2ROW_MAX_W;
static const int IM2ROW_MAX_CHANNELS = CNN_ACCEL_IM2ROW_MAX_CHANNELS;

// 第一版卷积窗口固定为3×3、Padding=1、Stride=1；K 包含9个空间位置及全部物理通道。
static const int IM2ROW_KERNEL = 3;
static const int IM2ROW_PADDING = 1;
static const int IM2ROW_STRIDE = 1;
static const int IM2ROW_MAX_K =
    IM2ROW_KERNEL * IM2ROW_KERNEL * IM2ROW_MAX_CHANNELS;
// P 路列 Bank 中每个 Bank 的最大地址深度，以及一个 Tile 覆盖的 P+2 个输入逻辑列。
static const int IM2ROW_MAX_BLOCKS_PER_ROW =
    (IM2ROW_MAX_W + IM2ROW_P - 1) / IM2ROW_P;
static const int IM2ROW_STRIP_WIDTH =
    IM2ROW_P + IM2ROW_KERNEL - 1;

// 首 Tile 的 Bottom 最多需要 P+1 个标量输入，必须在 Top/Middle 的6拍内读取完毕。
// 当前双 Strip 无气泡调度因此支持 1<=P<=5；本工程默认 P=4。
static_assert(IM2ROW_P > 0 && IM2ROW_P <= 5,
              "当前 Im2Row 双 Strip 调度要求 1 <= IM2ROW_P <= 5");

// 对称八位有符号量化激活，数值0既是量化零点，也是卷积 Padding 的填充值。
typedef ap_int<8> activation_t;

// 一个 pixel_t 表示一个空间位置的全部物理输入通道。
// 运行时仅 channel[0...Cin-1] 有效；Im2Row 输出前会清零其余通道，便于 Systolic 固定宽度接收。
struct pixel_t {
    activation_t channel[IM2ROW_MAX_CHANNELS];
};

// activation_pack_t 是 Im2Row 到 Systolic 的基本流事务：P 个像素属于同一输出行、
// 同一 Tile 和同一个 (kh,kw)。Systolic 的接口目标是在每个成功接收周期消费一个 Pack。
// spatial_valid 只标记空间 Lane 是否真实存在，在同一 Tile 的9拍内保持不变；
// 卷积 Padding 对应的 Lane 仍有效，而行尾不存在的 Lane 为0且其 pixel 在9拍内均为全零。
struct activation_pack_t {
    pixel_t pixel[IM2ROW_P];
    ap_uint<IM2ROW_P> spatial_valid;
};

#endif  // CNN_ACCELERATOR_TYPES_H_
