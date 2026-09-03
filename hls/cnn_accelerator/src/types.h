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
// 独立保留位宽常量，供模块内部构造完整通道打包字，避免重复硬编码。
static const int ACTIVATION_BITS = 8;
typedef ap_int<ACTIVATION_BITS> activation_t;

// Systolic 第一版固定使用 8 位有符号权重和 32 位有符号累加器。虽然当前
// IM2ROW_P 与输出通道并行度都为4，但二者在概念上独立，避免后续将空间
// 并行度变化无意传递为 Cout 并行度变化。
typedef ap_int<ACTIVATION_BITS> weight_t;
typedef ap_int<32> acc_t;

static const int SYSTOLIC_SPATIAL_PAR = IM2ROW_P;
static const int SYSTOLIC_COUT_PAR = 4;
static const int SYSTOLIC_CIN_PAR = IM2ROW_MAX_CHANNELS;
static const int SYSTOLIC_MAX_COUT = 16;
static const int SYSTOLIC_KERNEL_TOKENS = IM2ROW_KERNEL * IM2ROW_KERNEL;
static const int SYSTOLIC_MAX_COUT_TILES =
    SYSTOLIC_MAX_COUT / SYSTOLIC_COUT_PAR;
static const int SYSTOLIC_WEIGHT_BUFFER_DEPTH =
    SYSTOLIC_MAX_COUT_TILES * SYSTOLIC_KERNEL_TOKENS;

static_assert(SYSTOLIC_SPATIAL_PAR == 4,
              "第一版 Systolic 固定要求4个空间Lane");
static_assert(SYSTOLIC_COUT_PAR == 4,
              "第一版 Systolic 固定要求4个Cout Lane");
static_assert(SYSTOLIC_CIN_PAR == 8,
              "第一版 Systolic 固定要求8个物理输入通道");

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

// 一个 weight_lane_t 对应当前 Cout Tile 中的一条输出通道，并固定打包8个
// 物理输入通道。Conv1 由权重导出端将 channel[1..7] 填零。
struct weight_lane_t {
    weight_t channel[SYSTOLIC_CIN_PAR];
};

// Weight stream 的一个事务对应相同 (cout_tile, kh, kw) 下的全部4条 Cout
// Lane。其 stream 顺序为 cout_tile -> token(kh*3+kw)。
struct weight_pack_t {
    weight_lane_t lane[SYSTOLIC_COUT_PAR];
};

// Systolic 输出的一个事务包含4个相邻空间位置和当前4条输出通道的 INT32
// 累加结果；空间尾 Lane 的有效性不由数值本身表达，必须继续携带该掩码。
struct acc_tile_t {
    acc_t value[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
    ap_uint<SYSTOLIC_SPATIAL_PAR> spatial_valid;
};

#endif  // CNN_ACCELERATOR_TYPES_H_
