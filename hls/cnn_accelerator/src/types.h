#ifndef CNN_ACCELERATOR_TYPES_H_
#define CNN_ACCELERATOR_TYPES_H_

#include <ap_int.h>
#include <hls_stream.h>

// 本分支针对固定 MNIST 网络综合：不再通过编译宏生成不同硬件规模。
// P 是每个 Pack 的空间 Lane 数；Cin 物理宽度固定为8，Conv1 仅使用 channel0。
static const int IM2ROW_P = 4;
static const int IM2ROW_MAX_H = 28;
static const int IM2ROW_MAX_W = 28;
static const int IM2ROW_MAX_CHANNELS = 8;

// 唯一允许的两组运行时接口配置。接口保留 H/W/Cin/Cout，以便同一物理模块
// 按层分时复用；综合态仅由 Cin 选择模式，调用方必须提供下列合法组合。
static const int CNN_CONV1_H = 28;
static const int CNN_CONV1_W = 28;
static const int CNN_CONV1_CIN = 1;
static const int CNN_CONV1_COUT = 8;
static const int CNN_CONV2_H = 14;
static const int CNN_CONV2_W = 14;
static const int CNN_CONV2_CIN = 8;
static const int CNN_CONV2_COUT = 16;

// 固定卷积窗口为3×3、Padding=1、Stride=1。
static const int IM2ROW_KERNEL = 3;
static const int IM2ROW_PADDING = 1;
static const int IM2ROW_STRIDE = 1;
// P=4 时，每个列 Bank 深度固定为 ceil(28/4)=7；每个 Strip 覆盖4+2列。
static const int IM2ROW_MAX_BLOCKS_PER_ROW = 7;
static const int IM2ROW_STRIP_WIDTH = 6;

// 对称八位有符号量化激活，数值0既是量化零点，也是卷积 Padding 的填充值。
// 独立保留位宽常量，供模块内部构造完整通道打包字，避免重复硬编码。
static const int ACTIVATION_BITS = 8;
typedef ap_int<ACTIVATION_BITS> activation_t;

// Systolic 第一版固定使用 8 位有符号权重和 32 位有符号累加器。虽然当前
// IM2ROW_P 与输出通道并行度都为4，但二者在概念上独立，避免后续将空间
// 并行度变化无意传递为 Cout 并行度变化。
typedef ap_int<ACTIVATION_BITS> weight_t;
typedef ap_int<32> acc_t;

static const int SYSTOLIC_SPATIAL_PAR = 4;
static const int SYSTOLIC_COUT_PAR = 4;
static const int SYSTOLIC_CIN_PAR = 8;
static const int SYSTOLIC_KERNEL_TOKENS = IM2ROW_KERNEL * IM2ROW_KERNEL;
static const int SYSTOLIC_MAX_COUT_TILES = 4;
static const int SYSTOLIC_WEIGHT_BUFFER_DEPTH = 36;

// 一个 pixel_t 表示一个空间位置的全部8个物理输入通道。
// Im2Row 在输入读取时完成规范化：Conv1 仅 channel0 有效，Conv2 使用全部通道。
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

// PostProcess 输出保持与 acc_tile_t 相同的 spatial_tile -> cout_tile 事务
// 粒度，只将每个 INT32 累加结果 requantize 为 INT8 activation。尾部无效
// spatial lane 的 value 必须为零，不能由 bias 生成伪有效数据。
struct activation_tile_t {
    activation_t value[SYSTOLIC_SPATIAL_PAR][SYSTOLIC_COUT_PAR];
    ap_uint<SYSTOLIC_SPATIAL_PAR> spatial_valid;
};

// 一个参数 Pack 对应一个 Cout tile 的4条输出通道。所有字段都是已经由
// exporter 离线生成的有符号 INT32：bias 位于 accumulator 域，multiplier 与
// shift 共同表示 real_multiplier ~= multiplier / 2^shift。
struct postprocess_param_pack_t {
    ap_int<32> bias[SYSTOLIC_COUT_PAR];
    ap_int<32> multiplier[SYSTOLIC_COUT_PAR];
    ap_int<32> shift[SYSTOLIC_COUT_PAR];
};

#endif  // CNN_ACCELERATOR_TYPES_H_
