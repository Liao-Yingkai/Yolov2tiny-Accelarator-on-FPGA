#ifndef CNN_ACCELERATOR_CONFIG_H_
#define CNN_ACCELERATOR_CONFIG_H_

// 固定 MNIST 网络与本分支硬件规模的唯一编译期配置来源。这里仅保存会影响
// 数组尺寸、循环边界和模块间事务数量的常量；数据类型与 stream pack 定义位于
// types.h。两层卷积由同一套物理模块按 layer_mode_t 分时执行。

// 量化标量位宽。
static const int ACTIVATION_BITS = 8;
static const int WEIGHT_BITS = 8;
static const int ACCUMULATOR_BITS = 32;
static const int PRODUCT_BITS = ACTIVATION_BITS + WEIGHT_BITS;

// 固定卷积几何参数。
static const int CNN_CONV1_H = 28;
static const int CNN_CONV1_W = 28;
static const int CNN_CONV1_CIN = 1;
static const int CNN_CONV1_COUT = 8;
static const int CNN_CONV2_H = 14;
static const int CNN_CONV2_W = 14;
static const int CNN_CONV2_CIN = 8;
static const int CNN_CONV2_COUT = 16;

static const int IM2ROW_KERNEL = 3;
static const int IM2ROW_PADDING = 1;
static const int IM2ROW_STRIDE = 1;
static const int IM2ROW_P = 4;
static const int IM2ROW_MAX_H = CNN_CONV1_H;
static const int IM2ROW_MAX_W = CNN_CONV1_W;
static const int IM2ROW_MAX_CHANNELS = CNN_CONV2_CIN;
static const int IM2ROW_MAX_BLOCKS_PER_ROW =
    (IM2ROW_MAX_W + IM2ROW_P - 1) / IM2ROW_P;
static const int IM2ROW_STRIP_WIDTH = IM2ROW_P + IM2ROW_KERNEL - 1;

// Broadcast Systolic 阵列维度。
static const int SYSTOLIC_SPATIAL_PAR = IM2ROW_P;
static const int SYSTOLIC_CIN_PAR = IM2ROW_MAX_CHANNELS;
static const int SYSTOLIC_COUT_PAR = IM2ROW_P;
static const int SYSTOLIC_KERNEL_TOKENS = IM2ROW_KERNEL * IM2ROW_KERNEL;
static const int SYSTOLIC_ACTIVATION_LANE_BITS =
    SYSTOLIC_CIN_PAR * ACTIVATION_BITS;
static const int SYSTOLIC_WEIGHT_LANE_BITS = SYSTOLIC_CIN_PAR * WEIGHT_BITS;
static const int SYSTOLIC_SUM2_BITS = PRODUCT_BITS + 1;
static const int SYSTOLIC_SUM4_BITS = PRODUCT_BITS + 2;
static const int SYSTOLIC_DOT_BITS = PRODUCT_BITS + 3;

//Output-stationary Systolic
static const int SYSTOLIC_EXCU_TIME = (SYSTOLIC_KERNEL_TOKENS-1) + (SYSTOLIC_COUT_PAR-1) + (SYSTOLIC_SPATIAL_PAR-1) + 1;

// 每层卷积由固定几何参数派生的stream事务形状。
static const int CNN_CONV1_TILES_PER_ROW =
    (CNN_CONV1_W + SYSTOLIC_SPATIAL_PAR - 1) / SYSTOLIC_SPATIAL_PAR;
static const int CNN_CONV2_TILES_PER_ROW =
    (CNN_CONV2_W + SYSTOLIC_SPATIAL_PAR - 1) / SYSTOLIC_SPATIAL_PAR;
static const int CNN_CONV1_SPATIAL_TILES =
    CNN_CONV1_H * CNN_CONV1_TILES_PER_ROW;
static const int CNN_CONV2_SPATIAL_TILES =
    CNN_CONV2_H * CNN_CONV2_TILES_PER_ROW;
static const int CNN_CONV1_COUT_TILES = CNN_CONV1_COUT / SYSTOLIC_COUT_PAR;
static const int CNN_CONV2_COUT_TILES = CNN_CONV2_COUT / SYSTOLIC_COUT_PAR;
static const int CNN_CONV1_INPUT_TRANSACTIONS = CNN_CONV1_H * CNN_CONV1_W;
static const int CNN_CONV2_INPUT_TRANSACTIONS = CNN_CONV2_H * CNN_CONV2_W;
static const int CNN_CONV1_IM2ROW_TOKENS =
    CNN_CONV1_SPATIAL_TILES * SYSTOLIC_KERNEL_TOKENS;
static const int CNN_CONV2_IM2ROW_TOKENS =
    CNN_CONV2_SPATIAL_TILES * SYSTOLIC_KERNEL_TOKENS;
static const int CNN_CONV1_WEIGHT_TRANSACTIONS =
    CNN_CONV1_COUT_TILES * SYSTOLIC_KERNEL_TOKENS;
static const int CNN_CONV2_WEIGHT_TRANSACTIONS =
    CNN_CONV2_COUT_TILES * SYSTOLIC_KERNEL_TOKENS;
static const int CNN_CONV1_PARAM_TRANSACTIONS = CNN_CONV1_COUT_TILES;
static const int CNN_CONV2_PARAM_TRANSACTIONS = CNN_CONV2_COUT_TILES;
static const int CNN_CONV1_LAST_TILE_X = CNN_CONV1_TILES_PER_ROW - 1;
static const int CNN_CONV2_LAST_TILE_X = CNN_CONV2_TILES_PER_ROW - 1;
static const int CNN_CONV1_PRELAST_TILE_X = CNN_CONV1_LAST_TILE_X - 1;
static const int CNN_CONV2_PRELAST_TILE_X = CNN_CONV2_LAST_TILE_X - 1;
static const unsigned int CNN_FULL_SPATIAL_VALID_MASK =
    (1u << SYSTOLIC_SPATIAL_PAR) - 1u;
// ((W - 1) % P) + 1 在W恰好整除P时也会给出P个有效lane，避免tail mask
// 退化为零；对当前Conv1为4，对Conv2为2。
static const int CNN_CONV1_TAIL_LANES =
    ((CNN_CONV1_W - 1) % SYSTOLIC_SPATIAL_PAR) + 1;
static const int CNN_CONV2_TAIL_LANES =
    ((CNN_CONV2_W - 1) % SYSTOLIC_SPATIAL_PAR) + 1;
static const unsigned int CNN_CONV1_TAIL_SPATIAL_VALID_MASK =
    (1u << CNN_CONV1_TAIL_LANES) - 1u;
static const unsigned int CNN_CONV2_TAIL_SPATIAL_VALID_MASK =
    (1u << CNN_CONV2_TAIL_LANES) - 1u;
static const int CNN_CONV1_LAST_BOTTOM_LOADS =
    CNN_CONV1_W - (CNN_CONV1_LAST_TILE_X * IM2ROW_P + 1);
static const int CNN_CONV2_LAST_BOTTOM_LOADS =
    CNN_CONV2_W - (CNN_CONV2_LAST_TILE_X * IM2ROW_P + 1);
static const unsigned int CNN_CONV1_PRELAST_NEW_COLUMN_VALID_MASK =
    (1u << CNN_CONV1_LAST_BOTTOM_LOADS) - 1u;
static const unsigned int CNN_CONV2_PRELAST_NEW_COLUMN_VALID_MASK =
    (1u << CNN_CONV2_LAST_BOTTOM_LOADS) - 1u;
static const int SYSTOLIC_MAX_COUT_TILES = CNN_CONV2_COUT_TILES;
static const int SYSTOLIC_WEIGHT_BUFFER_DEPTH =
    SYSTOLIC_MAX_COUT_TILES * SYSTOLIC_KERNEL_TOKENS;

// 固定2x2、stride-2 MaxPool几何参数。输入使用卷积输出Tile，输出为紧密
// NHWC channel pack。
static const int POOL_KERNEL = 2;
static const int POOL_STRIDE = 2;
static const int POOL_HALF_BITS = SYSTOLIC_COUT_PAR * ACTIVATION_BITS;
static const int CNN_POOL1_H = CNN_CONV1_H / POOL_STRIDE;
static const int CNN_POOL1_W = CNN_CONV1_W / POOL_STRIDE;
static const int CNN_POOL1_C = CNN_CONV1_COUT;
static const int CNN_POOL2_H = CNN_CONV2_H / POOL_STRIDE;
static const int CNN_POOL2_W = CNN_CONV2_W / POOL_STRIDE;
static const int CNN_POOL2_C = CNN_CONV2_COUT;
static const int CNN_POOL1_CHANNEL_PACKS = CNN_POOL1_C / IM2ROW_MAX_CHANNELS;
static const int CNN_POOL2_CHANNEL_PACKS = CNN_POOL2_C / IM2ROW_MAX_CHANNELS;
static const int CNN_POOL1_INPUT_TRANSACTIONS =
    CNN_CONV1_SPATIAL_TILES * CNN_CONV1_COUT_TILES;
static const int CNN_POOL2_INPUT_TRANSACTIONS =
    CNN_CONV2_SPATIAL_TILES * CNN_CONV2_COUT_TILES;
static const int CNN_POOL1_OUTPUT_TRANSACTIONS =
    CNN_POOL1_H * CNN_POOL1_W * CNN_POOL1_CHANNEL_PACKS;
static const int CNN_POOL2_OUTPUT_TRANSACTIONS =
    CNN_POOL2_H * CNN_POOL2_W * CNN_POOL2_CHANNEL_PACKS;
static const int POOL_LINE_BUFFER_DEPTH =
    (CNN_CONV1_TILES_PER_ROW * CNN_CONV1_COUT_TILES >
     CNN_CONV2_TILES_PER_ROW * CNN_CONV2_COUT_TILES)
        ? CNN_CONV1_TILES_PER_ROW * CNN_CONV1_COUT_TILES
        : CNN_CONV2_TILES_PER_ROW * CNN_CONV2_COUT_TILES;
static const int POOL_MAX_TILES_PER_ROW =
    (CNN_CONV1_TILES_PER_ROW > CNN_CONV2_TILES_PER_ROW)
        ? CNN_CONV1_TILES_PER_ROW : CNN_CONV2_TILES_PER_ROW;
static const int POOL_MAX_COUT_TILES = SYSTOLIC_MAX_COUT_TILES;

// 固定FC 784 -> 10；输入为紧密的Pool2 NHWC stream。
static const int FC_CIN_PAR = IM2ROW_MAX_CHANNELS;
static const int FC_COUT_PAR = 5;
static const int FC_OUTPUTS = 10;
static const int FC_INPUT_DIM = CNN_POOL2_H * CNN_POOL2_W * CNN_POOL2_C;
static const int FC_INPUT_PACKS = FC_INPUT_DIM / FC_CIN_PAR;
static const int FC_OUT_TILES = FC_OUTPUTS / FC_COUT_PAR;
static const int FC_ACTIVATION_BUFFER_DEPTH = FC_INPUT_PACKS;
static const int FC_ACTIVATION_WORD_BITS = FC_CIN_PAR * ACTIVATION_BITS;
static const int FC_SUM2_BITS = PRODUCT_BITS + 1;
static const int FC_SUM4_BITS = PRODUCT_BITS + 2;
static const int FC_DOT_BITS = PRODUCT_BITS + 3;

// Top 固定执行 Conv1 与 Conv2 两个顺序阶段。层内 hls::stream 均显式采用
// 同一初始FIFO深度；Pool1阶段间FIFO必须容纳完整的Pool1输出。
static const int TOP_PHASE_COUNT = 2;
static const int TOP_STREAM_DEPTH = 2;
static const int TOP_POOL1_FEATURE_FIFO_DEPTH =
    CNN_POOL1_OUTPUT_TRANSACTIONS;

// 调用级模式，不是per-token字段。卷积模块把LAYER2_MODE解释为Conv2，
// MaxPool把它解释为Pool2。
typedef bool layer_mode_t;
static const layer_mode_t LAYER1_MODE = false;
static const layer_mode_t LAYER2_MODE = true;

static_assert(CNN_CONV1_COUT % SYSTOLIC_COUT_PAR == 0,
              "Conv1 Cout must fill systolic output tiles");
static_assert(CNN_CONV2_COUT % SYSTOLIC_COUT_PAR == 0,
              "Conv2 Cout must fill systolic output tiles");
static_assert(CNN_CONV1_H % POOL_STRIDE == 0 &&
              CNN_CONV1_W % POOL_STRIDE == 0 &&
              CNN_CONV2_H % POOL_STRIDE == 0 &&
              CNN_CONV2_W % POOL_STRIDE == 0,
              "Pool input dimensions must be divisible by stride");
static_assert(CNN_POOL1_C % IM2ROW_MAX_CHANNELS == 0 &&
              CNN_POOL2_C % IM2ROW_MAX_CHANNELS == 0,
              "Pool output channels must fill channel packs");
static_assert(POOL_HALF_BITS * 2 == IM2ROW_MAX_CHANNELS * ACTIVATION_BITS,
              "Two pooled halves must compose one channel pack");
static_assert(CNN_CONV1_TAIL_LANES >= 1 &&
              CNN_CONV1_TAIL_LANES <= SYSTOLIC_SPATIAL_PAR &&
              CNN_CONV2_TAIL_LANES >= 1 &&
              CNN_CONV2_TAIL_LANES <= SYSTOLIC_SPATIAL_PAR,
              "Tail lane count must fit spatial parallelism");
static_assert(FC_OUTPUTS % FC_COUT_PAR == 0,
              "FC outputs must fill FC output tiles");
static_assert(FC_INPUT_DIM == FC_INPUT_PACKS * FC_CIN_PAR,
              "Pool2 flatten must fill FC input packs");
static_assert(CNN_POOL2_H * CNN_POOL2_W * CNN_POOL2_C == FC_INPUT_DIM,
              "Pool2 flatten must equal FC input dimension");
static_assert(SYSTOLIC_WEIGHT_BUFFER_DEPTH >=
              CNN_CONV1_COUT_TILES * SYSTOLIC_KERNEL_TOKENS &&
              SYSTOLIC_WEIGHT_BUFFER_DEPTH >=
              CNN_CONV2_COUT_TILES * SYSTOLIC_KERNEL_TOKENS,
              "Weight buffer must hold either convolution layer");
static_assert(POOL_LINE_BUFFER_DEPTH >=
              CNN_CONV1_TILES_PER_ROW * CNN_CONV1_COUT_TILES &&
              POOL_LINE_BUFFER_DEPTH >=
              CNN_CONV2_TILES_PER_ROW * CNN_CONV2_COUT_TILES,
              "Pool line buffer must hold one complete input row");
static_assert(TOP_PHASE_COUNT == 2,
              "Top must execute exactly the two fixed convolution phases");
static_assert(TOP_STREAM_DEPTH >= 2,
              "Top FIFO depth must preserve the selected initial setting");
static_assert(TOP_POOL1_FEATURE_FIFO_DEPTH == CNN_POOL1_OUTPUT_TRANSACTIONS &&
              TOP_POOL1_FEATURE_FIFO_DEPTH == CNN_CONV2_INPUT_TRANSACTIONS,
              "Pool1 FIFO must exactly bridge Pool1 output and Conv2 input");

#endif  // CNN_ACCELERATOR_CONFIG_H_
