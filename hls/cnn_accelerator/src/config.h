#ifndef CNN_ACCELERATOR_CONFIG_H_
#define CNN_ACCELERATOR_CONFIG_H_

// 旧 MNIST 模块与新 YOLO Im2Row 的共同配置来源。这里仅保存会影响
// 数组尺寸、循环边界和模块间事务数量的常量；数据类型与 stream pack 定义位于
// types.h。两层卷积由同一套物理模块按 layer_mode_t 分时执行。

// 编译期选择：1 使用 Output-stationary，0 使用 Broadcast。
#ifndef CNN_SYSTOLIC_USE_OS
#define CNN_SYSTOLIC_USE_OS 1
#endif
#if CNN_SYSTOLIC_USE_OS != 0 && CNN_SYSTOLIC_USE_OS != 1
#error "CNN_SYSTOLIC_USE_OS must be 0 or 1"
#endif

// 量化标量位宽。
constexpr int ACTIVATION_BITS = 8;
constexpr int WEIGHT_BITS = 8;
constexpr int ACCUMULATOR_BITS = 32;
constexpr int PRODUCT_BITS = ACTIVATION_BITS + WEIGHT_BITS;

// 固定物理宽度独立于网络逻辑 Cin/Cout。
constexpr int CONV_SPATIAL_PAR = 4;
constexpr int CONV_CIN_PAR = 8;
constexpr int CONV_COUT_PAR = 4;
constexpr int FEATURE_PACK_CHANNELS = 8;
constexpr int FC_CIN_PAR = 8;
constexpr int ceil_div(int n, int d) { return (n + d - 1) / d; }
constexpr int config_max(int a, int b) { return a > b ? a : b; }
// bits 必须包含 one-past 终点，而不仅是最后有效地址。
constexpr int bits_for_max(int value) {
    return value < 2 ? 1 : 1 + bits_for_max(value >> 1);
}

// 固定卷积几何参数。
constexpr int CNN_CONV1_H = 28;
constexpr int CNN_CONV1_W = 28;
constexpr int CNN_CONV1_CIN = 1;
constexpr int CNN_CONV1_COUT = 8;
constexpr int CNN_CONV2_H = 14;
constexpr int CNN_CONV2_W = 14;
constexpr int CNN_CONV2_CIN = 8;
constexpr int CNN_CONV2_COUT = 16;

constexpr int IM2ROW_KERNEL = 3;
constexpr int IM2ROW_PADDING = 1;
constexpr int IM2ROW_STRIDE = 1;
constexpr int IM2ROW_P = CONV_SPATIAL_PAR;
// 共用调用级几何结构；旧conv_geometry(bool)仅供尚未迁移的MNIST模块使用。
struct ConvGeometry {
    int h, w, cin, cout;
    int tiles_per_row, spatial_tiles, cout_tiles;
    int reduction_steps, im2row_tokens, weight_transactions;
    int last_tile_x, prelast_tile_x, last_bottom_loads;
    unsigned int tail_spatial_valid_mask, prelast_new_column_valid_mask;
    int kernel, padding, stride, cin_groups, input_transactions;
};
enum class YoloTinyLayer {
    Conv1, Conv2, Conv3, Conv4, Conv5, Conv6, Conv7, Conv8, Prediction, Count
};
constexpr int YOLO_TINY_LAYER_COUNT = static_cast<int>(YoloTinyLayer::Count);
// 只负责从固定层表的四个逻辑参数统一派生全部字段，不是任意配置的支持承诺。
// 使用单return形式保持C++11常量表达式兼容，供下方容量计算直接求值。
constexpr ConvGeometry make_yolo_geometry(int h, int cin, int cout, int kernel) {
    return ConvGeometry{
        h, h, cin, cout,
        ceil_div(h, IM2ROW_P), h * ceil_div(h, IM2ROW_P), ceil_div(cout, CONV_COUT_PAR),
        kernel * kernel * ceil_div(cin, CONV_CIN_PAR),
        h * ceil_div(h, IM2ROW_P) * kernel * kernel * ceil_div(cin, CONV_CIN_PAR),
        ceil_div(cout, CONV_COUT_PAR) * kernel * kernel * ceil_div(cin, CONV_CIN_PAR),
        ceil_div(h, IM2ROW_P) - 1, ceil_div(h, IM2ROW_P) - 2,
        config_max(0, h - ((ceil_div(h, IM2ROW_P) - 1) * IM2ROW_P + 1)),
        (1u << (((h - 1) % IM2ROW_P) + 1)) - 1u,
        (1u << config_max(0, h - ((ceil_div(h, IM2ROW_P) - 1) * IM2ROW_P + 1))) - 1u,
        kernel, kernel == 3 ? 1 : 0, 1, ceil_div(cin, CONV_CIN_PAR),
        h * h * ceil_div(cin, CONV_CIN_PAR)};
}
// 唯一YOLOv2-tiny VOC层表；Im2Row、几何构造器和物理容量均从此表派生。
constexpr ConvGeometry YOLO_TINY_GEOMETRIES[YOLO_TINY_LAYER_COUNT] = {
    make_yolo_geometry(416, 3, 16, 3),
    make_yolo_geometry(208, 16, 32, 3),
    make_yolo_geometry(104, 32, 64, 3),
    make_yolo_geometry(52, 64, 128, 3),
    make_yolo_geometry(26, 128, 256, 3),
    make_yolo_geometry(13, 256, 512, 3),
    make_yolo_geometry(13, 512, 1024, 3),
    make_yolo_geometry(13, 1024, 1024, 3),
    make_yolo_geometry(13, 1024, 125, 1)};
constexpr int yolo_max_field(int ConvGeometry::*field, int count = YOLO_TINY_LAYER_COUNT) {
    return count == 0 ? 0 : config_max(YOLO_TINY_GEOMETRIES[count - 1].*field,
                                     yolo_max_field(field, count - 1));
}
constexpr int yolo_max_line_depth(int count = YOLO_TINY_LAYER_COUNT) {
    return count == 0 ? 0 : config_max(
        YOLO_TINY_GEOMETRIES[count - 1].tiles_per_row * YOLO_TINY_GEOMETRIES[count - 1].cin_groups,
        yolo_max_line_depth(count - 1));
}
constexpr int IM2ROW_MAX_H = yolo_max_field(&ConvGeometry::h);
constexpr int IM2ROW_MAX_W = yolo_max_field(&ConvGeometry::w);
constexpr int IM2ROW_MAX_CIN = yolo_max_field(&ConvGeometry::cin);
constexpr int IM2ROW_MAX_GROUPS = yolo_max_field(&ConvGeometry::cin_groups);
// 逐层取ceil(W/P)*G的最大值，不能将最大W与最大G相乘。
constexpr int IM2ROW_LINE_BANK_DEPTH = yolo_max_line_depth();
constexpr int IM2ROW_STRIP_BANK_DEPTH = 2 * IM2ROW_MAX_GROUPS;
static_assert(IM2ROW_MAX_H == 416 && IM2ROW_MAX_W == 416 &&
              IM2ROW_MAX_CIN == 1024 && IM2ROW_MAX_GROUPS == 128 &&
              IM2ROW_LINE_BANK_DEPTH == 512 && IM2ROW_STRIP_BANK_DEPTH == 256,
              "YOLO Im2Row容量必须与冻结层表一致");
// 兼容旧源码与 testbench：这是物理 channel pack 宽度，不是逻辑最大 Cin。
constexpr int IM2ROW_MAX_CHANNELS = FEATURE_PACK_CHANNELS;
constexpr int IM2ROW_MAX_BLOCKS_PER_ROW =
    (IM2ROW_MAX_W + IM2ROW_P - 1) / IM2ROW_P;
constexpr int IM2ROW_STRIP_WIDTH = IM2ROW_P + IM2ROW_KERNEL - 1;

// Broadcast 与 Output-stationary Systolic 共用的阵列规模。
constexpr int SYSTOLIC_SPATIAL_PAR = CONV_SPATIAL_PAR;
constexpr int SYSTOLIC_CIN_PAR = CONV_CIN_PAR;
constexpr int SYSTOLIC_COUT_PAR = CONV_COUT_PAR;
constexpr int SYSTOLIC_KERNEL_TOKENS = IM2ROW_KERNEL * IM2ROW_KERNEL;
constexpr int SYSTOLIC_ACTIVATION_LANE_BITS =
    SYSTOLIC_CIN_PAR * ACTIVATION_BITS;
constexpr int SYSTOLIC_WEIGHT_LANE_BITS = SYSTOLIC_CIN_PAR * WEIGHT_BITS;
constexpr int SYSTOLIC_SUM2_BITS = PRODUCT_BITS + 1;
constexpr int SYSTOLIC_SUM4_BITS = PRODUCT_BITS + 2;
constexpr int SYSTOLIC_DOT_BITS = PRODUCT_BITS + 3;

// Output-stationary Systolic 波前步数。
constexpr int SYSTOLIC_WAVEFRONT_STEPS =
    (SYSTOLIC_KERNEL_TOKENS - 1) + (SYSTOLIC_COUT_PAR - 1) +
    (SYSTOLIC_SPATIAL_PAR - 1) + 1;

// 每层卷积由固定几何参数派生的stream事务形状。
constexpr int CNN_CONV1_TILES_PER_ROW =
    (CNN_CONV1_W + SYSTOLIC_SPATIAL_PAR - 1) / SYSTOLIC_SPATIAL_PAR;
constexpr int CNN_CONV2_TILES_PER_ROW =
    (CNN_CONV2_W + SYSTOLIC_SPATIAL_PAR - 1) / SYSTOLIC_SPATIAL_PAR;
constexpr int CNN_CONV1_SPATIAL_TILES =
    CNN_CONV1_H * CNN_CONV1_TILES_PER_ROW;
constexpr int CNN_CONV2_SPATIAL_TILES =
    CNN_CONV2_H * CNN_CONV2_TILES_PER_ROW;
constexpr int CNN_CONV1_COUT_TILES = CNN_CONV1_COUT / SYSTOLIC_COUT_PAR;
constexpr int CNN_CONV2_COUT_TILES = CNN_CONV2_COUT / SYSTOLIC_COUT_PAR;
constexpr int CNN_CONV1_INPUT_TRANSACTIONS = CNN_CONV1_H * CNN_CONV1_W;
constexpr int CNN_CONV2_INPUT_TRANSACTIONS = CNN_CONV2_H * CNN_CONV2_W;
constexpr int CNN_CONV1_IM2ROW_TOKENS =
    CNN_CONV1_SPATIAL_TILES * SYSTOLIC_KERNEL_TOKENS;
constexpr int CNN_CONV2_IM2ROW_TOKENS =
    CNN_CONV2_SPATIAL_TILES * SYSTOLIC_KERNEL_TOKENS;
constexpr int CNN_CONV1_WEIGHT_TRANSACTIONS =
    CNN_CONV1_COUT_TILES * SYSTOLIC_KERNEL_TOKENS;
constexpr int CNN_CONV2_WEIGHT_TRANSACTIONS =
    CNN_CONV2_COUT_TILES * SYSTOLIC_KERNEL_TOKENS;
constexpr int CNN_CONV1_PARAM_TRANSACTIONS = CNN_CONV1_COUT_TILES;
constexpr int CNN_CONV2_PARAM_TRANSACTIONS = CNN_CONV2_COUT_TILES;
constexpr int CNN_CONV1_LAST_TILE_X = CNN_CONV1_TILES_PER_ROW - 1;
constexpr int CNN_CONV2_LAST_TILE_X = CNN_CONV2_TILES_PER_ROW - 1;
constexpr int CNN_CONV1_PRELAST_TILE_X = CNN_CONV1_LAST_TILE_X - 1;
constexpr int CNN_CONV2_PRELAST_TILE_X = CNN_CONV2_LAST_TILE_X - 1;
constexpr unsigned int CNN_FULL_SPATIAL_VALID_MASK =
    (1u << SYSTOLIC_SPATIAL_PAR) - 1u;
// ((W - 1) % P) + 1 在W恰好整除P时也会给出P个有效lane，避免tail mask
// 退化为零；对当前Conv1为4，对Conv2为2。
constexpr int CNN_CONV1_TAIL_LANES =
    ((CNN_CONV1_W - 1) % SYSTOLIC_SPATIAL_PAR) + 1;
constexpr int CNN_CONV2_TAIL_LANES =
    ((CNN_CONV2_W - 1) % SYSTOLIC_SPATIAL_PAR) + 1;
constexpr unsigned int CNN_CONV1_TAIL_SPATIAL_VALID_MASK =
    (1u << CNN_CONV1_TAIL_LANES) - 1u;
constexpr unsigned int CNN_CONV2_TAIL_SPATIAL_VALID_MASK =
    (1u << CNN_CONV2_TAIL_LANES) - 1u;
constexpr int CNN_CONV1_LAST_BOTTOM_LOADS =
    CNN_CONV1_W - (CNN_CONV1_LAST_TILE_X * IM2ROW_P + 1);
constexpr int CNN_CONV2_LAST_BOTTOM_LOADS =
    CNN_CONV2_W - (CNN_CONV2_LAST_TILE_X * IM2ROW_P + 1);
constexpr unsigned int CNN_CONV1_PRELAST_NEW_COLUMN_VALID_MASK =
    (1u << CNN_CONV1_LAST_BOTTOM_LOADS) - 1u;
constexpr unsigned int CNN_CONV2_PRELAST_NEW_COLUMN_VALID_MASK =
    (1u << CNN_CONV2_LAST_BOTTOM_LOADS) - 1u;
constexpr int CNN_CONV1_CIN_GROUPS = ceil_div(CNN_CONV1_CIN, CONV_CIN_PAR);
constexpr int CNN_CONV2_CIN_GROUPS = ceil_div(CNN_CONV2_CIN, CONV_CIN_PAR);
constexpr int CNN_CONV1_REDUCTION_STEPS = SYSTOLIC_KERNEL_TOKENS * CNN_CONV1_CIN_GROUPS;
constexpr int CNN_CONV2_REDUCTION_STEPS = SYSTOLIC_KERNEL_TOKENS * CNN_CONV2_CIN_GROUPS;
constexpr int SYSTOLIC_ACTIVATION_REPLAY_CAPACITY =
    config_max(CNN_CONV1_REDUCTION_STEPS, CNN_CONV2_REDUCTION_STEPS);
constexpr int SYSTOLIC_MAX_COUT_TILES =
    config_max(CNN_CONV1_COUT_TILES, CNN_CONV2_COUT_TILES);
constexpr int SYSTOLIC_WEIGHT_BUFFER_DEPTH =
    config_max(CNN_CONV1_WEIGHT_TRANSACTIONS, CNN_CONV2_WEIGHT_TRANSACTIONS);
constexpr int SYSTOLIC_WEIGHT_ADDRESS_BITS = bits_for_max(SYSTOLIC_WEIGHT_BUFFER_DEPTH);

// 固定2x2、stride-2 MaxPool几何参数。输入使用卷积输出Tile，输出为紧密
// NHWC channel pack。
constexpr int POOL_KERNEL = 2;
constexpr int POOL_STRIDE = 2;
constexpr int POOL_HALF_BITS = SYSTOLIC_COUT_PAR * ACTIVATION_BITS;
constexpr int CNN_POOL1_H = CNN_CONV1_H / POOL_STRIDE;
constexpr int CNN_POOL1_W = CNN_CONV1_W / POOL_STRIDE;
constexpr int CNN_POOL1_C = CNN_CONV1_COUT;
constexpr int CNN_POOL2_H = CNN_CONV2_H / POOL_STRIDE;
constexpr int CNN_POOL2_W = CNN_CONV2_W / POOL_STRIDE;
constexpr int CNN_POOL2_C = CNN_CONV2_COUT;
constexpr int CNN_POOL1_CHANNEL_PACKS = CNN_POOL1_C / IM2ROW_MAX_CHANNELS;
constexpr int CNN_POOL2_CHANNEL_PACKS = CNN_POOL2_C / IM2ROW_MAX_CHANNELS;
constexpr int CNN_POOL1_INPUT_TRANSACTIONS =
    CNN_CONV1_SPATIAL_TILES * CNN_CONV1_COUT_TILES;
constexpr int CNN_POOL2_INPUT_TRANSACTIONS =
    CNN_CONV2_SPATIAL_TILES * CNN_CONV2_COUT_TILES;
constexpr int CNN_POOL1_OUTPUT_TRANSACTIONS =
    CNN_POOL1_H * CNN_POOL1_W * CNN_POOL1_CHANNEL_PACKS;
constexpr int CNN_POOL2_OUTPUT_TRANSACTIONS =
    CNN_POOL2_H * CNN_POOL2_W * CNN_POOL2_CHANNEL_PACKS;
constexpr int POOL_LINE_BUFFER_DEPTH =
    (CNN_CONV1_TILES_PER_ROW * CNN_CONV1_COUT_TILES >
     CNN_CONV2_TILES_PER_ROW * CNN_CONV2_COUT_TILES)
        ? CNN_CONV1_TILES_PER_ROW * CNN_CONV1_COUT_TILES
        : CNN_CONV2_TILES_PER_ROW * CNN_CONV2_COUT_TILES;
constexpr int POOL_MAX_TILES_PER_ROW =
    (CNN_CONV1_TILES_PER_ROW > CNN_CONV2_TILES_PER_ROW)
        ? CNN_CONV1_TILES_PER_ROW : CNN_CONV2_TILES_PER_ROW;
constexpr int POOL_MAX_COUT_TILES = SYSTOLIC_MAX_COUT_TILES;

// 固定FC 784 -> 10；输入为紧密的Pool2 NHWC stream。
constexpr int FC_COUT_PAR = 5;
constexpr int FC_OUTPUTS = 10;
constexpr int SOFTMAX_SCALE_FRAC_BITS = 20;
constexpr int SOFTMAX_PROBABILITY_SCALE = 10000;
constexpr int SOFTMAX_EXP_BITS = 21;
constexpr int SOFTMAX_SUM_BITS = 24;
constexpr int FC_INPUT_DIM = CNN_POOL2_H * CNN_POOL2_W * CNN_POOL2_C;
constexpr int FC_INPUT_PACKS = FC_INPUT_DIM / FC_CIN_PAR;
constexpr int FC_OUT_TILES = FC_OUTPUTS / FC_COUT_PAR;
constexpr int FC_ACTIVATION_BUFFER_DEPTH = FC_INPUT_PACKS;
constexpr int FC_ACTIVATION_WORD_BITS = FC_CIN_PAR * ACTIVATION_BITS;
constexpr int FC_SUM2_BITS = PRODUCT_BITS + 1;
constexpr int FC_SUM4_BITS = PRODUCT_BITS + 2;
constexpr int FC_DOT_BITS = PRODUCT_BITS + 3;

// Top 固定执行 Conv1 与 Conv2 两个顺序阶段。层内 hls::stream 均显式采用
// 同一初始FIFO深度；Pool1阶段间FIFO必须容纳完整的Pool1输出。
constexpr int TOP_PHASE_COUNT = 2;
constexpr int TOP_STREAM_DEPTH = 2;
constexpr int TOP_POOL1_FEATURE_FIFO_DEPTH =
    CNN_POOL1_OUTPUT_TRANSACTIONS;

// 调用级模式，不是per-token字段。卷积模块把LAYER2_MODE解释为Conv2，
// MaxPool把它解释为Pool2。
using layer_mode_t = bool;
constexpr layer_mode_t LAYER1_MODE = false;
constexpr layer_mode_t LAYER2_MODE = true;


// 旧消费者的字段和值不变，新增Im2Row字段显式置零，标记尚未迁移；
// 该构造器的返回值不满足新Im2Row的YOLO调用契约。
inline ConvGeometry conv_geometry(layer_mode_t mode) {
#pragma HLS INLINE
    return mode == LAYER2_MODE
        ? ConvGeometry{CNN_CONV2_H, CNN_CONV2_W, CNN_CONV2_CIN, CNN_CONV2_COUT,
            CNN_CONV2_TILES_PER_ROW, CNN_CONV2_SPATIAL_TILES, CNN_CONV2_COUT_TILES,
            CNN_CONV2_REDUCTION_STEPS, CNN_CONV2_IM2ROW_TOKENS, CNN_CONV2_WEIGHT_TRANSACTIONS,
            CNN_CONV2_LAST_TILE_X, CNN_CONV2_PRELAST_TILE_X, CNN_CONV2_LAST_BOTTOM_LOADS,
            CNN_CONV2_TAIL_SPATIAL_VALID_MASK, CNN_CONV2_PRELAST_NEW_COLUMN_VALID_MASK,
            0, 0, 0, 0, 0}
        : ConvGeometry{CNN_CONV1_H, CNN_CONV1_W, CNN_CONV1_CIN, CNN_CONV1_COUT,
            CNN_CONV1_TILES_PER_ROW, CNN_CONV1_SPATIAL_TILES, CNN_CONV1_COUT_TILES,
            CNN_CONV1_REDUCTION_STEPS, CNN_CONV1_IM2ROW_TOKENS, CNN_CONV1_WEIGHT_TRANSACTIONS,
            CNN_CONV1_LAST_TILE_X, CNN_CONV1_PRELAST_TILE_X, CNN_CONV1_LAST_BOTTOM_LOADS,
            CNN_CONV1_TAIL_SPATIAL_VALID_MASK, CNN_CONV1_PRELAST_NEW_COLUMN_VALID_MASK,
            0, 0, 0, 0, 0};
}
// 表外枚举返回空配置，仅为避免非法索引；实际调用必须使用表内层枚举。
constexpr ConvGeometry yolov2_tiny_geometry(YoloTinyLayer layer) {
    return static_cast<int>(layer) >= 0 && static_cast<int>(layer) < YOLO_TINY_LAYER_COUNT
        ? YOLO_TINY_GEOMETRIES[static_cast<int>(layer)] : ConvGeometry{};
}
#ifndef __SYNTHESIS__
// 仅主机静态/调试构建使用的严格契约校验，不综合比较器、循环或拒绝分支。
inline bool im2row_geometry_supported(const ConvGeometry& g) {
    for (int layer = 0; layer < YOLO_TINY_LAYER_COUNT; ++layer) {
        const ConvGeometry e = yolov2_tiny_geometry(static_cast<YoloTinyLayer>(layer));
        if (g.h == e.h && g.w == e.w && g.cin == e.cin && g.cout == e.cout &&
            g.kernel == e.kernel && g.padding == e.padding && g.stride == e.stride &&
            g.cin_groups == e.cin_groups && g.tiles_per_row == e.tiles_per_row &&
            g.spatial_tiles == e.spatial_tiles && g.cout_tiles == e.cout_tiles &&
            g.reduction_steps == e.reduction_steps && g.im2row_tokens == e.im2row_tokens &&
            g.weight_transactions == e.weight_transactions &&
            g.input_transactions == e.input_transactions && g.last_tile_x == e.last_tile_x &&
            g.prelast_tile_x == e.prelast_tile_x && g.last_bottom_loads == e.last_bottom_loads &&
            g.tail_spatial_valid_mask == e.tail_spatial_valid_mask &&
            g.prelast_new_column_valid_mask == e.prelast_new_column_valid_mask) return true;
    }
    return false;
}
#endif
static_assert(CNN_CONV1_CIN_GROUPS == 1 && CNN_CONV2_CIN_GROUPS == 1,
              "Legacy MNIST consumers still implement one Cin group");
static_assert(IM2ROW_KERNEL == 3 && IM2ROW_PADDING == 1 && IM2ROW_STRIDE == 1,
              "Fixed nine-case Im2Row scheduler requires 3x3 same convolution");
static_assert(CONV_SPATIAL_PAR == 4 && CONV_COUT_PAR == 4,
              "Fixed bank selectors and OS skew require 4x4");
static_assert(CONV_CIN_PAR == 8 && FC_CIN_PAR == 8 &&
              FEATURE_PACK_CHANNELS == CONV_CIN_PAR,
              "Fixed dot8 and feature packs require eight channels");
static_assert(POOL_KERNEL == 2 && POOL_STRIDE == 2 &&
              2 * CONV_COUT_PAR == FEATURE_PACK_CHANNELS,
              "Pool combines two four-channel halves");

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
