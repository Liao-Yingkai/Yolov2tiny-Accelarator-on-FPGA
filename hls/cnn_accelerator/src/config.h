#ifndef CNN_ACCELERATOR_CONFIG_H_
#define CNN_ACCELERATOR_CONFIG_H_

// YOLO固定层表与尚未迁移的MNIST模块共用的配置文件。这里只定义形状、
// 并行度、缓存容量和事务数量；数据类型与stream事务结构位于types.h。
// YOLO按层枚举选配置；旧layer_mode_t接口仅留给尚未迁移的MNIST模块。

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
constexpr int config_min(int a, int b) { return a < b ? a : b; }
// 固定YOLO层表的G均为2的幂；只在编译期生成移位量。
constexpr int config_log2(int n) { return n <= 1 ? 0 : 1 + config_log2(n >> 1); }
// bits 必须包含 one-past 终点，而不仅是最后有效地址。
constexpr int bits_for_max(int value) {
    return value < 2 ? 1 : 1 + bits_for_max(value >> 1);
}

// Legacy MNIST参数：仅供尚未迁移的模块使用，下面的YOLO层表不引用它们。
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
constexpr int IM2ROW_FIRST_TILE_INPUT_COLUMNS = IM2ROW_P + IM2ROW_PADDING;  // 3x3首Tile固定读取的真实输入列数。
// 共用调用级几何结构；旧conv_geometry(bool)仅供尚未迁移的MNIST模块使用。
struct ConvGeometry {
    int h;  // 输入与stride-1卷积输出的行数，不含Pool。
    int w;  // 输入与stride-1卷积输出的列数，不含Pool。
    int cin;  // 逻辑输入通道数。
    int cout;  // 逻辑卷积输出通道数。
    int tiles_per_row;  // 每行ceil(W/4)个空间Tile，不跨行。
    int spatial_tiles;  // 卷积阶段H*ceil(W/4)个空间Tile。
    int cout_tiles;  // 整层ceil(Cout/4)个计算Tile。
    int reduction_steps;  // R=Kh*Kw*ceil(Cin/8)，每个结果的完整归约长度。
    int im2row_tokens;  // 单次输入扫描的窗口输出事务数，不乘Cout Tile数。
    int weight_transactions;  // 整层权重Pack数=cout_tiles*R，不是驻留缓存深度。
    int last_tile_x;  // 每行末空间Tile编号。
    int prelast_tile_x;  // 每行倒数第二个空间Tile编号。
    int last_bottom_loads;  // 3x3末Tile尚需读取的新Bottom真实列数。
    unsigned int tail_spatial_valid_mask;  // 末空间Tile的有效Lane掩码。
    unsigned int prelast_new_column_valid_mask;  // 倒数第二Tile预取的新列有效掩码。
    int kernel;  // 卷积核边长，仅3或Prediction的1。
    int padding;  // 卷积四侧补零宽度，仅1或0，不是Pool补边。
    int stride;  // 卷积步长，固定1。
    int cin_groups;  // G=ceil(Cin/8)，每个真实空间位置的输入Pack数。
    int input_transactions;  // 单次扫描读取的H*W*G个真实输入Pack。
    int cin_group_shift;  // log2(G)，供Line地址移位，不在调用时译码。
    int tail_spatial_lanes;  // 末空间Tile有效位置数，整除P时为P。
    int pool_kernel;  // Pool核边长；0表示Bypass。
    int pool_stride;  // Pool步长；0表示Bypass。
    int pool_padding_right;  // Pool仅右侧补边宽度，Conv6为1，其余为0。
    int pool_padding_bottom;  // Pool仅下侧补边宽度，Conv6为1，其余为0。
    int output_h;  // 经过Pool/Bypass后写回的逻辑行数。
    int output_w;  // 经过Pool/Bypass后写回的逻辑列数。
    int output_channel_groups;  // ceil(Cout/8)，末组不足8也占一个Pack。
    int output_transactions;  // 整层逻辑输出Pack数，不含DDR对齐或缓存块重扫。
    bool use_leaky_relu;  // 前八层启用固定alpha=0.1，Prediction禁用。
    int resident_cout_tiles;  // 当前缓存上限min(8,cout_tiles)，Conv1仅4。
    int cache_blocks;  // 整层ceil(cout_tiles/8)个Cout缓存块。
    int last_cache_tiles;  // 最后缓存块实际驻留的计算Tile数，不额外补Tile。
    unsigned int last_cout_valid_mask;  // 整层末4-Cout Tile有效Lane掩码，Prediction为1。
};
enum class YoloTinyLayer {
    Conv1, Conv2, Conv3, Conv4, Conv5, Conv6, Conv7, Conv8, Prediction, Count
};
constexpr int YOLO_TINY_LAYER_COUNT = static_cast<int>(YoloTinyLayer::Count);
constexpr int YOLO_TINY_COUT_CACHE_CHANNELS = 32;
constexpr int YOLO_TINY_COUT_CACHE_TILES = YOLO_TINY_COUT_CACHE_CHANNELS / CONV_COUT_PAR;
// pool_stride为2/1/0分别表示下采样/保持尺寸/Bypass；仅用于下方固定层表。
constexpr int yolo_stage_output_size(int h, int pool_stride) {
    return pool_stride == 2 ? h / 2 : h;
}
// 只负责从固定层表的逻辑参数统一派生全部字段，不是任意配置的支持承诺。
// 使用单return形式保持C++11常量表达式兼容，供下方容量计算直接求值。
constexpr ConvGeometry make_yolo_geometry(int h, int cin, int cout, int kernel,
                                          int pool_stride, bool use_leaky_relu) {
    return ConvGeometry{
        h, h, cin, cout,
        ceil_div(h, IM2ROW_P), h * ceil_div(h, IM2ROW_P), ceil_div(cout, CONV_COUT_PAR),
        kernel * kernel * ceil_div(cin, CONV_CIN_PAR),
        h * ceil_div(h, IM2ROW_P) * kernel * kernel * ceil_div(cin, CONV_CIN_PAR),
        ceil_div(cout, CONV_COUT_PAR) * kernel * kernel * ceil_div(cin, CONV_CIN_PAR),
        ceil_div(h, IM2ROW_P) - 1, ceil_div(h, IM2ROW_P) - 2,
        h - ((ceil_div(h, IM2ROW_P) - 1) * IM2ROW_P + 1),
        (1u << (((h - 1) % IM2ROW_P) + 1)) - 1u,
        (1u << (h - ((ceil_div(h, IM2ROW_P) - 1) * IM2ROW_P + 1))) - 1u,
        kernel, kernel == 3 ? 1 : 0, 1, ceil_div(cin, CONV_CIN_PAR),
        h * h * ceil_div(cin, CONV_CIN_PAR),
        config_log2(ceil_div(cin, CONV_CIN_PAR)), ((h - 1) % IM2ROW_P) + 1,
        pool_stride == 0 ? 0 : 2, pool_stride,
        pool_stride == 1 ? 1 : 0, pool_stride == 1 ? 1 : 0,
        yolo_stage_output_size(h, pool_stride), yolo_stage_output_size(h, pool_stride),
        ceil_div(cout, FEATURE_PACK_CHANNELS),
        yolo_stage_output_size(h, pool_stride) * yolo_stage_output_size(h, pool_stride) *
            ceil_div(cout, FEATURE_PACK_CHANNELS),
        use_leaky_relu,
        config_min(YOLO_TINY_COUT_CACHE_TILES, ceil_div(cout, CONV_COUT_PAR)),
        ceil_div(ceil_div(cout, CONV_COUT_PAR), YOLO_TINY_COUT_CACHE_TILES),
        ((ceil_div(cout, CONV_COUT_PAR) - 1) % YOLO_TINY_COUT_CACHE_TILES) + 1,
        (1u << (((cout - 1) % CONV_COUT_PAR) + 1)) - 1u};
}
// 唯一YOLOv2-tiny VOC层表；Im2Row、几何构造器和物理容量均从此表派生。
// 各列：H=W、Cin、Cout、卷积核边长、Pool步长(0=Bypass)、是否LeakyReLU。
constexpr ConvGeometry YOLO_TINY_GEOMETRIES[YOLO_TINY_LAYER_COUNT] = {
    make_yolo_geometry(416, 3, 16, 3, 2, true),
    make_yolo_geometry(208, 16, 32, 3, 2, true),
    make_yolo_geometry(104, 32, 64, 3, 2, true),
    make_yolo_geometry(52, 64, 128, 3, 2, true),
    make_yolo_geometry(26, 128, 256, 3, 2, true),
    make_yolo_geometry(13, 256, 512, 3, 1, true),
    make_yolo_geometry(13, 512, 1024, 3, 0, true),
    make_yolo_geometry(13, 1024, 1024, 3, 0, true),
    make_yolo_geometry(13, 1024, 125, 1, 0, false)};
constexpr int yolo_max_field(int ConvGeometry::*field, int count = YOLO_TINY_LAYER_COUNT) {
    return count == 0 ? 0 : config_max(YOLO_TINY_GEOMETRIES[count - 1].*field,
                                     yolo_max_field(field, count - 1));
}
constexpr int yolo_max_line_depth(int count = YOLO_TINY_LAYER_COUNT) {
    return count == 0 ? 0 : config_max(
        YOLO_TINY_GEOMETRIES[count - 1].tiles_per_row * YOLO_TINY_GEOMETRIES[count - 1].cin_groups,
        yolo_max_line_depth(count - 1));
}
constexpr int yolo_max_resident_weights(int count = YOLO_TINY_LAYER_COUNT) {
    return count == 0 ? 0 : config_max(
        YOLO_TINY_GEOMETRIES[count - 1].resident_cout_tiles *
            YOLO_TINY_GEOMETRIES[count - 1].reduction_steps,
        yolo_max_resident_weights(count - 1));
}
// YOLO Systolic 缓存容量；不替换尚未迁移的 MNIST 缓存常量。
constexpr int YOLO_TINY_ACTIVATION_REPLAY_CAPACITY = yolo_max_field(&ConvGeometry::reduction_steps);
constexpr int YOLO_TINY_WEIGHT_BUFFER_DEPTH = yolo_max_resident_weights();
constexpr int YOLO_TINY_WEIGHT_ADDRESS_BITS = bits_for_max(YOLO_TINY_WEIGHT_BUFFER_DEPTH);
constexpr int YOLO_TINY_MAX_RESIDENT_COUT_TILES = yolo_max_field(&ConvGeometry::resident_cout_tiles);
// 逐层/相邻层详尽校验放在build静态检查TU，不增加生产配置校验框架。
constexpr int YOLO_TINY_PREDICTION_TAIL_LAST_LANE =
    YOLO_TINY_GEOMETRIES[static_cast<int>(YoloTinyLayer::Prediction)].tail_spatial_lanes - 1;
static_assert(YOLO_TINY_LAYER_COUNT == 9 && YOLO_TINY_GEOMETRIES[8].kernel == 1 &&
              YOLO_TINY_GEOMETRIES[8].padding == 0 && YOLO_TINY_GEOMETRIES[8].w == 13 &&
              YOLO_TINY_PREDICTION_TAIL_LAST_LANE == 0,
              "Prediction专门化要求无Padding的1x1、W13且尾Tile只有Lane0");
static_assert(YOLO_TINY_COUT_CACHE_CHANNELS % FEATURE_PACK_CHANNELS == 0,
              "Cout缓存通道数必须组成完整的8通道Pack");
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

// OS 阵列最后一个 PE 相对入口的传播延迟；注入完 R 个 token 后排空这几步。
constexpr int SYSTOLIC_WAVEFRONT_DRAIN_STEPS =
    (SYSTOLIC_COUT_PAR - 1) + (SYSTOLIC_SPATIAL_PAR - 1);

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


// Legacy：旧消费者的字段和值不变，新增YOLO字段显式置零，标记尚未迁移；
// 该构造器的返回值不满足新Im2Row的YOLO调用契约。
inline ConvGeometry conv_geometry(layer_mode_t mode) {
#pragma HLS INLINE
    return mode == LAYER2_MODE
        ? ConvGeometry{CNN_CONV2_H, CNN_CONV2_W, CNN_CONV2_CIN, CNN_CONV2_COUT,
            CNN_CONV2_TILES_PER_ROW, CNN_CONV2_SPATIAL_TILES, CNN_CONV2_COUT_TILES,
            CNN_CONV2_REDUCTION_STEPS, CNN_CONV2_IM2ROW_TOKENS, CNN_CONV2_WEIGHT_TRANSACTIONS,
            CNN_CONV2_LAST_TILE_X, CNN_CONV2_PRELAST_TILE_X, CNN_CONV2_LAST_BOTTOM_LOADS,
            CNN_CONV2_TAIL_SPATIAL_VALID_MASK, CNN_CONV2_PRELAST_NEW_COLUMN_VALID_MASK,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0}
        : ConvGeometry{CNN_CONV1_H, CNN_CONV1_W, CNN_CONV1_CIN, CNN_CONV1_COUT,
            CNN_CONV1_TILES_PER_ROW, CNN_CONV1_SPATIAL_TILES, CNN_CONV1_COUT_TILES,
            CNN_CONV1_REDUCTION_STEPS, CNN_CONV1_IM2ROW_TOKENS, CNN_CONV1_WEIGHT_TRANSACTIONS,
            CNN_CONV1_LAST_TILE_X, CNN_CONV1_PRELAST_TILE_X, CNN_CONV1_LAST_BOTTOM_LOADS,
            CNN_CONV1_TAIL_SPATIAL_VALID_MASK, CNN_CONV1_PRELAST_NEW_COLUMN_VALID_MASK,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0, 0, 0, 0};
}
// 调用方必须使用合法层枚举；不为表外枚举生成兜底配置。
constexpr ConvGeometry yolov2_tiny_geometry(YoloTinyLayer layer) {
    return YOLO_TINY_GEOMETRIES[static_cast<int>(layer)];
}
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
