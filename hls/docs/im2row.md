# Im2Row 模块设计文档

## 1. 文档状态

- 当前阶段：架构设计完成，代码尚未实现。
- 适用工具：Vitis HLS 2022.2。
- 目标模块：`cnn_accelerator/src/im2row/`。
- 本文档用于指导后续 C Simulation、HLS 实现、C Synthesis 和 RTL Co-simulation。
- 代码实现、综合结果或模块接口发生变化后，必须同步更新本文档。

## 2. 设计范围

Im2Row 模块负责把 NHWC 顺序的输入 Feature Map 转换为广播型 `P×P` PE 阵列所需的激活数据流。模块不显式生成完整的 Im2Row 矩阵，而是使用流式 Padding、P 路 Line Buffer、滑动窗口和 Tile 缓冲在线生成数据。

本模块负责：

- 每个输入事务接收 P 个空间像素；
- 每个像素一次携带全部物理 Channel；
- 在线生成 `padding=1` 的零填充；
- 使用两行 Line Buffer 形成 3×3 邻域；
- 每周期最多生成 P 个候选窗口；
- 将有效窗口按全局空间顺序连续收集为 P 路空间 Tile；
- 按 `kh → kw → cin` 顺序沿 K 维串行输出；
- 在 K 输出循环中实现每周期 P 个 activation；
- 为最后不足 P 个空间位置的 Tile 产生有效 Lane 掩码。

本模块暂不负责：

- 输入 Feature Map 从 DDR 或其他外部存储器的加载；
- 权重加载、权重缓存和权重调度；
- `Cout` 方向的分 Tile；
- Systolic 激活重放；
- Bias、Requantization、激活函数和输出 Channel 组装；
- Pool、FC 以及顶层控制协议；
- 非 3×3 Kernel、非 1 Stride 或非 1 Padding 的卷积。

当前阶段统一使用 `hls::stream` 简化模块连接。外部存储器和 AXI 数据搬运将在后续顶层设计中补充。

## 3. 总体数据流

```text
NHWC、P 路、Channel-packed 输入流
                  │
                  ▼
        输入行尾 Lane 处理
                  │
                  ▼
        Padding Pack Generator
                  │
                  ▼
          P-Bank Line Buffer
                  │
                  ▼
        3×(P+2) Sliding Strip
                  │
                  ▼
     P 路 Candidate Window + Valid
                  │
                  ▼
       跨行连续窗口收集器
                  │
                  ▼
          P×K Patch Tile Buffer
                  │
                  ▼
       K 维 Scheduler，目标 II=1
                  │
                  ▼
 activation_pack[P] + spatial_valid
                  │
                  ▼
        广播型 P×P Systolic
```

## 4. 参数定义

### 4.1 编译期参数

以下参数决定物理硬件规模，必须在 HLS 综合前确定：

```text
P                 空间并行度，同时表示每个输入包的像素数量
MAX_H             支持的最大输入高度
MAX_W             支持的最大输入宽度
MAX_CHANNELS      支持的最大输入 Channel 数量
KH = 3            Kernel 高度
KW = 3            Kernel 宽度
PADDING = 1       零填充宽度
STRIDE = 1        卷积步长
```

第一版计划使用：

```text
P = 4
KH = 3
KW = 3
PADDING = 1
STRIDE = 1
```

`MAX_H`、`MAX_W` 和 `MAX_CHANNELS` 在进入代码实现前结合完整 CNN 结构确定。

### 4.2 运行时参数

```text
H      当前输入 Feature Map 高度
W      当前输入 Feature Map 宽度
Cin    当前有效输入 Channel 数量
```

必须满足：

```text
0 < H   <= MAX_H
0 < W   <= MAX_W
0 < Cin <= MAX_CHANNELS
```

当前卷积参数固定，因此输出尺寸为：

```text
Ho = H
Wo = W
K  = KH × KW × Cin = 9 × Cin
```

`Cin` 是运行时参数，但物理接口、Line Buffer 和窗口寄存器仍按 `MAX_CHANNELS` 分配。只有前 `Cin` 个 Channel 有效。

## 5. 数据类型和接口契约

下面的数据结构为设计意图，最终声明统一放入 `types.h`，具体命名可在实现时调整。

### 5.1 激活数据类型

当前量化方案为对称 int8，激活零点为 0：

```cpp
typedef ap_int<8> activation_t;
```

因此 Padding 产生整数 0 是正确的量化零值。

### 5.2 Channel-packed Pixel

```cpp
struct pixel_t {
    activation_t channel[MAX_CHANNELS];
};
```

一个 `pixel_t` 表示一个空间位置的全部物理 Channel。运行时只使用：

```text
channel[0 ... Cin-1]
```

其余 Channel 不参与窗口输出。

### 5.3 P 路输入包

```cpp
struct pixel_pack_t {
    pixel_t pixel[P];
    ap_uint<P> spatial_valid;
};
```

输入接口：

```cpp
hls::stream<pixel_pack_t>& input_stream
```

输入流约定：

- 一个 Stream Token 包含 P 个空间像素，而不是在同一个 Stream 上执行 P 次读取；
- 每个输入包中的有效像素必须属于同一行；
- Lane 顺序为从左到右；
- 每一行独立按 P 对齐；
- 行尾不足 P 个像素时，通过 `spatial_valid` 标记有效 Lane；
- 输入包不能跨越两行；
- 输入 Feature Map 的整体顺序为 NHWC、逐行从上到下。

每帧输入 Token 数量为：

```text
input_tokens = H × ceil(W / P)
```

例如 `W=14、P=4` 时，每行输入四个 Token，最后一个 Token 只有前两个 Lane 有效。

### 5.4 P 路激活输出包

```cpp
struct activation_pack_t {
    activation_t data[P];
    ap_uint<P> spatial_valid;
};
```

输出接口：

```cpp
hls::stream<activation_pack_t>& output_stream
```

一个输出 Token 表示：

- P 个空间位置；
- 同一个 k 维元素；
- `spatial_valid` 表示对应空间 Lane 是否属于有效输出位置。

同一个空间 Tile 的 K 个输出 Token 使用相同的 `spatial_valid`。

Systolic 根据自身的 K 计数器识别 Tile 边界，因此第一版不在每个 Token 中携带显式 `k`、`tile_start` 或 `tile_end` 字段。

### 5.5 函数接口

计划接口：

```cpp
void im2row(
    hls::stream<pixel_pack_t>& input_stream,
    hls::stream<activation_pack_t>& output_stream,
    int H,
    int W,
    int Cin
);
```

Im2Row 作为 CNN Accelerator 内部模块时，只使用 `hls::stream`，不在该内部函数上添加 AXI4-Stream 接口编译指示。AXIS 和 AXI-Lite 接口统一由最终顶层函数定义。

## 6. 数据顺序

### 6.1 输入空间顺序

输入 Feature Map 采用 NHWC 逻辑布局：

```text
for row
    for col
        pixel[channel]
```

Channel 已经打包在 `pixel_t` 内，因此空间流只需要遍历 H 和 W。

### 6.2 Window Flatten 顺序

Window 按以下顺序展开：

```text
kh → kw → cin
```

索引为：

```text
k = (kh × KW + kw) × Cin + cin
```

权重模块必须使用完全相同的 K 顺序。

为避免运行时除法和取模，Scheduler 采用三层嵌套循环，而不是从单一 `k_counter` 反算 `kh/kw/cin`。

### 6.3 空间 Tile 顺序

所有有效窗口按照：

```text
spatial_index = output_row × W + output_col
```

连续收集。空间 Tile 可以跨行。例如 `W=14、P=4` 时，一个 Tile 可以包含：

```text
row r   : col 12、col 13
row r+1 : col 0、col 1
```

Systolic 只把它们视为 P 个独立空间位置。卷积输出端后续需要行对齐 Repacker，把 Channel-packed 输出重新整理为每个包不跨行的格式，供 Pool 或下一层 Im2Row 使用。该 Repacker 不属于本模块。

## 7. Padding Pack Generator

### 7.1 设计原则

模块不创建完整的 Padding Feature Map，而是在读取输入流时生成虚拟 Padding 行列。

逻辑 Padding 后的宽度为：

```text
PADDED_W = W + 2
```

为了每周期处理 P 个位置，内部每行按 P 对齐：

```text
PADDED_STRIDE = ceil((W + 2) / P) × P
```

物理最大行宽为：

```text
MAX_PADDED_STRIDE = ceil((MAX_W + 2) / P) × P
```

### 7.2 左右 Padding

原始列与 Padding 列的映射：

```text
padded_col = 0       左侧 Padding，值为 0
padded_col = 1...W   对应原始列 0...W-1
padded_col = W+1     右侧 Padding，值为 0
padded_col > W+1     P 路对齐填充，值为 0，不产生输出
```

Padding 列由运行时 W 决定，不固定在 `MAX_W+1`。

`W=14、P=4` 时，一行内部数据为：

```text
[0,   x0,  x1,  x2 ]
[x3,  x4,  x5,  x6 ]
[x7,  x8,  x9,  x10]
[x11, x12, x13, 0  ]
```

`W=28、P=4` 时，Padding 后逻辑宽度为 30，物理对齐宽度为 32：

```text
0, x0, x1, ..., x27, 0, 0, 0
```

最后两个零只用于 P 路对齐。

### 7.3 上下 Padding

内部扫描逻辑行范围：

```text
padded_row = 0 ... H+1
```

行为如下：

```text
padded_row = 0       生成整行零，不读取输入流
padded_row = 1...H   读取原始行 0...H-1
padded_row = H+1     生成整行零，不读取输入流
```

这样底部最后一行窗口会在虚拟零行输入时自然完成，不需要在 Window Generator 中对底部数据做特殊随机访问。

### 7.4 输入尾部 Lane

原始输入按行对齐。当 `W % P != 0` 时，每行最后一个输入 Token 存在无效 Lane。Padding Generator 必须结合：

- 当前原始列；
- 输入 `spatial_valid`；
- 运行时 W；

只消费和转发有效像素。无效 Lane 不得进入下一行，也不得被解释为 Feature Map 数据。

## 8. P-Bank Line Buffer

### 8.1 缓冲规模

3×3 Kernel 需要保存前两行：

```text
line_buffer[2][MAX_PADDED_STRIDE]
```

每个元素是一个 Channel-packed `pixel_t`。

### 8.2 P 路分 Bank

每周期处理 P 个连续 `padded_col`，因此列方向必须提供 P 路独立访问能力。

逻辑映射：

```text
bank    = padded_col % P
address = padded_col / P
```

可以通过 cyclic partition，也可以直接把 Bank 作为显式数组维度。实现后必须通过综合报告确认实际生成了 P 个独立存储 Bank。

### 8.3 同列读后写

每个有效内部位置采用同列读后写：

```text
top_pixel    = line_buffer[0][padded_col]
middle_pixel = line_buffer[1][padded_col]

line_buffer[0][padded_col] = middle_pixel
line_buffer[1][padded_col] = current_pixel
```

这不是整行复制。每周期只更新当前 P 个列位置。

每个 Bank 每周期需要一次读和一次写，应检查综合后是否映射为满足该访问模式的双端口 RAM。

### 8.4 Channel 并行

`pixel_t.channel[MAX_CHANNELS]` 计划在 Channel 维完全分区，使一个像素的全部 Channel 可以并行读写。Line Buffer 的空间宽度维不能完全分区，否则会把整行存储展开成大量寄存器。

## 9. P 路 Window Generator

### 9.1 Sliding Strip

每周期注入 P 个新列位置。为同时形成 P 个相邻 3×3 窗口，需要维护三行、每行 `P+2` 个像素的滑动条带：

```text
strip[3][P+2]
```

相邻窗口共享绝大多数数据，因此不复制 P 个完全独立的 3×3 Shift Register。

### 9.2 更新顺序

每个内部输入包到达时必须：

1. 先把旧条带向左移动 P 个位置；
2. 再把 Line Buffer 和当前行提供的 P 个新像素注入最右侧；
3. 最后根据更新后的条带形成 P 个候选窗口。

禁止先覆盖右侧再执行左移，否则新数据会被复制到多个列位置并丢失旧窗口内容。

### 9.3 候选窗口有效条件

Padding 的零值已经在 Window Generator 之前生成。Window Generator 不负责选择越界像素，只负责根据当前窗口右下角的 Padding 坐标判断候选窗口是否对应有效输出。

对于标量坐标，窗口右下角满足以下范围时产生有效输出：

```text
padded_row = 2 ... H+1
padded_col = 2 ... W+1
```

P 路实现中，每个 Lane 都有自己的 `padded_col`，因此产生：

```text
candidate_window[P]
candidate_valid[P]
```

左侧预热位置、右侧 P 对齐位置、顶部预热行和其他非输出位置的 Valid 必须为 0。

## 10. 跨行连续窗口收集器

### 10.1 设计目标

Window Generator 每周期最多产生 P 个有效窗口，但每行尾部可能少于 P 个。收集器把所有有效窗口按照全局空间顺序压紧，避免在 Systolic 内部为每一行的尾部 Lane 浪费计算。

### 10.2 Carry Buffer

收集器维护少于 P 个尚未组成完整 Tile 的窗口。新候选窗口到达后：

1. 按 Lane 顺序过滤 `candidate_valid=0` 的窗口；
2. 把有效窗口追加到 Carry Buffer；
3. 累计达到 P 个时输出一个完整 Patch Tile；
4. 剩余窗口继续保留到下一周期，允许跨行组合。

每周期最多输入 P 个有效窗口，Carry 中最多已有 `P-1` 个窗口，因此一次更新后最多产生一个完整 Tile，并留下最多 `P-1` 个窗口。

### 10.3 最终不足 P 个窗口

帧结束时，如果 Carry Buffer 中仍有窗口，则输出一个尾 Tile：

- 有效窗口保持原始顺序；
- 无效 Lane 的 Patch 数据置零；
- `spatial_valid` 标记实际有效 Lane。

当前计划中的常用尺寸：

```text
28×28 = 784，可被 P=4 整除
14×14 = 196，可被 P=4 整除
```

因此这些尺寸没有最终尾 Tile，但通用实现仍保留 Mask 支持。

## 11. Patch Tile Buffer

### 11.1 缓冲内容

一个空间 Tile 包含：

```text
P × KH × KW × MAX_CHANNELS
```

逻辑上等价于：

```text
P × MAX_K
```

其中：

```text
MAX_K = KH × KW × MAX_CHANNELS
```

运行时只有前述 `kh → kw → cin` 中 `cin < Cin` 的元素有效。

### 11.2 双缓冲

Window Generator 产生空间 Tile 的速度快于 K Scheduler 消费 Tile 的速度。Scheduler 输出一个 Tile 需要 K 个周期，因此使用两个 Patch Tile Buffer：

```text
Buffer A：Scheduler 正在沿 K 维读取
Buffer B：Window Generator 正在准备下一个空间 Tile
```

建议状态：

```text
EMPTY    可由窗口收集器写入
READY    已写入完整 Tile，等待 Scheduler
SENDING  Scheduler 正在输出 K 个 Token
```

当两个 Buffer 都不可写时，反压必须逐级传回 Padding Generator 和输入流。禁止覆盖尚未发送完成的 Tile。

Im2Row 的 Patch Tile Buffer 与后续 Systolic 为 `Cout Tile` 重放而设置的激活缓冲在第一版中保持独立，以减少模块耦合。重复存储的资源开销在当前规模下可以接受。

## 12. K 维 Scheduler

### 12.1 输出循环

Scheduler 对每个空间 Tile 执行：

```text
for kh = 0 ... 2
    for kw = 0 ... 2
        for cin = 0 ... Cin-1
            输出 activation_pack[P]
```

每个周期输出 P 个空间位置在同一 `(kh, kw, cin)` 上的激活值。

### 12.2 性能定义

K 循环目标：

```text
II = 1
```

正确的吞吐定义为：

- 每周期输出一个 `activation_pack_t`；
- 每个 Token 含 P 个 activation；
- 每 K 个周期完成一个 P 空间位置 Tile；
- 不是每周期完成 P 个完整窗口。

每帧输出 Token 数量为：

```text
spatial_tiles = ceil(H × W / P)
output_tokens = spatial_tiles × K
```

例如 `H=14、W=14、Cin=8、P=4`：

```text
K             = 72
spatial_tiles = 49
output_tokens = 49 × 72 = 3528
```

不考虑首尾延迟和下游反压时，Scheduler 至少需要 3528 个输出周期。

## 13. 与广播型 Systolic 的接口

广播型 `P×P` PE 阵列每周期接收：

```text
P 个 activation，对应 P 个空间位置
P 个 weight，对应 P 个输出通道
```

执行：

```text
PE[i][j] += activation[i] × weight[j]
```

持续 K 个周期后得到：

```text
P 个空间位置 × P 个输出通道
```

当 `Cout > P` 时，同一个 `P×K` 激活 Tile 需要针对多个输出通道 Tile 重放。该重放由 Systolic 前端激活缓冲负责，Im2Row 每个空间 Tile 只输出一次，不接收 `Cout` 参数。

Systolic 输出最终会把全部 `Cout` 通道重新组装为 Channel-packed Pixel，使：

```text
当前层 Cout = 下一层 Cin
```

该行为用于定义模块间数据契约，不属于本阶段 Im2Row 实现范围。

## 14. HLS 实现要求

### 14.1 Pipeline

以下循环目标为 `II=1`：

- P 路内部 Padding Pack 处理；
- P-Bank Line Buffer 同列读写；
- Sliding Strip 更新和候选窗口生成；
- K Scheduler 输出循环。

如果综合结果没有达到 II=1，必须检查：

- Line Buffer Bank 数量；
- RAM 端口冲突；
- 数组依赖分析；
- 动态索引产生的多路选择器；
- Stream 或 Tile Buffer 反压；
- 运行时循环边界。

### 14.2 Array Partition

计划分区方向：

- Pixel Channel 维：Complete Partition；
- P 路空间 Lane：Complete Partition；
- Sliding Strip：空间小数组 Complete Partition；
- Line Buffer 列：Cyclic Partition，Factor=P；
- Line Buffer 整行宽度：禁止 Complete Partition。

最终编译指示必须作用于实际变量及正确维度，不能把文档中的逻辑名称直接照搬为无效 pragma。

### 14.3 Storage Binding

Line Buffer 每个 Bank 需要满足一读一写。实现时根据器件和综合结果选择适当的双端口 RAM 类型。不能只根据 C++ 数组声明假定工具一定生成目标端口结构。

### 14.4 Dataflow

第一版计划把功能分为两个主要并发阶段：

```text
窗口生产阶段：Padding + Line Buffer + Window + Collector
Tile 发送阶段：K Scheduler
```

两个阶段通过显式双缓冲连接。是否进一步拆分为更多 DATAFLOW 子函数，由综合报告和 Dataflow Viewer 决定。

每个阶段的 Token 数量、终止条件和 Buffer 状态必须完全一致，避免 C Simulation 正常但 RTL 因有限深度缓冲出现停顿或死锁。

### 14.5 运行时循环边界

H、W、Cin 是运行时参数。综合时应为相关循环提供合理的最大边界和 `LOOP_TRIPCOUNT` 信息，以改善性能估计。任何运行时参数都不能改变已综合的数组大小和端口宽度。

## 15. 状态复位和帧边界

每次函数调用处理一帧、Batch 固定为 1。

每帧开始时必须复位：

- 原始输入行列计数器；
- Padding 行列计数器；
- Sliding Strip 有效状态；
- Carry Buffer 计数；
- Patch Tile Buffer 状态；
- Scheduler 的 kh、kw、cin 和 Tile 计数。

顶部虚拟零行和左右 Padding 会覆盖窗口预热所需的数据，但实现仍不能依赖未初始化数组值或上一次函数调用残留状态。

Line Buffer 不要求通过整块清零循环清除全部存储；应通过明确的 Padding 扫描和有效状态保证旧数据不会成为当前帧有效窗口的一部分。

## 16. 验证计划

### 16.1 软件参考模型

Testbench 中建立普通软件 Im2Row Golden Model：

- 输入为 NHWC 数组；
- 显式执行 Padding；
- 按 `kh → kw → cin` 展开；
- 按全局空间顺序形成 P 路 Tile；
- 对尾 Tile 生成相同的 `spatial_valid`。

HLS 输出必须逐 Token、逐 Lane 与 Golden Model 比较。

### 16.2 功能测试

至少覆盖：

- `H/W = 1、2、3` 的极小尺寸；
- `H=3、W=5、P=4` 等不能被 P 整除的尺寸；
- `H=14、W=14、Cin=8`；
- `H=28、W=28、Cin=1`；
- 全零输入；
- 全一输入；
- 单点脉冲；
- 可复现随机输入；
- 四个角、四条边和中心位置；
- 输入行尾无效 Lane；
- 跨行空间 Tile；
- 最后不足 P 个窗口的 Tile；
- 连续处理两帧，检查状态残留；
- 输出 Token 总数和 Mask；
- 下游暂停造成的反压。

### 16.3 HLS 验证流程

```text
1. C Simulation
2. C Synthesis
3. 检查 II、Latency、LUT、FF、BRAM 和 RAM 端口
4. 检查 Dataflow Viewer
5. RTL Co-simulation
6. 与 Systolic 接口联调
```

综合后重点检查：

- K Scheduler 是否达到 II=1；
- Line Buffer 是否生成 P 个 Bank；
- 每个 Bank 是否满足一读一写；
- Channel 和 P Lane 是否按预期并行；
- 是否出现意外的整行复制电路；
- 是否出现运行时除法或取模电路；
- 双缓冲切换是否覆盖未完成 Tile；
- Tile 边界是否产生额外气泡。

## 17. 已知风险和后续事项

### 17.1 Channel-packed 总线宽度

输入物理宽度约为：

```text
P × MAX_CHANNELS × activation_bits
```

这种设计简化了 Im2Row 和 Pool 的 Channel 处理，也允许 Channel 维 UNROLL，但当 `MAX_CHANNELS` 增大时会增加接口宽度、Line Buffer 带宽和布线压力。第一版继续采用 Channel-packed 方案，综合后根据资源和时序评估。

### 17.2 P 路 Padding 重排复杂度

原始输入按行 P 对齐，而内部需要插入左右 Padding 并按 `PADDED_STRIDE` 对齐。实现必须正确处理输入包边界、行尾无效 Lane 和跨内部 Pack 的像素搬移。该部分应作为独立逻辑首先验证。

### 17.3 动态 Cin

运行时 Cin 提供灵活性，但硬件仍按 `MAX_CHANNELS` 分配。动态循环边界和动态选择可能影响 HLS 调度与时序，需要通过综合报告确认。

### 17.4 数据加载暂缓

当前使用 `hls::stream` 作为输入输出。DDR、AXI Master、DMA、权重加载和多层调度不在本阶段设计范围，后续顶层设计时补充。

### 17.5 目标器件和时钟暂未确定

目标 FPGA Part、目标时钟周期以及接口时钟域尚未确定。实现初期可以使用临时综合配置，但最终优化结论必须基于实际器件和目标频率。

## 18. 实施步骤

1. 在 `types.h` 中定义统一的激活、Pixel Pack 和 Activation Pack 类型。
2. 在 `im2row.h` 中声明参数、内部常量和模块接口。
3. 在 `test_im2row.cpp` 中先实现软件 Golden Model。
4. 实现并独立验证 Padding Pack Generator。
5. 实现 P-Bank Line Buffer 和 Sliding Strip。
6. 实现 Candidate Window Valid。
7. 实现跨行连续窗口收集器和尾 Tile Mask。
8. 实现 Patch Tile 双缓冲状态机。
9. 实现 `kh → kw → cin` K Scheduler。
10. 完成 C Simulation。
11. 完成 C Synthesis，依据报告调整 Partition、RAM 类型和流水结构。
12. 完成 RTL Co-simulation。
13. 与 Systolic 激活接口联调。
14. 根据实现和验证结果更新本文档及项目 README。
