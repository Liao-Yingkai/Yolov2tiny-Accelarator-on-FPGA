# MNIST FPGA Accelerator

## 项目简介

本项目使用 Vitis HLS 开发面向 MNIST CNN 推理的 FPGA 加速器。设计重点是通过流式模块、片上循环缓冲、可配置并行度和数据复用减少外部访存，并逐步完成 Im2Row、卷积阵列、池化、全连接和顶层控制。

当前工程处于 Im2Row 优化设计完成、源码等待重构的阶段，尚未形成经过综合验证的完整加速器。

本 README 是项目级设计文档，记录跨模块约定、总体数据流、当前状态和关键设计决策。模块的周期级微架构、地址映射、HLS 编译指示和验证计划放在同目录的模块文档中。

## 当前目录结构

```text
mnist_fpga_accelerator/
└── hls/
    ├── docs/
    │   ├── README.md
    │   └── im2row.md
    ├── cnn_accelerator/
    │   ├── src/
    │   │   ├── top.cpp
    │   │   ├── cnn_accel.h
    │   │   ├── types.h
    │   │   ├── conv/
    │   │   ├── im2row/
    │   │   ├── systolic/
    │   │   ├── pool/
    │   │   └── fc/
    │   ├── testbench/
    │   └── script/
    └── .vscode/
```

## 总体设计原则

- 使用 Vitis HLS 2022.2 完成 C/C++ 到 RTL 的综合。
- 模块之间优先使用阻塞式 `hls::stream`。
- Stream 满或下游未就绪时通过反压暂停，禁止丢弃或跳过 Token。
- 当前阶段暂不设计 DDR、DMA、AXI Master、输入 Loader 和权重 Loader。
- 激活采用对称 int8 量化，零点为 0。
- Feature Map 逻辑布局采用 NHWC。
- 一个空间位置的全部物理 Channel 打包为 `pixel_t`。
- 数据打包宽度和 PE 内计算并行度分开定义，Channel-packed 不等于必须全 Channel 并行计算。
- 物理硬件规模使用编译期参数，H、W、Cin、Cout 在允许范围内作为运行时参数。
- 模块注释、文档、提交信息和工程说明统一使用中文。
- 所有 II、Latency、资源和时序结论最终以 C Synthesis 和 RTL Co-simulation 为准。

## 计划中的加速器数据流

```text
输入 Loader / 片上输入缓冲（待设计）
                 │
                 ▼
        标量 Channel-packed Pixel Stream
                 │
                 ▼
┌────────────────────────────────────┐
│ Conv                               │
│  Im2Row                            │
│  广播型 P×P Systolic              │
│  激活 Tile 重放                    │
│  Bias / Requantization / Activation│
│  Cout Channel 组装                 │
└────────────────────────────────────┘
                 │
                 ▼
               Pool
                 │
                 ▼
             后续 Conv
                 │
                 ▼
                FC
                 │
                 ▼
              分类结果
```

具体网络层数、各层 H/W/Cin/Cout、权重布局和顶层加载方式将在后续模块设计中补充。

## 跨模块数据约定

### Channel-packed Pixel

目标激活类型：

```cpp
typedef ap_int<8> activation_t;

struct pixel_t {
    activation_t channel[MAX_CHANNELS];
};
```

一个空间位置的全部 Channel 统一打包：

```text
pixel = [channel0, channel1, ..., channel(MAX_CHANNELS-1)]
```

运行时只有前 `Cin` 或 `Cout` 个 Channel 有效。

采用完整 Channel 打包的原因：

- Im2Row 的物理输入流只遍历空间维；
- Im2Row 每个窗口元素可以直接携带完整 Channel；
- Systolic 输出可重新组装为下一层直接使用的 Pixel；
- Pool 可对不同 Channel 独立比较；
- 当前层 `Cout` 可以直接作为下一层 `Cin`。

风险是总线和缓存宽度随 `P×MAX_CHANNELS` 增长。第一版保留 Channel-packed 接口，综合后再决定是否引入 Channel Tile。

### 数据打包和计算并行度分离

`pixel_t` 的物理 Channel 数为 `MAX_CHANNELS`，但 Systolic 的 Channel 并行度使用独立编译期参数：

```text
C_PAR
```

两种情况：

```text
C_PAR = MAX_CHANNELS
    每个 PE 每周期完成完整 Channel 点积

C_PAR < Cin
    Systolic 锁存 pixel_t，在 PE 内执行多个 Channel Fold
```

Im2Row 不沿 `cin` 维重复输出数据。

### 空间打包约定

Im2Row 输入为标量空间流：

```text
1 个 Stream Token = 1 个 pixel_t
```

Im2Row 输出为 P 路空间流：

```text
1 个 Stream Token = P 个 pixel_t + spatial_valid
```

P 个输出 Lane 按全局 `row×W+col` 顺序连续分组，允许跨越输出行。只有整帧最后一个 Tile 可能包含无效 Lane。

卷积完成后输出仍是全局连续空间顺序。若 Pool 或外部接口要求每个包按行对齐，应在对应模块入口或卷积输出组装阶段增加 Repacker；是否需要 Repacker 在 Pool 设计阶段确定。

## Im2Row 模块

详细设计见 [Im2Row 模块详细设计](im2row.md)。

### 固定卷积参数

```text
Kernel  = 3×3
Padding = 1
Stride  = 1
P       = 4
```

运行时参数：

```text
H
W
Cin
```

当前优化方案增加约束：

```text
W >= P
```

该约束保证一个 P 路连续空间 Tile 最多跨越两行，并用于证明 Line Buffer Bank 无冲突。

### 目标接口

输入：

```cpp
hls::stream<pixel_t>& input_stream
```

输入契约：

- 一个 Token 是一个空间像素；
- 每帧恰好输入 `H×W` 个 Token；
- 顺序为 NHWC 的空间行优先顺序；
- 输入不携带 Padding；
- 上游必须遵守反压。

输出：

```cpp
struct activation_pack_t {
    pixel_t pixel[P];
    ap_uint<P> spatial_valid;
};
```

一个输出 Token 包含 P 个空间位置在同一个 `(kh,kw)` 上的完整 Channel 激活。

输出顺序：

```text
kh → kw
```

每个空间 Tile 固定输出9个 Token。

### 优化后的内部结构

```text
首行预装
    ↓
全局连续 P 路输出坐标生成
    ↓
每 Lane 独立 Padding 判断
    ↓
2 行奇偶 × P 列 Bank 循环 Line Buffer
    ↕
小型 Pending / Forward Buffer
    ↓
每周期 P 路 Channel-packed 读取
    ↓
9周期 kh→kw Scheduler
    ↓
广播型 P×P Systolic
```

已取消旧版结构：

- P 路空间输入包；
- 物理 Padding 行列；
- `3×(P+2)` Sliding Strip；
- P 个完整3×3候选窗口；
- 跨行窗口 Carry Collector；
- P×K Patch Tile Buffer；
- Im2Row 内部 Channel 串行循环。

### 循环 Line Buffer

目标组织：

```cpp
pixel_t line_buffer[2][P][ceil(MAX_W/P)];
```

地址：

```text
row_bank = input_y & 1
col_bank = input_x % P
address  = input_x / P
```

输入行 `y+2` 在元素完成最后一次读取后覆盖输入行 `y` 的同列槽位。覆盖按列渐进发生，因此同一个物理 Row Bank 可以同时包含旧行尾部和新行头部。

每个有效 Lane 使用自己的输出行列坐标计算 `(input_y,input_x)`。跨行 Tile 的两段 Lane 属于相邻输入行，进入不同 Row Bank；同一段内是至多 P 个连续列，进入不同 Col Bank。因此在 `W>=P` 条件下，每周期 P 路读取不存在相同 `(row_bank,col_bank)` 冲突。

### Pending Buffer

新输入像素可能已经是当前 Tile 的 Bottom 数据，但两行以前的同列槽位仍会被后续窗口使用，因此不能提前覆盖。

第一版使用：

```text
PENDING_DEPTH = P+2
```

Pending 项保存：

```text
pixel
真实 row
真实 col
valid
```

读取优先级：

```text
Padding 0 → Pending 精确坐标旁路 → Line Buffer
```

旧槽位完成最后一次读取后，Pending 项才提交到对应 Line Buffer Bank。

### Padding

不再扫描物理 Padding 行，也不把 Padding 写入 RAM。每个 Lane 独立计算：

```text
input_y = output_y + kh - 1
input_x = output_x + kw - 1
```

坐标越界时直接产生全零 `pixel_t`。

### 输入消费速率和反压

开始输出前预装原始第0行，共读取 W 个输入 Token。

每个 Tile 的前6个周期，即 `kh=0` 和 `kh=1` 阶段，按需读取至多一个新输入；后3周期不读取外部输入。P=4 时第一批最多额外需要5个真实像素，稳态每 Tile 平均推进4个真实输入像素。

模块不会无条件每周期读取输入。当前不需要数据、Pending 已满或输出被阻塞时，通过 Stream 反压暂停上游。

### 性能预期

```text
输出 II                    = 1 Token/周期
每空间 Tile               = 9 周期
空间吞吐                  = P/9 个输出位置/周期
每帧输出 Token            = ceil(H×W/P)×9
启动预装                  = W 个输入读取周期
```

上述结果要求下游每周期接受一个 Token。“9周期”表示9次连续成功的 Stream 输出；若 Systolic 正在执行 Channel Fold 或其他操作并施加反压，Im2Row 保持状态，实际墙钟周期相应增加。实际 Latency 和资源必须通过综合确认。

### Im2Row 与 Systolic

广播型 `P×P` PE 阵列每个 `(kh,kw)` 接收：

```text
P 个 Channel-packed activation
P 个输出通道对应的 Channel-packed weight
```

PE 逻辑：

```text
PE[i][j] += dot(activation[i], weight[j])
```

完整 Channel 并行时，9周期得到：

```text
P 个空间位置 × P 个输出通道
```

当 `Cout>P` 时，同一个激活 Tile 必须针对多个输出通道 Tile 重放。该缓冲位于 Systolic 前端，Im2Row 每个空间 Tile只输出一次。

### Im2Row 当前状态

```text
优化架构设计              已确定
详细设计文档              已更新
目标输入输出接口          已确定
连续跨行 Tile             已确定
Line Buffer Bank 映射     已确定，待仿真和综合证明
安全覆盖与 Pending 方案   已确定，待周期实现验证
Golden Model              未实现
HLS 源码                  仍是旧架构，等待重构
C Simulation              未执行
C Synthesis               未执行
RTL Co-simulation         未执行
Systolic 联调             未执行
```

## 与 Pool 的关系

卷积输出最终组装为 Channel-packed Pixel。Pool 对各 Channel 独立执行比较，可以在 Channel 维按资源情况进行 UNROLL。

由于 Im2Row 和 Systolic 的空间 Tile 允许跨行，Pool 是否直接接受全局连续 P 路包，或者在入口增加行对齐 Repacker，需要在 Pool 的 Line Buffer 和窗口并行方案确定后统一设计。

## 模块开发状态

| 模块 | 设计文档 | 代码 | C Simulation | C Synthesis | RTL Co-simulation |
|---|---|---|---|---|---|
| Im2Row | 优化版已完成 | 旧版待重构 | 未执行 | 未执行 | 未执行 |
| Systolic | 讨论中 | 未实现 | 未执行 | 未执行 | 未执行 |
| Conv | 待设计 | 未实现 | 未执行 | 未执行 | 未执行 |
| Pool | 待设计 | 未实现 | 未执行 | 未执行 | 未执行 |
| FC | 待设计 | 未实现 | 未执行 | 未执行 | 未执行 |
| Top | 待设计 | 未实现 | 未执行 | 未执行 | 未执行 |

## 当前待定事项

- 完整 MNIST CNN 网络结构和每层 H/W/Cin/Cout；
- 最终 `MAX_H`、`MAX_W`、`MAX_CHANNELS`；
- 目标 FPGA Part 和目标时钟；
- `C_PAR` 及 PE 内 Channel Fold；
- 输入 Feature Map Loader 和片上缓冲；
- 权重加载、更新和片上存储；
- Systolic 激活 Tile 重放；
- 累加位宽、Bias、Requantization 和激活函数；
- Pool 是否接受跨行 P 路包；
- FC 并行度；
- 顶层 AXI4-Stream、AXI4 Master 和 AXI-Lite 接口；
- 多层之间的 FIFO 深度和反压路径；
- 最终 HLS Tcl 工程配置。

## 文档维护规则

1. 编码前在 `hls/docs/` 中记录模块职责、接口、数据顺序、硬件结构和验证计划。
2. 代码实现后更新模块文档中的实际数据类型、状态机和 HLS 编译指示。
3. C Simulation 后记录测试尺寸、边界、Bank 冲突、覆盖时机和结果。
4. C Synthesis 后记录目标时钟、实际 II、Latency、LUT、FF、BRAM/LUTRAM 和端口结构。
5. RTL Co-simulation 后记录输入气泡、输出反压和接口时序结果。
6. 模块状态变化后同步更新本 README 的总体数据流、跨模块约定和状态表。
7. 如果综合结果迫使架构调整，必须同时更新代码、模块文档和本 README，避免文档与实现失配。
