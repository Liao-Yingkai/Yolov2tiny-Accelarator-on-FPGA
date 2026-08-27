# MNIST FPGA Accelerator

## 项目简介

本项目计划使用 Vitis HLS 开发一个面向 MNIST 推理的 FPGA CNN 加速器。设计重点是使用流式数据通路、片上缓冲和可配置并行度减少中间数据搬运，并逐步完成卷积、池化、全连接和顶层控制模块。

当前工程处于架构设计阶段，尚未形成可综合的完整加速器。

本 README 是项目级设计文档，将在开发期间持续维护。各模块的详细微架构、接口、HLS 编译指示和测试方案保存在同一目录下的模块文档中；本文件只记录跨模块约定、总体数据流、开发状态和重要设计决策。

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
- 模块之间优先使用 `hls::stream` 建立流式数据通路。
- 当前阶段暂不设计 DDR、AXI Master、DMA、输入数据加载和权重加载。
- 激活采用对称 int8 量化，零点为 0。
- Feature Map 逻辑布局采用 NHWC。
- 一个空间像素的全部物理 Channel 打包为一个 `pixel_t`。
- 每个空间数据包包含 P 个 Pixel，实现 P 路空间并行。
- 物理硬件规模由编译期最大参数决定，实际 H、W、Cin、Cout 可在允许范围内作为运行时参数。
- 模块代码注释、设计文档、提交信息和工程说明统一使用中文。
- 所有性能结论必须以 C Synthesis 和 RTL Co-simulation 结果为准。

## 计划中的加速器数据流

```text
输入/数据加载（待设计）
          │
          ▼
      Conv 模块
  ┌─────────────────────┐
  │ Im2Row              │
  │ 广播型 P×P Systolic │
  │ Bias/Requant/激活   │
  │ Cout Channel 组装   │
  └─────────────────────┘
          │
          ▼
      Pool 模块
          │
          ▼
      后续 Conv
          │
          ▼
       FC 模块
          │
          ▼
       分类结果
```

具体网络层数、各层 Channel 数量、权重布局和顶层加载方式将在后续设计中补充。

## 跨模块数据约定

### Channel-packed Pixel

一个空间位置的全部 Channel 统一打包：

```text
pixel = [channel0, channel1, ..., channel(MAX_CHANNELS-1)]
```

运行时只有前 `Cin` 或 `Cout` 个 Channel 有效。采用 Channel-packed 的主要原因是：

- Im2Row 不需要额外遍历物理输入 Stream 的 Channel 维；
- Systolic 输出可以重新组装为下一层直接使用的 Pixel；
- Pool 对不同 Channel 相互独立，可在 Channel 维使用 UNROLL；
- 当前层的 `Cout` 可以直接作为下一层的 `Cin`。

风险是总线宽度随 `P×MAX_CHANNELS` 增长，可能增加寄存器、RAM 带宽和布线压力。第一版先采用完整 Channel 打包，综合后再评估是否需要 Channel Tile。

### P 路空间数据包

模块间一个 Stream Token 最多携带 P 个空间像素。进入需要 Line Buffer 的模块前，数据包应按行对齐，同一个包不能跨越两行。行尾不足 P 个像素时使用空间有效 Mask。

卷积计算内部允许把有效空间位置跨行连续收集，以减少 Systolic 的无效 PE Lane。卷积输出端通过行对齐 Repacker 恢复为每个数据包不跨行的格式，再交给 Pool 或下一层 Im2Row。

## Im2Row 模块

详细设计见 [Im2Row 模块设计文档](im2row.md)。

### 模块职责

Im2Row 把 NHWC Feature Map 在线转换成广播型 `P×P` PE 阵列所需的 K 维激活流，不显式生成完整 Im2Row 矩阵。

当前固定卷积参数：

```text
Kernel  = 3×3
Padding = 1
Stride  = 1
P       = 4（第一版计划）
```

运行时参数：

```text
H
W
Cin
```

物理资源按以下编译期参数分配：

```text
MAX_H
MAX_W
MAX_CHANNELS
P
```

### Im2Row 输入

- 输入为 `hls::stream`；
- 一个 Token 包含 P 个 Channel-packed Pixel；
- 输入按行对齐，Token 不跨行；
- 行尾不足 P 个像素时使用空间有效 Mask；
- 逻辑顺序为 NHWC。

### Im2Row 内部结构

```text
输入行尾处理
    ↓
虚拟 Padding Pack Generator
    ↓
P-Bank 双行 Line Buffer
    ↓
3×(P+2) Sliding Strip
    ↓
P 路候选窗口
    ↓
跨行连续窗口收集
    ↓
P×K Patch Tile 双缓冲
    ↓
K Scheduler
```

Padding 不通过完整中间 Feature Map 实现：

- 左右边界在每行内部插入零；
- 顶部和底部生成虚拟零行；
- 内部 Padding 行宽按 P 对齐；
- Window Generator 根据 Padding 坐标判断候选窗口是否为有效输出。

Line Buffer 在列方向划分为 P 个 Bank，每周期更新 P 个连续位置。窗口更新顺序固定为先左移、再注入新像素。

有效窗口按全局 `row×W+col` 顺序连续收集。空间 Tile 可以跨行，最后不足 P 个窗口时使用 `spatial_valid`。

### Im2Row 输出

Window Flatten 顺序：

```text
kh → kw → cin
```

```text
K = 3 × 3 × Cin
```

每个输出 Token 包含 P 个空间位置在同一个 k 上的 activation。K 输出循环目标为 `II=1`：

- 每周期输出 P 个 activation；
- 每 K 个周期完成一个 P 空间位置 Tile；
- Im2Row 每个空间 Tile 只输出一次。

当 `Cout > P` 时，激活针对不同输出通道 Tile 的重放由 Systolic 前端缓冲负责，不属于 Im2Row。

### Im2Row 当前状态

```text
架构设计             已完成
接口和数据顺序       已确定
Padding 方案         已确定
P 路 Line Buffer     已确定
跨行窗口收集         已确定
Patch Tile 双缓冲    已确定
软件参考模型         未实现
HLS 代码             未实现
C Simulation         未执行
C Synthesis          未执行
RTL Co-simulation    未执行
Systolic 联调        未执行
```

## 与 Systolic 的关系

当前计划使用广播型 `P×P` PE 阵列。每周期：

- Im2Row 提供 P 个空间激活；
- 权重模块提供 P 个输出通道权重；
- PE 阵列执行 P×P 次 MAC。

经过 K 个周期后得到 P 个空间位置、P 个输出通道的结果。

当 `Cout > P` 时，同一个 `P×K` 激活 Tile 需要从 k=0 重放。计划在 Systolic 前端设置激活双缓冲：当前 Buffer 遍历全部输出通道 Tile，另一个 Buffer 可以接收下一空间 Tile。

权重在计算阶段计划存放于片上可寻址存储，并按照 `weight_pack[P]` 布局，每周期读取一个 P 路宽字。当前为了简化模块开发，权重连接暂时使用 `hls::stream`；具体的权重加载和片上存储方案后续设计。

Systolic 完成每个输出通道 Tile 后，将 Bias、量化和激活后的结果写入 `P×MAX_CHANNELS` 输出组装缓冲。全部 `Cout` 完成后，输出 P 个 Channel-packed Pixel。

Systolic 的累加位宽、结果导出流水、量化参数和具体实现将在 Systolic 模块设计阶段讨论。

## 与 Pool 的关系

卷积输出重新组装为 Channel-packed Pixel 后，Pool 可以对每个 Channel 独立执行比较。Channel 数量较小时，可以在 Channel 维完全 UNROLL；当 `MAX_CHANNELS` 增大时，需要根据资源和目标频率决定部分展开因子。

Pool 的 Line Buffer、空间并行方式、输入输出打包和边界处理将在 Pool 模块设计阶段补充。

## 模块开发状态

| 模块 | 设计文档 | 代码 | C Simulation | C Synthesis | RTL Co-simulation |
|---|---|---|---|---|---|
| Im2Row | 已完成初版 | 未实现 | 未执行 | 未执行 | 未执行 |
| Systolic | 讨论中 | 未实现 | 未执行 | 未执行 | 未执行 |
| Conv | 待设计 | 未实现 | 未执行 | 未执行 | 未执行 |
| Pool | 待设计 | 未实现 | 未执行 | 未执行 | 未执行 |
| FC | 待设计 | 未实现 | 未执行 | 未执行 | 未执行 |
| Top | 待设计 | 未实现 | 未执行 | 未执行 | 未执行 |

## 当前待定事项

- 完整 MNIST CNN 网络结构和各层 H/W/Cin/Cout；
- `MAX_H`、`MAX_W`、`MAX_CHANNELS`；
- 目标 FPGA Part 和目标时钟周期；
- 输入 Feature Map 加载方式；
- 权重加载、权重更新和片上存储方式；
- Systolic 累加位宽和输出 Epilogue；
- Pool 和 FC 的并行度；
- 顶层 AXI4-Stream、AXI4 Master 和 AXI-Lite 接口；
- 多层之间的 Buffer 和反压控制；
- 最终 HLS Tcl 工程配置。

## 文档维护规则

每个模块开发时应执行以下维护流程：

1. 编码前在 `hls/docs/` 中记录模块职责、接口、数据顺序、硬件结构和验证计划。
2. 代码实现后更新模块文档中的实际数据类型、状态机和 HLS 编译指示。
3. C Simulation 后记录覆盖的测试尺寸、边界条件和结果。
4. C Synthesis 后记录目标时钟、实际 II、Latency 和资源使用。
5. RTL Co-simulation 后记录接口和时序验证结果。
6. 模块状态变化后同步更新本 README 的总体数据流、跨模块接口和开发状态表。
7. 如果综合结果迫使架构调整，必须同时更新代码、模块文档和本 README，避免文档与实现失配。
