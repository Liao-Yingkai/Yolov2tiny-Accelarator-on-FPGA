# MNIST FPGA Accelerator

本项目使用 Vitis HLS 2022.2 开发面向 MNIST CNN 推理的 FPGA 加速器。设计采用流式接口、片上缓存与空间并行。当前 Im2Row 已形成稳定实现，下一阶段是单物理实例复用的 Broadcast / Output-Stationary Systolic 模块；PostProcess、Pool、FC 和顶层控制将在后续阶段实现。

## 工程原则

- 模块间使用阻塞式 `hls::stream`；
- 激活采用对称八位有符号量化，零点为0；
- 特征图逻辑布局采用 NHWC；
- `pixel_t` 打包一个空间位置的全部物理通道；
- 物理并行度是编译期参数，H、W、Cin、Cout 是运行时参数；
- 注释、文档、断言和工程说明统一使用中文；
- 性能和资源结论必须以综合与协同仿真报告为准。

## 当前数据流

```text
输入加载器或片上输入缓冲
           │
           ▼
标量、通道打包 pixel_t 流
           │
           ▼
Im2Row ── P 路 activation_pack_t ──► 广播型 P×P Systolic
                                              │
                                              ▼
                                 输出通道组装 / Pool / 后续 Conv / FC
```

## 单物理卷积流水引擎

系统目标是只实例化一套物理 Im2Row、Systolic、PostProcess 和 Pool。Conv1、Conv2 不并发，而是分两个阶段顺序复用同一套模块；每个卷积层内部仍通过 `hls::stream<>` 形成模块间流水：

```text
阶段 1：
Input → Im2Row → Systolic → PostProcess → Pool → Pool1 Feature Buffer

阶段 2：
Pool1 Feature Buffer → Im2Row → Systolic → PostProcess → Pool → FC
```

Pool1 Feature Buffer 保存 `14×14×8` 个 INT8 activation，逻辑容量为 1,568 Byte。具体 top-level 单实例调度和 HLS 资源共享方式在各计算模块完成后设计。

## Im2Row 当前状态

Im2Row 固定支持 `3×3`、`stride=1`、`same padding`，每行独立划分 `ceil(W/P)` 个 Tile，不允许 Tile 跨行。尾部不足 P 的 Lane 在9拍中输出全零并携带 `spatial_valid=0`；卷积边界 Padding 也输出零，但有效 Lane 保持 `spatial_valid=1`。

当前网络为两层卷积：Conv1 的 Im2Row 输入是 `28×28`，第一层 Pool 后 Conv2 的 Im2Row 输入是 `14×14`。Conv2 后的 Pool 虽得到 `7×7`，但该数据不会进入第三层卷积，因此 `7×7` 不属于本 Im2Row 的实际工作尺寸。`W<P` 等小宽度仅保留作运行时接口健壮性与边界检查，不参与当前网络的时序和资源决策。

内部采用两个行 Bank 与 `strip[2][3][P+2]` 双 Strip Ping-Pong：

- `current Strip` 连续9拍输出给 Systolic；
- `next Strip` 在这9拍中并行准备；
- `token=8` 后交换，因此 Tile 间和换行间的目标是不出现输出气泡；
- 全局 Token 循环每次迭代只写出一个 `activation_pack_t`，目标 `II=1`；
- Bottom 输入在当前 Tile 的 Token 0～5 中按需突发读取，峰值为每周期一个 `pixel_t`；
- 首行预装后开始输出，输入总量为 `H×W`，输出总量为 `H×ceil(W/P)×9`。

完整的周期时序、双 Strip 所有权、换行两 Beat 预取、反压语义和未确认风险见 [Im2Row 模块详细设计](im2row.md)。

Im2Row 已经过多轮综合，现有结果确认稳态 Token 循环达到 `II=1`，Line Buffer 映射到 BRAM。正式功能验证、RTL Co-simulation 和硬件反压行为仍待完成；反压下的 token 守恒和内部状态停顿必须在后续验证阶段重点检查。

## Systolic 第一版设计

Systolic 采用 `4 spatial × 4 Cout` Broadcast / Output-Stationary PE Array，每个 PE 固定并行处理 8 个物理输入通道，理论物理并行度为 128 个 INT8 multiplication/cycle。

- Conv1 使用运行时 `Cin=1、Cout=8`，无效的 7 个物理通道填零；
- Conv2 使用运行时 `Cin=8、Cout=16`；
- 同一 spatial tile 的 9 个 activation token通过本地 buffer replay；
- 每层只在第一个 spatial tile 期间读取一次整层权重，后续 spatial tile 从 Weight Buffer replay；
- Weight Buffer 使用四个 64-bit BRAM bank，每拍聚合提供 256-bit 权重；
- 输出保持 `output_y → tile_x → cout_tile` 顺序；
- Bias、Requant、ReLU 不进入 PE，后续由独立 PostProcess 完成；
- 第一版允许 Cout tile 边界存在额外控制或输出周期，不要求严格每 9 周期输出一个结果 Tile。

完整接口、Buffer、时序、事务守恒、HLS 约束和验证要求见 [Systolic / Broadcast PE Array 设计](systolic.md)。
