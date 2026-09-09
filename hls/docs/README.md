# MNIST FPGA Accelerator

本项目使用 Vitis HLS 2022.2 开发面向 MNIST CNN 推理的 FPGA 加速器。设计采用流式接口、片上缓存与空间并行。当前 Im2Row、Broadcast / Output-Stationary Systolic、ConvPostProcess、Pool 与独立FC MAC/后处理已完成第一版源码；顶层控制将在后续阶段实现。

## 工程原则

- 模块间使用阻塞式 `hls::stream`；
- 激活采用对称八位有符号量化，零点为0；
- 特征图逻辑布局采用 NHWC；
- `pixel_t` 打包一个空间位置的全部物理通道；
- 物理并行度固定为 `4 spatial × 4 Cout × 8 Cin`；接口仍保留 H、W、Cin、Cout 以支持同一物理模块分时复用；
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
                               ConvPostProcess ──► Pool / 后续 Conv / 独立 FC
```

## 单物理卷积流水引擎

系统目标是只实例化一套物理 Im2Row、Systolic、ConvPostProcess 和 Pool。Conv1、Conv2 不并发，而是分两个阶段顺序复用同一套模块；每个卷积层内部仍通过 `hls::stream<>` 形成模块间流水：

```text
阶段 1：
Input → Im2Row → Systolic → ConvPostProcess → Pool → Pool1 Feature Buffer

阶段 2：
Pool1 Feature Buffer → Im2Row → Systolic → ConvPostProcess → Pool → 独立 FC
```

Pool1 Feature Buffer 保存 `14×14×8` 个 INT8 activation，逻辑容量为 1,568 Byte。其属于顶层阶段间缓冲，因为Conv1和Conv2复用同一卷积流水硬件。Pool2默认直接流入独立FC；FC在第一个输出 Tile 消费 Pool2 的98个8-channel Pack，并写入自身的98x64-bit BRAM，供第二个输出 Tile 回放。具体顶层调度仍在后续阶段设计。

当前分支只支持下列两个层调用；在综合态，Im2Row/Systolic 都只使用 `Cin` 形成一位 `is_conv2` 模式选择，H/W/Cout 不进入核心调度。非综合构建会断言完整配置，防止测试误用接口。

| 模式 | Im2Row / Systolic 配置 | Spatial Tile 数 | Cout Tile 数 |
|---|---|---:|---:|
| Conv1 | `(H,W,Cin,Cout)=(28,28,1,8)` | 196 | 2 |
| Conv2 | `(H,W,Cin,Cout)=(14,14,8,16)` | 56 | 4 |

## Im2Row 当前状态

Im2Row 固定支持 `3×3`、`stride=1`、`same padding`，每行独立划分 `ceil(W/P)` 个 Tile，不允许 Tile 跨行。尾部不足 P 的 Lane 在9拍中输出全零并携带 `spatial_valid=0`；卷积边界 Padding 也输出零，但有效 Lane 保持 `spatial_valid=1`。

当前网络为两层卷积：Conv1 的 Im2Row 输入是 `28×28`，第一层 Pool 后 Conv2 的 Im2Row 输入是 `14×14`。Conv2 后的 Pool 虽得到 `7×7`，但该数据不会进入第三层卷积，因此 `7×7` 不属于本 Im2Row 的工作尺寸。

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
- Bias、Requant、ReLU 不进入 PE，后续由独立 ConvPostProcess 完成；
- 第一版允许 Cout tile 边界存在额外控制或输出周期，不要求严格每 9 周期输出一个结果 Tile。

完整接口、Buffer、时序、事务守恒、HLS 约束和验证要求见 [Systolic / Broadcast PE Array 设计](systolic.md)。

## Pool 第一版设计

Pool固定实现2×2、stride=2的INT8有符号MaxPool，消费ConvPostProcess保持的
`row -> tile_x -> cout_tile` 事务顺序。偶数输入行写入16×128-bit单端口Line
Buffer，奇数行读取上一行并完成四值有符号最大值；之后按 `pooled_H -> pooled_W
-> channel_pack` 紧密输出。Pool1产生196个8-channel pack，Pool2产生98个。

Pool1的输出由顶层写入阶段间Feature Buffer供Conv2使用；Pool2默认直连独立FC，
不复用当前Systolic，也不要求Pool侧FC Input Buffer。详细设计见
[Pool 模块详细设计](pool.md)。本阶段仅完成源码与静态检查，尚未进行C仿真、协同仿真或综合；BRAM、II、时序和资源结论待验证。

## ConvPostProcess 第一版设计

ConvPostProcess 消费 Systolic 的 `acc_tile_t`，保持 `spatial_tile(row -> tile_x) -> cout_tile`
顺序输出 `activation_tile_t`。每个输出通道使用独立的 INT32 bias、multiplier、
shift；首个 spatial tile 读取并缓存所有 Cout tile 的参数，之后回放寄存器缓存。

- 数值顺序固定为 `INT32 bias add → ReLU → INT64 multiply → 直接算术右移 → 上限127饱和`；
- 运行时不加入任何 rounding offset，保持与 golden model 当前 requant 的直接截断一致；
- 无效 spatial lane 强制输出零，不经过 bias；
- Conv1 处理 `196×2=392` 个 Tile、读取2个参数 Pack；Conv2 处理 `56×4=224` 个 Tile、读取4个参数 Pack；
- spatial loop 已确认达到 `II=1`，每周期完成1条 spatial lane 的4条 Cout lane requant；
- 在 `xc7z020-clg400-2`、10 ns 约束下，csynth 估计时钟为 7.050 ns，外层为13 cycles/Tile；
  Conv1/Conv2 分别为5,097/2,913 cycles，快于现有 Systolic 的19 cycles/Tile，因而不是当前瓶颈；
- 资源为0 BRAM、12 DSP、3,126 FF、2,933 LUT；4个32×32→64 位乘法器各占3 DSP，
  动态右移约占716 LUT，参数缓存为寄存器。

详细接口、位宽、缓存、验证范围、已确认综合结果及后续两级右移候选见 [ConvPostProcess 设计](conv_postprocess.md)。

## 独立 FC 第一版设计

固定FC将 Pool2 的 `7×7×16=784` 个 INT8 activation 分类为10个 signed INT8 logit。输入顺序保持 Pool2 的紧密 NHWC：`y -> x -> 8-channel pack`，共98个 Pack；权重顺序为 `out_tile(2) -> token(98)`，每个320-bit逻辑 Pack 有5条输出 lane和8条输入 lane。第一输出 Tile 边计算边写入98x64-bit FC Input Buffer，第二 Tile 从该 BRAM 回放，两个 Tile 各输出一个5路 INT32 accumulator Pack。

FC MAC 后使用 `src/postprocess/fc_postprocess.*` 的独立后处理：每组5 lane 串行复用一套32x32 signed requant数据通路，执行 bias、INT64乘法、直接算术右移和 `[-128,127]` 双向饱和，不做 ReLU 或额外 rounding。输出一次完整10路 logit Pack。FC 与卷积 Systolic 的滑窗/空间并行计算不同，既不复用其阵列也不复用其权重缓存。详细接口、缓冲和未验证项见 [独立 FC 设计](fc.md)；本阶段未运行C仿真、协同仿真或综合。

## 相对 dev 通用实现的固定网络化

| 内容 | 本分支变化 | 保持不变 |
|---|---|---|
| 参数与调度 | 删除可覆盖的 P/H/W/C 宏、通用宽高/Tile 计算和综合态非法参数返回；固定为 Conv1/Conv2 两种模式 | Im2Row、Systolic 的函数签名和 stream struct 字段布局 |
| Im2Row 边界 | 固定 Conv1/Conv2 的 Tile 数、tail valid、Bottom load 和预取 mask；输入读出后立即完成 Conv1 通道零填充 | 2-row circular line buffer、4-way banking、双 Strip、9-token 顺序和 write-first 反压语义 |
| Systolic 调度 | `row × tile_x` 合并成固定数量的 `spatial_tile`；Cout tile 固定为2或4 | 唯一 `4×4×8` PE、activation/weight replay BRAM、dot tree、累加和输出顺序 |

## 生成物目录

今后的 csynth/csim/cosim 工程与日志统一写入仓库根目录 `build/hls/<run-name>/`，不要写入 `hls` 源码树；该目录由 Git 忽略。

## 模块级验证

`cnn_accelerator/testbench/` 保留两组独立的模块 testbench：原有 `test_*.cpp`
使用确定性程序数据，新增 `test_*_pytorch.cpp` 直接读取 Python exporter 和
`dump_trace.py` 的单图 golden 数据。后者每个文件只调用一个 DUT top，输入直接来自
PyTorch 的对应上游边界，不将多个 HLS 模块串接；输出按硬件 stream 顺序恢复坐标后逐项
比较，不缓存完整 feature map 做重排。运行时 CSV 位于
`simulation_data/pytorch/<module>/`。当前 Python header 使用绝对路径 include，替换
trace 或模型结构后须同步检查 shape 断言和索引映射。
