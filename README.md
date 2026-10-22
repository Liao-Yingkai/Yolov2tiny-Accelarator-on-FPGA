# MNIST FPGA Accelerator

面向 Zynq-7000 的 INT8 MNIST CNN 推理加速器练手项目，目标器件为 `xc7z020-clg400-2`。仓库包含 Vitis HLS 计算与 DDR 接口源码、Vivado 硬件集成工程，以及 Zynq PS 端的 Vitis 裸机测试工程，使用工具版本为 **2022.2**。

计算模块包括 Im2Row、Broadcast / Output-Stationary Systolic、ConvPostProcess、MaxPool、独立 FC 及后处理、Argmax。模块间使用流式接口，DDR 系统入口通过 `m_axi` 搬运数据、`s_axilite` 设置地址和控制运行，最终输出10个 INT8 logits 与预测类别。

已有统一卷积 Loader 版本的 HLS 综合及连续三图 Verilog/XSim co-sim 验证记录，见下方“最新：统一卷积 Loader 验证”。HLS 结果不等于 Vivado 实现时序或上板结果；仓库中保留的 IP/XSA 也不会随 HLS 源码修改自动更新，使用时须保证各版本匹配。

## 仓库目录

```text
.
├── README.md                         仓库总览与 HLS 设计摘要
├── hls/
│   ├── cnn_accelerator/
│   │   ├── src/                      可综合 C++、固定配置及 DDR 布局
│   │   ├── testbench/                模块级与完整推理功能测试
│   │   └── script/                   Stream / DDR 核心 C Simulation 入口
│   └── docs/                         各模块与顶层设计文档
├── vivado/
│   ├── README.md                     硬件工程打开、重生成与版本管理说明
│   ├── cnn_accel_ddr/
│   │   ├── cnn_accel_ddr.xpr          Vivado 工程入口
│   │   ├── cnn_accel_ddr.srcs/        Block Design 与 IP 配置
│   │   ├── ip_repo/export/           导出的 HLS 自定义 IP
│   │   └── vitis/                    PS 裸机应用、平台与硬件交接文件
│   └── cnn_accel_cam/                OV5640 640×480 HDMI 预览与 PS 取图工程
│       └── vitis/cnn_accel_cam/      停车取帧、PS预处理与调试数组
└── build/hls/                        本地 HLS 工程、日志和报告（Git 忽略）
```

## 固定网络与数值格式

Batch 固定为1，FPGA 内部特征图采用 NHWC，网络如下：

| 阶段 | 运算 | 输出形状 |
|---|---|---|
| 输入 | 灰度图 | `28×28×1` |
| Conv1 | `3×3, stride=1, padding=1`，ReLU | `28×28×8` |
| Pool1 | `2×2, stride=2` MaxPool | `14×14×8` |
| Conv2 | `3×3, stride=1, padding=1`，ReLU | `14×14×16` |
| Pool2 | `2×2, stride=2` MaxPool | `7×7×16` |
| FC | `784 → 10`，无 ReLU | 10个 logits |
| Argmax | 并列最大值取最小索引 | 类别 `0～9` |

Activation/weight 为 signed INT8，accumulator/bias 为 INT32，零点为0；requant 使用整数 multiplier 与 shift，直接算术右移，不增加运行时舍入偏置。卷积先加 bias、ReLU 再 requant；FC 保留负数并饱和到 `[-128,127]`。

Conv1/Conv2 顺序复用唯一一套 `Im2Row → Systolic → ConvPostProcess → Pool`。层内 DATAFLOW，层间完整 barrier；Pool1 中间结果通过片上 BRAM FIFO 保存。FC 为独立硬件模块，直接接收 Pool2 输出。

## 各工程入口与使用流程

### HLS：计算源码与仿真

- Stream 计算核心：[cnn_core.cpp](hls/cnn_accelerator/src/core/cnn_core.cpp)，Top 函数为 `cnn_accelerator`。
- DDR 系统顶层：[cnn_accel_ddr.cpp](hls/cnn_accelerator/src/top/cnn_accel_ddr.cpp)，Top 函数为 `cnn_accelerator_ddr`，用于导出 Vivado IP。
- 网络配置：[config.h](hls/cnn_accelerator/src/config.h)；事务类型：[types.h](hls/cnn_accelerator/src/types.h)；DDR 数据协议：[ddr_layout.h](hls/cnn_accelerator/src/axi/ddr_layout.h)。
- 详细设计：[顶层](hls/docs/top.md)、[DDR Wrapper](hls/docs/ddr_wrapper.md)、[Im2Row](hls/docs/im2row.md)、[Systolic](hls/docs/systolic.md)、[卷积后处理](hls/docs/conv_postprocess.md)、[Pool](hls/docs/pool.md)、[FC](hls/docs/fc.md)、[Argmax](hls/docs/argmax.md)。

在已配置 Vitis HLS 2022.2 环境的终端中，从仓库根目录执行，例如：

```bash
mkdir -p build/hls
cd build/hls
vitis_hls -f ../../hls/cnn_accelerator/script/hls_core.tcl
vitis_hls -f ../../hls/cnn_accelerator/script/hls_ddr.tcl
```

两个脚本分别执行各自 Top 的 C Simulation，不自动运行综合、RTL co-sim 或 IP 导出。工程与启动日志放在 `build/hls/`，不要在源码目录启动工具。

PyTorch 训练和导出工程不包含在本仓库。HLS PyTorch testbench 当前通过绝对路径引用 `/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/` 下的模型与 trace 头文件；在其他机器上运行前，需准备相同导出数据并调整 include 路径。现有完整推理 testbench 连续测试三张图，每张比较全部 logits；DDR 测试同时检查预测类别与结果区边界。

### Vivado：PL 硬件集成

使用 Vivado 2022.2 打开 [cnn_accel_ddr.xpr](vivado/cnn_accel_ddr/cnn_accel_ddr.xpr)，工程集成 Zynq PS 与 DDR 加速器 IP。克隆后先为 Block Design 重新生成 Output Products，再运行综合、实现及 bitstream 生成。已有自定义 IP 保存在 `ip_repo/export/`，不需要先运行 HLS 才能打开现有工程；修改 HLS 后则需重新导出并更新 IP。

工程保留必要源文件和直接引用的 wrapper，缓存与运行产物按需重生成。具体保留项、迁移注意事项及历史工程打开验证见 [Vivado 工程说明](vivado/README.md)。

### Vitis：PS 裸机测试

Vitis 工作区位于 `vivado/cnn_accel_ddr/vitis/`，主要包含：

- `system_wrapper`：基于 XSA 的平台、BSP 配置和 FSBL 源码。
- `cnn_accel_ddr`：裸机应用，[main.c](vivado/cnn_accel_ddr/vitis/cnn_accel_ddr/src/main.c) 负责 DDR 缓冲区准备、缓存维护、设置 IP 地址、启动及轮询完成，最后核对 logits 与类别。
- `cnn_accel_ddr_system`：应用系统工程。

先构建平台/BSP，再构建应用；使用与当前 HLS/DDR 布局一致的 bitstream 和 XSA，完成 PS 初始化及 PL 配置后运行 ELF。应用使用本地 `mnist_test_data.h`，默认依次测试三张图，不依赖训练目录的绝对 include；更新模型时，用应用中的 `tools/export_test_data.py` 重新提取模型和测试数据。详细步骤、结果格式和缓存说明见 [裸机应用说明](vivado/cnn_accel_ddr/vitis/cnn_accel_ddr/src/README.txt)。

Vitis 工程元数据仍含原机器绝对路径，迁移工作区后可能需要重新绑定平台/XSA，再重建 BSP；不保证直接在任意路径打开即可编译。HLS、导出的 IP、bitstream、XSA、驱动及应用数据必须保持匹配，不能混用旧版 DDR 权重布局。

### 摄像头：480p 预览、PS 取帧与CNN推理

摄像头工程入口为 [cnn_accel_cam.xpr](vivado/cnn_accel_cam/cnn_accel_cam.xpr)，应用入口为 [main.c](vivado/cnn_accel_cam/vitis/cnn_accel_cam/src/main.c)。640×480 HDMI 预览已由用户上板验证；PS应用现已接入用户新增的CNN IP，取帧、预处理及新CNN推理链路仍需联合上板验证。

PS 通过短暂固定 VDMA MM2S 读取帧，借助已有 Dynamic Genlock 保持目标帧稳定，将完整 640×480 原始 BGR888 数据复制到 CPU 自有快照缓冲，并在解除停车前逐字节核对复制结果。采样目标周期为 100ms，不改变摄像头采集或 HDMI 刷新率，也不等于已证明每秒取得十张不同的新帧。

成功取帧后，PS 对中央280×280 ROI做整数灰度、10×10均值缩小至28×28、反色，再按训练端float32归一化及模型输入scale量化为单通道signed int8，不使用LUT。camera_cnn模块把784字节输入送入DDR CNN IP，模型只加载一次，通过轮询读取16字节结果并检查类别/首次argmax/保留字节。正常每轮仅输出`digit=<类别>`，不打印调试数组地址、统计或耗时；取帧失败跳过推理并重试，CNN busy/5秒超时等错误立即结束，须板级复位后重启，不能复用可能仍被PL访问的缓冲。使用`python3 vivado/cnn_accel_cam/tools/export_camera_model.py --generated-dir /path/to/generated`同一命令同步12个模型数组与输入scale，无trace图片。详细DDR布局、缓存方向、上板步骤及验证边界见 [摄像头推理说明](vivado/cnn_accel_cam/tools/CAMERA_INPUT.md)。100ms调度目标不保证10fps或识别精度，新链路尚未完成板测。

## 版本管理与验证范围

保留 HLS 源码/测试/设计文档、Vivado 工程及自定义 IP、Vitis 应用源码/工程描述/XSA/BSP 配置。`build/hls/`、工具缓存、日志、可再生成产物和各级 `.vscode/` 不纳入 Git，规则见 [.gitignore](.gitignore)。下文指向 `build/hls/` 的报告仅在相应本地验证产物存在时可打开，不随仓库分发。

下文保留各模块设计摘要和按开发阶段记录的验证结论；其中“本轮”指对应记录形成时的阶段，最新 HLS 集成结论以“最新：统一卷积 Loader 验证”为准。本次 README 迁移没有重新运行仿真、综合、Vivado 或板测。

## 工程原则

- 模块间使用阻塞式 `hls::stream`；
- 激活采用对称八位有符号量化，零点为0；
- 特征图逻辑布局采用 NHWC；
- `pixel_t` 打包一个空间位置的全部物理通道；
- 物理并行度固定为 `4 spatial × 4 Cout × 8 Cin`；卷积相关模块通过一个调用级 `layer_mode_t` 复用同一物理硬件；
- 注释、文档、断言和工程说明统一使用中文；
- 性能和资源结论必须以综合与协同仿真报告为准。

手写HLS代码按C++14整理：类型别名使用`using`，普通整数编译期配置使用
`constexpr`，文件内辅助函数由匿名namespace限定，不重复添加`static`。
跨调用保存状态的局部`static`、原生硬件数组、初始化和显式位宽转换仍按硬件语义
保留；Vitis的C应用及工具生成的RTL/驱动不套用这套C++写法。

## 当前数据流

```text
输入加载器或片上输入缓冲
           │
           ▼
NHWC 通道打包 pixel_t 流
           │
           ▼
Im2Row ── P 路 activation_pack_t ──► P×P Systolic（OS / Broadcast 编译期选择）
                                              │
                                              ▼
                               ConvPostProcess ──► Pool / 后续 Conv / 独立 FC
```

## 单物理卷积流水引擎

系统目标是只实例化一套物理 Im2Row、Systolic、ConvPostProcess 和 Pool。Conv1、Conv2 不并发，而是由顶层固定两次顺序调用同一个`run_layer_phase()`；每次调用内部通过 `hls::stream<>` 形成模块间流水：

```text
阶段 1：
Input → Im2Row → Systolic → ConvPostProcess → Pool → Pool1 Feature FIFO

阶段 2：
Pool1 Feature FIFO → Im2Row → Systolic → ConvPostProcess → Pool → 独立 FC
```

Pool1 Feature FIFO保存 `14×14×8` 个 INT8 activation，逻辑容量为1,568 Byte。它是
`run_layer_phase()`内持久的`static hls::stream<channel_pack_t>`，深度固定为196并约束为
BRAM FIFO：Layer1由Output Router写满，Layer2由Input Router按原顺序读空。由于Input Router
在源码中先于Output Router调用，普通C++语义也符合两phase先写后读的顺序。Pool2直接流入独立FC；FC在第一个输出 Tile 消费 Pool2 的98个8-channel Pack，并写入自身的98x64-bit BRAM，供第二个输出 Tile 回放。Top的具体接口、Barrier和尚待工具验证的反馈通道风险见 [Stream 计算核心设计](hls/docs/top.md)。

`hls/cnn_accelerator/src/config.h` 是固定网络、并行度及派生事务数量的唯一编译期来源。卷积相关模块只接收一个调用级模式：`LAYER1_MODE=false` 选择 Conv1/Pool1，`LAYER2_MODE=true` 选择 Conv2/Pool2；模式不进入每个stream事务。

| 模式 | 固定配置 | Spatial Tile 数 | Cout Tile 数 |
|---|---|---:|---:|
| Conv1 | `LAYER1_MODE`，`28×28×1 -> 8` | 196 | 2 |
| Conv2 | `LAYER2_MODE`，`14×14×8 -> 16` | 56 | 4 |

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

完整的周期时序、双 Strip 所有权、换行两 Beat 预取、反压语义和未确认风险见 [Im2Row 模块详细设计](hls/docs/im2row.md)。

Im2Row 已经过多轮综合，历史结果确认稳态 Token 循环达到 `II=1`，Line Buffer 映射到 BRAM。本轮没有修改其源码，程序数据和 PyTorch 数据两组 testbench 功能检查通过；新版集成的 RTL Co-simulation 和硬件反压仍须另行验证。

## Systolic 第一版设计

> 本轮修订：Systolic 自行处理有效 Cin，保留 `pixel_t`/`weight_pack_t`；Loader 不补零，权重 header 采用紧凑硬件顺序。接口契约见 [Systolic 设计](hls/docs/systolic.md)，统一 Loader 后的集成综合与三图 RTL 验证见 [DDR Wrapper](hls/docs/ddr_wrapper.md)。

Systolic 采用 `4 spatial × 4 Cout` Broadcast / Output-Stationary PE Array，每个 PE 固定并行处理 8 个物理输入通道，理论物理并行度为 128 个 INT8 multiplication/cycle。

- Conv1（`LAYER1_MODE`）只使用 channel0 的乘积，其余物理通道即使两侧都非零也不参与点积；Im2Row 保持原通道规范化契约；
- Conv2（`LAYER2_MODE`）使用全部8个物理输入通道；
- 同一 spatial tile 的 9 个 activation token通过本地 buffer replay；
- 每层只在第一个 spatial tile 期间读取一次整层权重，后续 spatial tile 从 Weight Buffer replay；
- Weight Buffer 使用四个 64-bit BRAM bank，每拍聚合提供 256-bit 权重；
- 输出保持 `output_y → tile_x → cout_tile` 顺序；
- Bias、Requant、ReLU 不进入 PE，后续由独立 ConvPostProcess 完成；
- 第一版允许 Cout tile 边界存在额外控制或输出周期，不要求严格每 9 周期输出一个结果 Tile。

完整接口、Buffer、时序、事务守恒、HLS 约束和验证要求见 [Systolic / Broadcast PE Array 设计](hls/docs/systolic.md)。

## 传统 Output-Stationary 脉动阵列

另提供的 `systolic_OS.cpp`：Activation 向右传播、Weight 向下传播，边界按
lane 延迟0/1/2/3步，部分和保留在 PE 内。每个 PE 仍是8路 Cin SIMD 点积，
不是将 Cin 再串行化的标量阵列。当前计算核心和两个 Systolic testbench
统一读取`config.h`中的`CNN_SYSTOLIC_USE_OS`：默认`1`选择此实现，`0`选择广播式。
也可通过编译选项`-DCNN_SYSTOLIC_USE_OS=0`切换，但必须在全部相关编译单元中
保持一致。这是编译期选择，不是额外实例化第二套卷积阵列。

## Pool 第一版设计

Pool固定实现2×2、stride=2的INT8有符号MaxPool，消费ConvPostProcess保持的
`row -> tile_x -> cout_tile` 事务顺序。偶数输入行写入16×128-bit单端口Line
Buffer，奇数行读取上一行并完成四值有符号最大值；之后按 `pooled_H -> pooled_W
-> channel_pack` 紧密输出。Pool1产生196个8-channel pack，Pool2产生98个。

Pool1的输出由顶层写入阶段间Feature FIFO供Conv2使用；Pool2默认直连独立FC，
不复用当前Systolic，也不要求Pool侧FC Input Buffer。详细设计见
[Pool 模块详细设计](hls/docs/pool.md)。本轮程序数据与 PyTorch 数据两组 C++ 功能检查通过，未重新综合或协同仿真；不能据此确认新版集成的 BRAM、II、时序和资源。

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

详细接口、位宽、缓存、验证范围、已确认综合结果及后续两级右移候选见 [ConvPostProcess 设计](hls/docs/conv_postprocess.md)。

## 独立 FC 第一版设计

固定FC将 Pool2 的 `7×7×16=784` 个 INT8 activation 分类为10个 signed INT8 logit。输入顺序保持 Pool2 的紧密 NHWC：`y -> x -> 8-channel pack`，共98个 Pack；权重顺序为 `out_tile(2) -> token(98)`，每个320-bit逻辑 Pack 有5条输出 lane和8条输入 lane。第一输出 Tile 边计算边写入98x64-bit FC Input Buffer，第二 Tile 从该 BRAM 回放，两个 Tile 各输出一个5路 INT32 accumulator Pack。

FC MAC 后使用 `hls/cnn_accelerator/src/postprocess/fc_postprocess.*` 的独立后处理：每组5 lane 串行复用一套32x32 signed requant数据通路，执行 bias、INT64乘法、直接算术右移和 `[-128,127]` 双向饱和，不做 ReLU 或额外 rounding。输出一次完整10路 logit Pack。FC 与卷积 Systolic 的滑窗/空间并行计算不同，既不复用其阵列也不复用其权重缓存。详细接口、缓冲和未验证项见 [独立 FC 设计](hls/docs/fc.md)；本轮四个 FC/FCPostProcess testbench 功能检查通过，未重新综合或协同仿真。

## 独立 logits Argmax

独立 `hls/cnn_accelerator/src/postprocess/logits_argmax.*` 每调用消费一个10路 signed INT8 logits Pack，输出一个 `ap_uint<4>` 类别索引；并列最大值选择最小索引。比较采用完整展开和分区的 `10 -> 5 -> 3 -> 2 -> 1` 平衡树，共9个 signed 比较器、4层比较深度，避免串行逐项扫描。独立 C Simulation 通过；历史独立综合在 `xc7z020-clg400-2`、10 ns 约束下 latency 为2 cycles、interval为3 cycles，估算时钟6.826 ns，资源为59 FF、183 LUT、0 BRAM、0 DSP。DDR wrapper 已通过 logits_split 将 FCPostProcess 输出分别交给 argmax 和 writer，本轮保留该连接。详细历史结果和综合报告见 [独立 logits Argmax 设计](hls/docs/argmax.md)。

## 相对 dev 通用实现的固定网络化

| 内容 | 本分支变化 | 保持不变 |
|---|---|---|
| 参数与调度 | 删除可覆盖的 P/H/W/C 宏、通用宽高/Tile 计算和综合态非法参数返回；`config.h`集中保存固定配置，接口统一为一位模式 | stream struct字段布局、Conv1/Conv2的token和事务顺序 |
| Im2Row 边界 | 固定 Conv1/Conv2 的 Tile 数、tail valid、Bottom load 和预取 mask；输入读出后立即完成 Conv1 通道零填充 | 2-row circular line buffer、4-way banking、双 Strip、9-token 顺序和 write-first 反压语义 |
| Systolic 调度 | `row × tile_x` 合并成固定数量的 `spatial_tile`；Cout tile 固定为2或4 | 唯一 `4×4×8` PE、activation/weight replay BRAM、dot tree、累加和输出顺序 |

## DDR Wrapper

新增 `cnn_accelerator_ddr` 将 DDR image/weights/params 加载为现有 core 的
stream，并通过独立 logits 分流/argmax 写出两个64-bit结果 word（16字节：
10个 logits、2个保留字节、uint32类别）。参数每层按完整 bias/multiplier/shift
数组连续保存，独立 params AXI 的 widening 上限为32 bit。原 stream
Top 的 phase 调度及 stream 接口保持不变；Systolic 现已自行选择有效 Cin。
新版权重 header v2 直接采用消费顺序且仅含有效字节，PS 三次 `memcpy` 拼接，
DDR 权重为1133个64-bit word（9064字节），与旧1196-word布局不兼容。
Python metadata v5 区分逻辑 NPY 与硬件 header 布局，golden model 解码后计算。
接口布局、PS 复制、缓存维护及有限 FIFO/层次待验证项见
[DDR Wrapper 设计](hls/docs/ddr_wrapper.md)。以下为旧版布局的历史验证记录，不代表本轮 RTL 已验证：GCC 11 与系统 Clang 14
静态语法检查通过；Vitis HLS 2022.2
C Simulation 仅运行一次，三图30个 logits、3个类别、守卫及只读区检查全部通过，
错误数0，input/output CSV 含表头分别2353/34行，日志无 WARNING/ERROR。报告见
[Wrapper C Simulation 日志](build/hls/ddr_layout_verify_16byte/project/solution1/csim/report/cnn_accelerator_ddr_csim.log)。
本轮同步更新 Vitis 裸机应用及数据提取工具，但不改 Vivado、旧 IP 或 bitstream。
本轮不运行综合、RTL Co-simulation 或上板验证；C Simulation 不验证有限深度 FIFO、AXI、HLS pragma 或缓存维护行为。

## 源码分层与两个入口

```text
hls/cnn_accelerator/
├── src/
│   ├── core/cnn_core.{h,cpp}        流式计算核心
│   ├── top/cnn_accel_ddr.{h,cpp}    DDR 系统顶层
│   ├── axi/ddr_io.{h,cpp}          DDR 加载、分流与写回
│   ├── axi/ddr_layout.h            共享 DDR 布局
│   ├── im2row/ systolic/ pool/ fc/ postprocess/
│   └── config.h types.h
├── testbench/test_cnn_core.cpp
├── testbench/test_cnn_accel_ddr.cpp
└── script/hls_core.tcl hls_ddr.tcl
```

`core/cnn_core.cpp` 的 `cnn_accelerator` 是普通 stream 计算入口，独立验证时不依赖
DDR 搬运或 argmax；`top/cnn_accel_ddr.cpp` 的 `cnn_accelerator_ddr` 是四指针 DDR
系统入口，通过 `ddr_io` actor 调用同一个计算核心并输出 logits 与类别。目录与文件
更名不改变函数名、计算、接口、DDR 布局、FIFO 深度或调度顺序。

## C Simulation 入口与历史验证

在安装并配置好 Vitis HLS 2022.2 的环境中，用 `vitis_hls -f` 分别运行
`hls/cnn_accelerator/script/hls_core.tcl` 或 `hls/cnn_accelerator/script/hls_ddr.tcl`；
脚本可从任意工作目录调用，分别选取唯一对应 testbench，使用
`xc7z020-clg400-2`、10 ns 约束，仅执行 C Simulation 并退出。
Vitis HLS 启动日志仍写入启动工作目录，建议先进入仓库根目录的
`build/hls/structure_refactor_verify/` 等生成物目录，再通过脚本的绝对路径
运行 `vitis_hls -f`；不要从 `hls` 源码目录启动工具。脚本的路径解析能力
不代表工具启动日志会自动重定向。
工程分别写入仓库根目录的 `build/hls/cnn_core_csim/` 和
`build/hls/cnn_ddr_csim/`，不重置或覆盖其他工程。不自动综合、协同仿真或导出；
需要时可自行追加相应命令，或在图形界面中设置。

上文归档日志属于结构迁移前的历史验证，保留原链接与原始文件。本轮迁移后，
GCC 11 与系统 Clang 14 对全部25个源码及 testbench `.cpp` 的静态检查通过。
修正 Vitis HLS 2022.2 的工程打开路径兼容性后，两个新脚本各完成一次 C Simulation：

- `hls_core.tcl`：三图30路 logits 与 stream 守恒全部通过，0 errors，无 WARNING；
  input/output CSV 含表头分别2353/31行，见 [核心 C Simulation 报告](build/hls/cnn_core_csim/solution1/csim/report/cnn_accelerator_csim.log)。
- `hls_ddr.tcl`：三图30路 logits、3个类别、守卫与只读区检查全部通过，0 errors，无 WARNING；
  input/output CSV 含表头分别2353/34行，见 [DDR C Simulation 报告](build/hls/cnn_ddr_csim/solution1/csim/report/cnn_accelerator_ddr_csim.log)。

本轮未运行综合或 RTL Co-simulation；DDR 系统入口的综合、RTL 反压和硬件验证仍待后续。
核心或独立模块的历史结果不代表 DDR 系统入口已经综合。

## 生成物目录

今后的 csynth/csim/cosim 工程与日志统一写入仓库根目录 `build/hls/<run-name>/`，不要写入 `hls` 源码树；该目录由 Git 忽略。

## 最新：统一卷积 Loader 验证

Conv1/Conv2 在唯一调用点顺序使用同一 `load_conv_weights()`，按 Pack/Cout lane 消费紧凑权重字节，不改1133-word DDR布局或上下游接口。通用 lane loop 达到 II=1；每个完整 Pack 的综合估计为18周期（含流水填充/排空和控制），不是每周期输出一个 Pack。

全部15个主机 testbench、Loader专项检查、Vitis CSim 和连续三图 Verilog/XSim co-sim 通过，30个 logits及类别0错误；PS host mock 三图也通过。HLS估计资源为42 BRAM18K、51 DSP、26351 FF、36496 LUT，时钟估计7.300 ns（目标10 ns）；RTL仿真单图 latency 为11979～12006周期。综合层次确认共享卷积模块各一个实例。

无 II violation；保留 FIFO 深度/反馈通道等警告，本次三图无死锁，未自动调整FIFO。完整结果、资源及警告分类见 [DDR Wrapper 验证结果](hls/docs/ddr_wrapper.md)，[co-sim 报告](build/hls/unified_weight_loader_verify/vitis_unified_loader/unified_weight_loader/solution1/sim/report/cnn_accelerator_ddr_cosim.rpt)。所有生成物位于 `build/hls/unified_weight_loader_verify/`，Vivado现有工程/IP/bitstream未修改；HLS估计和仿真不是布局布线或板测结果。

## 历史：紧凑权重与有效 Cin 首次修订验证

- Python：重新导出 v5 metadata/v2 权重头，解码后的三层数学权重及 bias/multiplier/shift 与旧版逐元素相同；原三图 trace 未修改，每图13组中间/最终张量逐元素一致。
- Loader：图像784个 pixel、Conv1 的9个 word/18个 Pack（含 pack8/9 跨 tile）、Conv2 的36个 Pack、FC 的196个 Pack 及无效字段透传检查通过。
- 全部15个 testbench 通过主机 C++ 功能检查；独立 Systolic 包含 Conv1 无效 activation/weight 双侧非零测试。
- Vitis HLS 2022.2 的 DDR `csim_design` 三图、30个 logits 和类别检查通过，0错误。见 [本轮 Vitis 日志](build/hls/cin_layout_update_verify/vitis_ddr_csim.log)。
- Vitis 裸机 main 通过真实 BSP 头文件的 ARM `-Wall -Wextra -Werror` 编译；接真实 HLS 顶层的隔离 host mock 三图类别0/4/1、错误数均0。

全部验证件位于仓库根 `build/hls/cin_layout_update_verify/`。本轮不运行综合、RTL co-sim 或板测，不更新 Vivado/IP/bitstream；新 loader/mode 选择的 II、资源、时序和 RTL 反压仍须重新确认。旧1196-word权重布局不兼容新版本。

## 模块级验证

`hls/cnn_accelerator/testbench/` 保留两组独立的模块 testbench：原有 `test_*.cpp`
使用确定性程序数据，新增 `test_*_pytorch.cpp` 直接读取 Python exporter 和
`dump_trace.py` 的 golden 数据。模块 testbench 输入直接来自 PyTorch 的对应上游边界，
不将多个 HLS 模块串接；输出按硬件 stream 顺序恢复坐标后逐项比较，不缓存完整 feature
map 做重排。运行时 CSV 位于 `simulation_data/pytorch/<module>/`。当前 Python header
使用绝对路径 include，替换 trace 或模型结构后须同步检查 shape 断言和索引映射。

Top 级 `test_cnn_core.cpp` 是连续多图例外：它同时 include
`trace_data_0000.h`、`trace_data_0001.h`、`trace_data_0002.h`，在 testbench 内用
三个独立 namespace 隔离同名数组；同一组 stream 连续调用 `cnn_accelerator` 三次，
每次完整重灌输入、两层卷积参数和 FC 参数，并逐图比较10路 logits、检查 stream 守恒。
其 `input.csv`/`output.csv` 均带 `sample` 列，以区分三张图；三组 input/logits 的
元素数量均有编译期 shape 断言。
