# YOLOv2-tiny 重构方案草案

记录日期：2026-11-01。状态：讨论暂存，尚未实施或完成资源、时序验证。

本文作为后续从 MNIST 扩展到 YOLOv2-tiny VOC 的设计输入，整理最近讨论的结论，并区分已确定方向、优先评估方案和待定事项。它不表示当前 HLS 已支持 YOLO，也不直接授权修改所有工程。

## 1. 重构边界与目标网络

近期首先重构 HLS 计算与存储调度，保留单套物理计算流水。PyTorch exporter、Golden Model、testbench 随接口和数值契约确定后同步更新；Vivado、Vitis、摄像头和检测结果显示不在当前实施范围内。

软件参考工程：`/home/liaoyingkai/python_project/minist_train_fpga/YOLOV2_TINY/`，重点参考 `model.py`、`quant.py` 和 `README.md`。该工程目前尚未完成硬件整数参数导出，不能认为其数值规则已经全部冻结。

输入为 RGB，参考分辨率为 416×416，Batch 第一版为 1。下表的空间尺寸均采用 H×W：

| 卷积阶段 | 输入 H×W×Cin | Kernel / Padding / Stride | Cout | 后续 Pool | 写回的输出 H×W×C |
|---|---|---|---:|---|---|
| Conv1 | 416×416×3 | 3×3 / 1 / 1 | 16 | 2×2，stride=2 | 208×208×16 |
| Conv2 | 208×208×16 | 3×3 / 1 / 1 | 32 | 2×2，stride=2 | 104×104×32 |
| Conv3 | 104×104×32 | 3×3 / 1 / 1 | 64 | 2×2，stride=2 | 52×52×64 |
| Conv4 | 52×52×64 | 3×3 / 1 / 1 | 128 | 2×2，stride=2 | 26×26×128 |
| Conv5 | 26×26×128 | 3×3 / 1 / 1 | 256 | 2×2，stride=2 | 13×13×256 |
| Conv6 | 13×13×256 | 3×3 / 1 / 1 | 512 | 2×2，stride=1，右/下补边 | 13×13×512 |
| Conv7 | 13×13×512 | 3×3 / 1 / 1 | 1024 | 无 | 13×13×1024 |
| Conv8 | 13×13×1024 | 3×3 / 1 / 1 | 1024 | 无 | 13×13×1024 |
| Prediction | 13×13×1024 | 1×1 / 0 / 1 | 125 | 无 | 13×13×125 |

前八层为 BN 折叠后的 Conv 与 LeakyReLU，当前软件斜率为 0.1；Prediction 无 BN、无激活。YOLO 主数据通路不实例化 MNIST 的 FC、FCPost 和十分类 Softmax。检测解码与 NMS 的实现位置留待后续确定，不把 MNIST 概率接口直接套到检测头。

本轮不扩展任意卷积 stride、任意 kernel 或任意网络图，仅覆盖上述网络所需操作。

## 2. 必须统一的事务和 Tile 含义

保留以下物理并行度，不改成逐通道串行，也不把接口扩大成一次传全部 Cin：

| 名称 | 含义 |
|---|---|
| Channel Pack | 一个空间位置的连续 8 个通道，64 bit |
| Spatial Tile | 同一图像行中最多 4 个连续 W 位置，不跨图像行 |
| Cout 计算 Tile | 同一次 PE 归约处理的 4 个输出通道 |
| Activation Pack | 4 个空间位置 × 同一 Cin Group 的 8 个通道，数据 256 bit |
| Weight Pack | 4 个 Cout Lane × 同一 Cin Group 的 8 个权重，数据 256 bit |
| Acc / Activation Tile | 4 个空间位置 × 4 个输出通道的结果，分别为 INT32 / INT8 |
| Cout 缓存块 | 片上同时驻留权重的多个 Cout 计算 Tile，大小尚未冻结 |

`value[spatial_lane][cout_lane]` 的两个维度是空间位置与输出通道，不能当作图像上的 4 行×4 列。Pool 对同一通道的 H/W 邻域比较，不对这个数值矩阵直接做二维池化。

## 3. 总体存储与计算调度

已确定方向：

```text
DDR 输入 Feature Map
    → Feature Loader
    → Im2Row
    → 单套传统 4×4 OS Systolic，PE 内 Dot8
    → ConvPostProcess
    → Pool / Bypass
    → Formatter / DDR Writer
    → DDR 输出 Feature Map
```

- DDR 保存完整模型权重、参数和层间 Feature Map，不保存完整 Im2Row 展开矩阵。
- 权重按 Cout 缓存块加载到 BRAM，跨空间位置复用；PE 内部保持 Output-Stationary。两种描述分别针对存储调度和 PE 数据流，不冲突。
- 一层的每个 Cout 缓存块处理完成后，加载下一缓存块并重新扫描输入 Feature Map；重新扫描包含 Im2Row 的预装和启动开销。
- 每层完成并确认 DDR 输出写入结束后，再切换下一层；所有层复用同一个计算调用路径，避免按层复制 DATAFLOW 网络。
- Conv 与后续 Pool 流式融合，不必先写回池化前完整 Feature Map。无 Pool 的层采用 Bypass。
- 层间 Feature Map 使用 DDR 两个互不重叠的区域交替读写，不继续用 MNIST 的整层 BRAM FIFO 保存层间结果。
- 近期继续采用 `m_axi + s_axilite`，模块内部使用 `hls::stream`。外置 AXI DMA、独立 RTL 交付及平台依赖问题暂缓，不以“m_axi 不能导出 RTL”为设计前提。

层内的优先调度方向如下，确切函数接口待缓存容量和 DDR 布局确定后设计：

```text
layer
    → cout_cache_block
        → 加载该缓存块的权重及后处理参数
        → 扫描一次输入 Feature Map
        → output_y
            → tile_x
                → local_cout_tile
                    → kh
                        → kw
                            → cin_group
```

同一 Spatial Tile 的第一个 local_cout_tile 直接使用 Im2Row 数据；缓存块含多个计算 Tile 时，可同时写入 Activation Replay，随后从本地回放。只驻留一个 4-Cout 计算 Tile 时，不需要为此配置 Replay。

Cin Group 之间不清零、不单独输出、不单独执行 Bias/Requant。完整归约结束后，才排空阵列并输出当前结果 Tile。回放期间不读取新 Activation，依靠 stream 反压，不新增 pause/stop 接口。

## 4. BRAM 预算先于 Cout 缓存数量

不冻结缓存 8、16 或其他固定数量的 Cout。计算并行度固定为 4，但缓存数量由实际 BRAM 预算决定，可以按层选择不同数量。

定义：

```text
Cin Groups = ceil(Cin / 8)
K Tokens   = Kh × Kw × Cin Groups

单个 4-Cout 权重块的数据容量 = K Tokens × 32 Byte
单个 Spatial Tile 的 Activation Replay = K Tokens × 32 Byte
```

缓存越多 Cout，通常能减少 Feature DDR 重读和 Im2Row 重启次数，但不能先把 BRAM 全部分给权重。预算必须同时覆盖：

- Im2Row Line Buffer、Strip 和预取暂存；
- Weight Buffer，以及需要时的 Activation Replay；
- Pool 原值行缓存；
- 必要的 Formatter / Burst 缓冲、FIFO 和 AXI 适配器内部存储；
- Bank、端口及宽深配置带来的 BRAM 碎片和实现余量。

权重缓存应只有一个明确的所有者，不在 Loader 和 Systolic 中重复保存同一份完整缓存块。参数在缓存块有效期间驻留并复用，不随每个 Spatial Tile 重读 DDR。

数组容量应按支持层中实际最大的工作集确定，不能机械地把最大 W 与最大 Cin 相乘。缓存层数、单双缓冲、Bank 组织和各层可驻留 Cout 数量均须先核算，再决定是否增加预取重叠。

## 5. DDR 布局与 PyTorch 导出：仍待冻结

必须分清三个顺序：DDR 物理布局、stream 事务顺序、PE 归约顺序。不能因 DDR 布局变化而把一个输出的 Cin Group 拆成多次独立归约。

Feature 输入 stream 的设计方向是：

```text
H → W → cin_group → Pack 内 channel_lane
```

每次发送一个 8-channel Pack；实际逻辑 Cin 尾部有效性由配置确定。

Feature 的 DDR 物理布局尚未确定：

- 普通 NHWC 与逐位置读取全部 Cin Group 自然匹配，但分 Cout 计算的输出写地址不连续。
- `[channel_group8][H][W][8]` 可在及时组装出 8 通道时改善分组连续写入，但下一层可能需要读取重排；它不是已经选定的方案。
- 只驻留 4 个 Cout 时，Blocked8 本身不能自动解决半 Pack 组装问题。
- 非连续地址可以正确写入；局部缓存用于满足打包和改善 Burst，不应把它描述为所有布局下功能正确的必要条件。

确定布局时，要一起写清 Loader 的连续读取块、重排存储量、Writer 的输出粒度、尾通道 Padding、行/块对齐和地址公式。两个 DDR Feature 区域容量按所有实际保存 Tensor 的物理大小取上限，包含输入和存储对齐，不能直接固定为约 0.7 MB。

权重逻辑仍对应 `W[k][cout]`，其中 `k=((kh*Kw)+kw)*Cin+cin`。Exporter 可离线打包成硬件消费顺序，暂按下列方向评估：

```text
cout_cache_block → local_cout_tile → kh → kw → cin_group
    → cout_lane → cin_lane
```

是否包含物理尾部字节、与缓存块大小是否绑定、各层参数偏移均待 ABI 冻结后同步到 exporter、loader、testbench 和 PS。不能通过只改一侧解决布局问题。

## 6. Im2Row：多 Group 的 BRAM 窗口方案

详细接口、BRAM组织、候选预取调度和待确认要求见[YOLOv2-tiny Im2Row设计](im2row.md)。2026-11-02已明确Im2Row只支持本YOLO层表，采用扩展后的ConvGeometry结构配置，不保留旧二层bool接口的设计要求；源码尚未实施。

### 6.1 输出顺序和边界

每个 Spatial Tile 输出 `Kh×Kw×Cin Groups` 个 Activation Pack，顺序为：

```text
kh → kw → cin_group
```

每个 Token 都包含 4 个空间位置的同一个 `(kh,kw,cin_group)`，每位置仍然只有 8 通道。

- 3×3：stride=1，四侧 Padding=1，卷积填充值为量化零点 0。
- 1×1：stride=1，无 Padding，不走 3×3 滑窗读取；仍需收集 4 个空间位置并按 Group 输出。
- 每行单独切 Spatial Tile；例如 W=13 的末 Tile 仅一个 Lane 有效。
- Cin 尾组按 `group*8+lane<Cin` 判定，不能继续用 MNIST 的层模式特判。
- Padding 不使输出空间 Lane 无效；不存在的空间 Lane 则 `spatial_valid=0`。

### 6.2 优先评估：保留空间结构，将 Group 存入 BRAM 深度

保留两行循环 Line Buffer、双 Strip 及窗口重叠关系。每个 Word 为 64 bit，不使用一次容纳全部 Cin 的超宽 Word。

Strip 的逻辑内容为：

```text
[ping_pong][3行][P+2列][cin_group]
```

不同 Group 使用不同存储地址。空间维根据并行访问做 Banking；Group 维保留为 RAM 深度，不 complete partition 成全部 FF。Ping-Pong 的物理分 Bank 或合并地址方式待端口排表确定。

这解决的是“保存全部待复用 Group，但不全部展开为寄存器”。不是只保存一个 Group、覆盖后再从已失效的 Line Buffer 找回旧值。

Line Buffer 的覆盖规则必须逐位置、逐 Group 成立：旧行数据先进入仍需要它的 Strip，之后才能用下一输入行覆盖对应地址。当前输出使用当前 Strip，下一 Strip 在另一 Ping-Pong 区域准备；重叠列也按 Group 保留或复制。

3×3的kernel位置q仍为0～8，每个q内遍历G个Group；若使用扁平Token，则范围是0～9G-1。复制/预取必须覆盖全部Group，不能只在扁平token等于3/6/7/8时各做一次。实施前须明确BRAM延迟、端口与反压；操作重叠不一定冲突，一读一写Bank和已读Word转发可能支持原来的连续输出目标。

1×1优先评估缓存前三个位置、在第四个位置的各Group到来时直接组合输出，避免先收齐4G个输入再独立输出G个Token。单路64-bit输入下的收集时间不等于PE必然空闲，须结合Systolic Replay调度分析。

本方案为优先评估方向，尚未证明 BRAM容量、端口和目标 II 均满足；若需改变两行/Strip结构，应先说明取舍，不自行替换架构。

## 7. Systolic 与 PostProcess

### 7.1 Systolic

- 保留传统 4×4 OS，Activation 向右、Weight 向下传播，PE 内 Dot8，INT32 累加。
- 支持可变 K Tokens；1×1 也连续归约全部 Cin Group，不按 Group 重启波前。
- 保留错拍、Valid 传播、初始化和排空机制。`K+6`仅为4×4阵列的逻辑步数模型，不等于实测 Tile 间隔。
- 分开处理传播 Token Valid、Spatial Valid、Cout尾部有效性和Cin尾组有效性。
- Prediction 的125通道最后一个4-Cout Tile只有一个有效通道；若按8通道打包，最后一组则有5个有效通道，不能混淆两种尾部。
- 不把逻辑 Cin 有效性依赖于上游恰好填零；具体屏蔽位置在接口设计时明确。

### 7.2 PostProcess 数值方向

当前 YOLO QAT 在 Conv 输出和 LeakyReLU 输出各有一次独立 FakeQuant。优先保持现有 QAT 结构，在一个 PostProcess 模块中明确表达两个数值边界：

```text
Accumulator + Bias
    → Conv 输出 Requant、舍入、INT8 饱和
    → LeakyReLU 与激活输出 Requant
    → 舍入、INT8 饱和
```

第二阶段可由 exporter 生成正负分支 multiplier/shift，分别表示 `scale_conv/scale_activation` 与 `0.1*scale_conv/scale_activation`，不需要硬件浮点反量化。

这是暂定数值基线，不是已经冻结的全部整数算法。两次边界的舍入、负数右移、饱和、bias 范围和参数格式，必须先由 YOLO exporter / Golden Model 确定。不能未经证明直接合并乘数并消掉中间舍入、饱和，也不能直接声称 MNIST 的截断规则等于当前 YOLO QAT。

逻辑上保留两阶段，不等于必须实现两套完全并行乘法器；物理复用按吞吐和综合结果决定。Prediction 不执行 LeakyReLU，输出原始预测量的量化表示，供后续检测处理使用。

## 8. Pool：原值行缓存与边界暂存

本节采用最后讨论的方案：Line Buffer 保存原始输入，不保存水平 max，也不保存最终池化结果。之前讨论的“水平 max 行缓存”不作为本次重构基线。

### 8.1 新调度下缓存什么

若当前只处理 C0～C3，Pool 扫描整张图的这4个通道，行缓存只需保存上一图像行对应的原始值。若一个缓存块含多个 local_cout_tile，则它们各自保留行状态及边界状态，不等待全层所有 Cout 到齐。

池化在同一通道内进行；跨 Cout Tile 的8通道组装是 Formatter / Writer 问题，不是 Pool 窗口跨通道取 max。

### 8.2 Stride=1 的跨空间 Tile 处理

固定一个通道，两个相邻输入 Tile 如下；A/B分别表示上一图像行和当前图像行，数字下标表示图像列：

```text
                  Tile0，第0～3列       Tile1，第4～7列
上一图像行：       A0 A1 A2 A3          A4 A5 A6 A7
当前图像行：       B0 B1 B2 B3          B4 B5 B6 B7
```

处理 Tile0：

1. 读出 Line Buffer 中的 A0～A3，锁存旧值，再使用当前输入 B0～B3。
2. 计算横向起点为0、1、2的三个2×2窗口。
3. 暂存原始边界值 A3、B3，每个当前处理通道各保留这两份值。
4. 将 B0～B3写回 Line Buffer，作为下一图像行所需的上一行原始值。

处理 Tile1：

1. 读出尚未覆盖的 A4～A7，并取得 B4～B7。
2. 用 `max(A3,A4,B3,B4)` 完成横向起点为3的窗口。
3. 继续处理 Tile1 内部窗口，暂存 A7、B7，再写回 B4～B7。

必须先保护旧行边界，再覆盖相应地址。边界寄存器按行和 local_cout_tile 隔离，行结束后不能把最后一列误当成下一行的左邻居。

输出按窗口空间坐标递增；跨边界窗口等下一 Tile 到达后完成，不能因此先输出更靠右的窗口。输出打包、尾 Tile 的延迟和排空需在接口阶段补全。

### 8.3 Stride、Padding 与 RAM 端口

- Pool1～Pool5：2×2、stride=2、无有效窗口补边。
- Pool6：2×2、stride=1，仅右侧/下侧各补1，输出保持13×13。
- Pool 填充值为 signed INT8 最小值 -128，不使用0。输入无效 Lane 的零 Payload 不能误参加负数窗口比较。
- Pool6 右边界使用右补边，最末输入行之后还须完成下补边窗口；带 Padding 的合法输出不能被当作无效位置丢弃。
- Stride=1必须重新设计读旧值、锁存、写当前行的时序，不能沿用隔行只读/只写的单口前提。优先评估一读一写双端口 BRAM；具体类型、读延迟和目标 II 待排表及综合确认。

## 9. 实施前必须补齐的设计

| 待定事项 | 需要形成的结论 |
|---|---|
| 片上存储预算 | 各 Buffer容量、Bank及端口组织、实际BRAM余量 |
| Cout驻留数量 | 每层缓存块大小、尾块处理、是否需要Activation Replay |
| Feature DDR布局 | Loader与Writer共同使用的地址公式、Burst和局部重排需求 |
| 导出ABI | 权重/参数/Feature物理排列、位序、尾通道与对齐规则 |
| Im2Row多Group调度 | Strip BRAM访问表、Line Buffer覆盖时刻、启动及行尾处理 |
| 1×1路径 | 四空间位置收集、Group读取与无Padding路径 |
| YOLO整数数值 | 两次量化边界、LeakyReLU、舍入及饱和Golden |
| Pool输出契约 | 当前Cout块结果的顺序、延迟输出、8通道组装位置 |
| 调用配置 | 已确定采用ConvGeometry；仍需补齐YOLO字段、派生值校验及调用方迁移 |
| 性能目标 | 目标分辨率、频率、fps及有效DDR带宽预算 |

没有完成这些结论前，不把缓存8/16个Cout、Blocked8布局、全部FIFO固定depth=2或某个II视为已冻结条件。

先前约2.84 fps仅是416输入、100 MHz、II=1、每Tile按K+6推进并含行尾空Lane的估算，不是综合或板测结论，也不是设计目标。性能评估须同时计入DDR重读、Im2Row预装、权重加载和流水停顿，不能只凭读取字节数认定某层必然受带宽限制。

## 10. 后续实施与验证顺序

1. 先确定存储预算、调度顺序、DDR布局和模块事务契约，再开始逐模块实现。
2. 扩展 Im2Row 与 Systolic：多Cin Group、3×3/1×1、尾通道和尾空间Lane；同步独立testbench。
3. 完成 YOLO exporter / 整数Golden，再实现PostProcess及其逐元素数值验证。
4. 按原值行缓存方案实现Pool，重点验证stride=1、跨Tile、右/下Padding及状态切换。
5. 实现Loader、Formatter、Writer和统一层控制，验证DDR双缓冲与单套硬件复用。
6. 先单层，再连续多层，最后做系统性能和上板验证；Vivado/Vitis更新另行安排。

测试至少覆盖Cin=3/16/1024、1×1长归约、Cout=125尾部、W=13末Tile、多Cout驻留块与连续调用。Pool应使用原始2×2四元素公式作为独立reference，不能让reference重复DUT缓存实现。

每个模块建立stream读写数量守恒；按获准阶段分别进行静态检查、C-sim、综合和co-sim。综合检查BRAM实际映射、端口冲突、归约II、阵列实例数；co-sim检查反压、FIFO深度、边界排空和层切换。不能把语法通过当成功能、资源或时序已经验证。

本次仅新增本文，未修改源码、导出程序或工程，也未运行仿真、综合。
