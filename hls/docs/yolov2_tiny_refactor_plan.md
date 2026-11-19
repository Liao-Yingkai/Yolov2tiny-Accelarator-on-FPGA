# YOLOv2-tiny 设计记录与待优化事项

记录日期：2026-11-01，整理日期：2026-11-19。当前YOLO系统已完成测试及上板功能验证
（用户确认）；本文保留架构取舍与后续优化方向，不再作为待实现模块的清单。

本文记录当前 YOLOv2-tiny VOC 的接口、数值契约和存储调度，并保留后续性能优化及验证事项。九层 DDR Top 已实现；已完成检查、综合结果与尚未覆盖的 RTL/板级边界见 [Top](top.md)。

## 1. 重构边界与目标网络

当前HLS计算与存储调度保留单套物理计算流水，已与Vivado、Vitis、摄像头和检测结果显示集成。后续变更接口或数值契约时仍需同步Python exporter、Golden Model和testbench；本轮仅整理工程，不改变现有实现和结构。

软件参考为外部Python工程的`YOLOV2_TINY/`目录。使用其`export.py`冻结的整数权重BIN布局和`dump_trace.py`逐层整数数据做对照；运行路径统一通过`YOLO_DATA_DIR`指定，见[数据说明](data.md)。

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

前八层为BN折叠后的Conv与LeakyReLU，斜率为0.1；Prediction无BN、无激活。HLS输出原始Prediction Tensor，检测解码及NMS由PS执行，Overlay负责绘图。

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
| Cout 缓存块 | 暂定最多32个Cout，即8个计算Tile；不足32的层只加载实际Tile，后续按综合BRAM占用调整 |

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
- 层间 Feature Map 使用 DDR 两个互不重叠的区域交替读写，不使用整层 BRAM FIFO 保存层间结果。
- 近期继续采用 `m_axi + s_axilite`，模块内部使用 `hls::stream`。外置 AXI DMA、独立 RTL 交付及平台依赖问题暂缓，不以“m_axi 不能导出 RTL”为设计前提。

层内按最多32个Cout一个缓存块调度，控制与 DDR 契约见
[Loader、Writer 与层间 DDR 缓冲](yolov2_tiny_ddr_io.md)：

```text
layer
    → cout_cache_block
        → 从整层流消费当前块的权重及后处理参数
        → 重新调用 Feature Loader 和 Im2Row，各扫描一次
        → output_y
            → tile_x
                → local_cout_tile
                    → kh
                        → kw
                            → cin_group
```

同一 Spatial Tile 的第一个 local_cout_tile 直接使用 Im2Row 数据；缓存块含多个计算 Tile 时，可同时写入 Activation Replay，随后从本地回放。只驻留一个 4-Cout 计算 Tile 时，不需要为此配置 Replay。

Cin Group 之间不清零、不单独输出、不单独执行 Bias/Requant。完整归约结束后，才排空阵列并输出当前结果 Tile。回放期间不读取新 Activation，依靠 stream 反压，不新增 pause/stop 接口。

块是流顺序，不要求所有模块按块调用。Systolic OS 的现有入口按块运行，上游层级调度连续供给各块；PostProcess 一次调用处理整层，在内部按相同计数切块。整网集成不能在当前块结束时等待整层 PostProcess 返回，否则下一块无法供给。层间仍保留整层 DDR 写回 barrier。

## 4. 首版32个Cout缓存及BRAM预算

首版各层统一按最多32个Cout驻留设计，包括Conv8，即最多8个4-Cout计算Tile。PE计算并行度仍为4，不扩展成32列阵列。后续依据整条流水综合后的BRAM占用再调整驻留数量；这不是已验证可装入目标器件的资源结论。

每个缓存块从全局Cout Tile编号开始，加载`min(8, 总Cout Tile数-起始Tile号)`个连续Tile。Conv1只有16个Cout，只加载4个Tile，不为填满缓存而额外导出16个通道。Prediction导出至128个物理Cout，其中最后3个Lane无效；最后一个32-Cout块有29个逻辑有效通道。

定义：

```text
Cin Groups = ceil(Cin / 8)
K Tokens   = Kh × Kw × Cin Groups

单个 4-Cout 权重块的数据容量 = K Tokens × 32 Byte
32-Cout 权重缓存最大容量 = 8 × 1152 × 32 Byte = 288 KiB
单个 Spatial Tile 的 Activation Replay = K Tokens × 32 Byte
Activation Replay最大数据容量 = 1152 × 32 Byte = 36 KiB
```

权重按4个64-bit Cout Lane Bank组织，逻辑地址为`local_cout_tile*K Tokens+token`，最大深度9216。每路显式拆为8192主段和1024尾段，以避免单个9216深度RAM在旧综合中映射过粗；拆分后的实际资源仍待复核。不要将8个local_cout_tile维完全分区为8套浅RAM，也不默认增加权重双缓冲。288 KiB是有效数据容量，实际BRAM占用须计入宽深配置和端口带来的碎片。

当前Im2Row独立综合占52个BRAM_18K；该存储与Activation Replay、Pool、Writer、AXI等必须一起预算，不能只评估Weight Buffer。

缓存越多 Cout，通常能减少 Feature DDR 重读和 Im2Row 重启次数，但不能先把 BRAM 全部分给权重。预算必须同时覆盖：

- Im2Row Line Buffer、Strip 和预取暂存；
- Weight Buffer，以及需要时的 Activation Replay；
- Pool 原值行缓存；
- 必要的 Formatter / Burst 缓冲、FIFO 和 AXI 适配器内部存储；
- Bank、端口及宽深配置带来的 BRAM 碎片和实现余量。

权重缓存由Systolic唯一拥有，Loader只按顺序读DDR并输出weight_stream，不重复保存同一份完整缓存块。参数在缓存块有效期间驻留并复用，不随每个 Spatial Tile 重读 DDR。覆盖权重前须完成旧缓存块的所有使用。第一版整块预加载后再计算；后续明确安排加载重叠，优先恢复首 Spatial Tile 边算边填，再评估跨块预取。见 [OS 实现计划](systolic_os.md#后续性能优化重叠权重加载)。

数组容量应按支持层中实际最大的工作集确定，不能机械地把最大 W 与最大 Cin 相乘。缓存层数、单双缓冲、Bank 组织和各层可驻留 Cout 数量均须先核算，再决定是否增加预取重叠。

## 5. 首版 DDR Feature 与模型布局

必须分清三个顺序：DDR 物理布局、stream 事务顺序、PE 归约顺序。不能因 DDR 布局变化而把一个输出的 Cin Group 拆成多次独立归约。

Feature 输入 stream 的设计方向是：

```text
H → W → cin_group → Pack 内 channel_lane
```

每次发送一个 8-channel Pack。Feature Loader 在发送时补齐最后一组的无效 Cin，
不把补零字节写回 DDR；Im2Row 的空间 Padding 与 `spatial_valid` 保持不变。

Feature DDR 首版采用普通 NHWC：首图为紧凑 RGB，中间通道均对齐到 8，
Prediction 按 128 个物理槽存 125 个逻辑通道。Writer 只缓存一个空间 Tile
的当前 Cout 块，按像素写短连续段；跨像素跳地址是这一布局的已知取舍。
A/B 两区各至少 692,224 B，整层写回完成后交换，不跨层覆盖或预读。

权重逻辑仍对应 `W[k][cout]`，其中 `k=((kh*Kw)+kw)*Cin+cin`。逻辑NPY保留`[Kh*Kw*Cin,Cout]`，物理BIN采用如下独立于缓存块容量的布局：

```text
global_cout_tile → kh → kw → cin_group
    → cout_lane → cin_lane
```

物理Shape为`[ceil(Cout/4),Kh*Kw*ceil(Cin/8),4,8]`，每个事务32字节。Exporter
将 Cin/Cout 物理尾权重置零，Weight Loader 整层顺序恢复 Pack，不重复补零。
Feature 的 Cin 尾零由 Feature Loader 生成，Systolic 不重复处理；空间与 Cout
的 mask 仍独立生效。

32-Cout 缓存块只是 Systolic 每次从整层流消费的连续范围；调整驻留数量不应改变
BIN 布局。模型 BIN 与 metadata 约定由 PyTorch 工程的 `YOLOV2_TINY/README.md`
维护。HLS 的 DDR 地址、参数组包和写回规则见[搬运规格](yolov2_tiny_ddr_io.md)。

## 6. Im2Row：多 Group 的 BRAM 窗口方案

详细接口、BRAM组织、调度和验证边界见[YOLOv2-tiny Im2Row设计](im2row.md)。2026-11-02独立HLS综合达到II=1；后续已接入九层Top，整网功能与综合记录见 [Top](top.md)，RTL反压尚未验证。

### 6.1 输出顺序和边界

每个 Spatial Tile 输出 `Kh×Kw×Cin Groups` 个 Activation Pack，顺序为：

```text
kh → kw → cin_group
```

每个 Token 都包含 4 个空间位置的同一个 `(kh,kw,cin_group)`，每位置仍然只有 8 通道。

- 3×3：stride=1，四侧 Padding=1，卷积填充值为量化零点 0。
- 1×1：stride=1，无 Padding，不走 3×3 滑窗读取；仍需收集 4 个空间位置并按 Group 输出。
- 每行单独切 Spatial Tile；例如 W=13 的末 Tile 仅一个 Lane 有效。
- Feature Loader 按 `group*8+lane<Cin` 补齐 Cin 尾组，Im2Row 只接收
  规范化后的固定 8-lane Pack。
- Padding 不使输出空间 Lane 无效；不存在的空间 Lane 则 `spatial_valid=0`。

### 6.2 已实现的存储方向：保留空间结构，将 Group 存入 BRAM 深度

保留两行循环 Line Buffer、双 Strip 及窗口重叠关系。每个 Word 为 64 bit，不使用一次容纳全部 Cin 的超宽 Word。

Strip 的逻辑内容为：

```text
[ping_pong][3行][P+2列][cin_group]
```

不同 Group 使用不同存储地址。空间维根据并行访问做 Banking；Group 维保留为 RAM 深度，不 complete partition 成全部 FF。当前Ping-Pong已合并到每个Strip Bank的深度地址，采用18个64-bit、一读一写BRAM Bank。

这解决的是“保存全部待复用 Group，但不全部展开为寄存器”。不是只保存一个 Group、覆盖后再从已失效的 Line Buffer 找回旧值。

Line Buffer 的覆盖规则必须逐位置、逐 Group 成立：旧行数据先进入仍需要它的 Strip，之后才能用下一输入行覆盖对应地址。当前输出使用当前 Strip，下一 Strip 在另一 Ping-Pong 区域准备；重叠列也按 Group 保留或复制。

3×3的kernel位置q仍为0～8，每个q内遍历G个Group；若使用扁平Token，则范围是0～9G-1。复制/预取覆盖全部Group，不能只在扁平token等于3/6/7/8时各做一次。当前采用一拍BRAM读延迟、逐Bank统一写入口与已读Word转发达到独立综合II=1；反压和数据正确性仍待仿真。

1×1已采用缓存前三个位置、在第四个位置的各Group到来时直接组合输出，避免先收齐4G个输入再独立输出G个Token。单路64-bit输入下的收集时间不等于PE必然空闲，须结合Systolic Replay调度分析。

该存储方向已在独立Im2Row综合中实现并达到II=1，但不代表整条流水的BRAM预算或功能已通过；后续改变两行/Strip结构仍须先说明取舍，不自行替换架构。

## 7. Systolic 与 PostProcess

### 7.1 Systolic

详细接口、循环、缓存和验收步骤见 [YOLOv2-tiny Systolic 重构计划](systolic_os.md#yolov2-tiny-重构计划)。首版各层统一按最多 32 Cout 驻留，Conv8 不例外；一次 Systolic 调用处理一个驻留块。

- 保留传统 4×4 OS，Activation 向右、Weight 向下传播，PE 内 Dot8，INT32 累加。
- 支持可变 K Tokens；1×1 也连续归约全部 Cin Group，不按 Group 重启波前。
- 保留错拍、Valid 传播、初始化和排空机制。`K+6`仅为4×4阵列的逻辑步数模型，不等于实测 Tile 间隔。
- 分开处理传播 Token Valid、Spatial Valid、Cout尾部有效性和Cin尾组有效性。
- Prediction 的125通道最后一个4-Cout Tile只有一个有效通道；若按8通道打包，最后一组则有5个有效通道，不能混淆两种尾部。
- 权重 Cin 尾组由导出器补零，Feature Cin 尾组由 Feature Loader 补零；
  `cout_valid` 按全局计算 Tile 生成，在权重注入端合入传播 valid，并随结果交给下游。

### 7.2 一次性PostProcess与QAT边界调整

PostProcess 调用与参数切块见 [YOLOv2-tiny ConvPostProcess 设计](conv_postprocess.md)。首版单份 32-Cout 参数缓存，每块先加载再计算，走完全部空间位置后读取下一块。参数预取作为后续优化单独实现，不改变数值或 stream 契约。

Conv1～Conv8采用卷积输出、激活之前不设置activation FakeQuant的数值契约；配套PyTorch量化结构应为：

```text
输入FakeQuant
    → Conv（保留per-output-channel weight FakeQuant，BN提前折叠）
    → LeakyReLU(alpha=0.1)
    → 输出activation FakeQuant
```

Conv1～Conv8只在LeakyReLU之后形成INT8输出边界；Pool不增加量化边界，沿用输入scale。Prediction没有激活函数，保留预测卷积输出的activation FakeQuant。不能将Conv的整个qconfig设为None而丢失权重FakeQuant，也不能让容器或Identity额外插入中间activation FakeQuant。

这是对QAT结构的主动修改，不是仅将旧模型的两个scale代数合并。旧QAT的中间舍入/饱和不可逆，新方案不能宣称逐位等价；需用新结构重新QAT微调并验证精度。保留FP32 checkpoint初始化路径，旧双边界QAT checkpoint不能静默当成新方案使用。

HLS MAC与PostProcess仍分离。令`a=accumulator+bias`，每个输出通道仅导出一组`M, S`，使用INT32 bias及Q31/frexp multiplier、总右移shift表示：

```text
r[c] = input_scale * weight_scale[c] / output_scale
r[c] ≈ M[c] / 2^S[c]
```

其中output_scale直接取LeakyReLU后的FakeQuant，Prediction则取其卷积输出FakeQuant；不再保留中间scale_conv或第二组scale换算参数。

alpha不进入DDR参数，由HLS使用相同frexp/Q31方式生成的固定整数常量表示：`ALPHA_MULTIPLIER=1717986918`、`ALPHA_SHIFT=34`，近似0.1。为统一后续实现与整数Golden，加载参数时派生：

```text
M_positive = M
M_negative = (INT64(M) * ALPHA_MULTIPLIER + 2^33) >> 34
```

上述非负系数派生采用最近值、half-up；它是每通道参数准备，不是对逐元素requant乘积加rounding offset。负分支继续使用同一个S，不再做frexp归一化，也不从文件读取另一组系数。常量乘法及系数缓存的资源仍需后续综合确认。

每个有效结果只执行一次requant：

```text
M_selected = LeakyReLU启用且a<0 ? M_negative : M_positive
product = INT64(a) * INT64(M_selected)
scaled = product arithmetic-right-shift S
output = signed INT8 clamp(scaled, -128, 127)
```

逐元素右移沿用当前整数硬件的直接截断，不加rounding offset，负值按算术右移向负无穷取整。中途不先将a截成INT8，不先对负INT8乘alpha再截断；bias相加须满足INT32范围契约，乘法保留INT64。Prediction禁用激活函数，所有符号都使用M_positive。

整数Golden必须使用相同的alpha常量、负系数派生及最终截断规则。其与PyTorch FakeQuant最近偶数舍入、FP32 bias和系数近似之间的差异需单独评估，不能宣称调整后QAT与整数推理天然逐位一致。无效spatial/Cout Lane不参与bias与requant，结果为零。

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

输入和输出均按 `Cout 块 → y → tile_x → local_cout_tile` 排列，输出空间坐标
按池化后宽度紧密分成连续四位置的 Tile。跨边界窗口等下一 Tile 到达后完成，
行尾按同一 Cout 顺序排空。stride=2 将相邻输入空间 Tile 的两个半结果合并，
末输入 Tile 无配对时直接发出带 mask 的结果。8 通道组装留给 Writer。

### 8.3 Stride、Padding 与 RAM 端口

- Pool1～Pool5：2×2、stride=2、无有效窗口补边。
- Pool6：2×2、stride=1，仅右侧/下侧各补1，输出保持13×13。
- Pool 填充值为 signed INT8 最小值 -128，不使用0。输入无效 Lane 的零 Payload 不能误参加负数窗口比较。
- Pool6 右边界使用右补边，最末输入行之后还须完成下补边窗口；带 Padding 的合法输出不能被当作无效位置丢弃。
- 首版使用共享的 `ram_s2p/bram`（416×128-bit，请求 1 拍读延迟），stride=1
  先锁存旧值、后写当前行。内层 Cout Tile 循环目标 II=1，实际排程和映射待综合。
- Conv7、Conv8、Prediction 原样透传。详细接口与验证见 [Pool 设计](pool.md)。

## 9. 当前集成决策与后续测量

| 事项 | 当前状态或待验证结论 |
|---|---|
| 片上存储预算 | 各 Buffer容量、Bank及端口组织、实际BRAM余量 |
| Cout驻留数量 | 首版各层最多32通道，BRAM综合后复核；按实际Tile数处理小层/尾块，配置Activation Replay |
| Feature DDR布局 | 首版 NHWC 与 A/B 两区已定；实际 burst、端口分组及吞吐待验证 |
| 导出ABI | 权重/参数/BIN 约定由 PyTorch 工程文档维护，HLS 与调用方按搬运规格接入 |
| Im2Row多Group调度 | 已实现Strip BRAM调度及Line Buffer覆盖门控；后续调整访问表时重新验证启动、行尾及反压 |
| 1×1路径 | 已实现四空间位置收集、Group读取与无Padding路径 |
| YOLO整数数值 | 已采用一次性PostProcess，固定alpha常量及负分支系数派生；更换模型后重新导出并对照整数Golden，精度单独评估 |
| Pool输出契约 | 已确定紧密的池化后 4×4 Tile；Writer 按输出坐标组装 8 通道并写 DDR |
| 调用配置 | 同一ConvGeometry九层表包含Pool/Bypass、输出、激活及32-Cout缓存字段，并由各模块与Top共用 |
| 性能目标 | 目标分辨率、频率、fps及有效DDR带宽预算 |

缓存数量首版按32个Cout设计，仍受后续综合预算约束。分块 DDR 布局保留为性能备选；
不预先冻结全部 FIFO 深度或整条流水 II。

先前约2.84 fps仅是416输入、100 MHz、II=1、每Tile按K+6推进并含行尾空Lane的估算，不是综合或板测结论，也不是设计目标。性能评估须同时计入DDR重读、Im2Row预装、权重加载和流水停顿，不能只凭读取字节数认定某层必然受带宽限制。

## 10. 后续实施与验证顺序

1. 已实现固定层表、多Cin Group、3×3/1×1、尾通道/尾空间Lane，以及对应独立testbench。
2. PostProcess和原值行缓存Pool已实现；独立功能记录见各模块文档，模型精度及QAT误差仍需单独评估。
3. Loader、Writer及九层计算已接入 [DDR Top](top.md)；已有整网C仿真与HLS综合记录，尚须RTL验证DDR A/B层间交接和浅FIFO反压。
4. 综合和RTL复核实际存储预算、实例复用、AXI顺序及吞吐，测量整网基线后评估权重/参数加载重叠。
5. 已完成Vivado/Vitis集成及上板功能验证；后续性能优化应先测量同一配置的整网基线，不沿用旧估算作为当前实测。

测试至少覆盖Cin=3/16/1024、1×1长归约、Cout=125尾部、W=13末Tile、多Cout驻留块与连续调用。Pool应使用原始2×2四元素公式作为独立reference，不能让reference重复DUT缓存实现。

每个模块建立stream读写数量守恒；按获准阶段分别进行静态检查、C-sim、综合和co-sim。综合检查BRAM实际映射、端口冲突、归约II、阵列实例数；co-sim检查反压、FIFO深度、边界排空和层切换。不能把语法通过当成功能、资源或时序已经验证。

当前 Top 直接组织层内、块内流水。上述 Im2Row 综合结果来自此前的独立工程。
