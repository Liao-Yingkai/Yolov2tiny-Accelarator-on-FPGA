# ConvPostProcess 设计

前五节记录现有 MNIST 实现。新版接口和调度见 [YOLOv2-tiny 重构方案](#6-yolov2-tiny-重构方案)，尚未实现。

## 1. 定位与接口

ConvPostProcess 位于 Systolic 与 Pool 之间，完成每输出通道的 bias、ReLU 和
requant；不改变 Im2Row 或 Systolic 的数据布局和事务顺序。

```text
acc_tile_t (INT32)
       │
       ▼
ConvPostProcess
       │
       ▼
activation_tile_t (INT8)
```

```cpp
void conv_postprocess(hls::stream<acc_tile_t>& input_stream,
                      hls::stream<postprocess_param_pack_t>& param_stream,
                      hls::stream<activation_tile_t>& output_stream,
                      layer_mode_t layer_mode);
```

`LAYER1_MODE`选择Conv1，`LAYER2_MODE`选择Conv2。固定Cout、spatial tile数和参数Pack数均由`src/config.h`的编译期常量派生。
两个数据 Tile 都是 `4 spatial lane × 4 Cout lane`：

```cpp
struct activation_tile_t {
    activation_t value[4][4];
    ap_uint<4> spatial_valid;
};
```

一个参数 Pack 恰好对应一个 Cout tile，且所有参数均是已由 exporter 离线
生成的 signed INT32：

```cpp
struct postprocess_param_pack_t {
    ap_int<32> bias[4];
    ap_int<32> multiplier[4];
    ap_int<32> shift[4];
};
```

参数流顺序为 `cout_tile -> cout_lane`；输入和输出事务顺序严格保持为
`spatial_tile(row -> tile_x) -> cout_tile`。

## 2. 冻结数值规则

对每个有效 spatial lane 和 Cout lane：

```text
biased     = accumulator + bias              // INT33 临时值，契约要求能装回INT32
relu_value = max(biased, 0)
product    = signed INT32(relu_value) * signed INT32(multiplier)  // 结果为INT64
scaled     = product >> shift                 // signed arithmetic right shift
output     = min(scaled, 127)                 // INT8 activation
```

这与当前 golden model 的运行时 requant 一致：`shift=31-exponent` 是
exporter Q31/frexp 表示的总右移量，不能再额外右移31位。运行时**绝不**加入
rounding offset、halfway 判断或 sticky bit；低位直接截断。离线导出 bias 和
multiplier 时使用的舍入已经固化在参数中，不属于本模块。

`multiplier` 和 `shift` 分别要求非负；非综合态会断言。一般非负 INT32 shift
都被支持：`0..63` 使用算术右移，`shift>=64` 返回符号扩展值。当前 ReLU 后
product 非负，因此后者为0。无 output zero point，且 ReLU 与非负 multiplier
使最终有效输出范围为 `[0,127]`。

## 3. 参数缓存与调用调度

一次调用处理一个完整卷积层：

| 层 | spatial tile | Cout tile | 参数 Pack 读取数 | 输入/输出 Tile 数 |
|---|---:|---:|---:|---:|
| Conv1 | 196 | 2 | 2 | 392 |
| Conv2 | 56 | 4 | 4 | 224 |

参数缓存为完整分区的 `4 cout_tile × 4 cout_lane` INT32 寄存器阵列：bias、
multiplier、shift 各一份。在第一个 spatial tile 中，模块按 Cout tile 从
`param_stream` 读取一个 Pack、立即用于计算并写入缓存；后续 spatial tile
只回放缓存且不读取参数流。这样参数流的消费量与 Systolic 的权重流同样只和
Cout tile 数有关。

## 4. Valid、循环与吞吐

`spatial_valid` 原样传递。无效 spatial lane 不进入 bias、ReLU 或 requant，
而是强制写出4个零；这避免零 accumulator 被非零 bias 变成伪像素。

主循环为：

```text
spatial_tile
    -> cout_tile
        -> spatial_lane (PIPELINE II=1)
            -> cout_lane (UNROLL)
```

因此每周期处理一条 spatial lane 的4个 Cout lane，设计含4套并行
signed INT32×INT32→INT64 requant datapath（不是64×64乘法）。`activation_tile_t` 放在完全分区的寄存器中，
4个 spatial lane 完成后写出一个 stream 事务。实现没有指定特定 HLS 版本的
DSP binding pragma，让工具按目标器件映射乘法器。

## 5. 验证与综合结果

独立 C-sim 测试覆盖直接截断、bias 后负/零/正、127/128饱和、`shift>=64`、
invalid lane、完整等效 Conv1/Conv2 per-Cout 参数集、2/4个参数 Pack 消费量及
全部输入输出 Tile 守恒。reference 使用纯 C++ `int64_t`，不依赖 PyTorch。

Vitis HLS 2022.2 已对目标器件 `xc7z020-clg400-2`、时钟约束 10 ns 完成
`csynth`。估计时钟周期为 7.050 ns；spatial-lane 流水循环达到 `II=1`、
depth 为 5。内层流水模块的 latency/interval 均为 9 个周期，外层每个
Tile iteration 的 latency 为 13 个周期。

综合资源为 0 BRAM、12 DSP、3,126 FF、2,933 LUT。四个
`mul_31ns_32s_63` 实例各使用 3 个 DSP，实现的是 32 位有符号数相乘并生成
63 位乘积；不存在 64×64 乘法器。参数缓存实现为寄存器。一般动态右移逻辑
约使用 716 LUT，是后续资源优化时最值得关注的部分。

以该实现的外层 13 cycles/Tile 计，Conv1 总计 5,097 cycles，Conv2 总计
2,913 cycles。现有 Systolic 综合结果为 19 cycles/Tile，因此 ConvPostProcess
已确认快于 Systolic，不是当前流水瓶颈。

若动态移位器成为瓶颈，可在另一次经验证的优化中，针对 exporter 明确保证
`shift>=31` 的模型研究两级右移；当前实现不作这一限制，也不硬编码当前模型
的39/40。候选等价变换为：

```text
product >> shift = (product >> 31) >> (shift - 31)
```

采用前仍须限制或验证 `shift>=31`，并以边界值逐位比对直接 64 位右移的结果，
再重新比较 LUT 与时序。

## 6. YOLOv2-tiny 重构方案

### 6.1 一次调用处理整层，内部遍历 Cout 驻留块

`conv_postprocess` 接收整层 `ConvGeometry`，输入为 `yolo_acc_tile_t`，输出为含 `spatial_valid`、`cout_valid` 的 `4 spatial × 4 Cout` INT8 Tile。Top 按层调用 PostProcess，不传当前块的 `first_cout_tile` 或 `block_cout_tiles`；模块按与 Systolic 相同的块划分自行推导。PostProcess 不遍历 Cin Group，输入已完成整个卷积归约。

整个输入流顺序是 `Cout 块 → spatial_tile（y → tile_x）→ local_cout_tile`。32 Cout 块包含 **8 个** 4-Cout 计算 Tile；例子中 local tile `0…3` 对应 Conv1 的 16 Cout 小块，完整 32 Cout 块则为 `0…7`。先处理当前块在所有空间位置的结果，才切到下一组通道。

```text
for 每个 Cout 块：
    first = 本块的起始全局 Cout Tile
    B = min(geometry.resident_cout_tiles, geometry.cout_tiles - first)
    按 local_cout_tile = 0…B-1 读 B 个参数 Pack，准备本块缓存
    for spatial_tile = 0…geometry.spatial_tiles-1：
        for local_cout_tile = 0…B-1：
            读一个 INT32 Tile
            只用本地缓存[local_cout_tile]，生成一个 INT8 Tile
            按原顺序写出，传递两个 valid mask
```

每块读 `B` 个参数 Pack 和 `spatial_tiles*B` 个累加 Tile，写同样多的 INT8 Tile。整层读 `geometry.cout_tiles` 个参数 Pack，输入输出各 `geometry.spatial_tiles*geometry.cout_tiles` 个 Tile。块边界由计数确定，不额外发送块头或坐标。参数流按全局 Cout Tile 递增；BIN 里只有逻辑 Cout 个 bias/M/S，Loader 先按 `oc=(first+local_cout_tile)*4+lane` 判界再读，无效 lane 在 Pack 内填零，不读越界地址。Prediction 最后一个 Pack 只含 C124 一个有效 lane。

Systolic 现有 OS 入口仍一次处理一块，不因本次 PostProcess 重构而改变。整网集成时，上游层级调度依次调用 OS，把各块接续到同一条输出流；PostProcess 同时按整层运行，不能让上游等待整层 PostProcess 返回后才供给下一块。Loader 须保证对应块参数可以到达，不能先因写满累加路径相关的浅 FIFO 而停在参数发送之前。Pool/Writer 按同一块顺序消费，层切换仍等待整层写回完成。

### 6.2 单份参数缓存与一次 signed requant

首版只保留当前块的 `bias`、`M_pos`、`M_neg`、`S`，最多 32 通道，不保存整层参数或 Feature Map。块内按 4 Cout lane 并行读，local Cout Tile 为缓存地址。参数加载阶段用共享运算路径逐通道派生 `M_neg`，不复制 32 份常量乘法器：

```text
M_pos = 导出的 M
M_neg = (INT64(M_pos) * 1717986918 + 2^33) >> 34

a = accumulator + bias
m = geometry.use_leaky_relu && a < 0 ? M_neg : M_pos
out = clamp((INT64(a) * m) 算术右移 S, -128, 127)
```

负支系数只在装载时计算一次；`+2^33` 仅用于系数派生，逐元素乘移不舍入。正负支共用 S，且只做一次逐元素乘法。Prediction 不派生 LeakyReLU 系数，正负都用原始 M/S。bias 相加须满足 INT32 契约；`S>=64` 按符号扩展。任一 mask 无效时，lane 不加 bias、不做乘移，输出零但不删掉 Tile。保留 4 路 Cout 并行、逐 spatial lane 处理；循环目标 `II=1`，实际 Tile 间隔、DSP 和移位器 LUT 留待综合确认。

### 6.3 参数预取：首版不做，后续优化

**首版先加载并准备当前块的全部参数，再消费该块首个累加 Tile。** 当前块最后一个 INT8 Tile 成功写入输出流后，才覆盖缓存、读下一块参数。无需等 Pool/Writer 写完 DDR 才复用参数缓存：写出的 Tile 已经是完整 INT8 结果，不再引用缓存。首版不读取下一块参数、不做 ping-pong；当前块等待时使用 stream 自然反压。

**后续实施项：**测量每块参数加载和负支系数准备有多少时间暴露在关键路径上，其中部分可能与上游 Systolic 的权重加载重叠。若仍有明显切块等待，优先评估两份小参数缓存：当前块只读活跃缓存，后台读入下一块并生成负支系数，当前块最后一个结果提交后才切换到已就绪缓存。也可对比首 Spatial Tile 边读参数边计算、随后缓存复用的方案。两者都须验证参数顺序、块边界、阻塞及输出一致性；不改变 Python BIN 或 Systolic 的 Tile 顺序。

独立验收：用同一套 Python 导出的 Systolic INT32、bias/quant BIN 对照 PostProcess INT8 BIN，覆盖 Conv1 小块、多块参数切换、Prediction 最后一个 Cout Tile、两种无效 lane、bias 改变符号、负支截断和上下饱和。核对每层/每块事务数，不将定向测试数据混入真实 BIN。
