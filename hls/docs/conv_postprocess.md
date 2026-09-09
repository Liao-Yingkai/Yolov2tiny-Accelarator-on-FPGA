# ConvPostProcess 设计

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
