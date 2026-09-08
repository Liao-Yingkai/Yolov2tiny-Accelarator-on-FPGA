# Systolic / Broadcast PE Array 设计

## 1. 模块定位与当前状态

Systolic 位于 Im2Row 与 ConvPostProcess 之间，只负责卷积的 INT8 乘法和 INT32 累加：

```text
activation_pack_t
        │
        ▼
Broadcast PE Array
        │
        ▼
INT32 acc_tile_t
        │
        ▼
Bias / Requant / ReLU
```

本模块不是经典的 A、B 双向传播 wavefront systolic array，而是一个 `spatial × Cout` 的 Broadcast、Output-Stationary PE Array。

本文记录第一版已冻结的架构契约。Systolic 已实现，并已完成独立综合；当前实现的 token loop 达到 `II=1`。资源映射、时钟频率和 RTL 级反压行为仍必须随集成持续验证。

## 2. 单物理实例与运行方式

系统只实例化一套物理 Im2Row、Systolic、ConvPostProcess 和 Pool。Conv1、Conv2 分两个阶段顺序使用同一套硬件：

```text
阶段 1：Conv1
Input → Im2Row → Systolic → ConvPostProcess → Pool → Pool1 Feature Buffer

阶段 2：Conv2
Pool1 Feature Buffer → Im2Row → Systolic → ConvPostProcess → Pool → FC
```

每个卷积层内部的模块通过 `hls::stream<>` 连接，并计划在上层形成 DATAFLOW。Conv1 和 Conv2 之间不并发；Pool1 的完整输出必须先写入片上 Feature Buffer，再作为 Conv2 的输入。

Pool1 Feature Buffer 的有效数据量为：

```text
14 × 14 × 8 × 8 bit = 12,544 bit = 1,568 Byte
```

具体 top-level 共享控制和 HLS 实例约束留到各模块完成后设计。Systolic 本身保持“一次运行处理一个完整卷积层”的单层语义。

## 3. 第一版物理参数

第一版固定：

```text
SPATIAL_PAR = 4
COUT_PAR    = 4
CIN_PAR     = 8
Kernel      = 3×3
Stride      = 1
Padding     = 1
MAX_COUT    = 16
MAX_COUT_TILES = 4
```

`IM2ROW_P=4` 与 Cout 并行度都固定为4，但两者在概念上不同：

```text
spatial lane：同一输出行中连续的 4 个 W 位置
Cout lane：当前 Cout tile 中连续的 4 个输出通道
```

同一套物理硬件只支持并分时复用下列两组配置：

| 层 | H×W | Cin | Cout | Spatial tiles | Cout tiles |
|---|---:|---:|---:|---:|---:|
| Conv1 | 28×28 | 1 | 8 | `28×ceil(28/4)=196` | 2 |
| Conv2 | 14×14 | 8 | 16 | `14×ceil(14/4)=56` | 4 |

函数签名保留 `H/W/Cin/Cout`，但综合态只以 `Cin==8` 形成一位 `is_conv2` 选择：Conv1 的 `spatial_tiles=196、cout_tiles=2`，Conv2 的 `spatial_tiles=56、cout_tiles=4`。完整四元组只在非综合构建中断言；综合态不保留通用拒绝路径。Cout tail 不支持，Conv2 的空间 tail 继续使用 `spatial_valid=0011`。

## 4. Activation 接口契约

Systolic 直接消费现有 Im2Row 输出：

```cpp
struct activation_pack_t {
    pixel_t pixel[IM2ROW_P];
    ap_uint<IM2ROW_P> spatial_valid;
};
```

一个 Pack 表示：

```text
4 个 spatial lane
×
每个 lane 的 8 个物理 activation channel
```

Im2Row 对一个 spatial tile 连续输出 9 个 token：

| token | kh | kw |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 1 | 0 |
| 4 | 1 | 1 |
| 5 | 1 | 2 |
| 6 | 2 | 0 |
| 7 | 2 | 1 |
| 8 | 2 | 2 |

Systolic 不需要显式接收 `kh`、`kw`，而是维护 `token=0..8` 计数器。

Im2Row 在输入 pixel 刚被消费时完成通道规范化：Conv1 清零 channel 1..7，Conv2 保留全部8通道。因此 Systolic 的 PE 不再进行运行时 `channel < Cin` 判断，而是固定计算全部8个物理 channel。

对于 Conv1：

```text
channel[0]    = 有效 activation
channel[1..7] = 0
```

对于 Conv2，8 个 channel 全部有效。

## 5. PE Array 数学映射

PE 阵列的逻辑维度为：

```text
PE[spatial_lane][cout_lane]
```

每个周期，一个 PE 计算：

```text
dot[i][j] = Σ(c=0..7) activation[i][c] × weight[j][c]
```

并更新当前输出的 accumulator：

```text
acc[i][j] ← acc[i][j] + dot[i][j]
```

完整 3×3 reduction 为：

```text
acc[i][j]
= Σ(kh=0..2) Σ(kw=0..2) Σ(c=0..7)
   activation[i][kh][kw][c] × weight[j][kh][kw][c]
```

一个计算周期的物理乘法并行度为：

```text
4 spatial × 4 Cout × 8 Cin = 128 个 INT8 multiplication
```

Conv1 的 channel 1～7 为零，因此只有 16 个乘法产生有效贡献。由于同一物理阵列随后供 Conv2 使用，这不会增加静态硬件面积。第一版接受 Conv1 的低利用率，不增加动态 channel/Cout 重映射，以避免额外的 mux、控制和 LUT 消耗。

## 6. 数值位宽

类型必须显式保持有符号语义：

```text
activation : signed INT8
weight     : signed INT8
product    : signed 16 bit
8 路 dot   : 至少 signed 19 bit
accumulator: signed INT32
```

极值 `(-128)×(-128)×8=131072` 超出 signed 18-bit 正范围，因此 8 路 dot 不能使用 18-bit signed 类型。

第一版可先让 HLS 综合完全展开的 channel reduction，但不得依赖不明确的表达式自动位宽。综合后重点检查加法结构、关键路径、LUT/DSP 映射和 accumulator recurrence。

## 7. Accumulator 调度

每个 PE 对当前 Cout tile 保持一个 INT32 accumulator。第一版不设置独立的 accumulator clear 周期，而在 token 0 直接用第一个 dot 覆盖旧值：

```text
token 0：acc = dot
token 1～8：acc = acc + dot
```

token 8 的 dot 必须包含在输出结果中。实现可以使用更新后的 `acc`，也可以使用临时 `next_acc`：

```text
next_acc = (token == 0) ? dot : acc + dot
token < 8：acc = next_acc
token = 8：result = next_acc
```

`next_acc` 只是对本拍组合结果的明确表达，不要求增加一个独立累加器。

一个 Cout tile 的逻辑 reduction 长度固定为 9 个 token，但第一版不要求 Cout tile 边界严格零气泡。结果组织、stream write 或控制切换可以产生额外周期。第一版优先保证功能正确和 token 守恒，不实现 accumulator ping-pong。

若后续要求稳定每 9 周期输出一个结果 Tile，需要进一步处理：

- channel dot-product 的流水延迟；
- accumulator 的循环相关；
- token 和 valid 随计算流水线的对齐；
- token 8 结果输出与下一 Cout tile token 0 的重叠；
- downstream 阻塞时的在途状态保存。

## 8. Activation Replay Buffer

同一个 spatial tile 会用于该层的全部 Cout tile，因此使用四个显式的 spatial-lane replay bank：

```text
activation_bank[4][9]

activation_bank[0]：spatial lane 0，9 个 64-bit word
activation_bank[1]：spatial lane 1，9 个 64-bit word
activation_bank[2]：spatial lane 2，9 个 64-bit word
activation_bank[3]：spatial lane 3，9 个 64-bit word
```

每个 64-bit word 显式打包一个 `pixel_t` 的 8 个 INT8 activation，位序固定为：

```text
bit [7:0]   = channel 0
bit [15:8]  = channel 1
...
bit [63:56] = channel 7
```

逻辑容量仍为：

```text
9 × 4 spatial × 8 channel × 8 bit = 2304 bit
```

每个 spatial tile 的调度为：

```text
Cout tile 0：
    activation = activation_stream
    同时把4个 spatial lane 分别写入 activation_bank[lane][token]

Cout tile 1 及之后：
    同时从4个 Bank 读取 activation_bank[lane][token] 并解包
    不读取 activation_stream
```

lane 维 complete partition 后，每拍可并行访问四个单端口 64-bit BRAM bank，即固定得到 `4×64=256 bit/cycle` 的完整 token 带宽。`spatial_valid` 在同一个 spatial tile 的9个 token中不变，故不写入 replay BRAM，而是由 token 0 捕获到 `tile_spatial_valid`，在 replay 时重新附加。

当前 spatial tile 的全部 Cout tile 完成后，buffer 可被下一个 spatial tile 覆盖。第一版不使用 activation ping-pong；replay 期间依靠 stream backpressure 暂停 Im2Row。

## 9. Weight Stream 与布局

建议逻辑类型为：

```text
weight_pack_t
    [4 Cout lanes]
    [8 physical input channels]
```

每个 Pack 含 32 个 INT8 weight：

```text
4 × 8 × 8 bit = 256 bit = 32 Byte
```

Weight stream 的严格顺序为：

```text
cout_tile
    → token = kh*3 + kw
        → cout_lane
            → channel
```

导出端应满足：

```text
weight_pack[cout_lane][channel]
= ConvWeight[cout_tile*COUT_PAR + cout_lane]
             [channel]
             [kh]
             [kw]
```

等价 GEMM flatten 顺序仍为：

```text
kh → kw → cin
```

Conv1 的 `channel[1..7]` 权重由导出端填零。即使 activation 已经保证这些通道为零，权重填零仍有利于接口一致性、波形检查和降低无意义翻转。

## 10. Weight Buffer：四个 BRAM Bank

第一版采用四个独立的 64-bit BRAM bank，不使用 LUTRAM：

```text
weight_bank[0]：Cout lane 0，每字 8×INT8 = 64 bit
weight_bank[1]：Cout lane 1，每字 8×INT8 = 64 bit
weight_bank[2]：Cout lane 2，每字 8×INT8 = 64 bit
weight_bank[3]：Cout lane 3，每字 8×INT8 = 64 bit
```

每个 bank 的最大逻辑深度为：

```text
MAX_COUT_TILES × 9 = 4 × 9 = 36
```

地址可以组织为：

```text
address = cout_tile * 9 + token
```

每个周期四个 bank 各读取一个 64-bit word，总读取带宽为：

```text
4 × 64 bit = 256 bit/cycle
```

逻辑总容量为：

```text
4 × 36 × 64 bit = 9216 bit
```

实际实现应明确请求 BRAM，并通过综合报告确认没有落入 LUTRAM。第一版不使用两个 true-dual-port BRAM 组合四个 lane 的方案；该方向仅作为未来 BRAM 数量优化候选。

## 11. 每层只加载一次权重

“每次加载”指每个完整卷积层开始时加载一次，绝不表示 spatial tile 换行时重新加载。

一次 Systolic 运行处理一个完整卷积层。进入该层时，权重状态视为未加载；在该层第一个 spatial tile 期间边计算、边写 Weight Buffer：

```text
Conv1：
    第一个 spatial tile：
        Cout tile 0 的 9 个 token → 读取并缓存权重
        Cout tile 1 的 9 个 token → 读取并缓存权重
    后续 195 个 spatial tile：
        全部从 Weight Buffer replay

Conv2：
    第一个 spatial tile：
        4 个 Cout tile × 9 token → 读取并缓存权重
    后续 55 个 spatial tile：
        全部从 Weight Buffer replay
```

行内 Tile 切换和输出行换行都不会重新读取 weight stream。

进入下一卷积层时重新开始一次新的层调用，并覆盖 Weight Buffer：

```text
Conv1 开始 → 加载 Conv1 权重一次
Conv1 完成
Conv2 开始 → 覆盖并加载 Conv2 权重一次
Conv2 完成
```

第一版不要求权重跨层或跨图片常驻。不得使用含义不清晰、跨调用保持的 static `weights_loaded`；权重有效状态只属于当前层运行过程。

## 12. Spatial Valid

同一个 spatial tile 的 9 个 activation token 应携带完全相同的 `spatial_valid`。Systolic 在 token 0 记录该 mask，并把它附加到该 spatial tile 的每个 Cout result tile。

第一版不 gate invalid PE row，因为 Im2Row 已把无效 spatial lane 的 activation 清零。因此 Systolic 的对应 accumulator 自然为零。

但是 ConvPostProcess 加入 bias 后，无效 lane 不再自然保持零。因此 `spatial_valid` 必须继续经过：

```text
Systolic → ConvPostProcess → Pool
```

ConvPostProcess 应继续传递 mask，并建议在输出时把 invalid spatial lane 强制清零，防止 bias/requant 结果意外参与后续处理。

非综合测试应检查同一 spatial tile 的 9 个输入 token 具有一致的 `spatial_valid`。

## 13. 输出类型与顺序

建议 INT32 输出类型为：

```cpp
struct acc_tile_t {
    acc_t value[SPATIAL_PAR][COUT_PAR];
    ap_uint<SPATIAL_PAR> spatial_valid;
};
```

其中：

```text
value[spatial_lane][cout_lane]
```

表示同一输出行中最多 4 个相邻 W 位置和当前 4 个 Cout。

输出顺序固定为：

```text
spatial_tile = output_y × tiles_per_row + tile_x
    → cout_tile
```

`spatial_tile` 只是内部合并计数器；外部可观察顺序仍严格等价于 `output_y → tile_x → cout_tile`。第一版不显式输出 H/W 坐标或 `cout_tile_index`。ConvPostProcess 和 Pool 根据固定模式和固定事务顺序维护自己的计数器。

如果 output stream 阻塞，Systolic 不得覆盖当前结果、推进 Cout/spatial 计数器或消费属于下一计算组的 activation。第一版允许 PE 在 result write 阻塞时整体停顿。

## 14. Stream 事务守恒

固定模式下：

```text
Conv1：spatial_tiles = 196，cout_tiles = 2
Conv2：spatial_tiles = 56， cout_tiles = 4
```

一次完整层运行必须满足：

```text
activation stream reads = spatial_tiles × 9
weight stream reads     = cout_tiles × 9
result stream writes    = spatial_tiles × cout_tiles
```

具体为：

| 层 | Activation Pack 读取 | Weight Pack 读取 | Result Tile 输出 |
|---|---:|---:|---:|
| Conv1 | `196×9 = 1764` | `2×9 = 18` | `196×2 = 392` |
| Conv2 | `56×9 = 504` | `4×9 = 36` | `56×4 = 224` |

Activation replay 不会增加 activation stream 的读取数量；Weight replay 不会增加 weight stream 的读取数量。

## 15. 实际控制层级

第一版逻辑循环为：

```text
for each spatial tile:
    capture tile spatial_valid
    weight_base = 0

    for each Cout tile:
        for token 0..8:
            Cout tile 0:
                activation stream read + activation buffer write
            其他 Cout tile:
                activation buffer replay

            第一个 spatial tile:
                weight stream read + Weight Buffer write
            其他 spatial tile:
                Weight Buffer read

            PE dot + accumulate

        output one acc_tile_t
        weight_base += 9
```

实现使用单一 `spatial_tile` 循环，不再在 Systolic 内部根据运行时 H/W 计算或嵌套 `output_y × tile_x`。`weight_base` 为窄地址基址，按每个 Cout tile 加9；这与原有 `cout_tile×9+token` 地址和权重流顺序完全等价。第一版允许 Cout tile 边界气泡，不实现 accumulator ping-pong。

## 16. HLS 结构要求

当前实现：

- spatial lane 循环完全 UNROLL；
- Cout lane 循环完全 UNROLL；
- channel 乘法循环完全 UNROLL；
- accumulator 的 spatial/Cout 两维完全 partition；
- PE 输入所需的 activation/weight lane 和 channel 必须支持同周期并行访问；
- Weight Buffer 采用四个 64-bit BRAM bank；
- token reduction loop 已约束并在独立综合中达到 `II=1`；
- Systolic 内部第一版不使用 DATAFLOW；
- 顶层层内模块连接以后再使用 DATAFLOW。

集成后的综合仍必须检查：

- 实际 token loop II；
- accumulator recurrence constraint；
- channel reduction 是否形成过长的串行加法链；
- 128 个 INT8 multiplier 的 DSP/LUT 映射；
- adder tree 的 LUT 消耗；
- Weight Buffer 是否全部映射 BRAM；
- activation replay buffer 的资源映射和读写端口；
- result stream 阻塞时的 pipeline stall 行为。

## 17. 第一版明确不做的内容

第一版不实现：

- Conv1 的 channel multiplier 到额外 Cout lane 的动态重映射；
- activation ping-pong；
- accumulator ping-pong；
- Cout tail valid；
- Systolic 内部 Bias、Requant 或 ReLU；
- 显式 H/W 坐标或 Cout tile metadata；
- 跨层、跨图片权重常驻；
- top-level 单实例调度器；
- 为严格零气泡输出而设计的复杂多级状态机。

## 18. 验证要求

实现后至少需要：

1. 使用软件 golden convolution 逐 Tile、逐 lane、逐 Cout 比较 INT32 结果；
2. 分别覆盖 Conv1 和 Conv2 参数；
3. 验证负数 activation/weight 和 INT8 极值；
4. 验证 Conv2 `W=14` 的行尾 `spatial_valid=0011` 语义；
5. 检查同一 spatial tile 的 9 个 valid mask 一致；
6. 检查 activation、weight、result 三类 stream 事务守恒；
7. 验证第一个 spatial tile 后不再读取 weight stream；
8. 验证 output backpressure 不会丢失、重复或错位 token；
9. 综合确认 token loop II、BRAM、DSP、LUT、时钟频率；
10. RTL co-sim 验证带输入气泡和下游阻塞的行为。
