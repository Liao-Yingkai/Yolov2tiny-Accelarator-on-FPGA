# Im2Row 模块详细设计

## 1. 状态与目标

- 适用工具：Vitis HLS 2022.2。
- 对应源码：`cnn_accelerator/src/im2row/im2row.cpp`。
- 固定卷积：`3×3`、`stride=1`、`same padding`。
- 当前空间并行度：`P=IM2ROW_P=4`。

当前无气泡调度要求 `1<=P<=5`。原因是行首 Tile 最多需要在 Top/Middle 的6拍内读取 `P+1` 个 Bottom 标量；本工程默认 `P=4`。

本版本的首要接口目标是：启动完成且下游可接收时，Im2Row 每一个连续时钟周期向 Systolic 写出一个 `activation_pack_t`。一个 Pack 固定包含同一 `(kh,kw)` 下、同一输出行和同一 Tile 的 P 个空间 Lane。每一个 Tile 连续输出9拍，严格为：

```text
(0,0)、(0,1)、(0,2)、
(1,0)、(1,1)、(1,2)、
(2,0)、(2,1)、(2,2)
```

Tile 间和输出行间不插入准备气泡。源码以单一 `PIPELINE II=1` 的全局 Token 循环表达此目标，循环体中恰好一次 `output_stream.write`；实际 II、RAM 读延迟和资源结果仍必须以综合报告为准。

本模块已经过多轮综合，现有结果确认稳态 Token 循环达到 `II=1`，Line Buffer 映射到 BRAM。正式功能验证、RTL Co-simulation 和硬件反压验证仍未完成。

## 2. 接口与数据契约

```cpp
void im2row(
    hls::stream<pixel_t>& input_stream,
    hls::stream<activation_pack_t>& output_stream,
    int H,
    int W,
    int Cin
);
```

`pixel_t` 打包一个空间位置的全部物理输入通道。输入是严格行优先的标量空间流，每帧严格提供 `H×W` 个真实像素；卷积 Padding 和行尾无效数据都不在输入流中。

```cpp
struct activation_pack_t {
    pixel_t pixel[IM2ROW_P];
    ap_uint<IM2ROW_P> spatial_valid;
};
```

每个输出 Pack 的契约：

- P 个 `pixel_t` 的 `(kh,kw)` 完全一致；
- P 个 Lane 属于同一输出行、同一 `tile_x`；
- `channel[0...Cin-1]` 保留，其他物理通道强制置零；
- `spatial_valid` 在同一 Tile 的9拍中不变；
- 下游 Systolic 每拍接收这一个 P 路 Pack，并将它与同拍 P 路权重对应。

## 3. 逐行 Tile 与边界规则

空间 Tile 不跨行：

```text
tiles_per_row = ceil(W/P)
tile_base_x   = tile_x × P
lane_x        = tile_base_x + lane
spatial_valid[lane] = (lane_x < W)
```

最后 Tile 中超出宽度的 Lane 是尾部无效 Lane：9拍均输出全零且 `spatial_valid=0`。它们不消费输入，也不代表卷积 Padding。

有效 Lane 在卷积边界外的访问是卷积 Padding：像素为全零，但 `spatial_valid=1`。顶部、底部、左侧和右侧均按此规则处理。

当前网络固定为两层卷积：Conv1 的 Im2Row 输入为 `28×28`；第一层 Pool 后，Conv2 的 Im2Row 输入为 `14×14`。Conv2 之后再 Pool 虽会得到 `7×7`，但网络不再进入第三层卷积，因此 `7×7` 不是本 Im2Row 的实际工作尺寸。

运行时接口仍支持 `W<P` 等小宽度，用于健壮性和边界检查；这些尺寸不参与当前两层 CNN 的时序与资源决策。

## 4. 存储结构与所有权

### 4.1 Line Buffer

```text
line_buffer[2][P][ceil(MAX_W/P)]
```

第一个维度是两行物理 Bank，第二个维度按 `x % P` 分区。处理输出行 `y` 时：

```text
middle_bank = y % 2        保存输入行 y
top_bank    = 1-middle     保存输入行 y-1
```

`top_bank` 在当前 Tile 的 Bottom 读取期间按列覆盖为输入行 `y+1`。这是安全的，因为当前 Tile 的 Top 已在 `current Strip` 中；同行下一 Tile 仍会需要的更右列在覆盖之前由预取读取。实现不复制整行，也没有逐像素生命周期表、`pending` 队列或跨行 Tile 控制。

### 4.2 双 Strip

```text
strip[2][3][P+2]
```

以 P=4 为例，每个 Strip 是 `3×6`。一个 Tile 的第 `lane` 个窗口元素为：

```text
strip[kh][lane+kw]
```

两个 Strip 采用 Ping-Pong 角色切换，但 `current Strip` 并非严格只读：

```text
current Strip：提供当前 Tile 输出；在 Top/Middle 输出期间，其 Bottom 行按需写入下一输入行
next Strip：准备下一 Tile 已可提前构造的 Top/Middle、重叠列和 Bottom 初值
```

`current Strip` 的 Bottom 写入发生在该 Bottom 被 token 6～8 读取之前，是分阶段复用而不是同一数据的冲突读写。`token=8` 的输出写成功后才交换两个 Strip。

## 5. 全局 Token 调度

顶层没有无意义的 `DATAFLOW`。所有相关状态在同一全局循环中更新：

```text
row / tile_x / token / current_strip / next_strip
```

每次迭代的固定先后关系是：

```text
1. 由 current Strip 组织一个 P 路 activation_pack_t；
2. 执行唯一的 output_stream.write；
3. 写成功后，按当前 token 执行一次输入读取、Line Buffer 更新和 next Strip 准备；
4. token=8 时交换 Strip，并推进 Tile 或输出行。
```

将 `write` 放在所有状态更新之前，是反压语义的一部分：下游阻塞时，该次迭代停在写操作，Token、输入前沿、Line Buffer、Strip 和交换状态均不推进。

### 5.1 同行下一 Tile 的时序

设当前不是本行最后一个 Tile。下一 Tile 的逻辑列由当前 Strip 的最后两列和 P 个新列组成。

| 当前 Token | current Strip 输出 | next Strip 同时准备 |
|---:|---|---|
| 0～2 | Top | 当前 Tile Bottom 按需读取；Top 仍在使用 |
| 3 | Middle | 复制当前 Top 最后两列到 next Top 前两列 |
| 4～5 | Middle | 继续当前 Tile Bottom 读取 |
| 6 | Bottom | 复制当前 Middle 最后两列到 next Middle 前两列 |
| 7 | Bottom | 从两个行 Bank 并行预取 P 个新 Top/Middle 列 |
| 8 | Bottom | 写入预取结果；复制 Bottom 两列重叠；新 Bottom 列置为零 |
| 下一拍 | 下一个 Tile 的 Top `(0,0)` | 新 Tile 开始在自己的前6拍读取 Bottom |

同行 P 个新增列为：

```text
x = tile_base_x + P + 1 ... tile_base_x + 2P
```

它们的列模 P 互不冲突；Top 与 Middle 位于不同物理行 Bank，可在一个 Bank Beat 中并行读取。代码把 Token 7 作为预取阶段、Token 8 作为写入阶段，按 BRAM 读延迟为一拍保守安排。

### 5.2 换行首 Tile 的时序

下一个输出行的首 Tile 不复用当前 Strip，逻辑列是：

```text
-1、0、1、...、P
```

`-1` 是左 Padding，真实列 `0...P` 共 P+1 个。列0和列P落入同一个列 Bank，所以 Top/Middle 需要两个读取 Beat：

| 当前行最后 Tile 的 Token | 下一行首 Tile 准备 |
|---:|---|
| 6 | 预取真实列 `0...P-1` 的 Top/Middle |
| 7 | 写入这些结果到 Strip 列 `1...P`；预取列 P |
| 8 | 写入列 P 到 Strip 最后一列；构造 Bottom 全零初值 |
| 下一拍 | 直接输出下一行首 Tile `(0,0)` |

这两个 Beat 中每个 Beat 同时访问两个物理行 Bank：下一输出行的 Top 来自旧 Middle，下一输出行的 Middle 来自已填入 `top_bank` 的新行。当前行的最后 Tile 在 Token 0～5 已完成本段 Bottom 输入，因此 Token 6～8 不与这些读取竞争。

## 6. Bottom 输入突发

Bottom 数据归属于当前输出 Tile，而不是下一 Tile。在 Token 0～5 中每拍至多读取一个真实像素，Token 6 开始输出 Bottom 前已经就绪。

非底部输出行中，每行的读取区间如下：

```text
首 Tile：x=0 ... min(P,W-1)，最多 P+1 个
后续 Tile：x=tile_base_x+1 起，最多 P 个
```

P=4 时，当前两层 CNN 实际工作宽度的核心读取数量为：

| W | 每行各 Tile 的 Bottom 读取数量 |
|---:|---|
| 28 | 5, 4, 4, 4, 4, 4, 3 |
| 14 | 5, 4, 4, 1 |

以下小宽度仅用于运行时接口健壮性与边界检查，不参与当前网络的时序与资源决策：

| W | 每行各 Tile 的 Bottom 读取数量 |
|---:|---|
| 5 | 5, 0 |
| 1～4 | W |

各区间连续、不重复，单行总数恰好为 W。因此首行预装 W 个像素，加上后续 H-1 行各 W 个 Bottom 像素，输入总量严格为：

```text
W + (H-1)×W = H×W
```

最后一个输出行的 Bottom 全为 Padding，不读取输入流。

## 7. 启动、吞吐与反压

本实现选择简单启动：先以 `II=1` 连续预装完整首行，同时建立首 Tile 的 Middle；Top 和 Bottom 先按 Padding 构造。预装完成后，首 Tile 立即进入全局 Token 循环，其 Bottom 在 Token 0～5 读取第二行前 `min(P+1,W)` 个像素。

因此本实现从首个输入到首个输出的启动延迟约为 W 个周期（外加目标器件的寄存器调度延迟）。若采用更复杂的绝对 JIT 预取，理论下界约为：

```text
max(0, W-P)
```

本版本刻意不引入该复杂控制。

稳定输出速率的目标为每周期一个 P 路 Pack。输入接口的峰值读取能力为每周期一个 `pixel_t`，但平均需求远低于此值：P=4 时，每9个输出 Token 通常读取约4个新像素。未读周期依靠阻塞式 `hls::stream` 反压暂停上游；上游不能要求无限期每周期灌入一个像素而不接受反压。

## 8. 实现与验证边界

源码中不存在以下旧结构：

```text
stored_row
stored_last_tile
stored_last_token
released 数组
pending / pending_forward
全局跨行 Tile
```

当前实现以 `#pragma HLS PIPELINE II=1` 表达目标，并按两个行 Bank 的访问模式设计时序。已有综合已经确认稳态 Token 循环达到 `II=1`，Line Buffer 映射到 BRAM；以下结果仍需后续正式验证或继续观察：

- 不同运行参数下综合结果是否继续保持目标 II；
- BRAM 端口和读延迟在后续模块集成后的时序行为；
- 同周期预取、Bottom 写入和输出在集成后的端口与时序余量；
- 资源、频率和动态多路选择开销；
- C Simulation 下与 Golden Model 的逐 Token 数值一致性；
- 输入气泡、下游反压和多帧连续处理行为。
