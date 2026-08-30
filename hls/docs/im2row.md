# Im2Row 模块详细设计

## 1. 文档状态

- 适用工具：Vitis HLS 2022.2。
- 目标目录：`cnn_accelerator/src/im2row/`。
- 当前阶段：优化后的微架构已经确定，文档已更新，源码尚未按本方案重构。
- 当前源码仍是旧版 P 路输入、完整 Patch Tile 和 Channel 串行输出结构，不能作为本方案的实现依据。
- 本文档是下一阶段重写 `types.h`、`im2row.h` 和 `im2row.cpp` 的设计基线。
- 本阶段不执行 C Simulation、C Synthesis 或 RTL Co-simulation。

## 2. 最终设计结论

第一版 Im2Row 固定支持：

```text
Kernel  = 3×3
Padding = 1
Stride  = 1
P       = 4
```

优化后的核心决策如下：

1. 外部输入改为标量空间流，每个成功握手周期最多读取一个 Channel-packed `pixel_t`。
2. Line Buffer 保留 P 路列 Bank，每周期最多为 Systolic 并行读取 P 个空间数据。
3. 两个物理行缓存按输入行奇偶循环使用，不再执行整行复制，也不物理执行 `top=middle`。
4. 每个 Line Buffer 槽位完成最后一次读取后，允许被两行之后的同列输入覆盖。
5. Line Buffer 可以在行边界附近同时保存不同代的行数据，更新粒度是单个列槽位，不是整行。
6. 空间 Tile 按全局 `row×W+col` 连续分组，允许一个 Tile 跨越两行。
7. 不再为每一行补齐 P 个输出位置，只在整帧最后一个 Tile 使用 `spatial_valid`。
8. 每个 Lane 独立计算输出坐标和 Padding，Systolic 不需要知道空间坐标。
9. 取消完整 `3×(P+2)` Sliding Strip、P 个完整 3×3 Window 和 P×K Patch Tile Buffer。
10. 使用一个小型待提交输入缓冲解决“新数据已经到达，但旧槽位尚未完成最后一次读取”的情况。
11. Im2Row 每周期输出 P 个 `pixel_t`，一个 Token 对应同一个 `(kh,kw)`。
12. 一个 P 空间位置 Tile 固定输出 9 个 Token，顺序为 `kh → kw`。
13. Channel 在 `pixel_t` 中完整打包；PE 内实际 Channel 并行度 `C_PAR` 由 Systolic 决定，不属于 Im2Row 调度维度。
14. 输入和输出均使用阻塞式 `hls::stream`，下游暂停时，内部状态、Line Buffer 更新和输入读取必须一起暂停。

## 3. 模块职责和边界

Im2Row 负责把 NHWC Feature Map 在线转换为广播型 `P×P` PE 阵列需要的激活流，不显式生成完整 Im2Row 矩阵。

本模块负责：

- 接收逐像素、Channel-packed、行优先的输入流；
- 在片上保存卷积所需的历史行；
- 为 P 个连续全局输出位置生成 3×3 窗口数据；
- 在元素最后一次使用后安全覆盖 Line Buffer；
- 对每个 Lane 独立产生上下左右 Padding 零；
- 按 `kh → kw` 顺序，每周期输出 P 个 Channel-packed 激活；
- 生成整帧最后一个 Tile 的有效 Lane 掩码；
- 通过 Stream 反压保证输入、内部状态和输出严格同步。

本模块不负责：

- DDR、AXI Master、DMA 或外部输入数据加载；
- 权重加载、权重缓存和权重调度；
- `Cout` 分 Tile；
- 激活针对多个 `Cout Tile` 的重放；
- Bias、Requantization、激活函数和输出 Channel 组装；
- Pool、FC 和顶层控制；
- 非 3×3、非 Padding 1、非 Stride 1 的卷积。

## 4. 参数、约束和尺寸

### 4.1 编译期参数

```text
P                 空间并行度，第一版固定为 4
MAX_H             最大输入高度
MAX_W             最大输入宽度
MAX_CHANNELS      最大物理输入 Channel 数
KH = 3
KW = 3
PADDING = 1
STRIDE = 1
PENDING_DEPTH     待提交输入缓冲深度，第一版取 P+2
```

当前默认上限计划继续使用：

```text
MAX_H        = 28
MAX_W        = 28
MAX_CHANNELS = 8
P            = 4
PENDING_DEPTH = 6
```

### 4.2 运行时参数

```text
H      当前 Feature Map 高度
W      当前 Feature Map 宽度
Cin    当前有效输入 Channel 数
```

必须满足：

```text
0 < H   <= MAX_H
P <= W  <= MAX_W
0 < Cin <= MAX_CHANNELS
```

`W>=P` 是连续跨行 Tile 和无冲突 Bank 证明的前提。若未来需要支持 `W<P`，必须增加回退调度或重新设计 Bank 结构，不能直接复用本文档中的证明。

由于卷积参数固定：

```text
Ho = H
Wo = W
空间输出数量 = H×W
空间 Tile 数量 = ceil(H×W/P)
每个 Tile 的 Im2Row Token 数量 = 9
```

## 5. 数据类型和接口

以下声明表示目标接口，下一阶段编码时可以调整字段名，但不能改变数据含义。

### 5.1 激活和 Pixel

```cpp
typedef ap_int<8> activation_t;

struct pixel_t {
    activation_t channel[MAX_CHANNELS];
};
```

一个 `pixel_t` 表示一个空间位置的全部物理 Channel。运行时仅有：

```text
channel[0 ... Cin-1]
```

有效。其余 Channel 必须由输入约定置零，或由 Systolic 根据 `Cin` 忽略。

当前量化为对称 int8，激活零点为 0，因此 Padding 的 `pixel_t` 所有 Channel 均写 0。

### 5.2 输入接口

目标输入接口：

```cpp
hls::stream<pixel_t>& input_stream
```

输入契约：

- 一个 Token 只包含一个空间像素；
- 一个像素包含全部物理 Channel；
- Token 顺序严格为 NHWC 的空间行优先顺序；
- 由于 Channel 已打包，实际空间顺序为 `row → col`；
- 每帧恰好输入 `H×W` 个 Token；
- 输入流不携带 Padding，不携带每行尾部无效 Token；
- Padding 完全由 Im2Row 内部生成；
- 上游必须遵守 Stream 反压，只有成功握手的数据才算被消费。

### 5.3 输出接口

目标输出包：

```cpp
struct activation_pack_t {
    pixel_t pixel[P];
    ap_uint<P> spatial_valid;
};
```

目标输出接口：

```cpp
hls::stream<activation_pack_t>& output_stream
```

一个输出 Token 表示：

- P 个全局连续空间位置；
- 所有 Lane 使用同一个 `(kh,kw)`；
- 每个 Lane 的 Channel 已完整打包在 `pixel_t` 中；
- `spatial_valid[lane]` 表示该 Lane 是否仍位于本帧 `H×W` 范围内。

同一个空间 Tile 固定输出 9 个 Token：

```text
k=0  → kh=0, kw=0
k=1  → kh=0, kw=1
k=2  → kh=0, kw=2
k=3  → kh=1, kw=0
k=4  → kh=1, kw=1
k=5  → kh=1, kw=2
k=6  → kh=2, kw=0
k=7  → kh=2, kw=1
k=8  → kh=2, kw=2
```

### 5.4 顶层函数

目标函数接口：

```cpp
void im2row(
    hls::stream<pixel_t>& input_stream,
    hls::stream<activation_pack_t>& output_stream,
    int H,
    int W,
    int Cin
);
```

内部函数不添加 AXI 接口编译指示。AXI4-Stream、AXI4 Master 和 AXI-Lite 统一在最终顶层函数设计。

## 6. Channel 打包与 Systolic 契约

Im2Row 的输出不再包含独立的 `cin` 循环。每周期输出：

```text
P 个 pixel_t = P×MAX_CHANNELS 个物理激活 Lane
```

若 Systolic 完全并行处理 Channel：

```text
C_PAR = MAX_CHANNELS
一个空间 Tile 在 9 周期内完成输入累加
```

若 Systolic 只并行处理部分 Channel：

```text
C_PAR < Cin
```

则 Systolic 在读取一个 `activation_pack_t` 后锁存其中的 P 个 `pixel_t`，内部执行：

```text
ceil(Cin/C_PAR)
```

个 Channel Fold。Channel Fold 不能要求 Im2Row 重复读取输入 Feature Map。Systolic 尚未完成当前 Token 时，通过输出 Stream 反压暂停 Im2Row；`C_PAR`、权重切片和 PE 内加法树在 Systolic 文档中确定。

## 7. 全局空间 Tile 和坐标生成

### 7.1 全局连续分组

所有输出位置使用全局索引：

```text
spatial_index = output_y×W + output_x
```

Tile `t`、Lane `lane` 对应：

```text
spatial_index = t×P + lane
```

Tile 可以跨越两行。例如 `W=14、P=4` 时可能出现：

```text
lane 0 → (row r,   col 12)
lane 1 → (row r,   col 13)
lane 2 → (row r+1, col 0)
lane 3 → (row r+1, col 1)
```

Systolic 只把它们视为 P 个独立空间位置，不要求 Lane 属于同一行。

### 7.2 只在整帧尾部使用 Mask

```cpp
spatial_valid[lane] =
    (tile_base_index + lane) < H * W;
```

不再为每一行单独补齐 P。常见尺寸：

```text
28×28 = 784，可被 4 整除
14×14 = 196，可被 4 整除
7×7   = 49，最后一个 Tile 只有 1 个有效 Lane
```

### 7.3 避免运行时除法

热循环中禁止直接为每个 Lane 综合：

```cpp
output_y = spatial_index / W;
output_x = spatial_index % W;
```

第一版维护 Tile 起点：

```text
base_y
base_x
```

由于 `W>=P`，一个 Tile 最多跨越一次行边界。每个 Lane 坐标可写成：

```cpp
int lane_x_raw = base_x + lane;
bool wrapped = lane_x_raw >= W;

lane_y = base_y + (wrapped ? 1 : 0);
lane_x = wrapped ? lane_x_raw - W : lane_x_raw;
```

Tile 完成后：

```cpp
int next_x = base_x + P;
if (next_x >= W) {
    base_x = next_x - W;
    base_y = base_y + 1;
} else {
    base_x = next_x;
}
```

最后一个 Tile 仍按相同方式计算坐标，但无效 Lane 不访问 Line Buffer。

## 8. Line Buffer 物理组织

### 8.1 两行循环存储

目标存储：

```cpp
pixel_t line_buffer[2][P][MAX_BLOCKS_PER_ROW];
```

其中：

```text
MAX_BLOCKS_PER_ROW = ceil(MAX_W/P)
```

三个维度含义：

```text
dim 0：输入行奇偶 Bank
dim 1：列 Bank
dim 2：Bank 内地址
```

实际输入坐标 `(input_y,input_x)` 的地址为：

```cpp
row_bank = input_y & 1;
col_bank = input_x % P;
address  = input_x / P;
```

P=4 时，`%P` 和 `/P` 应综合为低位选择和移位，不应生成通用除法器。

### 8.2 行角色不通过复制实现

硬件不执行：

```text
top_line    = middle_line
middle_line = current_line
```

输入行 `y+2` 复用输入行 `y` 的 `row_bank`，在对应列槽位释放后直接覆盖：

```text
row y     和 row y+2 使用相同 row_bank
row y+1   使用另一个 row_bank
```

因为覆盖是逐列发生的，同一物理 Row Bank 在行边界附近可以同时包含旧行尾部和新行头部。读取时使用实际 `input_y` 决定行奇偶，不能假定整个 Bank 只属于一条完整逻辑行。

### 8.3 Channel 和 Bank 分区

目标分区：

- `pixel_t.channel`：Complete Partition；
- `line_buffer` 的 Row Bank 维：Complete Partition；
- `line_buffer` 的列 Bank 维：Complete Partition；
- Bank 内地址维：保持 RAM，不 Complete Partition。

每个物理 Bank 的目标端口模式：

```text
每周期最多 1 次读取 + 1 次写入
```

应根据综合结果绑定为适当的双端口 RAM 或 LUTRAM。不能只根据 C++ 数组声明假定 HLS 一定生成目标端口。

## 9. 每周期 P 路读取

### 9.1 Lane 独立地址

对于当前输出 Lane 和 `(kh,kw)`：

```cpp
input_y = lane_y + kh - 1;
input_x = lane_x + kw - 1;
```

若坐标有效，读取地址为：

```cpp
row_bank = input_y & 1;
col_bank = input_x % P;
address  = input_x / P;
```

### 9.2 Bank 无冲突条件

在 `W>=P` 条件下，一个 P 路 Tile 最多跨越两个相邻输出行。

对于固定 `(kh,kw)`：

- 同一输出行内的 Lane 对应连续且互不相同的输入列；
- 每段 Lane 数量不超过 P，因此 `input_x%P` 不重复；
- 跨行的两段对应相邻输入行，因此 `input_y&1` 不同；
- Padding Lane 不访问 RAM。

因此，不会有两个有效 Lane 在同一周期访问相同的：

```text
(row_bank, col_bank)
```

实现后仍必须在 Testbench 中枚举所有支持的 H、W、Tile、Lane、kh、kw，自动检查 Bank 唯一性，并在 C Synthesis 中确认每个 Bank 的实际端口数。

### 9.3 HLS 动态索引注意事项

即使数学上无冲突，完全展开的 Lane 循环中使用动态 `row_bank` 和 `col_bank` 仍可能让 HLS 生成较大的选择网络，甚至推导为多端口 RAM。

实现时优先考虑：

- 按物理 Bank 组织读请求；
- 显式 `switch` 或静态 Bank 选择；
- 对 P=4 固定展开；
- 避免在热循环中使用运行时 `%W` 和 `/W`。

## 10. Padding

Padding 不写入 Line Buffer，也不从输入 Stream 消费 Token。

每个 Lane 独立判断：

```cpp
bool padding =
    input_y < 0 || input_y >= H ||
    input_x < 0 || input_x >= W;
```

若为 Padding：

```text
输出一个所有 Channel 均为 0 的 pixel_t
不发起 Line Buffer 读取
不改变输入计数器
```

这样可以统一处理：

- 顶部 Padding；
- 底部 Padding；
- 左侧 Padding；
- 右侧 Padding；
- 同一个 Tile 中不同 Lane 位于不同边界的情况。

本方案不再构造 `W+2` 的物理 Padding 行，也不再计算 P 对齐后的 Padding 行宽。

## 11. 输入预取和待提交缓冲

### 11.1 为什么仍需要小型待提交缓冲

取消完整 Strip 不代表所有新输入都能立即写入 Line Buffer。

输入行 `new_y` 的像素 `(new_y,x)` 会覆盖两行之前的 `(new_y-2,x)`。若旧像素仍会被后续窗口使用，新像素必须暂存，不能提前覆盖。

跨行 Tile 的典型情况是：

- 新一行头部像素已经是当前 Tile 某些 Lane 的 Bottom 数据；
- 两行以前同列像素仍是下一个 Tile 某些 Lane 的 Top 数据；
- 新像素必须可被当前输出旁路读取，同时等待旧槽位释放后再提交到 Line Buffer。

因此保留一个小型、完全分割的 Pending Buffer：

```cpp
struct pending_entry_t {
    pixel_t pixel;
    int row;
    int col;
    bool valid;
};

pending_entry_t pending[PENDING_DEPTH];
```

第一版：

```text
PENDING_DEPTH = P+2 = 6
```

Pending Buffer 不是完整窗口、不是 Patch Tile，也不是用来吸收长期速率差的 FIFO。它只保存尚不能覆盖旧 Line Buffer 槽位的新输入，并为当前窗口提供旁路数据。

### 11.2 跨行覆盖示例

以 `W=6、P=4` 为例，第二个全局空间 Tile 为：

```text
lane 0 → 输出 (y,   4)
lane 1 → 输出 (y,   5)
lane 2 → 输出 (y+1, 0)
lane 3 → 输出 (y+1, 1)
```

在 `kh=2` 时：

```text
lane 0、1 需要输入行 y+1 的尾部
lane 2、3 需要输入行 y+2 的头部
```

输入流会依次读到：

```text
row y+1 的剩余真实像素
row y+2 的头部真实像素
```

此时 `row y+2, col 1` 与 `row y, col 1` 使用相同物理槽位。若 `row y, col 1` 仍会被下一个 Tile 的 Top 读取，则新像素不能立即覆盖它。处理方式是：

```text
row y+2, col 1 先进入 Pending
当前 Tile 的 Bottom 通过 Pending 旁路读取它
row y, col 1 完成最后一次 Top 读取后
再把 Pending 数据提交到 Line Buffer
```

因此，逐列更新思路仍然成立；Pending 只覆盖“新值已需要、旧值尚未释放”的短暂重叠期。

### 11.3 输入预装

开始输出前先把原始输入第 0 行完整读入 Line Buffer：

```text
读取 W 个 pixel_t
写入 row_bank = 0
```

这样第一个输出行的 Middle 数据已经就绪。顶部数据由 Padding 0 产生。

若 `H=1`，后续 Bottom 全部由 Padding 0 产生，不再读取第二行。

### 11.4 输入前沿

维护下一个尚未读取的真实输入坐标：

```text
load_y
load_x
load_index = load_y×W + load_x
```

每个 Tile 开始时，计算该 Tile 所有有效 Lane 的 3×3 窗口所需的最大真实输入位置。由于输入流是行优先顺序，只要读取前沿越过该位置，该 Tile 所有非 Padding 输入就已经在 Line Buffer 或 Pending Buffer 中。

可以定义：

```text
required_last_input_index =
    当前 Tile 所有有效、非 Padding (input_y,input_x)
    对应 input_y×W+input_x 的最大值
```

通常最大值来自 `kh=2` 的 Bottom 行。

### 11.5 9周期内的预取

Im2Row 在每个 Tile 的前 6 个 Token 周期，即 `kh=0` 和 `kh=1` 阶段，每周期最多读取一个新输入 Pixel：

```text
k0～k5：需要时读取 1 个新 pixel_t
k6～k8：停止外部输入读取，输出 Bottom 数据
```

P=4 时：

- 第一批输出在预装第0行后，最多需要额外读取下一行前 `P+1=5` 个真实像素；
- 稳态每个 Tile 平均只推进 P=4 个真实输入像素；
- 跨行时 Padding 不消费输入 Token；
- 6 个预取周期足以覆盖第一版 P=4 的最坏需求。

实现中必须设置断言或错误状态：进入 `k6` 前，若 `load_index` 尚未越过 `required_last_input_index`，说明调度假设被破坏，不能继续输出未就绪数据。

### 11.6 Pending 查询和提交

读取一个窗口元素时按以下优先级取数：

```text
1. 坐标越界：返回 Padding 0
2. Pending Buffer 中存在相同 (row,col)：返回 Pending 数据
3. 否则读取 Line Buffer
```

输入到达时：

```text
1. 计算目标 (row_bank,col_bank,address)
2. 若目标旧槽位已释放，可直接写入或经旁路后写入
3. 若目标旧槽位未释放，写入 Pending Buffer
4. Pending 满时停止 input_stream.read()，通过反压等待槽位释放
```

每周期输出读取结束后，检查可提交的 Pending 项。第一版每周期最多提交一个 Pending 项，保证每个物理 Bank 不超过一写；若综合和无冲突证明允许，可后续扩展为多 Bank 并行提交。

Pending 项提交后：

- 写入对应 Line Buffer 槽位；
- 更新该槽位保存的行代；
- 清除 Pending Valid；
- 新一代槽位重新标记为“尚未完成最后一次读取”。

## 12. 旧槽位最后一次使用与安全覆盖

### 12.1 最后使用位置

一个真实输入像素 `(input_y,input_x)` 参与的最晚输出位置为：

```text
last_output_y = min(H-1, input_y+1)
last_output_x = min(W-1, input_x+1)
```

对应 Kernel 位置：

```text
last_kh = input_y - last_output_y + 1
last_kw = input_x - last_output_x + 1
```

当该输出位置对应 Lane 的 `(last_kh,last_kw)` Token 完成读取后，该输入像素不会再被任何后续窗口使用，可以释放。

### 12.2 Released 状态

建议为每个物理列槽位维护轻量状态：

```cpp
bool released[2][P][MAX_BLOCKS_PER_ROW];
```

该状态数组只有少量 bit，可以完全分割或实现为寄存器。

规则：

- 预装或提交一个新像素后，`released=false`；
- 当前输出读取命中该像素的最后使用位置后，`released=true`；
- 新输入行小于2时，没有被覆盖的旧有效行，可以直接写入；
- 覆盖操作必须发生在旧值读取之后，采用明确的 read-before-write 语义；
- 若输出 Stream 发生反压，读取、释放、提交和输入计数器必须全部保持不变。

### 12.3 行代校验

为了防止读取到相同行奇偶但错误代的数据，建议在 C Simulation 版本中为每个槽位维护：

```cpp
int stored_row[2][P][MAX_BLOCKS_PER_ROW];
```

读取 Line Buffer 时断言：

```text
stored_row == requested input_y
```

综合版本可以根据验证结果决定是否删除该元数据，或缩减为必要的代标志。Golden Model 和 Debug 版本必须保留等价检查，以验证覆盖时机。

## 13. 9周期输出调度

每个空间 Tile 执行：

```cpp
for (int kh = 0; kh < 3; ++kh) {
    for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
        // 1. 计算 P 个 Lane 的输入坐标
        // 2. Padding / Pending / Line Buffer 三选一
        // 3. 组装 activation_pack_t
        // 4. 写 output_stream
        // 5. 标记本周期完成最后使用的旧槽位
        // 6. 提交至多一个已释放的 Pending 项
        // 7. kh<2 时按需预取至多一个真实输入
    }
}
```

建议的单周期语义顺序：

```text
A. 读取本周期输出所需的旧数据或 Pending 数据
B. 成功写出 activation_pack_t
C. 标记刚完成最后使用的槽位
D. 提交已经安全的 Pending 数据
E. 在 kh<2 且仍需预取时读取一个新输入
```

HLS 可能并行或重排无副作用操作，因此代码必须用清晰的数据依赖和局部变量保证逻辑上的 read-before-write，不能只依赖源代码书写顺序。

输出 Token 总数：

```text
output_tokens = ceil(H×W/P) × 9
```

不考虑初始化和反压，输出阶段目标为：

```text
II = 1 Token/周期
9 周期/空间 Tile
P/9 个输出空间位置/周期
```

## 14. 参考状态机

实现可以使用以下状态：

```text
CHECK_ARGS
    校验 H、W、Cin

PRELOAD_FIRST_ROW
    读取第0行 W 个真实像素

PREPARE_TILE
    计算 Lane 坐标、spatial_valid 和 required_last_input_index

EMIT_K
    运行 kh、kw 计数器，输出9个 Token
    前6周期执行输入预取
    每周期处理释放和 Pending 提交

ADVANCE_TILE
    更新 base_y、base_x、tile_base_index

DRAIN_PENDING
    帧尾如仍有必须提交的 Pending 项则完成提交
    若后续模块不再读取 Line Buffer，也可只做一致性检查后结束

DONE
```

如果采用循环而非显式状态机，也必须保持完全相同的状态转换语义。

## 15. 接近代码的总体伪代码

```cpp
validate(H, W, Cin);
clear_control_state();

// 预装第0行，输入流严格消费 W 个 Pixel。
for (int x = 0; x < W; ++x) {
    pixel_t p = input_stream.read();
    write_line_buffer(0, x, p);
    mark_slot_not_released(0, x);
}

load_y = 1;
load_x = 0;
base_y = 0;
base_x = 0;
tile_base_index = 0;

for (int tile = 0; tile < ceil_div(H * W, P); ++tile) {
    prepare_lane_coordinates(base_y, base_x, H, W);
    required_last = find_required_last_input_index();

    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
#pragma HLS PIPELINE II=1
            activation_pack_t pack;

            for (int lane = 0; lane < P; ++lane) {
#pragma HLS UNROLL
                if (!spatial_valid[lane]) {
                    pack.pixel[lane] = zero_pixel;
                } else {
                    int iy = lane_y[lane] + kh - 1;
                    int ix = lane_x[lane] + kw - 1;
                    pack.pixel[lane] = read_with_padding_and_pending(iy, ix);
                }
            }

            pack.spatial_valid = current_mask;
            output_stream.write(pack);

            release_last_used_slots(kh, kw, lane_coordinates);
            commit_one_ready_pending_entry();

            if (kh < 2 && next_input_index <= required_last) {
                prefetch_one_real_pixel_or_stall();
            }
        }
    }

    assert(next_input_index > required_last);
    advance_base_by_P(base_y, base_x, W);
    tile_base_index += P;
}

assert(total_input_reads == H * W);
assert(total_output_writes == ceil_div(H * W, P) * 9);
```

真实实现需要把 Pending 查询、释放和提交组织成 HLS 能够推导的静态循环，不能直接使用动态容器。

## 16. 反压和数据加载

### 16.1 Stream 反压

当前模块不会每周期无条件读取输入。只有在以下条件同时成立时才调用 `input_stream.read()`：

- 当前仍有真实输入需要预取；
- 当前处于允许预取的周期；
- Pending Buffer 有空间或目标槽位可直接提交；
- 模块没有因输出 Stream 阻塞而暂停。

若上游保持 `valid`，但 Im2Row 当前不读取，则数据留在上游或上游 FIFO 中，不能被跳过。

### 16.2 外部数据加载

当前 `hls::stream` 方案适用于片上模块连接和支持反压的 AXI4-Stream。

未来若输入来自 DDR，推荐由独立 Loader 使用 Burst 把一行、一个 Tile 或整张 Feature Map 加载到片上缓冲，再按 Im2Row 的实际消费节奏送入 Stream。DDR Loader 和 Im2Row 之间是否增加 Ping-Pong Buffer 在顶层设计阶段确定。

若数据源是不能反压的固定速率传感器，则必须使用帧缓存或保证长期计算吞吐不低于输入吞吐；有限深度 FIFO 不能解决长期平均速率不足。

## 17. HLS 实现要求

### 17.1 Pipeline 和 Unroll

目标：

- `kh/kw` 输出循环：`PIPELINE II=1`；
- P 个 Lane 的坐标、Padding、Pending 查询和读数据：完全 `UNROLL`；
- `pixel_t.channel`：根据完整 Channel 打包要求完全分区；
- Pending 查询：固定深度循环完全展开或按综合结果部分展开。

### 17.2 存储端口

综合报告必须确认：

- 生成 2×P 个独立 Line Buffer Bank；
- 每个 Bank 每周期不超过一次读和一次写；
- 未生成意外的整行复制电路；
- 未因为动态索引把多个 Bank 合并为一个多端口大 RAM；
- Pending 旁路没有成为目标频率的关键路径。

### 17.3 依赖编译指示

禁止在没有验证真实读写关系前盲目添加：

```cpp
#pragma HLS DEPENDENCE ... inter false
```

只有 C Simulation 的代检查、Bank 冲突检查和读后写证明全部通过后，才可以用依赖编译指示消除 HLS 的保守假依赖。

### 17.4 Dataflow

本方案把输入预取、Line Buffer 更新和9周期输出紧密耦合，不再需要旧版“窗口生产阶段 + Patch Tile Scheduler”两级 DATAFLOW。

第一版优先实现为一个明确的流水调度过程。是否拆分 Pending Loader、Bank Reader 和 Output Writer，需要根据综合后的 II、选择网络和反压路径决定。

## 18. 初始化和帧边界

每次函数调用处理一帧，Batch 固定为1。

每帧开始时必须重置：

- 输入读取计数和 `load_y/load_x`；
- Tile 起点 `base_y/base_x`；
- `kh/kw` 计数器；
- Pending Buffer Valid；
- Released 状态；
- Debug 行代信息；
- 输入和输出 Token 统计；
- `spatial_valid`。

Line Buffer 数据本体不要求整块清零，但任何读取都必须满足以下之一：

- 坐标是 Padding，直接返回零；
- Pending 中存在精确坐标；
- Line Buffer 的行代与请求行一致。

禁止依赖未初始化 RAM 或上一帧残留数据。

## 19. 验证计划

### 19.1 软件 Golden Model

Golden Model 直接对 NHWC 数组执行：

```text
for global spatial Tile
    for kh
        for kw
            for lane
                计算独立 output_y/output_x
                计算 input_y/input_x
                越界输出零，否则读取输入 pixel_t
```

逐 Token、逐 Lane、逐 Channel 比较 HLS 输出。

### 19.2 功能测试尺寸

至少覆盖：

- `H=1、W=4`；
- `H=2、W=5`；
- `H=3、W=5`；
- `H=7、W=7`；
- `H=14、W=14`；
- `H=28、W=28`；
- `Cin=1`；
- `Cin=MAX_CHANNELS`；
- 全零、全一、单点脉冲和固定随机输入；
- 四个角、四条边和中心；
- Tile 不跨行、跨行和整帧尾 Tile；
- 连续处理两帧；
- 输出 Stream 随机暂停；
- 输入 Stream 随机产生气泡。

### 19.3 必须增加的结构检查

Testbench 或独立检查函数必须枚举全部支持坐标，验证：

1. 每个 Token 中有效 Lane 的 `(row_bank,col_bank)` 唯一；
2. 每次覆盖前，旧值已经完成最后一次读取；
3. Pending Buffer 从不溢出；
4. Pending 中不存在重复 `(row,col)`；
5. Line Buffer 读取行代与请求行一致；
6. 进入 `kh=2` 前，当前 Tile 所需真实输入全部就绪；
7. 每帧恰好读取 `H×W` 个输入 Token；
8. 每帧恰好输出 `ceil(H×W/P)×9` 个 Token；
9. 只有整帧最后一个 Tile 可能出现无效 Lane。

### 19.4 HLS 验证顺序

```text
1. 完成周期级 C++ 实现
2. C Simulation
3. C Synthesis
4. 检查实际 II、Latency、LUT、FF、BRAM/LUTRAM 和端口
5. 检查是否存在运行时除法器和大选择网络
6. RTL Co-simulation，加入输入气泡和输出反压
7. 与 Systolic 接口联调
```

## 20. 性能预期

不考虑启动、帧尾和反压，且下游每周期接受一个 Token 时：

```text
Im2Row 输出吞吐 = 1 个 activation_pack_t / 周期
空间吞吐        = P/9 个输出位置 / 周期
P=4 时          = 4/9 个输出位置 / 周期
```

每帧输出周期下界：

```text
ceil(H×W/P)×9
```

启动阶段需要预装第0行：

```text
W 个输入读取周期
```

若输入数据已在上游 FIFO 中，预装可以连续进行；后续输入读取与前6个 `(kh,kw)` 输出周期重叠。最终 Latency 必须以 C Synthesis 和 RTL Co-simulation 为准。

“9周期/Tile”表示9次连续成功的输出事务。若 Systolic 因 Channel Fold、Cout Tile 重放或其他原因拉低就绪信号，Im2Row 会保持当前状态，实际墙钟周期按反压时间增加。

## 21. 已知风险

### 21.1 Pending 深度

第一版使用 `P+2` 是基于3×3、P=4、`W>=P` 和每 Tile 前6周期可预取的设计。必须通过全部 H/W 的穷举调度检查验证最坏占用。若深度不足，应先扩大 Pending Buffer，再根据综合结果优化，不能提前覆盖旧数据。

### 21.2 HLS Bank 推导

数学上的无冲突不保证 HLS 自动生成理想硬件。动态 Row Bank、Col Bank 和 Lane 交叉选择可能增加多路器或导致 RAM 端口推导失败。

### 21.3 Channel-packed 宽度

P=4、MAX_CHANNELS=8、int8 时，一个输出 Token 的激活数据宽度约为：

```text
4×8×8 = 256 bit
```

当前规模可接受，但未来增加 Channel 上限时必须重新评估布线、Stream FIFO 和 Systolic 输入寄存器。

### 21.4 运行时 W

W 是运行时参数。坐标推进必须使用比较和减法，避免在每个 Lane 上综合通用除法器。Bank 的 `%P` 和 `/P` 仅在 P 为编译期常量时可低成本实现。

### 21.5 输出反压

一个输出 Token 未成功写出时，不得提前执行槽位释放、Pending 提交或输入读取，否则恢复后会出现数据代错位。

## 22. 下一阶段实施顺序

1. 更新 `types.h`：输入改为标量 `pixel_t`，输出改为 `pixel_t[P]`。
2. 更新 `im2row.h`：替换函数接口并记录 `W>=P` 约束。
3. 先实现纯软件 Golden Model。
4. 实现无运行时除法的全局 Tile 坐标生成器。
5. 实现 2×P Bank、行奇偶循环 Line Buffer。
6. 实现 Padding、Pending 查询和精确坐标旁路。
7. 实现最后使用检测、Released 状态和安全提交。
8. 实现第0行预装和前6周期输入预取。
9. 实现9周期 `kh→kw` 输出调度。
10. 添加 Bank 冲突、行代、Pending 深度和 Token 数量断言。
11. 完成 C Simulation 后再删除或裁剪 Debug 元数据。
12. 完成 C Synthesis，根据报告调整存储绑定和静态 Bank 选择。
13. 完成 RTL Co-simulation和 Systolic 联调。
14. 将实际接口、II、Latency 和资源结果回写本文档与项目 README。
