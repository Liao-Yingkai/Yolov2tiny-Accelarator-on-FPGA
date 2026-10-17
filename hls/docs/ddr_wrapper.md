# DDR Loader / Writer Wrapper

源码为 `cnn_accelerator/src/top/cnn_accel_ddr.{h,cpp}`，HLS top 选择
`cnn_accelerator_ddr`。固定一次调用处理一张图像；原 `cnn_accelerator` 的
stream 接口、计算循环、两阶段 barrier 和 Pool1 FIFO 保持不变。
本轮同步 Python exporter、HLS/testbench 与 Vitis 裸机应用，Vivado/IP 导出与 bitstream 不修改。
权重 header 布局版本为2，Python metadata 为5；旧权重布局不能与新版 loader 混用。
以下历史结果不证明本轮新布局已完成 RTL 验证：
迁移前参数与16字节结果布局的独立静态检查及三图 C Simulation 已通过；
迁移后 GCC 11 与系统 Clang 14 的全25个源码及 testbench 静态检查通过，
`hls_ddr.tcl` 再次通过三图30路 logits、3个类别、守卫与只读区检查，
0 errors、无 WARNING；input/output CSV 含表头分别2353/34行，见
[迁移后 DDR C Simulation 报告](../../build/hls/cnn_ddr_csim/solution1/csim/report/cnn_accelerator_ddr_csim.log)。
本轮统一 Loader 已完成 wrapper 综合与三图 RTL Co-simulation，结果见文末；尚未上板或更新 Vivado 生成物。PS 应用位于
`vivado/cnn_accel_ddr/vitis/cnn_accel_ddr/src/main.c`，仅直接复制新 header 到 DDR。

## 控制与 AXI 接口

```cpp
void cnn_accelerator_ddr(const ap_uint<64>* image,
                         const ap_uint<64>* weights,
                         const ap_int<32>* params,
                         ap_uint<64>* result);
```

四个 DDR 基地址经 `s_axilite bundle=control` 配置，`return` 同属 control，
保留默认 `ap_ctrl_hs`。PS 写入地址后单次启动，等待完成再读取结果；不使用 DMA、
自动重启或批处理协议。DDR buffer 基地址与 IP 的 AXI-Lite 控制寄存器基地址不是
同一个概念；控制寄存器实际偏移与地址宽度应以工具生成的驱动/寄存器表为准。

| 指针 | 元素 | m_axi bundle | depth / 有效字节 |
|---|---|---|---|
| image | unsigned 64 bit | gmem_io | 98 / 784 |
| weights | unsigned 64 bit | gmem_weights | 1133 / 9064 |
| params | signed 32 bit | gmem_params | 102 / 408 |
| result | unsigned 64 bit | gmem_io | 2 / 16 |

三个独立 m_axi 接口可以通过互连连到同一 PS HP0；其 DDR 带宽受 HP0 和互连
共同仲裁，不是三个独占带宽通道。image/weights/result 显式设置
`max_widen_bitwidth=64`，params 独立接口设置 `max_widen_bitwidth=32`，
限制自动 widening 上限；params 的逻辑元素与布局均保持 signed 32 bit。
实际接口宽度、burst、调度和 pragma 接受情况仍须综合确认。

所有 DDR 区域必须互不重叠，并隔离到不同 cache line；image/weights/result 最小
8-byte 对齐，params 最小4-byte 对齐，推荐全部至少64-byte对齐，并按实际 CPU
cache line 大小分配和补齐。非一致性 HP 通路：启动前 clean/flush 输入、权重和
参数缓存；结果 buffer 启动前消除脏缓存影响，完成后 invalidate 结果 cache line
再读取，配合平台要求的内存屏障。不得对与其他活动对象共用的 cache line 做
缓存维护。数据区 padding 不属于有效 DDR 事务。

## 共享布局与位序

`src/axi/ddr_layout.h` 只包含 `../config.h` 与内建整数编译期常量，适合 PS
C++ 与 HLS 共用；由于 config 使用 bool/static_assert，它不是可直接用于 C 的
header。所有 offset 都是相应指针的元素 offset，而非 byte offset；尺寸由网络
配置派生，不依赖 C++ struct 的对象内存布局。

每个64-bit word 的低 byte 对应较早元素：byte c 位于 `[8*c+7:8*c]`。
有符号 INT8 以原始二补码传输，loader 显式恢复 signed INT8。

image 按 `y -> x` raster 紧密保存784个 signed INT8：pixel index p 对应
`image[p/8]` 的 byte `p%8`。每个 word 只读取一次，随后连续发出8个 pixel；
`pixel.channel[0]` 是图像值；其余物理通道透传当前已读 word 的字节作为确定的无效字段，不补零、不另读 DDR。Im2Row 按原契约只保留 Conv1 的 channel 0，内部清零其余通道；Systolic 同时自行保证只有 channel 0 参与 Conv1 点积。

### 权重：header 直接采用紧凑硬件顺序

| 层 | word offset | stream Pack 数 | DDR words/Pack | 总 words |
|---|---:|---:|---:|---:|
| Conv1 | 0 | 18 | 1/2 | 9 |
| Conv2 | 9 | 36 | 4 | 144 |
| FC | 153 | 196 | 5 | 980 |

Conv header 按 `[CoutTile][Kh][Kw][CoutLane][logical_Cin]` 紧密排列，不为 Conv1 加入无效权重。令 T 为 Cout tile，t 为 kernel token，L 为 Cout lane，c 为有效 Cin：

- header 字节索引 = `((T*9+t)*4+L)*logical_Cin+c`；
- DDR word 索引 = `layer_offset + header_byte_index/8`，字节位置 = `header_byte_index%8`；
- Conv1 每个 Pack 只含4个真实权重字节，一个 DDR word 包含两个 Pack。两个 Cout tile 各有9个 token，word 缓存必须跨 tile 连续使用，不能在 tile 边界丢弃剩余半个 word；
- Conv1 解包后各 Cout lane 仅 channel 0 有效；channel 1..7 透传当前 word 的确定字节，Systolic 按模式忽略，loader 不生成零填充；
- Conv2 每个 Pack 含32个真实字节，各 lane 的完整8路 Cin 正好占一个 word。

FC header 按 `[2][98][5][8]` 排列，字节索引为 `((T*98+t)*5+L)*8+c`；DDR word 索引为 `FC_WEIGHT_OFFSET+(T*98+t)*5+L`。784输入的 NHWC flatten 语义不变，不转回 NCHW。

三个 header 的原始字节依次拼接为 DDR 权重区，PS 仅 `memcpy`，不进行重排或逻辑通道补零。Python 数学矩阵/NPY 仍为 `[K_HWC,Cout]`（FC 为 `[784,10]`），golden model 根据新版 metadata 将硬件 header 还原成数学矩阵计算；header 与 NPY 不再具有相同线性排列。数据头和应用均检查 weight header 布局版本2，拒绝旧 header。

### 参数：每层完整字段连续

每个参数元素是 signed INT32，每层顺序固定为
`[完整 bias Cout][完整 multiplier Cout][完整 shift Cout]`，各数组内部按
输出通道递增。不是 pack 内交错字段，也不是 lane 内交错三个字段。

| 层 | layer base | Cout / field stride | bias 起点 | multiplier 起点 | shift 起点 | 总 words |
|---|---:|---:|---:|---:|---:|---:|
| Conv1 | 0 | 8 | 0 | 8 | 16 | 24 |
| Conv2 | 24 | 16 | 24 | 40 | 56 | 48 |
| FC | 72 | 10 | 72 | 82 | 92 | 30 |

表中起点和 stride 均以32-bit word 计，合计102 words / 408 bytes。
`CONV_PARAM_WORDS_PER_PACK` / `FC_PARAM_WORDS_PER_PACK` 仅表示逻辑 stream
pack 的元素数，不表示 DDR 上连续块；每层 `*_PARAM_FIELD_STRIDE` 等于 Cout。

field f=0/1/2 分别表示 bias/multiplier/shift。index 为
`layer_offset + f*Cout + tile*lanes + lane`，输出通道为
`tile*lanes+lane`。loader 保留 pack -> field -> lane 循环顺序，按上述地址
跨字段读取。每拍最多读一个32-bit元素，只保存当前 pack，
不增设整层参数 buffer。

### 结果：固定两个完整64-bit words（16 bytes）

| word | 内容 |
|---|---|
| 0 | bytes0..7 为 logits0..7 |
| 1 | bits7..0 为 logit8，bits15..8 为 logit9，bits31..16 为0，bits63..32 为 uint32 类别0..9 |

PS 可见字节布局为 bytes0..9 signed logits，bytes10..11 保留为0，
bytes12..15 为 little-endian uint32 class_id。`RESULT_CLASS_BYTE_OFFSET=12`，
对应 word offset 1、bit offset 32。PS C++ 可使用以下记录并检查 ABI；这只是
软件解码约定，HLS writer 仍显式操作64-bit word，不依赖 struct 自动 packing：

```cpp
#include <cstddef>
#include <cstdint>
struct ResultRecord {
    int8_t logits[10];
    uint8_t reserved[2];
    uint32_t class_id;
};
static_assert(sizeof(ResultRecord) == 16, "result size");
static_assert(offsetof(ResultRecord, class_id) == 12, "class offset");
```

所有 words 初始化为0，logit 打包显式取 unsigned byte，PS 读取后按 signed
INT8 解释。类别采用现有 logits_argmax 的 signed 比较，平局选择最小索引。

## DATAFLOW 与调度边界

`src/top/cnn_accel_ddr.cpp` 只保留外层接口、stream、DATAFLOW 编排及匿名命名空间
内的 compute_runner / argmax_runner。五个搬运与分流 actor 位于
`src/axi/ddr_io.cpp` 的 `ddr_io` 命名空间，公共签名由 `src/axi/ddr_io.h` 声明；
两层卷积共用一个 `load_conv_weights()`，与模板 load_params 一同留在该实现文件的匿名命名空间内并保留内联约束。`weight_loader()` 以两次顺序迭代、唯一调用点处理 Conv1/Conv2，不展开或流水化该层循环；通用函数内部按层模式选择地址、事务数和输出流，不使用动态 stream 引用。
共享 DDR 布局仍为 `src/axi/ddr_layout.h`，不随系统入口移动。

外层无条件调用七个 `INLINE off` actor：image_loader、weight_loader、
param_loader、compute_runner、logits_split、argmax_runner、result_writer。
argmax_runner 调用现有 logits_argmax，避免改变其独立模块源码。
compute_runner 唯一调用原 cnn_accelerator，隔离内部顺序 phase loop，
不向外层展开；最终物理层次与单引擎复用仍须综合报告确认。

weights 和 params 是两个独立并发 loader，各自依次输出 Conv1、Conv2、FC。
不能合并为先全部权重后全部参数的单一 loader：浅 FIFO 反压可能使权重
loader 停在 Conv2/FC，而 core 又等待 Conv1 参数，构成等待环。
所有 wrapper stream 显式 `depth=TOP_STREAM_DEPTH=2`，core 的196-depth
Pool1 FIFO 不变。logits_split 读一次结果，分别发送给 writer 与 argmax；
writer 等待两路，保留原10个 logits，不只写类别。

通用卷积 Loader 使用 `Pack → Cout lane` 循环，lane loop 目标 II=1，不展开 Cout lane。每个 lane 都完整定义8个物理通道；Conv1 的 channel0 按3-bit字节游标取值，其余字段透传已读 word。Conv1 每 lane 消费1字节，游标绕回0才读取新 word；Conv2 每 lane 消费8字节，游标保持0，每次读取新 word。64-bit缓存与字节游标在完整层内连续，层调用结束必须正好耗尽。

Conv1 仍以一个缓存 word 提供两个 Pack，但每个 Pack 现在经过4次 lane 迭代，不再采用原专用函数的并行快路径。此取舍减少重复实现，只影响首个 spatial tile 的装载，后续 Systolic 权重回放方式不变。Conv2 每 Pack 仍读取4个 word，FC 每 Pack 读取5个 word。循环 II、控制开销和实际吞吐由本轮综合/协同仿真报告确认。
同一 HP0 的竞争会增加停顿，目标 II=1 不代表端到端吞吐保证。不能据普通
C Simulation 的无限深 stream 行为断言有限深度 DATAFLOW 不死锁。

## 独立 testbench

`testbench/test_cnn_accel_ddr.cpp` 用 namespace include 三个绝对路径 trace
header 和新版模型 header。软件侧仅将三个权重 header 按原始字节拼接到 DDR；参数
按输出通道直接将三个完整 header 数组拷贝到各层 bias/multiplier/shift 区，
不照抄 DUT 的 pack/field/lane 地址循环。模型
只准备一次，三次 wrapper 调用复用地址，仅更新图像。检查30个 logits、独立
scalar first-argmax、结果保留字节清零、四个 buffer 守卫，以及所有只读输入
未改变。错误显式累计并返回1，不依赖 assert。输出
`simulation_data/pytorch/ddr_wrapper/output.csv`，含
`sample,output,expected,actual,match`，类别行 output 为 pred_class。
同目录 `input.csv` 保存 `sample,pixel,value`，三图共2352行输入数据。
这项检查验证功能与连续调用，不代替综合/RTL反压、AXI或上板验证。

## 统一卷积 Loader 验证结果

验证工程：仓库根 `build/hls/unified_weight_loader_verify/`，Vitis HLS 2022.2，
`xc7z020-clg400-2`，时钟目标10 ns。未改 Python/Vitis 数据布局、FIFO深度、
Vivado 工程、导出 IP 或 bitstream。

- 直接 Loader 检查、全部15个主机 testbench、Vitis CSim 均通过；PS host mock 重链新 Loader 后三图类别仍为0/4/1、各0错误。
- 综合确认一个 `weight_loader_U0` 和一个 `conv_lane_loop`，共享卷积流水中的 Im2Row、Systolic、ConvPostProcess、Pool 各一个实例。lane loop 达到 II=1、depth=12；每 Pack 包含4次 lane 迭代，加填充/排空及控制后，综合估计每 Pack 为18周期，并非每周期输出一个 Pack。
- Verilog/XSim co-sim 三笔交易全部 PASS，30个 logits、类别、保留字节及守卫检查0错误，无死锁。Latency 为11979～12006周期，平均11988；Interval 平均11992周期，总执行35964周期。见 [co-sim 报告](../../build/hls/unified_weight_loader_verify/vitis_unified_loader/unified_weight_loader/solution1/sim/report/cnn_accelerator_ddr_cosim.rpt)。这些是仿真 AXI 内存模型下的结果，不是板测。
- HLS 的 latency 估计为8545～14929周期，估计时钟7.300 ns；与 co-sim 的实际事务测量不同，不能据此承诺板上频率或吞吐。

| HLS 资源估计 | 使用量 | 器件容量 |
|---|---:|---:|
| BRAM18K | 42 | 280 |
| DSP | 51 | 220 |
| FF | 26351 | 106400 |
| LUT | 36496 | 53200 |

其中整个 `weight_loader` 为0 BRAM、0 DSP、1566 FF、1649 LUT。FC activation buffer 实际仍为98×64-bit BRAM（2个 BRAM18K）。以上不是 Vivado 布局布线报告。

综合无 II violation，63条 warning；co-sim 4条 warning，无 ERROR：

- FIFO建议：`conv_weight_stream` 建议3、`conv_param_stream` 建议4、`write_logits` 建议3；仍保持原depth=2。本次三图未发生死锁，不能据此证明所有外部停顿条件或最优吞吐。
- Pool1反向反馈通道与两个 FC guard 有通用死锁风险提示；本次真实 RTL 三图完成，未擅自更改阶段控制或FIFO。
- `BIND 205-102` 提示存储绑定可能忽略，但上述 FC buffer 的实际映射仍为 BRAM。其余综合 warning 为 loop flatten/bound 更新、变量索引 QoR、名字合法化、未使用子端口等生成处理；不将它们等同于功能错误。
- co-sim 另有通用 FIFO/AXI sizing 风险、`LIBRARY_PATH` 环境提示、生成 Im2Row 模块空端口及部分对象不可记录波形的提示；本次未阻止仿真或结果比较。

完整日志：[综合](../../build/hls/unified_weight_loader_verify/csim_csynth.log)、[co-sim](../../build/hls/unified_weight_loader_verify/cosim.log)。后续上板仍需用当前源码重新导出 IP 并生成 bitstream。
