# DDR Loader / Writer Wrapper

源码为 `cnn_accelerator/src/axi/cnn_accel_ddr.{h,cpp}`，HLS top 选择
`cnn_accelerator_ddr`。固定一次调用处理一张图像；原 `cnn_accelerator` 的
stream 接口、计算循环、两阶段 barrier 和 Pool1 FIFO 保持不变。
本阶段提供源码、独立 testbench 与接口约定，不包含 PS 软件、export 脚本或 BD；
本次参数与16字节结果布局的独立静态检查及三图 C Simulation 已通过；
未完成 wrapper 综合、RTL
Co-simulation 或上板验证，尚未实施 PS 实际数据打包程序。

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
| weights | unsigned 64 bit | gmem_weights | 1196 / 9568 |
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
`pixel.channel[0]` 是图像值，其余物理通道均为0。

### 权重：按硬件消费顺序离线预排

| 层 | word offset | Pack 数 | words/Pack | 总 words |
|---|---:|---:|---:|---:|
| Conv1 | 0 | 18 | 4 | 72 |
| Conv2 | 72 | 36 | 4 | 144 |
| FC | 216 | 196 | 5 | 980 |

Conv 按 `cout_tile -> token(kh*3+kw) -> cout_lane -> cin_byte`。
令 T 为 Cout tile，t 为 kernel token，L 为 Cout lane，c 为 Cin byte：

- DDR word index = `layer_offset + (T*9+t)*4+L`；
- 输出通道 o = `T*4+L`；
- 原 Conv1 `[K,Cout]` 数组值为 `conv1_weight[t*8+o]`，只有 byte0 有效，byte1..7 为0；
- 原 Conv2 `[K,Cout]` 的 K 为 `t*8+c`，值为 `conv2_weight[(t*8+c)*16+o]`。

FC 按 `out_tile -> token -> cout_lane -> cin_byte`。令 T 为 out tile、t 为
输入 token、L 为输出 lane、c 为输入 byte：DDR word index 为
`FC_WEIGHT_OFFSET+(T*FC_INPUT_PACKS+t)*FC_COUT_PAR+L`；原 `[784,10]`
数组值为 `fc_weight[(t*8+c)*10+(T*5+L)]`。784 输入已经是 Pool2 的 NHWC
flatten 顺序，不应再次转回 NCHW。

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

DDR 权重 word 循环不展开，每周期目标最多一次64-bit读，byte恢复展开；
因此 Conv Pack 至少4拍、FC Pack至少5拍，FC 将因权重供应速度而 stall。
同一 HP0 的竞争会增加停顿，目标 II=1 不代表端到端吞吐保证。不能据普通
C Simulation 的无限深 stream 行为断言有限深度 DATAFLOW 不死锁。

## 独立 testbench

`testbench/test_ddr_wrapper.cpp` 用 namespace include 三个绝对路径 trace
header 和现有模型 header。软件侧独立使用权重索引公式构造 DDR 数据；参数
按输出通道直接将三个完整 header 数组拷贝到各层 bias/multiplier/shift 区，
不照抄 DUT 的 pack/field/lane 地址循环。模型
只准备一次，三次 wrapper 调用复用地址，仅更新图像。检查30个 logits、独立
scalar first-argmax、结果保留字节清零、四个 buffer 守卫，以及所有只读输入
未改变。错误显式累计并返回1，不依赖 assert。输出
`simulation_data/pytorch/ddr_wrapper/output.csv`，含
`sample,output,expected,actual,match`，类别行 output 为 pred_class。
同目录 `input.csv` 保存 `sample,pixel,value`，三图共2352行输入数据。
这项检查验证功能与连续调用，不代替综合/RTL反压、AXI或上板验证。
