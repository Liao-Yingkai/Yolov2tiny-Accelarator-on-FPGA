# Top 初版设计

- 对应源码：`cnn_accelerator/src/top.cpp`、`cnn_accelerator/src/cnn_accel.h`。
- 本阶段只完成源码和静态检查；不运行Top Testbench、C Simulation、RTL Co-simulation或C Synthesis。

## 顶层接口与阶段顺序

`cnn_accelerator()`每次调用处理一张固定`28×28`图像，并保持普通
`hls::stream<>`接口和默认`ap_ctrl_hs`控制。每次调用必须提供：

| 流 | 事务数 |
|---|---:|
| `image_stream` | 784 `pixel_t` |
| `conv1_weight_stream` / `conv2_weight_stream` | 18 / 36 `weight_pack_t` |
| `conv1_param_stream` / `conv2_param_stream` | 2 / 4 `postprocess_param_pack_t` |
| `fc_weight_stream` / `fc_param_stream` | 196 / 2个FC Pack |
| `logits_stream` | 1 `logits_pack_t` |

顶层只有一个静态`run_layer_phase()`调用点，位于固定两次的phase循环中：

```text
phase 0: image → Conv1 → Pool1 → Feature Buffer
          run_layer_phase() 完整返回（Layer Barrier）
phase 1: Feature Buffer → Conv2 → Pool2 → FC → FCPost → logits
```

该顶层phase循环显式`UNROLL off`，没有`PIPELINE`或`DATAFLOW`。每次
`run_layer_phase()`返回意味着该次唯一DATAFLOW区域已drain；因此不需要额外的done、busy、
watermark或跨层ping-pong控制。

## 单一DATAFLOW区域

`run_layer_phase()`为`INLINE off`函数，内部只有一个DATAFLOW区域。以下process始终无条件
调用，mode只决定固定事务数量，不改变进程拓扑：

```text
FeatureBuffer Process ⇄ Input Router → Im2Row → Systolic → ConvPostProcess
                                                   → Pool → Output Router
Output Router → FeatureBuffer Process (Layer1)
Output Router → FC Guard → FCPost Guard (Layer2)
Weight Router ───────────────────────────────→ Systolic
ConvParam Router ────────────────────────────→ ConvPostProcess
```

Im2Row、Systolic、ConvPostProcess和Pool各只在此函数出现一个调用点。FC和FCPost也各只有
一个调用点，并由Guard在Layer1保持零事务。全部内部stream显式设置
`depth=TOP_STREAM_DEPTH=2`；没有整层深FIFO。四个计算模块本体不使用`INLINE off`：唯一的
DATAFLOW调用点已表达Top进程边界，同时不限制它们作为独立模块单独综合。

## Feature Buffer与路由

唯一`feature_buffer_process`在其函数内部持有：

```cpp
static ap_uint<64> pool1_feature_buffer[196];
```

该数组约束为`ram_1p/bram`，不作为两个DATAFLOW process之间的函数参数传递。

- Layer1：Output Router从Pool1读取196个pack，写入`buffer_write_stream`；Feature Buffer
  Process消费这196个pack并顺序写BRAM。
- Layer2：Feature Buffer Process顺序读196个BRAM word，解包为`pixel_t`并写入
  `buffer_read_stream`；Input Router只消费该stream。

位序固定为`channel[c] ↔ bits[8*c+7:8*c]`。打包和解包显式复制INT8补码位型，不依赖C++对象
内存布局。Layer1的Buffer读取路径、Layer2的Buffer写入路径以及Layer1的FC路径均为零事务；
各Router和Feature Buffer Process在非综合构建中对此执行事务守恒断言。

## 后续验证重点

- Vitis是否接受该单一DATAFLOW图中的模式互斥stream路径，且没有200-971类共享数组错误；
- RTL hierarchy是否只生成一套Im2Row、Systolic、ConvPostProcess和Pool；
- 全部depth=2 FIFO的死锁、stall、资源和时序；
- 连续两次Top调用的Feature Buffer覆盖与结果独立性。

DDR、DMA、AXI master、权重加载和FIFO性能调优均不属于本初版。
