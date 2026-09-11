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
phase 0: image → Conv1 → Pool1 → Pool1 Feature FIFO
          run_layer_phase() 完整返回（Layer Barrier）
phase 1: Pool1 Feature FIFO → Conv2 → Pool2 → FC → FCPost → logits
```

该顶层phase循环不添加`UNROLL`、`PIPELINE`或`DATAFLOW`。每次
`run_layer_phase()`返回意味着该次唯一DATAFLOW区域已drain；因此不需要额外的done、busy、
watermark或跨层ping-pong控制。

## 单一DATAFLOW区域

`run_layer_phase()`为`INLINE off`函数，内部只有一个DATAFLOW区域。以下process始终无条件
调用，mode只决定固定事务数量，不改变进程拓扑：

```text
Input Router → Im2Row → Systolic → ConvPostProcess → Pool → Output Router
Pool1 Feature FIFO: Output Router (Layer1) → Input Router (Layer2)
Output Router → FC Guard → FCPost Guard (Layer2)
Weight Router ───────────────────────────────→ Systolic
ConvParam Router ────────────────────────────→ ConvPostProcess
```

Im2Row、Systolic、ConvPostProcess和Pool各只在此函数出现一个调用点。FC和FCPost也各只有
一个调用点，并由Guard在Layer1保持零事务。全部内部stream显式设置
`depth=TOP_STREAM_DEPTH=2`；唯一例外是跨阶段Pool1 FIFO。四个计算模块本体不使用
`INLINE off`：唯一的DATAFLOW调用点已表达Top进程边界，同时不限制它们作为独立模块单独综合。

## Pool1阶段间FIFO与路由

`run_layer_phase()`内部持有一个跨两次phase调用持续存在的FIFO：

```cpp
static hls::stream<channel_pack_t> pool1_feature_fifo;
```

它显式设置`depth=TOP_POOL1_FEATURE_FIFO_DEPTH=196`并约束为`fifo/bram`；深度同时
等于Pool1输出事务数和Conv2输入事务数。`channel_pack_t`是`pixel_t`的别名，因此无需
手动位打包、解包或依赖C++对象布局。

- Layer1：Input Router只读外部图像；Output Router从Pool1读取196个pack并依序写入FIFO。
- Layer2：Input Router先从FIFO读出196个pack，直接作为Conv2 Im2Row输入；Output Router
  不写FIFO，只将Pool2输出送入FC。

在源码DATAFLOW调用序中，Input Router位于Output Router之前。忽略DATAFLOW pragma的普通
C++语义会先完整执行Layer1并填满FIFO，再在Layer2开头读空FIFO，因此不会出现旧架构中
旧Feature Buffer Process先读空内部stream的问题。Layer1的FIFO读取、Layer2的FIFO写入和
Layer1的FC路径均为零事务；Router在非综合构建中对此执行事务守恒断言。

## 后续验证重点

- Vitis是否接受同一静态FIFO被两个模式互斥Router端点跨phase使用；它在DATAFLOW图中可能被
  识别为反馈通道，需确认不会引入死锁、跨调用FIFO生命周期限制或不支持的实现；
- RTL hierarchy是否只生成一套Im2Row、Systolic、ConvPostProcess和Pool；
- 层内depth=2 FIFO及196-depth阶段间FIFO的死锁、stall、资源和时序；
- 连续两次Top调用的Pool1 FIFO清空、覆盖与结果独立性。

DDR、DMA、AXI master、权重加载和FIFO性能调优均不属于本初版。
