# 流式计算核心设计

- 对应源码：`cnn_accelerator/src/core/cnn_core.cpp`、`cnn_accelerator/src/core/cnn_core.h`。
- 迁移后 `hls_core.tcl` 已完成三图连续调用检查：30 路 logits 与 stream 守恒全部通过，
  错误数为0、无 WARNING；input/output CSV 含表头分别2353/31行，见
  [核心 C Simulation 报告](../../build/hls/cnn_core_csim/solution1/csim/report/cnn_accelerator_csim.log)。
- 本轮未运行 RTL Co-simulation 或 C Synthesis，不覆盖历史工程的验证记录。

流式核心入口仍为 `cnn_accelerator`，DDR 系统入口为 `cnn_accelerator_ddr`，见
[DDR Wrapper](ddr_wrapper.md)。

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

## Top Testbench 的三图连续调用

`cnn_accelerator/testbench/test_cnn_core.cpp` 同时 include
`trace_data_0000.h`、`trace_data_0001.h` 和 `trace_data_0002.h`。三个生成 header
仍保持相同的 `trace_*` 数组名；testbench 在全局 include `stdint.h` 后，分别用
`trace_data_0000`、`trace_data_0001`、`trace_data_0002` namespace 包裹 include，
因此 trace 数组名称和绝对 include 路径保持不变。权重 header 则使用新版紧凑分块布局（版本2），testbench 按该布局装入原有 weight stream；不再从 `[K,Cout]` header 重排。Im2Row 的 `pixel_t` 输入、两阶段调度及事务数不变。

`main` 只创建一组外部 stream，并按 0000 → 0001 → 0002 顺序连续调用顶层三次。
每次调用前重新写入完整的784个输入事务、两层卷积权重/参数和 FC 权重/参数；调用
返回后逐 lane 精确比较10个 INT8 logits，并检查7条输入/参数 stream 与1条 logits
stream 均已清空。`input.csv` 的列为 `sample,y,x,channel,value`，`output.csv`
的列为 `sample,output,expected,actual,match`；写入 CSV 的 INT8 值先提升为 `int`，
避免 `signed char` 被输出为字符。本次 C Simulation 已验证三次调用的顺序和守恒；RTL
Co-simulation 与综合仍按后续验证流程执行。

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
- 连续三次Top调用的Pool1 FIFO清空、覆盖与结果独立性。

原 stream Top 本身不含 DDR、DMA 或 AXI master。新增外层
`cnn_accelerator_ddr` 保留本 Top 接口/循环，负责独立 DDR loaders、定点Softmax
与24字节结果writer（10个uint16概率及uint32原logits first-max类别）。
新增Q20 scale控制参数，寄存器由HLS自动分配，PS使用重新生成的驱动；
旧IP/16字节协议不兼容，需重新导出XSA/BSP/bitstream。
本轮只静态检查，未功能验证、精度测量、综合、RTL或时序验证；历史core报告不覆盖新DDR后处理。
详见 [DDR Wrapper 设计](ddr_wrapper.md)。
Wrapper 的外层 DATAFLOW 与本层 phase barrier/共享 FIFO 是否保持预期层次和
有限深度反压行为，仍待综合与 RTL Co-simulation 确认。
