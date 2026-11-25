# YOLOv2-tiny DDR Top

顶层入口为 `cnn_accelerator_ddr(feature, weights, params)`，一次调用完成
416×416 紧凑 RGB 输入的九层推理，输出原始 Prediction Tensor，不执行检测解码。
计算与搬运是独立模块，DDR 数据格式见
[Loader、Writer 与层间 DDR 缓冲](yolov2_tiny_ddr_io.md)。

## 网络结构

固定输入为 416×416 RGB，batch 为 1。Conv1–Conv8 执行 3×3 卷积及 LeakyReLU；Prediction 执行 1×1 卷积，不使用激活函数。

| 卷积阶段 | 输入 H×W×Cin | Kernel / Padding / Stride | Cout | 后续 Pool | 写回的输出 H×W×C |
|---|---|---|---:|---|---|
| Conv1 | 416×416×3 | 3×3 / 1 / 1 | 16 | 2×2，stride=2 | 208×208×16 |
| Conv2 | 208×208×16 | 3×3 / 1 / 1 | 32 | 2×2，stride=2 | 104×104×32 |
| Conv3 | 104×104×32 | 3×3 / 1 / 1 | 64 | 2×2，stride=2 | 52×52×64 |
| Conv4 | 52×52×64 | 3×3 / 1 / 1 | 128 | 2×2，stride=2 | 26×26×128 |
| Conv5 | 26×26×128 | 3×3 / 1 / 1 | 256 | 2×2，stride=2 | 13×13×256 |
| Conv6 | 13×13×256 | 3×3 / 1 / 1 | 512 | 2×2，stride=1，右/下补边 | 13×13×512 |
| Conv7 | 13×13×512 | 3×3 / 1 / 1 | 1024 | 无 | 13×13×1024 |
| Conv8 | 13×13×1024 | 3×3 / 1 / 1 | 1024 | 无 | 13×13×1024 |
| Prediction | 13×13×1024 | 1×1 / 0 / 1 | 125 | 无 | 13×13×125 |

层表由 [`config.h`](../cnn_accelerator/src/config.h) 定义。Prediction 写入 DDR 时使用 128 通道物理步长，表中 125 是逻辑通道数。PS 负责检测解码和 NMS。

## 接口与模型区

| 参数 | 内容 | AXI master |
|---|---|---|
| `feature` | 一个可读写的 64-bit 字地址区，分成 A/B 两段 | `gmem_feature` |
| `weights` | 按固定层表依次拼接每层的 `weight.bin`，无层间填充 | `gmem_weights` |
| `params` | 按固定层表依次拼接每层的 `bias.bin`、`quant.bin`，无层间填充 | `gmem_params` |

三个指针基址和启动/完成控制通过 AXI-Lite 设置。三组 master 可接到同一片
DDR，不代表三套物理存储。模型基址由 PS 设置；层内偏移按各层实际长度递增，
不把地址表作为运行时输入。权重步长是 `weight_transactions × 4` 个 64-bit 字，
参数步长是 `3 × Cout` 个 INT32 字。`ddr_layout.h` 从固定层表推导缓冲容量
和偏移，当前模型的权重区、参数区分别为 15,859,328 B 和 38,172 B。

PS 每次推理前把 519,168 B 紧凑 RGB 原样复制到 A，不做 Cin 补零。
A/B 各 692,224 B（86,528 个 64-bit 字），同属一个连续分配，建议基址
64 B 对齐。Conv1 读 A 写 B；Conv2 读 B 写 A；此后继续交替。Conv2 会覆盖
A 中的原始图像。第九层写 B；结果从 Feature 基址偏移 692,224 B 开始，占
21,632 B，逻辑形状为 `13×13×125`，物理步长为每像素 128 字节。
PS 只解释前 125 个通道，最后 3 字节为零。在非一致性接口上，PS 启动前完成
输入及参数的缓存维护，结束后再使结果对应缓存失效并读取。

## 层间顺序、块内重扫与硬件复用

```text
Top：顺序调用 run_layer（九次，唯一调用点）
  run_layer：层级 DATAFLOW
    Weight Loader：按整层顺序供权重
    Param Loader：按整层顺序供参数
    run_conv_blocks：逐 Cout 块顺序调用 run_conv_block
      run_conv_block：块级 DATAFLOW
        Feature Loader → Im2Row → Systolic
    PostProcess → Pool/Bypass → Writer：按块顺序消费整层
```

层循环与 Cout 块循环都不展开、不跨迭代流水化。一个块内 Feature Loader 与
Im2Row 各调用一次；块切换后从同一 DDR 输入基址重新扫描，3×3 Im2Row 重新预填
首行。Systolic 从整层权重流取当前块所需 Pack；PostProcess、Pool 和 Writer
各只调用一次，通过几何参数和事务数识别块边界。无 Pool 的层仍走同一个 Pool
入口，在模块内部透传。

权重、参数、累加和结果 FIFO 属于层级作用域，像素和窗口 FIFO 属于块级
作用域，深度均为 2。块切换不要求 Writer 先写完当前块；层切换必须等待
整个 `run_layer()` 完成。源代码没有跨层预读，也没有中间 Feature FIFO。
AXI 上要验证上一层写响应已完成后下一层才发读请求；最后一笔 stream 写出
不等同于 DDR 已完成写回。

同一个层调用点和同一个块调用点用于复用计算模块；层级综合报告需确认只有
一套 Im2Row、Systolic、PostProcess、Pool。浅 FIFO 下的实际反压、层间
AXI 完成顺序和板级吞吐不由 C 仿真证明。

## 验证入口

`hls/cnn_accelerator/script/hls_ddr.tcl`只注册本Top使用的源文件，默认运行
`-O3` 编译的 C 仿真。设置 `YOLO_HLS_STAGE=csynth` 只做 HLS 综合。
工程在仓库根目录的 `build/hls/cnn_yolo_ddr/`；建议从生成物目录启动
Vitis HLS，避免工具启动日志进入源码树。

`test_cnn_accel_ddr.cpp` 读取样本 0000 的 RGB、模型 BIN 和 Prediction
golden。测试读取仓库的 `data/yolov2_tiny/`，文件准备见[模型与测试数据](data.md)。
测试逐元素核对 B 区结果和尾槽，并检查缓冲区外的哨兵，只打印摘要与有限
错误详情，不生成 CSV。

C 仿真验证数值与地址边界；时序、资源、FIFO 反压及 DDR 写回屏障需分别通过综合报告、RTL 验证和板上测试检查。HLS 估计不能代替 Vivado 布局布线结果。
