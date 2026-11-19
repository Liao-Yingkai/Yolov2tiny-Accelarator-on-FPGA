# 模型与测试数据

本仓库保存 HLS、PS 源码及两份小型模型配置头，不保存训练 checkpoint、图像数据集、
模型 BIN 或逐层 trace。训练和数据导出在外部 Python 工程的 `YOLOV2_TINY/` 中完成；
HLS 测试只读取导出结果，不自动训练、导出或改写模型。

## 导出目录与路径

准备以下目录，并让权重、参数、metadata 和 trace 来自同一次模型导出：

```text
generated/
├── metadata.json
├── bins/
│   ├── conv1_weight.bin
│   ├── conv1_bias.bin
│   ├── conv1_quant.bin
│   ├── ...                 # conv2～conv8，每层同样三份文件
│   ├── prediction_weight.bin
│   ├── prediction_bias.bin
│   └── prediction_quant.bin
└── trace/
    ├── trace_input_int8_hwc_0000.bin
    ├── trace_conv1_unfold_int8_0000.bin
    ├── trace_conv1_im2row_int8_0000.bin
    ├── trace_conv1_systolic_int32_hwc_0000.bin
    ├── trace_conv1_postprocess_int8_hwc_0000.bin
    ├── trace_pool1_int8_hwc_0000.bin
    └── ...                 # 其余层按下面的边界命名
```

五个 CNN testbench 都通过 `YOLO_DATA_DIR` 指定 `generated` 根目录，不再依赖作者电脑
上的路径。运行 Vitis HLS 前在同一终端设置，使其启动的测试程序继承环境变量：

```bash
export YOLO_DATA_DIR=/path/to/YOLOV2_TINY/generated
```

未设置或为空时，测试明确报错并返回非零；当前样本后缀固定为 `0000`。可直接从其他
位置读取大型 trace，无需复制到 HLS 源码目录。

## 模块测试需要哪些文件

`<layer>` 为 `conv1`～`conv8` 或 `prediction`，`<n>` 为 1～6。

| 测试 | 读取边界 |
|---|---|
| Im2Row | 上游 INT8 HWC、`trace_<layer>_unfold_int8_0000.bin`、`trace_<layer>_im2row_int8_0000.bin` |
| Systolic | Im2Row packed BIN、`bins/<layer>_weight.bin`、`trace_<layer>_systolic_int32_hwc_0000.bin` |
| ConvPostProcess | Systolic INT32 HWC、bias/quant BIN、`trace_<layer>_postprocess_int8_hwc_0000.bin` |
| Pool | 各层 PostProcess INT8 HWC；池化层另读 `trace_pool<n>_int8_hwc_0000.bin`，Bypass 对比原输入 |
| DDR Top | 全部 27 个模型 BIN、输入 RGB BIN、`trace_prediction_postprocess_int8_hwc_0000.bin` |

Im2Row 的上游分别为：Conv1 读 `input_int8_hwc`，Conv2～Conv7 读 Pool1～Pool6，
Conv8 读 Conv7 PostProcess，Prediction 读 Conv8 PostProcess。所有前后缀均按上表。
Overlay 测试使用程序构造的视频与框，不需要模型目录。

## 数值与文件契约

- Feature 和逐层 golden 为紧凑 HWC，激活/权重是有符号 INT8；Systolic golden、
  bias、multiplier、shift 是 little-endian INT32。BIN 不保存 C++ 结构体或 valid mask。
- 权重物理顺序为 `cout_tile → kh → kw → cin_group → cout_lane → cin_lane`，每组
  4 Cout × 8 Cin，共 32 B；无效权重通道由 exporter 填零。
- 每层 `bias.bin` 为逻辑 Cout 个 bias；`quant.bin` 依次保存全部 multiplier、全部
  shift，不补参数尾槽。两文件拼接为 `[bias][M][S]`。
- RGB 输入仅含每像素 3 B，不补 8 通道；Feature Loader 组包时补流上的无效槽。
- Im2Row packed BIN 是 `[H,ceil(W/4),Kh*Kw*ceil(Cin/8),4,8]`，而 unfold 是
  `[H*W,Kh*Kw*Cin]`。测试按各自逻辑坐标比较，不能按相同平坦下标混读。
- Prediction golden 是紧凑 `13×13×125`；PL 的 DDR 结果按 `13×13×128` 存放，最后
  3 个通道为零。Top 测试按两种不同步长取值，结果位于 Feature B。

更完整的地址和事务顺序见 [DDR 布局](yolov2_tiny_ddr_io.md) 和各模块文档。
测试保留文件长度、逐元素、valid、事务数及边界检查；不再导出 CSV 或创建
`simulation_data/`，失败返回非零，工具日志保存在源码树外的 `build/`。

## PS 配置与 SD 卡

在仓库根目录执行以下命令，默认更新现有 PS 配置头，不更改模型 BIN：

```bash
python3 vivado/cnn_accel_cam/tools/generate_camera_model_input.py
python3 vivado/cnn_accel_cam/tools/generate_camera_model_output.py
```

两脚本读取 `YOLO_DATA_DIR/metadata.json`，也可用 `--metadata /path/to/metadata.json`
显式覆盖。输出分别为 `camera_model_input.h` 和 `camera_model_output.h`，保存量化
标量、Prediction scale、anchors、类别及 checkpoint 校验标识；不保存大模型或 LUT。
更新模型后，两份头和 27 个 BIN 必须同步，重新编译 PS 应用。

板上只需把 `bins/` 下 27 个文件放到 SD 的 `0:/yolo/`，不要再嵌套 `bins/`；SD 不需要
trace、NPY 或 metadata。JTAG 运行不要求 `BOOT.BIN`，SD 自启动时则另需匹配硬件的
启动镜像。详细文件清单和 DDR 分配见 [PS 应用说明](../../vivado/cnn_accel_cam/vitis/cnn_accel_cam/README.md)。
