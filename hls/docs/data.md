# 模型与测试数据

本文路径以仓库根目录（包含`README.md`、`hls/`、`vivado/`的目录）为基准。
默认数据位置是根目录下的`data/yolov2_tiny/`，不是`hls/data/`。

## 从 Release 获取配套数据

Git保存HLS、PS源码及两份小型模型配置头，不跟踪训练checkpoint、图像数据集、模型BIN
或逐层trace。`data/yolov2_tiny/` 只跟踪目录说明，克隆后需自行放入数据。

配套上板模型的 27 个 BIN 通过本项目的 GitHub Releases 附件分发。选择与代码版本配套
的附件，按下文清单放入 SD 卡即可；Vitis 应用已包含配套的 PS 配置头，不需要 PyTorch、
checkpoint 或额外的导出操作。模型附件与 GitHub 自动生成的源码压缩包是分开的。

CNN 功能仿真还需要同一模型的 golden trace；若所用发布版本同时提供 trace，可直接使用。
只有在自行训练、修改模型、重新导出，或需要生成未随发布版提供的 trace 时，才需要下文的
[PyTorch 配套仓库](https://github.com/Liao-Yingkai/INT8-CNN-Trainer-for-CNN-Accelarator)。

## 从 PyTorch 工程生成数据

按 [YOLOv2-tiny 使用说明](https://github.com/Liao-Yingkai/INT8-CNN-Trainer-for-CNN-Accelarator/tree/main/YOLOV2_TINY)
安装依赖；需要训练或生成图像 trace 时，按说明准备 VOC 数据。已有兼容的
`single_postprocess_v1` QAT checkpoint 时，可跳过训练，直接从 **PyTorch 仓库根目录**执行：

```text
python YOLOV2_TINY/export.py --checkpoint YOLOV2_TINY/checkpoints/qat_model_best.pth
python YOLOV2_TINY/dump_trace.py --num-samples 1 --seed 1
```

将示例 checkpoint 路径替换为实际文件。第一条命令在默认的 `YOLOV2_TINY/generated/`
下写出 `bins/` 和 `metadata.json`。第二条只在需要仿真参考时运行，它读取刚导出的
整数参数，在同一目录的 `trace/` 生成样本 `0000`。重新导出模型后也要重新生成配套
trace；若旧文件仍在，先将它们另存或给导出和 trace 生成指定一个新的输出目录，以免混用。

## 取得数据后放在哪里

如果数据以压缩包提供，先解压；以包含 `bins/` 或 `trace/` 的目录为准，
不要把压缩包、外层版本目录或 `generated/` 整个再套进去。按目标准备文件：

| 目标 | 需要的文件及位置 |
|---|---|
| 上板运行仓库配套模型 | 27 个模型 BIN → SD 根目录 `yolo/`；不需要 metadata、trace 或 Python |
| 整网 HLS C 仿真 | 27 个模型 BIN → `data/yolov2_tiny/bins/`；样本 `0000` 的输入与 Prediction 参考 BIN → `data/yolov2_tiny/trace/` |
| 运行全部 CNN 模块测试 | 同一套模型 BIN 与样本 `0000` 的完整 trace，分别放入上述两个目录 |
| 更换模型并生成 PS 配置头 | 配套的 `metadata.json` → `data/yolov2_tiny/metadata.json`，或通过脚本 `--metadata` 显式指定 |
| Overlay C 仿真、仅做 HLS 综合 | 不需要模型文件 |

如果既要仿真又要上板，`data/yolov2_tiny/bins/` 和 SD 的 `yolo/` 分别存同一份
模型 BIN；两个位置不会自动同步。模型根目录不要混放不同 checkpoint 的文件。
本仓库的 HLS 测试不会自动训练、导出或更改数据。

## 导出目录与路径

准备以下目录，并让权重、参数、metadata 和 trace 来自同一次模型导出：

```text
data/yolov2_tiny/             # 与外部工程的generated根目录内容对应
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

五个 CNN testbench 都读取仓库根目录的 `data/yolov2_tiny/`。测试优先通过原始共享头文件
定位仓库，否则从仿真工作目录向上查找仓库。Tcl 和 GUI 工程都不需要给编译选项传入数据
路径；手动建立工程的路径要求见[手动模块工程](../README.md#手动模块工程与路径限制)。

供五个 CNN 测试使用的样本 `0000` 文件集包含 27 个模型 BIN 和 43 个 trace BIN。
`metadata.json` 是同批导出的说明和 PS 配置生成输入，CNN testbench 不读取它。
不需要 NPY、checkpoint 或完整训练数据集。文件名中的`0000`是样本编号，
不表示只验证第一层。更换模型时应整套更新，不能混用不同checkpoint的BIN和golden。
当前两份PS配置头记录了checkpoint校验标识；使用其他模型时需按下文重新生成。

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

需要Python 3.7或以上，两个生成器仅依赖Python标准库，不要求在本机安装PyTorch。
以下命令从仓库根目录执行，默认更新现有PS配置头，不更改模型BIN。请先确认所用命令
启动的是Python 3；`python3`和`py -3`是两种环境下的示例，不是跨系统固定名称。

Linux/Bash：

```bash
python3 scripts/generate_camera_model_input.py
python3 scripts/generate_camera_model_output.py
```

Windows/PowerShell（已安装Python Launcher时）：

```powershell
py -3 scripts/generate_camera_model_input.py
py -3 scripts/generate_camera_model_output.py
```

没有`py`命令时，用本机Python 3解释器命令或可执行文件路径替换；脚本输入输出位置
不因更换解释器命令而改变。此处给出调用语法，不声称已完成Windows端到端构建验证。

两脚本默认读取仓库内`data/yolov2_tiny/metadata.json`，根据脚本所在位置定位，与启动
目录无关。读取另一份 metadata 时，用 `--metadata` 指定文件。
输出分别为 `camera_model_input.h` 和 `camera_model_output.h`，保存量化
标量、Prediction scale、anchors、类别及 checkpoint 校验标识；不保存大模型或 LUT。
更新模型后，两份头和 27 个 BIN 必须同步，重新编译 PS 应用。

显式传入的相对 `--metadata`、`--output` 路径按进程当前目录解释；
只有默认路径按脚本位置定位。外部覆盖建议传绝对路径，以免在IDE和终端中指向不同文件。

板上只需把 `bins/` 下 27 个文件放到 SD 的 `0:/yolo/`，不要再嵌套 `bins/`；SD 不需要
trace、NPY 或 metadata。JTAG 运行不要求 `BOOT.BIN`，SD 自启动时则另需匹配硬件的
启动镜像。详细文件清单和 DDR 分配见 [PS 应用说明](../../vitis/cnn_accel_cam/README.md)。
