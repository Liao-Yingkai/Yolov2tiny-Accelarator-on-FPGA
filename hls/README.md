# HLS 工程与验证

本目录记录固定九层YOLOv2-tiny VOC设计。接口入口以[DDR Top](docs/top.md)为准，整体环境见
[仓库说明](../README.md)。文件路径以仓库根目录为基准；表中链接可直接定位源文件。
系统验证基线为2026-11-19，各模块保留的历史检查只代表对应版本与检查范围。

- [DDR Top](docs/top.md)：网络层表、顶层接口与层/块调度。
- [模型与测试数据](docs/data.md)：导出目录、BIN/trace及SD模型的对应关系。
- [Loader、Writer 与层间 DDR 缓冲](docs/yolov2_tiny_ddr_io.md)：模型布局、紧凑 RGB、NHWC 写回与 A/B 屏障。
- [Im2Row](docs/im2row.md)、[OS Systolic](docs/systolic_os.md)、[ConvPostProcess](docs/conv_postprocess.md)、[Pool](docs/pool.md)：模块接口、实现与各自验证边界。

## 独立视频叠加模块

[ROI遮罩与检测框显示](docs/roi_overlay.md) 位于 `hls/roi_overlay/`，不参与CNN计算流水。它在640×480视频中显示416×416 ROI，将外侧填黑，并接收PS完成NMS后提交的检测框和标签。独立Top为 `roi_overlay`，测试入口为 [`testbench/test_roi_overlay.cpp`](roi_overlay/testbench/test_roi_overlay.cpp)；不能将该测试与CNN的testbench一起注册到同一个Top。

当前Overlay为`ap_ctrl_none`自由运行和显式FLP流水，无需PS启动；命令完成通过sequence ACK确认。

## 当前 testbench

`hls/cnn_accelerator/testbench/`保留以下五个YOLO测试入口，每个文件含独立`main()`，
应按对应Top单独注册，不能将五个文件一起加入同一个testbench：

| 文件 | 对应 Top |
|---|---|
| [test_im2row_pytorch.cpp](cnn_accelerator/testbench/test_im2row_pytorch.cpp) | `im2row` |
| [test_systolic_pytorch.cpp](cnn_accelerator/testbench/test_systolic_pytorch.cpp) | `systolic_OS` |
| [test_conv_postprocess_pytorch.cpp](cnn_accelerator/testbench/test_conv_postprocess_pytorch.cpp) | `conv_postprocess` |
| [test_pool_pytorch.cpp](cnn_accelerator/testbench/test_pool_pytorch.cpp) | `maxpool` |
| [test_cnn_accel_ddr.cpp](cnn_accelerator/testbench/test_cnn_accel_ddr.cpp) | `cnn_accelerator_ddr` |

测试使用同一次模型导出的BIN/trace，样本为`0000`。默认目标目录是`data/yolov2_tiny/`，
但它不随Git提供，须先准备数据。所有测试只打印统计和有限错误，不写CSV，失败返回非零。
完整文件清单见[数据说明](docs/data.md)。

## 脚本入口与运行产物

这里的流程用于验证 HLS 源码。直接复现板上应用时，Vivado 已包含打包好的 CNN 和
Overlay IP，**不需要先运行本目录脚本**。只有修改 HLS 源码并希望变更进入硬件时，
才需要重新综合、导出 IP，并更新 Vivado 的 IP 仓库及 XSA。

CNN 和 Overlay 两个工程均使用 `src/`、`script/` 和 `testbench/`。接口和实现说明集中在
[`docs/`](docs/)。

| 入口 | 默认操作 | 工程输出（相对于仓库根目录） |
|---|---|---|
| [CNN DDR Top](cnn_accelerator/script/hls_ddr.tcl) | C 仿真；设置 `YOLO_HLS_STAGE=csynth` 可改为综合 | `build/hls/cnn_yolo_ddr/` |
| [ROI Overlay](roi_overlay/script/hls_roi_overlay.tcl) | 独立 C 仿真，不需要模型或 trace | `build/hls/roi_overlay/` |

先加载 Vitis HLS 2022.2 的工具环境，确保 `vitis_hls` 可执行。以下是从仓库根目录开始的
Linux/Bash 示例。进入 `build/hls/` 是为了让工具启动日志也留在构建目录：

```bash
mkdir -p build/hls
cd build/hls
```

选择要运行的一个测试。CNN Top C 仿真在准备好 [BIN 与参考输出](docs/data.md)后运行：

```bash
vitis_hls -f ../../hls/cnn_accelerator/script/hls_ddr.tcl
```

只验证 Overlay 时，无需数据即可运行：

```bash
vitis_hls -f ../../hls/roi_overlay/script/hls_roi_overlay.tcl
```

只做 CNN HLS 综合时，在 `build/hls/` 中执行：

```bash
YOLO_HLS_STAGE=csynth vitis_hls -f ../../hls/cnn_accelerator/script/hls_ddr.tcl
```

脚本按自身路径定位源码，不自动执行 RTL 协同仿真或导出 IP。CNN testbench 自行定位
数据目录，Tcl 不向 `-cflags` 写入数据路径。


## 手动模块工程与路径限制

手动建立工程时，选择Vitis HLS 2022.2的Vivado IP流程、C++14和目标器件
`xc7z020clg400-2`。每个模块选择上表对应的Top与唯一testbench；设计文件分别为：

| Top | 设计源文件（相对于`hls/cnn_accelerator/src/`） |
|---|---|
| `im2row` | `im2row/im2row.cpp` |
| `systolic_OS` | `systolic/systolic_OS.cpp` |
| `conv_postprocess` | `postprocess/conv_postprocess.cpp` |
| `maxpool` | `pool/pool.cpp` |
| `cnn_accelerator_ddr` | 上述四个文件，加`axi/ddr_io.cpp`和`top/cnn_accel_ddr.cpp` |

保留源码的相对目录关系及`yolo_data_path.h`，不要只将单个CPP复制到无关目录。CNN脚本
使用10 ns目标时钟；Overlay独立工程不套用此配置，约束应与其视频系统匹配。

测试优先从共享头文件的位置推导数据目录；若工具复制了头文件或记录的相对路径不可读，
测试会从仿真工作目录向上查找仓库。手动建工程时，请引用仓库原始源码；如果工具会复制
测试源码，建议将仿真工程建在仓库的 `build/hls/` 下。无需设置数据路径宏或环境变量。

### 操作系统边界

- 本仓库HLS流程与路径检查在Linux完成验证。
- 本仓库未验证 Windows 端到端 HLS 构建。不要在 PowerShell 或 CMD 中直接使用文档中的 Bash 命令。

这些限制属于软件验证入口，不改变PL接口或网络的数据契约。

历史`build/`报告与日志不随源码提供。文档中的历史PASS、资源或时序数值不保证新的工具、
主机系统、路径或模型也通过，需要使用对应环境重新检查。
