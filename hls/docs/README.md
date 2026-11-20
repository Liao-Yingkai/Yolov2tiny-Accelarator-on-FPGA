# YOLOv2-tiny HLS 文档

本目录记录固定九层YOLOv2-tiny VOC设计。接口入口以[DDR Top](top.md)为准，整体环境见
[仓库说明](../../README.md)。文件路径以仓库根目录为基准；表中链接可直接定位源文件。
系统验证基线为2026-11-19，各模块保留的历史检查只代表对应版本与检查范围。

- [DDR Top](top.md)：网络层表、顶层接口与层/块调度。
- [模型与测试数据](data.md)：导出目录、BIN/trace及SD模型的对应关系。
- [Loader、Writer 与层间 DDR 缓冲](yolov2_tiny_ddr_io.md)：模型布局、紧凑 RGB、NHWC 写回与 A/B 屏障。
- [Im2Row](im2row.md)、[OS Systolic](systolic_os.md)、[ConvPostProcess](conv_postprocess.md)、[Pool](pool.md)：模块接口、实现与各自验证边界。

## 独立视频叠加模块

[ROI遮罩与检测框显示](../roi_overlay/README.md) 位于 `hls/roi_overlay/`，不参与CNN计算流水。它在640×480视频中显示416×416 ROI，将外侧填黑，并接收PS完成NMS后提交的检测框和标签。独立Top为 `roi_overlay`，测试入口为 `test_roi_overlay.cpp`；不能将该测试与CNN的testbench一起注册到同一个Top。

当前Overlay为`ap_ctrl_none`自由运行和显式FLP流水，无需PS启动；命令完成通过sequence ACK确认。

## 当前 testbench

`hls/cnn_accelerator/testbench/`保留以下五个YOLO测试入口，每个文件含独立`main()`，
应按对应Top单独注册，不能将五个文件一起加入同一个testbench：

| 文件 | 对应 Top |
|---|---|
| [test_im2row_pytorch.cpp](../cnn_accelerator/testbench/test_im2row_pytorch.cpp) | `im2row` |
| [test_systolic_pytorch.cpp](../cnn_accelerator/testbench/test_systolic_pytorch.cpp) | `systolic_OS` |
| [test_conv_postprocess_pytorch.cpp](../cnn_accelerator/testbench/test_conv_postprocess_pytorch.cpp) | `conv_postprocess` |
| [test_pool_pytorch.cpp](../cnn_accelerator/testbench/test_pool_pytorch.cpp) | `maxpool` |
| [test_cnn_accel_ddr.cpp](../cnn_accelerator/testbench/test_cnn_accel_ddr.cpp) | `cnn_accelerator_ddr` |

测试使用同一次模型导出的BIN/trace，样本为`0000`。默认目标目录是`data/yolov2_tiny/`，
但它不随Git提供，须先准备数据。所有测试只打印统计和有限错误，不写CSV，失败返回非零。
完整文件清单见[数据说明](data.md)。

## 脚本入口与运行产物

`hls/cnn_accelerator/script/hls_ddr.tcl`仅建立DDR Top工程，默认执行C仿真；
`YOLO_HLS_STAGE=csynth`选择综合，空值同默认，其他值报错。它不自动执行co-sim或导出IP。
Linux/Bash调用示例见[仓库说明](../../README.md#运行-hls-验证)。

工程始终生成在`build/hls/cnn_yolo_ddr/`，路径由脚本位置定位；启动工具所在目录仍可能
产生`vitis_hls.log`，因此示例先进入`build/hls/`。这两个目录含义不能混淆。
默认数据路径通过`YOLO_DEFAULT_DATA_DIR`编译宏仅传给Top测试文件，不改变DUT配置。
整个仓库搬迁后需要重新执行脚本并编译，旧测试程序可能保留旧编译路径。

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

不使用 Tcl 脚本时，测试从共享头文件的路径推导默认目录。请在工程中添加仓库内的原始源码；若编译器记录的是相对路径，该路径在仿真运行目录中也必须可读。工具复制源码或改变目录结构时，可在 testbench 编译选项中把 `YOLO_DEFAULT_DATA_DIR` 定义为指向数据目录的 C++ 字符串常量。该设置是编译宏，不是环境变量。

### 操作系统边界

- 本仓库HLS流程与路径检查在Linux完成验证。
- Tcl 编译参数的引用按 POSIX shell 处理，本仓库未验证 Windows 端到端 HLS 构建。不要在 PowerShell 或 CMD 中直接使用文档中的 Bash 命令。

这些限制属于软件验证入口，不改变PL接口或网络的数据契约。

历史`build/`报告与日志不随源码提供。文档中的历史PASS、资源或时序数值不保证新的工具、
主机系统、路径或模型也通过，需要使用对应环境重新检查。
