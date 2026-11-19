# YOLOv2-tiny HLS 文档

本目录记录当前固定九层 YOLOv2-tiny VOC 设计。接口入口以 [DDR Top](top.md) 为准，整体使用入口见[仓库说明](../../README.md)。用户已于2026-11-19确认测试及上板功能验证完成；各模块保留的旧检查记录只代表对应版本，不等同于本次重新运行。

- [设计记录与待优化事项](yolov2_tiny_refactor_plan.md)：层表、数值契约、存储预算与后续优化。
- [DDR Top](top.md)：顶层接口、层/块调度与已有整网检查记录。
- [模型与测试数据](data.md)：导出目录、`YOLO_DATA_DIR`、BIN/trace及SD模型的对应关系。
- [Loader、Writer 与层间 DDR 缓冲](yolov2_tiny_ddr_io.md)：模型布局、紧凑 RGB、NHWC 写回与 A/B 屏障。
- [Im2Row](im2row.md)、[OS Systolic](systolic_os.md)、[ConvPostProcess](conv_postprocess.md)、[Pool](pool.md)：模块接口、实现与各自验证边界。

## 独立视频叠加模块

[ROI遮罩与检测框显示](../roi_overlay/README.md) 位于 `hls/roi_overlay/`，不参与CNN计算流水。它在640×480视频中显示416×416 ROI，将外侧填黑，并接收PS完成NMS后提交的检测框和标签。独立Top为 `roi_overlay`，测试入口为 `test_roi_overlay.cpp`；不能将该测试与CNN的testbench一起注册到同一个Top。

当前Overlay为`ap_ctrl_none`自由运行和显式FLP流水，无需PS启动；命令完成通过sequence ACK确认。

## 当前 testbench

`cnn_accelerator/testbench/` 保留以下五个 YOLO 测试入口，每个文件含独立 `main()`，应按对应 Top 单独注册：

| 文件 | 对应 Top |
|---|---|
| [test_im2row_pytorch.cpp](../cnn_accelerator/testbench/test_im2row_pytorch.cpp) | `im2row` |
| [test_systolic_pytorch.cpp](../cnn_accelerator/testbench/test_systolic_pytorch.cpp) | `systolic_OS` |
| [test_conv_postprocess_pytorch.cpp](../cnn_accelerator/testbench/test_conv_postprocess_pytorch.cpp) | `conv_postprocess` |
| [test_pool_pytorch.cpp](../cnn_accelerator/testbench/test_pool_pytorch.cpp) | `maxpool` |
| [test_cnn_accel_ddr.cpp](../cnn_accelerator/testbench/test_cnn_accel_ddr.cpp) | `cnn_accelerator_ddr` |

测试使用同一次外部Python模型导出的BIN/trace；五个CNN测试都要求设置`YOLO_DATA_DIR`为`generated/`根目录，样本为`0000`。所有测试只打印统计和有限错误，不再写CSV或创建输出数据目录；失败返回非零。完整文件清单见[数据说明](data.md)。

文档中的历史`build/`报告和日志只保存在本机，不随源码提供。2026-11-19整理仅做静态检查，不重跑testbench、仿真或综合，不更改DUT实现及硬件配置。
