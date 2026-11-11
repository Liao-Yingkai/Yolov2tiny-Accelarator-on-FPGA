# YOLOv2-tiny HLS 文档

本目录仅记录当前固定九层 YOLOv2-tiny VOC 设计。接口入口与整网验证边界以 [DDR Top](top.md) 为准；本文索引不新增验证结论。

- [设计记录与待优化事项](yolov2_tiny_refactor_plan.md)：层表、数值契约、存储预算与后续优化。
- [DDR Top](top.md)：顶层接口、层/块调度与已有整网检查记录。
- [Loader、Writer 与层间 DDR 缓冲](yolov2_tiny_ddr_io.md)：模型布局、紧凑 RGB、NHWC 写回与 A/B 屏障。
- [Im2Row](im2row.md)、[OS Systolic](systolic_os.md)、[ConvPostProcess](conv_postprocess.md)、[Pool](pool.md)：模块接口、实现与各自验证边界。

## 当前 testbench

`cnn_accelerator/testbench/` 保留以下五个 YOLO 测试入口，每个文件含独立 `main()`，应按对应 Top 单独注册：

| 文件 | 对应 Top |
|---|---|
| [test_im2row_pytorch.cpp](../cnn_accelerator/testbench/test_im2row_pytorch.cpp) | `im2row` |
| [test_systolic_pytorch.cpp](../cnn_accelerator/testbench/test_systolic_pytorch.cpp) | `systolic_OS` |
| [test_conv_postprocess_pytorch.cpp](../cnn_accelerator/testbench/test_conv_postprocess_pytorch.cpp) | `conv_postprocess` |
| [test_pool_pytorch.cpp](../cnn_accelerator/testbench/test_pool_pytorch.cpp) | `maxpool` |
| [test_cnn_accel_ddr.cpp](../cnn_accelerator/testbench/test_cnn_accel_ddr.cpp) | `cnn_accelerator_ddr` |

测试使用外部 Python 工程的配套 YOLO BIN/trace，默认导出目录为 `/home/liaoyingkai/python_project/minist_train_fpga/YOLOV2_TINY/generated/`。该真实路径中的 `minist_train_fpga` 保持原样；迁移数据目录的方式见各测试与模块文档。本次文档和旧测试清理未运行 testbench、仿真或综合。
