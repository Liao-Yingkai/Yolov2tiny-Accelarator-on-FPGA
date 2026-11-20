# YOLOv2-tiny 数据目录

此目录是 CNN testbench 的默认数据位置。克隆仓库只会得到本说明，不包含模型或测试数据。
运行 CNN 仿真时，把同批导出的 `bins/` 和 `trace/` 直接放在此目录下，不要再嵌套
`generated/`。只有重新生成 PS 配置头才需要 `metadata.json`。若只上板，按
[SD 卡文件清单](../../vitis/cnn_accel_cam/README.md#sd模型文件布局)直接把模型 BIN
放入卡根目录的 `yolo/`，不需要先填充本目录。

文件清单、格式和数据获取边界见[模型与测试数据](../../hls/docs/data.md)。
