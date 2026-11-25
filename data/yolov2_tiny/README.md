# YOLOv2-tiny 数据目录

此目录是 CNN testbench 的默认数据位置。克隆仓库只会得到本说明，不包含模型或测试数据。
运行 CNN 仿真时，把同批导出的 `bins/` 和 `trace/` 直接放在此目录下，不要再嵌套
`generated/`。只有重新生成 PS 配置头才需要 `metadata.json`。若只上板，按
[SD 卡文件清单](../../vitis/cnn_accel_cam/README.md#sd模型文件布局)直接把模型 BIN
放入卡根目录的 `yolo/`，不需要先填充本目录。

配套模型 BIN 从本项目的 GitHub Releases 获取。仿真可使用发布版提供的同批 trace（如有）；
仅在自行导出模型或生成未提供的 trace 时，才按 [PyTorch 导出说明](../../hls/docs/data.md#从-pytorch-工程生成数据)操作。
完整文件清单和格式见[模型与测试数据](../../hls/docs/data.md)。
