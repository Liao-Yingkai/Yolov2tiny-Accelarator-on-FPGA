# YOLOv2-tiny 数据目录

此目录是 CNN testbench 的默认数据位置。克隆仓库只会得到本说明，不包含模型或测试数据。
使用同一批导出的 `metadata.json`、`bins/` 和 `trace/`，直接放在此目录下，不要再嵌套
`generated/`。若只上板运行，按应用说明把模型 BIN 放入 SD 卡，无需 trace。

文件清单、格式和数据获取边界见[模型与测试数据](../../hls/docs/data.md)。
