# FPGA INT8 CNN Accelerator

本仓库从 Zynq-7000 上的 MNIST 加速器演进而来。当前 **HLS 源码**面向固定的
YOLOv2-tiny VOC 九层网络；**Vivado摄像头工程已接入YOLO IP和roi_overlay**。
Vitis PS应用正分阶段迁移，当前接入SD模型加载、416×416 ROI整数预处理、
连续YOLO推理、检测框解码和NMS，保留HDMI预览；检测列表上传尚未接入。

## 当前 HLS

- 目标器件为 `xc7z020-clg400-2`，工具版本为 Vitis HLS 2022.2。
- 输入为 `416×416×3` 紧凑 RGB INT8，内部使用 8 通道 Pack、4 个空间位置
  并行、4 个 Cout 并行的 OS Systolic。激活和权重为 signed INT8，累加为 INT32。
- 单层按最多 32 Cout 驻留。每个 Cout 块重新扫描 Feature，权重和参数按整层供流。
  层内通过 stream 流水，层与层之间通过 DDR A/B 两区交替，目标是九层复用
  一套 Im2Row、Systolic、PostProcess 和 Pool。
- 唯一系统入口是 `cnn_accelerator_ddr(feature, weights, params)`，不再有独立
  stream Core。PS 把紧凑 RGB 放在 Feature A 区，结果在 B 区；逻辑形状
  为 `13×13×125`，每像素物理步长 128 B。Top 不做检测框解码或 NMS。

详细契约见 [Top 设计](hls/docs/top.md)、[DDR 搬运与布局](hls/docs/yolov2_tiny_ddr_io.md)
和 [YOLOv2-tiny 总体规划](hls/docs/yolov2_tiny_refactor_plan.md)。模块文档：
[Im2Row](hls/docs/im2row.md)、[OS Systolic](hls/docs/systolic_os.md)、
[ConvPostProcess](hls/docs/conv_postprocess.md)、[Pool](hls/docs/pool.md)。

## 目录与验证

| 目录 | 内容 |
|---|---|
| [hls/cnn_accelerator/src](hls/cnn_accelerator/src) | HLS 模块、固定几何、DDR 布局和系统入口 |
| [hls/cnn_accelerator/testbench](hls/cnn_accelerator/testbench) | 模块级测试与 YOLO Top BIN 对照 |
| [hls/cnn_accelerator/script](hls/cnn_accelerator/script) | Vitis HLS 工程入口 |
| [hls/docs](hls/docs) | 设计、布局与验证记录 |
| [vivado/cnn_accel_cam](vivado/cnn_accel_cam) | 摄像头预览、YOLO/Overlay IP和分阶段迁移的Vitis PS工程 |
| `build/hls/` | 本地生成的工程、日志和报告（不提交 Git） |

安装 Vitis HLS 2022.2 后，在仓库根目录执行：

```bash
mkdir -p build/hls
cd build/hls
vitis_hls -f ../../hls/cnn_accelerator/script/hls_ddr.tcl
```

脚本默认只跑 C 仿真，工程写到 `build/hls/cnn_yolo_ddr/`。如需单独运行 HLS
综合，在命令前加 `YOLO_HLS_STAGE=csynth`；脚本不自动运行 RTL co-sim 或导出 IP。
Top testbench 读取 PyTorch 导出的 `bins/` 和 `trace/`，可通过
`YOLO_DATA_DIR` 指定导出根目录。只打印统计和有限错误，不写 CSV。

C++ 对照可以验证逐元素输出和 DDR 软件地址计算；单实例、有限 FIFO 反压、
AXI 写回顺序和板级时序需要分别检查综合报告、RTL 仿真与实现。

## 既有板级工程

[Vivado 工程说明](vivado/README.md)记录了OV5640 480p、HDMI预览与Vitis
裸机工程的打开方式。[PS应用说明](vivado/cnn_accel_cam/vitis/cnn_accel_cam/README.md)
列出27个BIN的SD目录、DDR拼接、ROI整数预处理、连续推理、解码/NMS和构建步骤；
[Overlay设计](hls/roi_overlay/README.md)说明新列表控制协议。
使用匹配当前硬件的XSA和BSP，Build Platform后再构建PS应用；软件检查和硬件Done不能代替板上数值及检测效果验证。
