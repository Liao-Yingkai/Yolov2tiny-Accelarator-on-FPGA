# YOLOv2-tiny INT8 FPGA Accelerator

基于 Vitis HLS / Zynq-7000 的 YOLOv2-tiny VOC 目标检测工程，包含 HLS 计算模块、
Vivado 摄像头/HDMI 工程和 Vitis 裸机应用。网络采用固定九层结构，面向学习与实现
FPGA 推理数据通路，不是任意网络的通用部署框架。

当前已完成测试及上板功能验证（2026-11-19，用户确认），能够连续执行摄像头取帧、
整数预处理、PL 推理、PS 解码/NMS 和 HDMI 检测框显示。具体帧率、精度及最终资源
应以对应版本的实测和实现报告为准，历史 HLS 估计不作为当前板测指标。

## 系统分工

```text
OV5640 → VDMA / DDR → PS：裁切416×416 ROI、RGB整数归一化/量化
                  → PL：九层卷积 / LeakyReLU / Pool → DDR Prediction
                  → PS：sigmoid / softmax / NMS → Overlay → HDMI
```

HDMI 保留实时预览；Overlay 位于视频输出路径，不写回 CNN 的采集数据。
模型在启动时从 SD 一次性加载，之后连续推理，不固定为 10 fps，也不逐帧打印 UART。

## HLS 计算结构

- 目标器件为 `xc7z020-clg400-2`，工具版本为 Vitis HLS 2022.2。
- 输入为 `416×416×3` 紧凑 RGB INT8，内部使用 8 通道 Pack、4 个空间位置
  并行、4 个 Cout 并行的 OS Systolic。激活和权重为 signed INT8，累加为 INT32。
- 单层按最多 32 Cout 驻留。每个 Cout 块重新扫描 Feature，权重和参数按整层供流。
  层内通过 stream 流水，层与层之间通过 DDR A/B 两区交替，九层复用
  一套 Im2Row、Systolic、PostProcess 和 Pool。
- 唯一系统入口是 `cnn_accelerator_ddr(feature, weights, params)`，不再有独立
  stream Core。PS 把紧凑 RGB 放在 Feature A 区，结果在 B 区；逻辑形状
  为 `13×13×125`，每像素物理步长 128 B。Top 不做检测框解码或 NMS。

详细契约见 [Top 设计](hls/docs/top.md)、[DDR 搬运与布局](hls/docs/yolov2_tiny_ddr_io.md)
和 [YOLOv2-tiny 设计记录](hls/docs/yolov2_tiny_refactor_plan.md)。模块文档：
[Im2Row](hls/docs/im2row.md)、[OS Systolic](hls/docs/systolic_os.md)、
[ConvPostProcess](hls/docs/conv_postprocess.md)、[Pool](hls/docs/pool.md)。

## 工程目录

| 目录 | 内容 |
|---|---|
| [hls/cnn_accelerator/src](hls/cnn_accelerator/src) | HLS 模块、固定几何、DDR 布局和系统入口 |
| [hls/cnn_accelerator/testbench](hls/cnn_accelerator/testbench) | 模块级测试与 YOLO Top BIN 对照 |
| [hls/cnn_accelerator/script](hls/cnn_accelerator/script) | Vitis HLS 工程入口 |
| [hls/roi_overlay](hls/roi_overlay) | 独立视频遮罩、检测框及文字叠加核 |
| [hls/docs](hls/docs/README.md) | 模块设计、DDR 契约和测试数据说明 |
| [vivado/cnn_accel_cam](vivado/cnn_accel_cam) | 摄像头预览、YOLO/Overlay IP和Vitis PS应用 |
| [vivado/cnn_accel_cam/tools](vivado/cnn_accel_cam/tools/CAMERA_INPUT.md) | 从模型 metadata 生成 PS 输入/输出配置 |
| `build/hls/`、`build/ps/` | 本地检查、工程、日志和报告，不纳入版本管理 |

保留现有目录与工程名称；本地目录名即使包含历史项目名称，也不影响当前 YOLO 实现。

## 运行 HLS 验证

需要 Vitis HLS 2022.2，以及同一次 Python 模型导出的 `metadata.json`、`bins/` 和
`trace/`。模型权重、测试图像和大型 trace 不包含在本仓库，文件要求见
[测试与模型数据](hls/docs/data.md)。

在仓库根目录执行（将示例路径换成自己的导出目录）：

```bash
export YOLO_DATA_DIR=/path/to/YOLOV2_TINY/generated
mkdir -p build/hls
cd build/hls
vitis_hls -f ../../hls/cnn_accelerator/script/hls_ddr.tcl
```

脚本默认只跑 C 仿真，工程写到 `build/hls/cnn_yolo_ddr/`。如需单独运行 HLS
综合，在命令前加 `YOLO_HLS_STAGE=csynth`；脚本不自动运行 RTL co-sim 或导出 IP。
五个 CNN testbench 统一要求设置 `YOLO_DATA_DIR`，使用样本 `0000`，只打印统计和有限
错误，不写 CSV。模块级测试须各自选择对应 Top，不要将多个 `main()` 加入同一工程；
入口对照见 [HLS 文档索引](hls/docs/README.md)。

C++ 对照可以验证逐元素输出和 DDR 软件地址计算；单实例、有限 FIFO 反压、
AXI 写回顺序和板级时序需要分别检查综合报告、RTL 仿真与实现。

## 打开板级工程

[Vivado 工程说明](vivado/README.md)记录了OV5640 480p、HDMI预览与Vitis
裸机工程的打开方式。[PS应用说明](vivado/cnn_accel_cam/vitis/cnn_accel_cam/README.md)
列出27个BIN的SD目录、DDR拼接、ROI整数预处理、连续推理、解码/NMS、检测框显示和构建步骤；
[Overlay设计](hls/roi_overlay/README.md)说明显示与列表控制协议。当前 Overlay 使用
`ap_ctrl_none` 和显式 FLP 流水，不调用 Start；CNN 则由 PS 按图像逐次启动。
使用匹配当前硬件的XSA和BSP，Build Platform后再构建PS应用；软件检查和硬件Done不能代替板上数值及检测效果验证。

首次在另一目录打开 Vitis 时需要重新关联本地 platform；工具生成物、BSP 库和 IDE 状态
不随源码保存，具体恢复步骤见 Vivado 工程说明。不要为消除本机路径提示而改硬件参数。
