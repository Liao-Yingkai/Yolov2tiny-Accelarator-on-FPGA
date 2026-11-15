# YOLO相机输入配置

当前PS使用416×416 RGB INT8输入；取帧、DDR和推理流程见[应用说明](../vitis/cnn_accel_cam/README.md)。此前MNIST灰度、反色和mean/std流程不再适用。

## 输入系数生成

在仓库根目录运行：

```bash
python3 vivado/cnn_accel_cam/tools/generate_camera_model_input.py \
  --metadata /path/to/YOLOV2_TINY/generated/metadata.json
```

默认metadata为`/home/liaoyingkai/python_project/minist_train_fpga/YOLOV2_TINY/generated/metadata.json`，默认输出为`../vitis/cnn_accel_cam/src/camera_model_input.h`（相对于本目录）。更换SD模型时必须用同一次导出的metadata重新生成此头并重编译PS。

生成器只读取metadata，不修改训练工程，不重新导出BIN；只生成少量M/S标量，不生成LUT或大数组。

## 数值与布局

- 源视频640×480 BGR888，安全抓拍只复制`[112,528)×[32,448)`的紧凑BGR ROI。
- 预处理交换B/R后，以H→W→RGB写紧凑INT8；不缩放、不灰度、不反色、不做mean/std，不补8通道。
- 离线参考为`FP32(pixel/255) × FP32(1/input_scale)`后最近偶数舍入和INT8饱和，与YOLO `dump_trace.quantize_input`一致。
- 生成器枚举256种输入值，寻找32位安全、逐值一致的M/S；失败时拒绝更新头文件。
- 运行时仅整数乘法、移位、最近偶数处理和饱和，没有浮点或查表。

权重及bias/M/S仍由SD的27个BIN加载，不再使用旧`export_camera_model.py`生成MNIST模型数组；`generate_softmax_params.py`也不是本阶段入口。当前不计算检测概率，解码/NMS后续接入。
