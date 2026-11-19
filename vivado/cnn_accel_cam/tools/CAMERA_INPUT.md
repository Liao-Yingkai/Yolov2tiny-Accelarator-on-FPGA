# YOLO相机输入配置

当前PS使用416×416 RGB INT8输入；取帧、DDR和推理流程见[应用说明](../vitis/cnn_accel_cam/README.md)。模型BIN从SD加载，输入量化和输出解码配置从同批metadata生成后编译进PS应用。

## 输入系数生成

在仓库根目录运行：

```bash
python3 vivado/cnn_accel_cam/tools/generate_camera_model_input.py \
  --metadata /path/to/YOLOV2_TINY/generated/metadata.json
```

默认读取仓库内`data/yolov2_tiny/metadata.json`，根据脚本位置定位，正常运行无需设置环境变量。`--metadata`可显式指定文件，非空`YOLO_DATA_DIR`可选指定数据根目录；优先级为`--metadata` → 环境变量 → 仓库默认目录。默认输出为`../vitis/cnn_accel_cam/src/camera_model_input.h`（相对于本目录）。更换SD模型时必须用同一次导出的metadata重新生成此头并重编译PS。

生成器只读取metadata，不修改训练工程，不重新导出BIN；只生成少量M/S标量，不生成LUT或大数组。

## 数值与布局

- 源视频640×480 BGR888，安全抓拍只复制`[112,528)×[32,448)`的紧凑BGR ROI。
- 预处理交换B/R后，以H→W→RGB写紧凑INT8；不缩放、不灰度、不反色、不做mean/std，不补8通道。
- 离线参考为`FP32(pixel/255) × FP32(1/input_scale)`后最近偶数舍入和INT8饱和，与YOLO `dump_trace.quantize_input`一致。
- 生成器枚举256种输入值，寻找32位安全、逐值一致的M/S；失败时拒绝更新头文件。
- 运行时仅整数乘法、移位、最近偶数处理和饱和，没有浮点或查表。

权重及bias/M/S由SD的27个BIN加载；模型BIN保持训练端导出的布局，不编译为模型数组。SD文件目录及DDR拼接约定见[应用说明](../vitis/cnn_accel_cam/README.md)。

## 输出解码配置

当前PS已接入解码/NMS。更换SD模型后，还需从同一metadata生成Prediction输出scale、anchors和类别名：

```bash
python3 vivado/cnn_accel_cam/tools/generate_camera_model_output.py \
  --metadata /path/to/YOLOV2_TINY/generated/metadata.json
```

默认输出`../vitis/cnn_accel_cam/src/camera_model_output.h`。生成器验证固定416输入、20类、5个anchor、125逻辑/128物理通道及零点0；校验完成后才更新配置头。当前输出scale为`0.19452644884586334`，不生成指数LUT或追加SD文件。输入、输出配置头及SD模型必须来自同一次导出。
