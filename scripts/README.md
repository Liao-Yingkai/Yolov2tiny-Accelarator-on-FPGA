# Python 辅助脚本：相机模型配置

PS 将 416×416 的 BGR888 裁剪数据转换为 RGB INT8；取帧、DDR和推理流程见
[应用说明](../vitis/cnn_accel_cam/README.md)。模型BIN从SD加载，输入量化与输出解码配置
从同一批metadata生成后编译进PS应用，不需要在PS上解析JSON。

## 什么时候需要运行

**使用 Release 的配套模型 BIN 时不用运行。** Vitis 应用已跟踪
[`camera_model_input.h`](../vitis/cnn_accel_cam/src/camera_model_input.h) 和
[`camera_model_output.h`](../vitis/cnn_accel_cam/src/camera_model_output.h)，可直接构建。
脚本只在使用与当前头文件不匹配的新模型导出，或需要从 metadata 重新生成配置时使用。
脚本不会下载权重、生成模型 BIN 或生成 HLS trace。新模型的 BIN 和 metadata 先在
[配套 PyTorch 工程中导出](../hls/docs/data.md#从-pytorch-工程生成数据)，再在本仓库运行下列脚本。

## 运行前提与默认位置

两个脚本要求Python 3.7或以上，仅使用标准库，不要求安装PyTorch。本文路径和命令
从仓库根目录计算；使用其他工作目录时，需要自行指定脚本文件的实际位置。

- 默认输入：`data/yolov2_tiny/metadata.json`，不随Git克隆提供，须预先准备。
- 默认输出：`vitis/cnn_accel_cam/src/camera_model_input.h`及
  `camera_model_output.h`。
- 默认输入、输出按脚本位置定位，不受启动目录影响；显式相对路径则按进程当前目录解释。
- `--metadata` 可显式指定输入；`--output` 可另存结果，避免覆盖配置头。默认运行无需设置数据环境变量。

metadata 需要满足 `format_version=2`、`weight_layout_version=3`，并与要部署的
27 个 BIN 属于同一次导出；这两个工具不会把任意 YOLO checkpoint 转成 FPGA 模型。

## 生成输入和输出配置

Linux/Bash，使用Python 3解释器命令：

```bash
python3 scripts/generate_camera_model_input.py
python3 scripts/generate_camera_model_output.py
```

Windows/PowerShell，安装Python Launcher时可使用：

```powershell
py -3 scripts/generate_camera_model_input.py
py -3 scripts/generate_camera_model_output.py
```

解释器命令不是固定要求：若本机使用`python`或解释器完整路径，请确认其版本后替换命令。
以上Windows示例是调用语法，不代表已完成Windows整套工程验证。导出文件清单及获取边界见[数据说明](../hls/docs/data.md)。

两个命令都会更新相应配置头。更换SD模型时，应从同一次导出的metadata同时生成两份头，
然后重编译PS应用；不得将新BIN与旧scale、anchors或类别名混用。生成器不训练模型、
不修改外部训练工程，也不重新导出BIN。

## 输入数值与布局

- 源视频为640×480 BGR888，抓拍复制`[112,528)×[32,448)`的紧凑BGR ROI。
- 预处理交换B/R，以H→W→RGB写紧凑INT8；不缩放、灰度、反色、mean/std或补8通道。
- 离线参考为`FP32(pixel/255) × FP32(1/input_scale)`，随后最近偶数舍入、INT8饱和。
- 输入生成器枚举256种像素值，寻找32位安全、逐值一致的M/S；不能满足时拒绝更新头。
- PS运行时只进行整数乘法、移位、最近偶数处理及饱和，不使用浮点或查表。

权重、bias及M/S仍由SD的27个BIN加载，不编译为大模型数组。这里的两个小配置头不替代
这些BIN；SD布局及DDR拼接见应用说明。

## 输出解码配置

输出生成器校验416输入、20类、5个anchor、125逻辑/128物理通道及零点0，随后生成
Prediction scale、anchors和类别名称。校验成功后才写入目标文件，不生成指数LUT或
额外SD文件。配置中的具体scale由模型决定，以同批metadata和生成头为准，不是可复用
到任意模型的固定常量。
