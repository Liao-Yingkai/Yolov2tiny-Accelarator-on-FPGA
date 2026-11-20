# YOLO相机预处理与推理

应用保留640×480摄像头和HDMI预览，启动时从SD一次性加载模型，随后连续执行416×416 ROI抓拍、RGB整数预处理、YOLO推理、解码/NMS，并把检测列表发送给`roi_overlay`。Overlay在视频帧边界整批更新框、类别和置信度；没有目标时清除旧框。2026-11-19基线已完成测试及上板功能验证。

本文文件路径以仓库根目录为基准，`src/`特指`vitis/cnn_accel_cam/src/`。
构建使用Vitis Classic IDE 2022.2及standalone Cortex-A9 BSP；工作区导入、平台重绑定和
操作系统验证范围见[Vitis 工程说明](../README.md)。本应用运行于板上裸机，不能将
它与开发主机使用Linux还是Windows混为一谈，也不能作为普通桌面程序直接运行。

`model_loader.c`负责SD加载；`camera_input.c`负责受保护的ROI复制；`camera_preprocess.c`负责RGB顺序和整数归一化/量化；`camera_cnn.c`负责DDR、单次启动、超时及缓存维护；`camera_postprocess.c`负责解码/NMS；`camera_display.c`负责坐标/文字打包、Overlay命令和确认。没有运行时JSON解析，也没有把大模型编译进ELF。

## SD模型文件布局

使用当前xilffs支持的分区格式，建议FAT32。在SD卡根目录创建小写`yolo`文件夹，将以下27个文件直接放进去，不要再套一层`bins`目录：

```text
SD卡根目录/
└── yolo/
    ├── conv1_weight.bin
    ├── conv1_bias.bin
    ├── conv1_quant.bin
    ├── conv2_weight.bin
    ├── conv2_bias.bin
    ├── conv2_quant.bin
    ├── conv3_weight.bin
    ├── conv3_bias.bin
    ├── conv3_quant.bin
    ├── conv4_weight.bin
    ├── conv4_bias.bin
    ├── conv4_quant.bin
    ├── conv5_weight.bin
    ├── conv5_bias.bin
    ├── conv5_quant.bin
    ├── conv6_weight.bin
    ├── conv6_bias.bin
    ├── conv6_quant.bin
    ├── conv7_weight.bin
    ├── conv7_bias.bin
    ├── conv7_quant.bin
    ├── conv8_weight.bin
    ├── conv8_bias.bin
    ├── conv8_quant.bin
    ├── prediction_weight.bin
    ├── prediction_bias.bin
    └── prediction_quant.bin
```

板上路径是`0:/yolo/conv1_weight.bin`等，`0:`是FatFs卷号，不是Windows盘符。所需27个BIN
来自同一次兼容模型导出；在本仓库默认数据目录中位于：

```text
data/yolov2_tiny/bins/
```

只需要这27个模型BIN；SD模型目录不需要`metadata.json`、NPY、trace或测试图片。不要将`conv1_quant.bin`拆成两个文件，也不要重新打包权重或补通道。所有文件必须来自同一次导出；长度检查不能识别两个相同网络、不同checkpoint之间的混用。

这些BIN不随Git克隆提供，数据来源与准备边界见[模型与测试数据](../../hls/docs/data.md)。
可使用主机文件管理器复制到SD，过程不要求特定Shell；不要把`bins/`目录本身嵌套到`yolo/`下。

若通过JTAG运行ELF，SD中只需上述模型目录；若从SD启动，根目录仍需匹配当前硬件和应用的`BOOT.BIN`，它与`yolo/`目录独立，不要覆盖原有启动文件。

## FatFs与长文件名

保留的BSP配置要求`use_lfn=1`。导入或重新生成平台后，应确认实际构建的xilffs采用该值：

- `0`：关闭长文件名，不能按上述长文件名访问。
- `1`：开启长文件名，文件名处理缓冲放在静态内存中，适合当前单线程裸机程序。
- `2/3`：分别将工作缓冲放到栈/堆；本工程没有必要使用。

该缓冲只保存文件名处理信息，不是模型、图像或SD卡缓存。不要手改生成的`ffconf.h`。修改BSP设置后需要Build Platform，使导出的头文件和库真正更新。

## DDR布局

三个数组均以64字节对齐，由应用链接脚本放入DDR，无需手动指定绝对地址；PS与PL各自维护相同布局约定。

| 区域 | 有效字节数 | 实际分配字节数 |
| --- | ---: | ---: |
| 权重 | 15,859,328 | 15,859,328 |
| bias/M/S参数 | 38,172 | 38,208 |
| Feature A+B | 1,384,448 | 1,384,448 |

每个Feature bank为692,224字节，B紧接A。预处理直接写A的前519,168字节，九层完成后在B的前21,632字节得到`13×13×128`物理输出，其中仅125个逻辑通道有效，交由YOLO解码/NMS处理。

权重按Conv1～Conv8、Prediction顺序直接连接。参数对每层执行`bias.bin → quant.bin`，因为quant本身是完整M数组后接完整S数组，所以每层自然成为`[bias][M][S]`。层与层、字段与字段之间**不额外插入对齐空间**；只有整个参数数组的最后允许补容量。Prediction的参数字段各125个INT32，不按128通道存储。

读取时使用64 KiB对齐临时缓冲分块读取，复制到最终数组；文件大小、读返回值和实际读数都必须正确。每个文件仅打开一次；“启动时一次性加载”不意味着只能调用一次`f_read`。模型全部读完后才做CPU→PL缓存维护并设置基址，后续不再读取SD模型。缓存维护API没有成功/失败返回值，所以不打印虚构的flush成功或失败状态。

PL侧地址单位与具体布局见[DDR设计](../../hls/docs/yolov2_tiny_ddr_io.md)。

## ROI与整数预处理

几何统一位于`camera_geometry.h`：视频640×480 BGR888，ROI为`[112,528)×[32,448)`，大小416×416，与Overlay一致。抓拍保留READ停车、两个新MM2S事件和READ/WRITE槽所有权检查；源帧invalidate后按行复制ROI，复制后再次核实所有权/DMA状态，再解除停车。所有停车后出口都解除停车；失败快照不可使用。快照由PS独占，不是VDMA第四帧。

只复制519,168字节紧凑BGR，不复制整帧或逐字节memcmp。解除停车后才执行归一化，缩短预览定格时间。预处理只做BGR→RGB及点对点量化，直接写Feature A，无INT8整图二次复制：

```text
q = clamp(round_nearest_even(FP32(pixel/255) * FP32(1/input_scale)), -128,127)
```

以上是YOLO训练端`to_tensor`及`dump_trace.quantize_input`的离线参考；运行时使用整数M/S和最近偶数舍入，全部256种像素值必须一致。没有mean/std、灰度、反色、平均池化、resize、LUT或8通道补零。DDR保存紧凑RGB，尾通道由PL Loader补零。

更换SD模型后，使用同批metadata运行`generate_camera_model_input.py`并重编译应用。
Python解释器、Linux/Bash与Windows/PowerShell调用示例见[配置生成说明](../../scripts/README.md)。

默认读取仓库内`data/yolov2_tiny/metadata.json`，根据脚本位置定位，无需环境变量；可通过 `--metadata` 指定输入文件。默认输出`src/camera_model_input.h`。生成器穷举256值，无法生成32位安全且严格等价的参数就拒绝更新。当前输入scale为`0.007842701859772205`，生成`M=524317、SHIFT=20`。数据组织见[模型与测试数据](../../hls/docs/data.md)。

## 连续推理及DDR所有权

```text
取得可写A地址 → 安全抓取ROI → 预处理写A → flush输入A → Start
→ 轮询Done → invalidate结果B → 解码/NMS → 上传整批检测并等帧边界确认 → 取下一帧
```

CNN的AutoRestart保持关闭，每次只启动一张图；不设固定10fps节拍，但保留安全取帧等待。
本次CNN完成前不能写入下一张图。

输出是B中的只读视图，按`(y*13+x)*128+c`定位，`c<125`。不转置，也不复制成紧凑125通道；视图只在本次完成到下一次启动之间有效。当前解码/NMS在复用Feature前同步消费完，结果复制到独立的`detections`列表。

Start前清旧Done，启动后只轮询`IsDone`，避免其他控制寄存器读取偷清COR完成事件。超时阈值为10秒；推理超时或异常Busy后锁住缓冲访问并停止程序，不自动复位或继续写A。

## 检测解码与NMS

每个网格的逻辑通道按`25*anchor+field`解释，字段是`tx,ty,tw,th,objectness,20类logits`。只读取0..124通道，忽略125..127填充。使用Prediction输出scale反量化；当前scale为`0.19452644884586334`、zero_point为0。

```text
center_xy = (sigmoid(tx,ty) + grid_xy) / 13
width_height = exp(tw,th) * anchor_width_height / 13
score[class] = sigmoid(objectness) * softmax(class_logits)[class]
```

PS用float、expf及稳定sigmoid/softmax计算，不改变PL量化规则，不增加PL DSP/BRAM占用。Softmax先减最大值，objectness不超过阈值时安全跳过，至少一个类别过阈值后才计算框。保留所有过阈值类别，不简化为每个anchor取一个argmax。

当前配置在`camera_postprocess.h`：score严格大于0.55，同类框IoU严格大于0.55时抑制，最多8个结果。没有单独的类别概率门限；屏幕上的概率文本表示`objectness×class_probability`。这些是PS软件配置，修改后重编译应用，不需重新综合PL。面积按连续坐标计算、不加1。先在归一化坐标中裁剪到[0,1]并丢弃零面积框，再NMS，最后乘416得到ROI内浮点像素坐标，不加视频偏移、不取整。

静态候选区容量`13×13×5×20=16900`，位于PS DDR，不占大栈或PL BRAM，没有预先Top-K截断。候选按分数降序，只与已经保留的同类框比较；达到`CAMERA_POSTPROCESS_MAX_DETECTIONS`（当前8）后停止。并列分数按anchor→y→x→class序号排序，可确定复现，但不能保证与PyTorch不同设备对同分框的任选顺序一致。与Python对照时应统一阈值、并列规则、零面积规则和最终数量，不直接套用评估程序的默认配置。

`camera_postprocess(prediction, &detections)`返回本帧列表，count为0表示无目标，不沿用上一帧。每项有ROI内left/top/right/bottom、最终score和class_id；`camera_class_name()`返回模型类别名。列表独立于B区，直接交给Overlay控制模块；正常主循环不打印结果。

更换模型时，还需从同一metadata运行`generate_camera_model_output.py`生成输出scale、
anchors和类别名；调用方式及可选参数见[配置生成说明](../../scripts/README.md)。

默认生成`src/camera_model_output.h`，不增加SD文件或运行时JSON解析。两份小型配置头及SD的27个BIN必须来自同一次导出。

## Overlay控制与显示

当前Overlay使用`ap_ctrl_none`自由运行及显式`PIPELINE II=1 style=flp`，不需要Start或AutoRestart；命令仍通过sequence ACK确认。此前为co-sim临时加入启动控制的接口不再使用，重新生成硬件时须保证IP、XSA、BSP和bitstream配套。详见[Overlay说明](../../hls/docs/roi_overlay.md)。

`camera_display_init()`在HDMI启动后执行，使用BSP生成的`XRoi_overlay`驱动，不硬编码AXI寄存器地址，也不调用Start（该核自由运行）。先读取当前command的sequence，等待视频已见SOF、旧命令已ACK且没有待提交批次，再从该sequence继续递增；随后提交空列表，清除上次运行留下的检测框。这样PS重新运行而PL未复位时，不会直接用sequence=1覆盖尚未完成的命令。

`camera_display_update(&detections)`在发送前检查并组装整批数据：

- 左上边界向下取整，右下边界向上取整，限制在416×416 ROI；加上`(112,32)`得到全视频半开坐标。
- 类别来自模型的20类名称，最多16个ASCII字节。概率是NMS使用的最终`objectness×class_probability`，格式为`87.3%`，不是只有分类Softmax的概率；百分比取整仅影响显示。
- 每框完整上传9个32位word，字符低字节在先；Overlay容量为16框，当前PS后处理最多输出8框。显示门限由`camera_postprocess.h`中的score阈值控制。
- 先BEGIN，逐框写9word，最后COMMIT(count)。每条先写payload，再写命令并轮询同sequence的ACK，确认前不发下一条。sequence按16位回绕。

COMMIT的ACK表示新列表已在SOF生效，而非仅表示命令收到。核会在非活动页准备数据，HDMI继续画完整的旧列表，到下一帧再切到新列表，不会逐个露出正在上传的框。PS不等待整个视频帧画完；显示与下一轮抓拍/推理各自运行。无目标也提交count=0，不保留旧框。

核控制依赖视频事务；停流、下游阻塞或没有SOF时命令可能无法确认。初始化及每条命令均有`CAMERA_DISPLAY_TIMEOUT_MS=500`毫秒超时。协议错误或超时后锁住Overlay接口并停止应用，不盲目重发；需检查视频流、时钟、当前bitstream/BSP和复位。非法软件列表在发BEGIN前拒绝，不发布半批数据。

抓拍从正常转为失败时清框一次，重复的取帧竞争不反复上传空列表；恢复后重新推理并显示。CNN/后处理等终止错误也尽力清除旧框；Overlay本身已出错时不再发送清框命令。主循环没有10fps限制，没有逐帧UART输出。

HDMI显示ROI内实时图像、ROI外黑色、绿色检测框，类别及概率在框内左下角；空间不够时由HLS隐藏文字但保留边框。绘图位于VDMA的MM2S输出之后，不写回采集DDR，不影响CNN输入。框来自较早的快照，物体移动时会滞后；帧边界更新只避免半新半旧列表，不保证检测框与当前预览来自同一采集帧。

完整字布局及命令字段见[Overlay设计](../../hls/docs/roi_overlay.md)。PS和HLS各自维护相同协议，修改其中一端时必须同步。

## 构建与运行边界

干净克隆后的平台导入、应用构建和 JTAG 下载顺序见 [Vitis 首次使用](../README.md)。
复现配套模型时保留仓库已有的两个模型配置头，**不运行生成脚本**。
若 UART 报 SD 挂载或文件错误，优先检查分区和 `yolo/` 目录结构；不要通过重建 FPGA
来处理模型 BIN 缺失或长度不符。

1. 使用与当前Vivado设计一致的XSA和bitstream。
2. 在Vitis中Build Platform，确认应用实际使用的导出目录包含`ff.h`、`ffconf.h`以及`libxilffs.a`，而不只是BSP设置界面已勾选。
3. 刷新应用并重新生成构建文件。Debug/Release的链接组均需包含`-lxilffs`和`-lm`。
4. Debug/Release均须编译`camera_preprocess.c`、`camera_postprocess.c`及`camera_display.c`；
   重新生成应用构建文件，不依赖其他工作区留下的Makefile。链接选项设置在应用项目
   `cnn_accel_cam`，不是系统项目`cnn_accel_cam_system`的System Builder。
5. 下载当前bitstream、初始化PS/DDR、运行新ELF；观察HDMI预览及UART。

实际连续运行建议使用Release（现有配置为`-O2`）；Debug的`-O0`适合单步检查，不适合作为预处理性能结果。不要启用fast-math来改变数值约定。

缺文件、文件长度不符、读取失败、DDR范围/对齐异常或CNN并非Idle都会使初始化失败；不启动CNN、不允许将半加载模型视为有效。已成功初始化时再次调用初始化函数不会重新加载SD。

## UART应看到什么

关键阶段有状态打印，成功过程大致为：

```text
[VIDEO] 640x480 preview ready
[OVERLAY] ready
DDR weights=0x... params=0x... feature A=0x... B=0x...
SD mount 0: start
SD mount 0: success
Model conv1: weights=1152 params=192 loaded
...（Conv2～Conv8每层各一行）
Model prediction: weights=131072 params=1500 loaded
Model: all 27 files loaded weights=15859328 params=38172
YOLO model ready; accelerator not started
[APP] Continuous YOLO detection/overlay ready
```

地址由链接器实际分配，B必须比A大692,224字节。初始化本身不启动CNN，随后主循环推理、解码/NMS并上传Overlay。正常循环不打印每帧完成、计数、张量、检测列表或耗时；串口保持安静不代表程序停止。只在初始化、错误状态变化、恢复或停机时打印信息。

文件错误会打印具体路径、FRESULT，或者预期/实际长度与读取偏移；主入口随后打印初始化失败原因。没有逐块读取进度、逐像素日志、正常flush提示或耗时刷屏。

`camera_model_input.h`保存YOLO输入系数，`camera_model_output.h`保存解码配置；模型权重及bias/M/S从SD加载。
