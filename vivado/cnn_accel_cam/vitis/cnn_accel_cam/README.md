# YOLO相机预处理与推理

当前保留640×480摄像头和HDMI预览，启动时从SD一次性加载模型，随后连续执行416×416 ROI抓拍、RGB整数预处理、YOLO推理、解码/NMS，并把检测列表发送给`roi_overlay`。Overlay在视频帧边界整批更新框、类别和置信度；没有目标时清除旧框。软件闭环已连接，板上显示与检测效果仍需实测。

`model_loader.c`负责SD加载；`camera_input.c`负责受保护的ROI复制；`camera_preprocess.c`负责RGB顺序和整数归一化/量化；`camera_cnn.c`负责DDR、单次启动、超时及缓存维护；`camera_postprocess.c`负责解码/NMS；`camera_display.c`负责坐标/文字打包、Overlay命令和确认。没有运行时JSON解析，也没有把大模型编译进ELF。

## SD卡必须如何放文件

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

板上路径是`0:/yolo/conv1_weight.bin`等。文件来源是同一次Python模型导出的：

```text
/home/liaoyingkai/python_project/minist_train_fpga/YOLOV2_TINY/generated/bins/
```

只需要这27个模型BIN；SD模型目录不需要`metadata.json`、NPY、trace或测试图片。不要将`conv1_quant.bin`拆成两个文件，也不要重新打包权重或补通道。所有文件必须来自同一次导出；长度检查不能识别两个相同网络、不同checkpoint之间的混用。

若通过JTAG运行ELF，SD中只需上述模型目录；若从SD启动，根目录仍需匹配当前硬件和应用的`BOOT.BIN`，它与`yolo/`目录独立，不要覆盖原有启动文件。

## 长文件名与“静态工作缓冲”

目前BSP的`system.mss`已经配置`use_lfn=1`，这是所需设置，无需另开一个“静态工作缓冲”开关：

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

PL侧地址单位与具体布局见[DDR设计](../../../../hls/docs/yolov2_tiny_ddr_io.md)。

## ROI与整数预处理

几何统一位于`camera_geometry.h`：视频640×480 BGR888，ROI为`[112,528)×[32,448)`，大小416×416，与Overlay一致。抓拍保留READ停车、两个新MM2S事件和READ/WRITE槽所有权检查；源帧invalidate后按行复制ROI，复制后再次核实所有权/DMA状态，再解除停车。所有停车后出口都解除停车；失败快照不可使用。快照由PS独占，不是VDMA第四帧。

只复制519,168字节紧凑BGR，删除整帧复制和逐字节memcmp诊断。解除停车后才执行归一化，缩短预览定格时间。预处理只做BGR→RGB及点对点量化，直接写Feature A，无INT8整图二次复制：

```text
q = clamp(round_nearest_even(FP32(pixel/255) * FP32(1/input_scale)), -128,127)
```

以上是YOLO训练端`to_tensor`及`dump_trace.quantize_input`的离线参考；运行时使用整数M/S和最近偶数舍入，全部256种像素值必须一致。没有mean/std、灰度、反色、平均池化、resize、LUT或8通道补零。DDR保存紧凑RGB，尾通道由PL Loader补零。

更换SD模型后，在仓库根目录重新生成同批输入系数并重编译：

```bash
python3 vivado/cnn_accel_cam/tools/generate_camera_model_input.py \
  --metadata /path/to/YOLOV2_TINY/generated/metadata.json
```

默认metadata为本机YOLO导出路径，默认输出`src/camera_model_input.h`。生成器穷举256值，无法生成32位安全且严格等价的参数就拒绝更新。当前输入scale为`0.007842701859772205`，生成`M=524317、SHIFT=20`。

## 连续推理及DDR所有权

```text
取得可写A地址 → 安全抓取ROI → 预处理写A → flush输入A → Start
→ 轮询Done → invalidate结果B → 解码/NMS → 上传整批检测并等帧边界确认 → 取下一帧
```

AutoRestart保持关闭，每次只启动一张图。取消原100 ms节拍，但保留安全取帧等待；本次CNN完成前不能写入下一张图。

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

默认配置在`camera_postprocess.h`：score严格大于0.25，同类框IoU严格大于0.45时抑制，最多16个结果。面积按连续坐标计算、不加1。先在归一化坐标中裁剪到[0,1]并丢弃零面积框，再NMS，最后乘416得到ROI内浮点像素坐标，不加视频偏移、不取整。

静态候选区容量`13×13×5×20=16900`，位于PS DDR，不占大栈或PL BRAM，没有预先Top-K截断。候选按分数降序，只与已经保留的同类框比较；16个幸存者后停止。并列分数按anchor→y→x→class序号排序，可确定复现，但不能保证与PyTorch不同设备对同分框的任选顺序一致。Python评估默认阈值0.01、最多300框，对照时应统一阈值、并列规则、零面积规则和最终数量。

`camera_postprocess(prediction, &detections)`返回本帧列表，count为0表示无目标，不沿用上一帧。每项有ROI内left/top/right/bottom、最终score和class_id；`camera_class_name()`返回模型类别名。列表独立于B区，直接交给Overlay控制模块；正常主循环不打印结果。

更换模型时，还需从同一metadata生成输出scale、anchors和类别名：

```bash
python3 vivado/cnn_accel_cam/tools/generate_camera_model_output.py \
  --metadata /path/to/YOLOV2_TINY/generated/metadata.json
```

默认生成`src/camera_model_output.h`，不增加SD文件或运行时JSON解析。两份小型配置头及SD的27个BIN必须来自同一次导出。

## Overlay控制与显示

`camera_display_init()`在HDMI启动后执行，使用BSP生成的`XRoi_overlay`驱动，不硬编码AXI寄存器地址，也不调用Start（该核自由运行）。先读取当前command的sequence，等待视频已见SOF、旧命令已ACK且没有待提交批次，再从该sequence继续递增；随后提交空列表，清除上次运行留下的检测框。这样PS重新运行而PL未复位时，不会直接用sequence=1覆盖尚未完成的命令。

`camera_display_update(&detections)`在发送前检查并组装整批数据：

- 左上边界向下取整，右下边界向上取整，限制在416×416 ROI；加上`(112,32)`得到全视频半开坐标。
- 类别来自模型的20类名称，最多16个ASCII字节。概率是NMS使用的最终`objectness×class_probability`，格式为`87.3%`，不是只有分类Softmax的概率；百分比取整仅影响显示。
- 每框完整上传9个32位word，字符低字节在先；最多16框。显示门限由`camera_postprocess.h`中的score阈值控制。
- 先BEGIN，逐框写9word，最后COMMIT(count)。每条先写payload，再写命令并轮询同sequence的ACK，确认前不发下一条。sequence按16位回绕。

COMMIT的ACK表示新列表已在SOF生效，而非仅表示命令收到。核会在非活动页准备数据，HDMI继续画完整的旧列表，到下一帧再切到新列表，不会逐个露出正在上传的框。PS不等待整个视频帧画完；显示与下一轮抓拍/推理各自运行。无目标也提交count=0，不保留旧框。

核控制依赖视频事务；停流、下游阻塞或没有SOF时命令可能无法确认。初始化及每条命令均有`CAMERA_DISPLAY_TIMEOUT_MS=500`毫秒超时。协议错误或超时后锁住Overlay接口并停止应用，不盲目重发；需检查视频流、时钟、当前bitstream/BSP和复位。非法软件列表在发BEGIN前拒绝，不发布半批数据。

抓拍从正常转为失败时清框一次，重复的取帧竞争不反复上传空列表；恢复后重新推理并显示。CNN/后处理等终止错误也尽力清除旧框；Overlay本身已出错时不再发送清框命令。主循环没有10fps限制，没有逐帧UART输出。

HDMI显示ROI内实时图像、ROI外黑色、绿色检测框，类别及概率在框内左下角；空间不够时由HLS隐藏文字但保留边框。绘图位于VDMA的MM2S输出之后，不写回采集DDR，不影响CNN输入。框来自较早的快照，物体移动时会滞后；帧边界更新只避免半新半旧列表，不保证检测框与当前预览来自同一采集帧。

完整字布局及命令字段见[Overlay设计](../../../../hls/roi_overlay/README.md)。PS和HLS各自维护相同协议，修改其中一端时必须同步。

## 构建与运行边界

1. 使用与当前Vivado设计一致的XSA和bitstream。
2. 在Vitis中Build Platform，确认应用实际使用的导出目录包含`ff.h`、`ffconf.h`以及`libxilffs.a`，而不只是BSP设置界面已勾选。
3. 刷新应用并重新生成构建文件。Debug/Release的链接组均需包含`-lxilffs`和`-lm`。
4. Debug/Release均须编译`camera_preprocess.c`、`camera_postprocess.c`及重写的`camera_display.c`（已取消旧的源文件排除）；不能单独使用历史Debug/Release目录中尚未重生成的Makefile。
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

## 上一轮模型加载软件检查（2026-11-15）

- 当前构建范围的8个应用C文件通过ARM Cortex-A9编译；使用现有链接脚本和Xilinx specs完成ELF链接。只有未修改的`vdma_api.c`保留原有的有符号/无符号比较警告。
- 当时平台导出目录尚缺FatFs头文件和库，检查在独立目录中按当前BSP配置重建`libxilffs.a`，与已导出的`libxil.a`链接；没有改写用户BSP和IDE生成文件。该结果不代表用户平台已经完成Build Platform。
- ELF中的模型、Feature、视频帧和临时读取缓冲均落在DDR，大小、64字节对齐和互不重叠检查通过。
- 主机文件系统模拟检查逐字节核对全部27个真实BIN的权重/参数DDR拼接，覆盖缺文件、大小不符、成功返回但实际短读、读取错误及容量不足。失败块不复制到目标区，挂载只执行一次。该检查没有连接真实SD控制器或CNN。
- 应用工程XML和本次修改范围的差异检查通过；用户原有平台/FSBL改动保持不动。

记录位于仓库根忽略目录`build/ps/sd_model_loader_check_20261115/`，包括ARM ELF/map、独立xilffs库和主机检查；不提交生成物。尚未下载ELF或进行板上SD读取、长文件名访问、HDMI及DDR总线实测。

## 上一轮预处理与推理软件检查（2026-11-15）

- 当前9个应用C文件完成ARM编译和ELF链接，使用当前平台实际导出的BSP及xilffs库；没有改动BSP、RTL或bitstream。
- C输入量化与独立binary32参考对全部256种字节一致；对416×416有规律BGR快照核对全部173,056个像素的RGB输出顺序及缓冲边界。Release对应的`-O2`预处理对象中量化已内联，未见浮点运算或数学helper依赖。
- 抓拍主机模拟核对精确ROI行偏移、两个帧事件、源帧invalidate，以及复制后所有权失败仍解除停车；不等于真实VDMA时序已验证。
- 推理主机模拟核对清旧Done、输入flush、屏障、Start、Done轮询、结果invalidate和屏障的顺序；重复调用均返回B视图，启动后没有额外IsIdle读取。超时后输出为空、输入不可写、不允许再次启动，异常Busy也锁住访问。
- 正常逐帧日志和对应计数器已移除，保留初始化、错误状态变化、恢复和停机日志。最新ELF不含逐帧打印。
- 本轮检查生成物均保存在忽略目录`build/ps/yolo_camera_infer_check_20261115/`；Git工作区只保留必要源码、配置和文档变更，未自动提交。

以上是软件构建和模拟接口检查，尚未下载新ELF或执行真实摄像头→YOLO推理，不声明板上帧率或检测精度。本轮没有运行HLS综合、C-sim或RTL co-sim。

完整检查记录：[validation.log](../../../../build/ps/yolo_camera_infer_check_20261115/validation.log)（本机生成，不提交）。

## 上一轮解码/NMS软件检查（2026-11-15）

- 10个应用C文件使用当前导出BSP、xilffs及libm完成ARM编译和链接；仅未修改的`vdma_api.c`保留原有符号比较警告。没有修改HLS、BSP或bitstream。
- 将真实`trace_prediction_postprocess_int8_hwc_0000.bin`的HWC125数据补成物理HWC128后，C与独立NumPy解码/NMS参考均保留1个检测；类别一致，分数/坐标在`rtol=3e-5、atol=2e-4`范围内一致。当前环境没有PyTorch/torchvision，因此没有直接执行其NMS内核。
- 空结果、已知网格/anchor/通道地址、填充通道污染、多类别候选、同类抑制/异类保留、裁边、零面积过滤、2,535个过阈值候选、16框上限、INT8极值、无效参数及连续调用清空结果检查均通过。
- 模型输出配置重新生成后与提交用头文件逐字节一致。候选数组为473,200字节静态DDR，与模型、Feature、视频缓冲和结果列表互不重叠；当前链接的newlib qsort无堆分配调用。
- 正常循环仍无逐帧UART日志，SD文件组织不变。Git差异检查通过；生成物只在忽略目录`build/ps/yolo_decode_nms_check_20261115/`，没有自动提交。

该轮尚未上板验证后处理耗时、精度或检测效果，当时没有接入Overlay传输。检查记录：[validation.log](../../../../build/ps/yolo_decode_nms_check_20261115/validation.log)。

## Overlay接入软件检查（2026-11-16）

- 当前11个应用C文件使用现有Vitis 2022.2 ARM工具链和BSP完成编译、链接；新增代码无警告，仅保留旧`vdma_api.c:134`的符号比较警告。
- 软件驱动模拟覆盖延迟ACK、旧命令未完成时的启动同步、sequence回绕、16框与空列表、坐标裁剪/取整/偏移、类别/百分比打包、非法列表拒绝、协议错误及超时锁存；再次启用地址和未定义行为检查后通过。
- Debug/Release均重新纳入`camera_display.c`；工程XML、主循环调用顺序、无逐帧UART和Git差异检查通过。现有忽略规则已覆盖全部检查生成物，无需新增忽略项。

上述模拟只验证PS软件的命令组织和错误处理，不验证实际RTL绘图、SOF时序或AXI互联。本轮没有修改HLS/Vivado/BSP，没有运行HLS仿真或综合，也没有下载上板。需在Vitis刷新并重新生成应用构建文件、Build Release，再使用匹配当前硬件的bitstream和ELF检查显示效果。

检查记录：[validation.log](../../../../build/ps/yolo_overlay_check_20261115/validation.log)（目录沿用检查启动时命名，生成物不提交）。
