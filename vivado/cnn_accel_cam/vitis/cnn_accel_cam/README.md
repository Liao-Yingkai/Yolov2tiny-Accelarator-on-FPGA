# YOLO SD模型加载验收

当前阶段保留640×480摄像头和HDMI预览，启动时将YOLOv2-tiny的九层模型从SD一次性加载到DDR，配置CNN三个基址后进入空闲。**不会启动推理，也没有接入新预处理、NMS和roi_overlay列表控制。**

`main.c`调用`camera_cnn_init()`；`camera_cnn.c`管理DDR数组与CNN驱动；`model_loader.c`只负责SD文件读取、长度检查和层顺序拼接。没有运行时JSON解析，也没有把大模型编译进ELF。

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

只需要这27个模型BIN；本阶段不需要`metadata.json`、NPY、trace、测试图片或旧MNIST的头文件。不要将`conv1_quant.bin`拆成两个文件，也不要重新打包权重或补通道。所有文件必须来自同一次导出；长度检查不能识别两个相同网络、不同checkpoint之间的混用。

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

每个Feature bank为692,224字节，B紧接A。后续每帧从A的前519,168字节读紧凑RGB输入，九层完成后在B的前21,632字节得到`13×13×128`的物理输出，其中仅125个逻辑通道有效。当前阶段不填输入、不读输出。

权重按Conv1～Conv8、Prediction顺序直接连接。参数对每层执行`bias.bin → quant.bin`，因为quant本身是完整M数组后接完整S数组，所以每层自然成为`[bias][M][S]`。层与层、字段与字段之间**不额外插入对齐空间**；只有整个参数数组的最后允许补容量。Prediction的参数字段各125个INT32，不按128通道存储。

读取时使用64 KiB对齐临时缓冲分块读取，复制到最终数组；文件大小、读返回值和实际读数都必须正确。每个文件仅打开一次；“启动时一次性加载”不意味着只能调用一次`f_read`。模型全部读完后才做CPU→PL缓存维护并设置基址，后续不再读取SD模型。缓存维护API没有成功/失败返回值，所以不打印虚构的flush成功或失败状态。

PL侧地址单位与具体布局见[DDR设计](../../../../hls/docs/yolov2_tiny_ddr_io.md)。

## 构建与运行边界

1. 使用与当前Vivado设计一致的XSA和bitstream。
2. 在Vitis中Build Platform，确认应用实际使用的导出目录包含`ff.h`、`ffconf.h`以及`libxilffs.a`，而不只是BSP设置界面已勾选。
3. 刷新应用并重新生成构建文件。Debug/Release的链接组均需包含`-lxilffs`。
4. 本阶段在两种配置中排除了旧`camera_preprocess.c`和`camera_display.c`。文件暂留供后续重构，不连接旧灰度/数字显示流程；不能单独使用历史Debug/Release目录中尚未重生成的Makefile。
5. 下载当前bitstream、初始化PS/DDR、运行新ELF；观察HDMI预览及UART。

缺文件、文件长度不符、读取失败、DDR范围/对齐异常或CNN并非Idle都会使初始化失败；不启动CNN、不允许将半加载模型视为有效。已成功初始化时再次调用初始化函数不会重新加载SD。

## UART应看到什么

关键阶段有状态打印，成功过程大致为：

```text
[VIDEO] 640x480 preview ready
DDR weights=0x... params=0x... feature A=0x... B=0x...
SD mount 0: start
SD mount 0: success
Model conv1: weights=1152 params=192 loaded
...（Conv2～Conv8每层各一行）
Model prediction: weights=131072 params=1500 loaded
Model: all 27 files loaded weights=15859328 params=38172
YOLO model ready; accelerator not started
[APP] Model ready; inference is not started in this stage.
```

地址由链接器实际分配，不要求与示例相同；B必须比A大692,224字节。到最后一行后程序空闲，HDMI继续预览，不再周期打印，也不会得到检测结果。这是当前验收入口的预期行为。

文件错误会打印具体路径、FRESULT，或者预期/实际长度与读取偏移；主入口随后打印初始化失败原因。没有逐块读取进度、逐像素日志、正常flush提示或耗时刷屏。

本阶段没有实现预处理、NMS或Overlay命令发送，不要将“model ready”误认为已经完成YOLO相机识别。旧`camera_model_data.h`、`camera_model_input.h`、`softmax_params.h`不再是当前模型加载入口；后续部署的scale等配置也必须与SD模型一致。

## 本轮软件检查（2026-11-15）

- 当前构建范围的8个应用C文件通过ARM Cortex-A9编译；使用现有链接脚本和Xilinx specs完成ELF链接。只有未修改的`vdma_api.c`保留原有的有符号/无符号比较警告。
- 当时平台导出目录尚缺FatFs头文件和库，检查在独立目录中按当前BSP配置重建`libxilffs.a`，与已导出的`libxil.a`链接；没有改写用户BSP和IDE生成文件。该结果不代表用户平台已经完成Build Platform。
- ELF中的模型、Feature、视频帧和临时读取缓冲均落在DDR，大小、64字节对齐和互不重叠检查通过。
- 主机文件系统模拟检查逐字节核对全部27个真实BIN的权重/参数DDR拼接，覆盖缺文件、大小不符、成功返回但实际短读、读取错误及容量不足。失败块不复制到目标区，挂载只执行一次。该检查没有连接真实SD控制器或CNN。
- 应用工程XML和本次修改范围的差异检查通过；用户原有平台/FSBL改动保持不动。

记录位于仓库根忽略目录`build/ps/sd_model_loader_check_20261115/`，包括ARM ELF/map、独立xilffs库和主机检查；不提交生成物。尚未下载ELF或进行板上SD读取、长文件名访问、HDMI及DDR总线实测。
