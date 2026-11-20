# Vitis 裸机工程

使用 Vitis Classic IDE 2022.2，应用运行在 Zynq Cortex-A9 的 standalone 域。
硬件由 [Vivado 工程](../vivado/README.md)提供；模型、SD 卡和应用行为见
[PS 应用说明](cnn_accel_cam/README.md)。

| 工作区内路径 | 用途 |
|---|---|
| `system_wrapper/` | 平台描述、BSP 配置与 FSBL 源码 |
| `cnn_accel_cam/` | 裸机应用 |
| `cnn_accel_cam_system/` | 应用所在的系统工程 |
| `drivers/` | IP 驱动仓库 |
| `system_wrapper.xsa` | 平台导入的硬件交接文件 |

## 打开已有工作区

选择仓库根目录下的 `vitis/` 作为工作区。已有平台输出、BSP 库和 ELF 时可继续使用；
目录迁移本身不需要重新综合硬件或生成平台。如果 IDE 仍显示旧平台路径，检查它能否解析
`system_wrapper/export/system_wrapper/system_wrapper.xpfm`；必要时给应用和系统工程
重新选择该文件。调试时使用配套的 bitstream、PS 初始化和 ELF。

## 从干净克隆准备平台

1. 启动 Vitis Classic 2022.2，将工作区设为仓库根目录下的 `vitis/`。如果项目未
   出现在 Explorer 中，使用 **File → Import** 导入已有工程，将搜索目录指向该工作区，
   选择 `system_wrapper`、`cnn_accel_cam` 和 `cnn_accel_cam_system`。
   源码已经在工作区内，不需再复制一份或新建空应用。
2. 确定使用哪份 XSA。使用仓库快照时选 `vitis/system_wrapper.xsa`，其中已包含
   bitstream；从 Vivado 重建时，在 `system_wrapper` 平台上选择
   **Update Hardware Specification**，选中[重新导出的 XSA](../vivado/README.md#打开和构建vivado)。
   不要把新 `.bit` 与旧 XSA、PS 初始化或 BSP 混用。
3. 构建 **`system_wrapper` 平台**，生成 BSP 和
   `vitis/system_wrapper/export/system_wrapper/system_wrapper.xpfm`。
   核对 standalone 域运行于 `ps7_cortexa9_0`，BSP 启用 `xilffs` 且 `use_lfn=1`。
   若应用或系统工程找不到旧开发机的 `.xpfm`，在平台关联设置中选择本机刚生成的
   `.xpfm`，并确认使用该 standalone 域。
4. 构建 **`cnn_accel_cam` 应用**，例如选择 Release；产物通常为
   `vitis/cnn_accel_cam/Release/cnn_accel_cam.elf`。若构建报缺少 FatFs 头或库，
   先检查平台导出的 BSP；若报 `expf` 等数学符号未定义，检查应用链接配置是否包含
   `-lm`。应用还需链接 `-lxilffs`。这些是应用项目设置，不在系统项目的 System Builder 里改。

使用配套模型时，应用直接编译已提交的 `camera_model_input.h` 和
`camera_model_output.h`，不需要 Python、模型 BIN 或 `metadata.json` 参与构建。
模型 BIN 由板上程序运行时从 SD 卡加载；更换模型才按[脚本说明](../scripts/README.md)更新头文件。

`system_wrapper/hw/system_wrapper.xsa` 是平台内部的硬件交接副本，同样随工程保留。
`system_wrapper/platform.tcl` 会创建及生成平台，不必为打开已有工作区运行该脚本。

## 首次通过 JTAG 运行

1. 按[应用的 SD 文件清单](cnn_accel_cam/README.md#sd模型文件布局)准备 FAT32 SD 卡：
   将 27 个模型 BIN 放到卡根目录的 `yolo/`。先接好 OV5640、HDMI 显示器、JTAG 和 UART，
   按板卡说明设置 JTAG 启动方式，再给板卡上电。
2. 在应用项目使用 **Debug As → Launch on Hardware (Single Application Debug)**，或创建
   对应的调试配置。检查目标为 Cortex-A9 核 0，选择当前平台对应的 bitstream 与
   PS 初始化脚本（`ps7_init.tcl`），打开 FPGA 编程及 PS 初始化选项，并使用刚构建的 ELF。
   若停在入口断点，继续执行。参见 [AMD 2022.2 裸机调试说明](https://docs.amd.com/r/2022.2-English/ug1400-vitis-embedded/Standalone-Application-Debug-Using-Xilinx-System-Debugger)。
3. 用 UART 115200、8N1 查看[初始化日志](cnn_accel_cam/README.md#uart应看到什么)。
   成功加载应显示 `Model: all 27 files loaded`，随后显示
   `[APP] Continuous YOLO detection/overlay ready`。正常推理不逐帧打印；用已知目标
   检查 HDMI 预览及检测框。没有目标时只显示 ROI 内的实时画面。

JTAG 运行使用 ELF，不需要 `BOOT.BIN`。若要从 SD 自启动，需另行用配套的
FSBL、bitstream 和 ELF 生成 `BOOT.BIN`，放到 SD 卡根目录；它不能替代 `yolo/` 中的模型。

Git 保留源码、工程描述、XSA 和 BSP 的 `system.mss`，不跟踪 IDE 状态、Debug/Release、
平台 export、生成的 BSP 库或日志。忽略规则不会删除本地文件。
