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

1. 在 Vitis Classic 中导入现有平台、应用和系统工程。
2. 平台使用工作区中的 `system_wrapper.xsa`。硬件发生变化时，使用平台的
   **Update Hardware Specification** 选择新 XSA。
3. Build Platform 生成 BSP 与 `system_wrapper/export/system_wrapper/system_wrapper.xpfm`，
   然后给应用与系统工程选择该本地平台。IDE 元数据可能保存开发机的绝对路径，不要直接使用旧路径。
4. 确认 BSP 启用 `xilffs` 且 `use_lfn=1`，应用链接配置包含 `xilffs` 和 `m`。
   构建应用后使用匹配的 bitstream 和 ELF。SD 卡模型 BIN 需另行准备。

`system_wrapper/hw/system_wrapper.xsa` 是平台内部的硬件交接副本，同样随工程保留。
`system_wrapper/platform.tcl` 会创建及生成平台，不必为打开已有工作区运行该脚本。

Git 保留源码、工程描述、XSA 和 BSP 的 `system.mss`，不跟踪 IDE 状态、Debug/Release、
平台 export、生成的 BSP 库或日志。忽略规则不会删除本地文件。
