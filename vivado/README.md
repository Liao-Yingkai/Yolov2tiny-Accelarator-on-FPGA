# Vivado 硬件工程

本目录包含摄像头、HDMI 与 YOLO 加速器的硬件工程。验证基线使用
`xc7z020clg400-2`、Vivado/Vitis 2022.2，已完成摄像头、YOLO推理、解码/NMS及显示功能验证。
本文路径均相对于仓库根目录；模型、SD文件和应用流程见
[Vitis 工程说明](../vitis/README.md)。

## 环境与硬件前提

使用 Vivado 2022.2。Vitis 是仓库根目录下的独立工程，两者通过配套 XSA 交接。

板级工程包含OV5640、640×480视频、VDMA、DDR、DVI/HDMI输出及两个HLS IP。器件型号相同
不等于板卡相同：使用其他板卡前须核对电源/接口、PS DDR/MIO配置、时钟和
`vivado/cnn_accel_cam/cnn_accel_cam.srcs/constrs_1/new/LCD.xdc`的引脚约束。
这些配置不是通用Zynq板卡模板。

## 工程文件分工

| 内容 | 仓库内位置 |
|---|---|
| Vivado入口 | `vivado/cnn_accel_cam/cnn_accel_cam.xpr` |
| BD、IP配置、约束及布局 | `vivado/cnn_accel_cam/cnn_accel_cam.srcs/` |
| XPR实际引用的wrapper | `vivado/cnn_accel_cam/cnn_accel_cam.srcs/sources_1/imports/system_wrapper.v` |
| 自定义IP实体 | `vivado/cnn_accel_cam/ip_repo/` |

`ip_repo`内HDL、`component.xml`、XGUI、驱动和`.dat`初始化文件都是当前BD的实际依赖，
不能因其由HLS生成就当作缓存删除。现有导出IP可用于Vivado重建；只有修改HLS实现时才
需要重新综合、导出并替换相应IP，不直接手改导出HDL。

## 打开和构建Vivado

如果只想运行仓库已有硬件，且板卡和硬件配置匹配，可以跳过本节，直接把
[`vitis/system_wrapper.xsa`](../vitis/system_wrapper.xsa) 交给 Vitis。该 XSA 已包含
`system_wrapper.bit`；仓库未单独提交 Vivado 实现缓存或独立的 `.bit` 输出目录。

1. 启动Vivado 2022.2，通过Open Project选择上表中的XPR。
2. 检查当前项目器件为`xc7z020clg400-2`，IP Repository解析到本工程的`ip_repo`，
   源文件列表中没有实际缺失的BD、wrapper或约束。
3. 打开`system` BD，先检查Report IP Status，再执行Validate Design。
   如有locked IP，按报告中的锁定原因处理，不直接删除互联或盲目升级所有IP。
4. 对 `system` BD 执行 **Generate Output Products**。保留工程已引用的
   `system_wrapper.v`；确认顶层是 `system_wrapper`，不需要另建一个同名 wrapper。
5. 在 Flow Navigator 执行 **Generate Bitstream**。未完成的综合和实现会先运行；
   检查运行状态、关键警告和实现时序报告。bitstream 通常位于
   `vivado/cnn_accel_cam/cnn_accel_cam.runs/impl_1/system_wrapper.bit`（对应默认实现 run）。
6. 选择 **File → Export → Export Hardware**，选择 **Include bitstream**，将 XSA
   导出到独立的本地构建目录，例如 `build/ps/handoff/system_wrapper.xsa`。不要为一次本地试验
   直接覆盖仓库快照。将这份新 XSA 通过 Vitis 平台的 **Update Hardware Specification**
   导入，再重新构建平台、BSP 和应用。参见 [AMD 2022.2 XSA 导出说明](https://docs.amd.com/r/2022.2-English/ug1400-vitis-embedded/Creating-a-Hardware-Design-XSA-File)。

部分XPR字段保存旧ImportPath、备用run或未启用增量缓存的历史位置，不应与当前实际
文件依赖混为一谈。移植时先检查实际Source/IP Repository解析结果，不因看到旧目录名
就改器件、连接或时钟。更换工具版本、板卡或时钟后，需要重新验证时序与功能。

CNN由PS逐次Start；Overlay采用`ap_ctrl_none`自由运行和显式FLP流水，不调用
Start/AutoRestart。两者不是同一种控制方式。

## 硬件交接与版本管理

重新生成硬件后，导出包含 bitstream 的 XSA，按 [Vitis 工程说明](../vitis/README.md)
更新平台。只移动软件工作区不需要重新综合、实现或导出硬件。

Git 保留 XPR、SRCs 和工程引用的自定义 IP。缓存、运行输出和日志仅保留在本地。
