# Vivado 与 Vitis 工程

本目录包含摄像头/HDMI硬件工程和Zynq裸机应用。验证基线使用
`xc7z020clg400-2`、Vivado/Vitis 2022.2，已完成摄像头、YOLO推理、解码/NMS及显示功能验证。
本文路径均相对于仓库根目录；模型、SD文件和应用流程见
[PS应用说明](cnn_accel_cam/vitis/cnn_accel_cam/README.md)。

## 环境与硬件前提

使用Vivado和Vitis Classic IDE 2022.2，不将新版Vitis Unified IDE的菜单和工程格式
混用于本流程。GUI操作以相应工具菜单为入口，不要求Linux终端；Windows构建迁移尚未在
本仓库验证。工具的系统要求应按AMD对应版本文档检查，不能由本项目的一次验证推及所有系统。
[Vitis启动与环境设置](https://docs.amd.com/r/2022.2-English/ug1400-vitis-embedded/Setting-Up-the-Environment-to-Run-the-Vitis-Software-Platform)
分别说明Linux shell与Windows桌面/命令行入口。

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
| Vitis工作区 | `vivado/cnn_accel_cam/vitis/` |

`ip_repo`内HDL、`component.xml`、XGUI、驱动和`.dat`初始化文件都是当前BD的实际依赖，
不能因其由HLS生成就当作缓存删除。现有导出IP可用于Vivado重建；只有修改HLS实现时才
需要重新综合、导出并替换相应IP，不直接手改导出HDL。

## 打开和构建Vivado

1. 启动Vivado 2022.2，通过Open Project选择上表中的XPR。
2. 检查当前项目器件为`xc7z020clg400-2`，IP Repository解析到本工程的`ip_repo`，
   源文件列表中没有实际缺失的BD、wrapper或约束。
3. 打开`system` BD，先检查Report IP Status，再执行Validate Design。
   如有locked IP，按报告中的锁定原因处理，不直接删除互联或盲目升级所有IP。
4. 按需Generate Output Products，再执行综合、实现及生成bitstream。
5. 导出包含bitstream的XSA，更新Vitis平台后再重建应用；不要只替换bitstream而保留
   不配套的硬件描述、驱动或初始化代码。

部分XPR字段保存旧ImportPath、备用run或未启用增量缓存的历史位置，不应与当前实际
文件依赖混为一谈。移植时先检查实际Source/IP Repository解析结果，不因看到旧目录名
就改器件、连接或时钟。更换工具版本、板卡或时钟后，需要重新验证时序与功能。

CNN由PS逐次Start；Overlay采用`ap_ctrl_none`自由运行和显式FLP流水，不调用
Start/AutoRestart。两者不是同一种控制方式。

## 导入和构建Vitis

工作区为`vivado/cnn_accel_cam/vitis/`，项目名称如下：

| 类型 | 名称 |
|---|---|
| Platform | `system_wrapper` |
| Application | `cnn_accel_cam` |
| System project | `cnn_accel_cam_system` |

1. 在Vitis Classic IDE 2022.2中选择该工作区。若Explorer没有这些项目，通过项目
   Import入口导入现有平台、应用及系统工程，不创建空应用替代源码。
2. 选中平台`system_wrapper`，检查其硬件描述。需要更新时，右键平台选择
   Update Hardware Specification，并选择当前硬件对应的XSA。
   [AMD硬件更新流程](https://docs.amd.com/r/2022.2-English/ug1400-vitis-embedded/Updating-the-Hardware-Specification)
   指定了此平台级操作。
3. Build Platform，生成BSP及
   `vivado/cnn_accel_cam/vitis/system_wrapper/export/system_wrapper/system_wrapper.xpfm`。
4. 迁移目录后，检查应用和系统工程关联的平台，重新选择上一步本地生成的`.xpfm`。
   `.prj/.sprj`保留过旧开发路径，而`platform.spr`使用平台目录占位表达式；
   不能把它们一概视为自动可迁移，也不靠硬编码旧include路径绕过平台生成。
5. 检查standalone域的BSP启用了`xilffs`、`use_lfn=1`；应用的Debug/Release链接配置
   包含`xilffs`及`m`。应设置应用项目的编译/链接选项，不是系统项目的System Builder。
6. Build Release。使用匹配当前硬件的bitstream、PS初始化和新ELF运行；
   JTAG调试与SD自启动是不同方式，SD模型与启动镜像的关系见应用说明。

两份硬件交接文件分别为
`vivado/cnn_accel_cam/vitis/system_wrapper.xsa`和
`vivado/cnn_accel_cam/vitis/system_wrapper/hw/system_wrapper.xsa`。
更新硬件时须保持平台使用的交接内容一致；文件名相同不证明内容配套。
FSBL源码、hook、链接脚本、初始化文件、平台描述和BSP的`system.mss`均保留在源码管理中。

`platform.tcl`按脚本位置定位硬件文件，但它是会创建/生成平台的脚本，不是只读恢复检查。
已有平台优先按上述GUI流程更新；不在仍使用的工作区中反复source。当前没有提供覆盖
所有主机系统、路径和IDE状态的一键重建保证。

## 源文件与生成物

版本管理保留XPR、SRCs、自定义IP、两份XSA和Vitis必要源码/描述。以下内容由各自
开发环境生成，不随Git提供：

- Vivado的`.cache/.runs/.sim/.hw/.ip_user_files/.gen`、日志及`.Xil`。
- Vitis工作区状态、`Debug/Release`、`_ide/_platform`、平台`export/logs`。
- BSP自动生成的头、库、驱动工作副本、编译产物及硬件解包副本（上述XSA除外）。
- PS配置器摘要报告、Python缓存和旧测试CSV。

因此克隆后需要Build Platform及生成Output Products，不能依赖开发者原有缓存。
忽略规则不会删除本地文件；必要工程配置仍跟踪，重建后应审查其diff，不统一忽略所有
modified文件，也不使用`assume-unchanged`隐藏真实配置变更。
