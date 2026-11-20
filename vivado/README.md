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

## 硬件交接与版本管理

重新生成硬件后，导出包含 bitstream 的 XSA，按 [Vitis 工程说明](../vitis/README.md)
更新平台。只移动软件工作区不需要重新综合、实现或导出硬件。

Git 保留 XPR、SRCs 和工程引用的自定义 IP。缓存、运行输出和日志仅保留在本地。
