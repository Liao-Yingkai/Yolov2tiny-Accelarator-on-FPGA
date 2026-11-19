# Vivado 与 Vitis 工程

当前工程已完成摄像头 → YOLO 推理 → 解码/NMS → HDMI 叠加的上板功能验证
（2026-11-19，用户确认）。硬件目标为 `xc7z020clg400-2`，使用 Vivado / Vitis
2022.2。模型、SD 文件及应用流程见 [PS 应用说明](cnn_accel_cam/vitis/cnn_accel_cam/README.md)。

## 保留的工程结构

- `cnn_accel_cam/cnn_accel_cam.xpr`：Vivado 工程入口。
- `cnn_accel_cam/cnn_accel_cam.srcs/`：BD、IP 配置、约束和图形布局。
- `cnn_accel_cam/cnn_accel_cam.srcs/sources_1/imports/system_wrapper.v`：XPR 实际引用的 wrapper。
- `cnn_accel_cam/ip_repo/`：完整的 CNN、Overlay、摄像头和 DVI 自定义 IP。
- `cnn_accel_cam/vitis/`：现有平台、应用和系统工程；不另建一套目录结构。

`ip_repo` 内的 HDL、`component.xml`、XGUI、驱动及 `.dat` 初始化文件是当前 BD 的
实际依赖。即使由 HLS 生成，也不是可以随意删除的缓存；修改 HLS 后须重新导出 IP，
不能手改已导出 HDL 来替代源码。

## 打开 Vivado

1. 用 Vivado 2022.2 打开 `cnn_accel_cam/cnn_accel_cam.xpr`。
2. 检查项目器件为 `xc7z020clg400-2`，IP Repository 指向本工程的 `ip_repo`。
3. 打开 `system` BD，检查 IP Status，再 Validate Design。
4. 首次打开时按需 Generate Output Products，然后运行综合、实现及 bitstream 生成。
5. 若修改了硬件，重新导出包含 bitstream 的 XSA，并同步 Vitis 平台。

实际 BD、wrapper、XDC 和 IP 仓库均在工程内。XPR 中旧的 ImportPath、备用 run
或未启用的增量缓存目录是迁移记录，不是缺失源文件；不要为消除这些记录而改器件、
连接或时钟。遇到 locked IP，先看 Report IP Status 的具体原因，不要直接删除互联。
保留同版本工具，避免无意升级 IP 改变已验证配置。

CNN 由 PS 逐次 Start；Overlay 是 `ap_ctrl_none` 自由运行、显式 FLP 流水，不调用
Start/AutoRestart。两者不是同一种控制方式。

## 打开 Vitis

现有项目名称为：

| 类型 | 名称 |
|---|---|
| Platform | `system_wrapper` |
| Application | `cnn_accel_cam` |
| System project | `cnn_accel_cam_system` |

1. 在 Vitis 2022.2 中使用 `cnn_accel_cam/vitis/` 作为工作区，导入现有平台及应用。
2. 确认平台硬件指向本地的 `system_wrapper.xsa`，必要时 Update Hardware Specification。
3. Build Platform，生成 BSP、库和 `export/system_wrapper/system_wrapper.xpfm`。
4. **迁移目录后重新选择应用及系统工程关联的平台**。IDE 的 `.prj/.sprj` 保存过旧机器
   的绝对平台路径，不能把缺失的旧 `.xpfm` 当成缺失源码，也不靠硬编码 include 路径绕过。
5. 检查 BSP 启用 `xilffs`，长文件名 `use_lfn=1`；应用链接包含 `xilffs` 和 `m`。
   Build Release 后，使用配套 bitstream、PS 初始化和新 ELF 运行。

`platform.tcl` 的硬件路径已改为相对于脚本位置解析，原平台生成步骤和 BSP 配置保持
不变。该文件是平台创建/生成脚本，不要在正在使用的平台上反复 source；已有平台优先
通过 IDE 更新、构建。本轮整理没有执行平台重建或验证全新路径下的 IDE 构建。

两个 `system_wrapper.xsa` 分别位于工作区根和 `system_wrapper/hw/`；平台描述会引用
它们，因此都保留。整理时两者内容一致。硬件、XSA、驱动与应用必须配套；不要只更新
其中一个。FSBL 源码、hook、链接脚本、初始化文件、平台描述及 BSP 的 `system.mss`
继续保留，不把它们按“生成文件”整体移除。

## 源文件与生成物

版本管理保留 XPR、完整 SRCs、完整自定义 IP、两份 XSA，以及 Vitis 的源文件、工程
描述、平台设置和 FSBL 源码。以下内容由本机重建，不纳入版本管理：

- Vivado 的 `.cache/.runs/.sim/.hw/.ip_user_files/.gen`、日志及 `.Xil`。
- Vitis 的工作区状态、`Debug/Release`、`_ide/_platform`、平台 `export/logs`。
- BSP 自动生成的头、库、驱动工作副本、编译产物和平台硬件解包副本（上述 XSA 除外）。
- PS 配置器的 `NA/ps7_summary.html` 等摘要报告、Python 缓存及旧测试的 CSV 输出。

这些忽略规则不删除已有本地文件。已有跟踪的生成报告单独取消跟踪；不清空本机工作区，
不改写 Git 历史。`.prj/.sprj/.cproject/platform.spr` 等必要配置仍跟踪，不使用
`assume-unchanged` 隐藏真正的设置变化。再次生成工程后，应审查必要配置的 diff，
不能以“所有 modified 都是垃圾”为理由统一忽略。
