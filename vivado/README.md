# Vivado 工程

安装 Vivado 2022.2，打开 `cnn_accel_ddr/cnn_accel_ddr.xpr`。首次打开后，为 `system` Block Design 重新生成 Output Products，再运行综合；缓存、综合、实现和仿真输出由本机生成，不纳入 Git。

版本管理保留以下内容：

- `.xpr` 工程文件与完整 `.srcs` 目录，包括 Block Design、IP 配置（`.xci`）及界面布局。
- `cnn_accel_ddr/ip_repo/export` 中完整的自定义 IP，包括 `component.xml`、HDL、驱动、XGUI 和数据文件。
- `cnn_accel_ddr/cnn_accel_ddr.gen/sources_1/bd/system/hdl/system_wrapper.v`。虽然它位于生成目录中，但当前 XPR 直接引用这一位置，因此原地保留此文件；其余 `.gen` 内容忽略并按需重生成。

`.cache`、`.runs`、`.sim`、`.hw`、`.ip_user_files` 及工具日志和临时缓存不纳入 Git。忽略规则不会删除本地已有产物。

此目录用于保留可打开的工程及所需源文件，不代表已经完成时序收敛或上板验证。请使用同版本 Vivado，避免自动升级 IP 带来的差异。

## Vitis 工作区版本管理

`cnn_accel_ddr/vitis` 与 `cnn_accel_cam/vitis` 均保留应用 `src/`（包括模型数据头、链接脚本、`Xilinx.spec` 和说明）、工具脚本、`.project`/`.cproject`/`.prj`、系统 `.sprj`、平台 `platform.spr`/`platform.tcl` 及小型资源配置。规则按工作区和平台目录层级匹配，不依赖 `system_wrapper` 的固定名称，也覆盖 `system_wrappe` 等平台命名变体；不排除其他平台的源码或工程描述。
保留工作区根 `system_wrapper.xsa` 和平台内 `system_wrapper/hw/system_wrapper.xsa` 两份硬件交接文件；后者由 `platform.spr` 的 `platIntHandOff` 引用，不能随解包目录一并排除。两个 BSP 的 `system.mss` 配置也保留。

忽略 IDE 状态（`.workspace`、`.metadata`、`.analytics`、`IDE.log`、`RemoteSystemsTempFiles`）、各级 `Debug`/`Release`、`_ide` 复制的 bitstream/PS 初始化文件、平台 `export`/`logs`/`.log`、`tempdsa` 临时硬件解包目录、编译生成的 `.elf`，以及 BSP 自动生成的 include/lib/libsrc 和构建文件。Vivado 目录下其他位置的 `.log`/`.jou` 也忽略。平台 `hw/` 中除上述 XSA 外的解包副本（驱动、bitstream、PS 初始化等）也忽略；原始自定义 IP 的完整源仍保留在 `ip_repo/export`，不受这些规则影响。

平台标记 `hasFsblMakeHasChanges=false`；与本机 Vitis 2022.2 的 `data/embeddedsw/lib/sw_apps/zynq_fsbl/src` 比较，共有源码（含 hook）当前一致，工作区另有硬件初始化和构建文件。仍保守保留 `zynq_fsbl` 的源码、hook、链接脚本、初始化源码与自身 Makefile，便于以后保留用户启动定制；仅排除其 BSP 生成物和 Debug/Release，不把所有 Makefile 或头文件一刀切忽略。

恢复工程后先 Build Platform/BSP，再 Build Application。`.prj` 和 `platform.tcl` 含原机器绝对路径，搬迁时需要重新绑定/导入平台并检查硬件路径；当前不改写元数据，也不承诺整个 Vitis 工作区可自动跨路径重建。

`.gitignore` 不会自动取消已有文件的跟踪。如果生成物已经提交，需仅对明确的生成物执行 `git rm --cached`，保留本地文件。本轮静态审计清理了摄像头工作区迁入的 IDE 状态、`_ide` 硬件副本、平台 `hw` 解包副本、两个 BSP 自动生成内容及 FSBL ELF；也清理了 `vivado/` 下一个 NFS 延迟删除临时文件的跟踪。未删除任何本地文件，也未恢复正在使用的 `workbench.xmi`。审计和本地文件校验记录位于忽略目录 `build/hls/camera_static_check/git_cleanup.log`。

摄像头工程保留完整 `.srcs`、`ip_repo` 和 XPR 实际引用的 `.srcs/sources_1/imports/system_wrapper.v`；XPR 中旧外部 `.gen` 路径只是导入来源记录，并非当前 wrapper 的实际路径。DDR 工程原地引用的 `.gen` wrapper 例外规则继续保留。两个工作区的原交接 XSA、平台内部 `hw/*.xsa`、BSP `system.mss` 和 FSBL 自身源码均继续跟踪；没有按 `.c`、`.h`、`.tcl` 或 `.dcp` 后缀统一排除。

重新构建仍可能修改必要的工程描述：`.cproject` 可能增加 `refreshScope`，`.prj`/`.sprj` 更新 `dirty`/`lastBuildOptions`，自动维护的 `platform.tcl` 可能追加 `platform clean/generate`。这些文件同时保存编译配置、平台关联或重建命令，因此继续跟踪，不能为消除 modified 而整体忽略，也不使用 `assume-unchanged` 隐藏将来的配置变更。

当前 XPR 还直接引用 `.srcs/utils_1/imports/synth_1/system_wrapper.dcp` 作为增量综合检查点，暂保留该文件，避免缺失引用。若以后不再保存增量综合缓存，应先在 Vivado 中移除该引用，再调整跟踪规则；不要直接忽略所有 DCP。

## 可移植性实测

以下是此前 Vivado-only 快照的历史实测，不是本轮加入 Vitis 规则后的计数；本轮未重新运行 Vivado。此前使用 Vivado 2022.2，将 217 个保留文件复制到新位置（不带 647 个忽略项）后，成功打开 XPR；识别到的 5 个 IP 均为 `Up-to-date`，未出现缺失文件或锁定 IP 错误。Block Design 可打开，`validate_bd_design -force` 通过。当时仅验证工程打开与 BD 校验，未运行综合、实现或上板测试。

首次打开时，缺少 `IPUserFilesDir`、`GeneratedRun` 的警告是未跟踪缓存及运行记录的预期现象；重新生成 Output Products 并运行相应流程可重建这些产物。强制 BD 校验另有 `BD 41-927` 警告，涉及 `/cnn_accelerator_ddr_0/ap_clk` 的 `FREQ_HZ = 1e+08` 属性同步，未阻止校验通过，也不是缺少源码；本次未修改工程参数。

本地验证日志：[vivado_open_force.log](../build/hls/vivado_portability_verify/vivado_open_force.log)（位于忽略的 `build/hls` 下，不随仓库分发）。
