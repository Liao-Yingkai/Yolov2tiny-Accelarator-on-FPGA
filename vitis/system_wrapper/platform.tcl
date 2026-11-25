# 
# Usage: To re-create this platform project launch xsct with below options.
# xsct path/to/vitis/system_wrapper/platform.tcl
# 
# OR launch xsct and run below command.
# source path/to/vitis/system_wrapper/platform.tcl
# 
# To create the platform in a different location, modify the -out option of "platform create" command.
# -out option specifies the output directory of the platform project.

# 路径相对于当前脚本解析；保留原平台生成步骤，不依赖作者电脑目录。
set platform_script_dir [file dirname [file normalize [info script]]]
set platform_workspace_dir [file dirname $platform_script_dir]
set platform_hardware_file [file join $platform_workspace_dir system_wrapper.xsa]

platform create -name {system_wrapper}\
-hw $platform_hardware_file\
-out $platform_workspace_dir

platform write
domain create -name {standalone_ps7_cortexa9_0} -display-name {standalone_ps7_cortexa9_0} -os {standalone} -proc {ps7_cortexa9_0} -runtime {cpp} -arch {32-bit} -support-app {empty_application}
platform generate -domains 
platform active {system_wrapper}
domain active {zynq_fsbl}
domain active {standalone_ps7_cortexa9_0}
platform generate -quick
platform generate
platform generate
platform active {system_wrapper}
platform config -updatehw $platform_hardware_file
platform generate -domains 
platform active {system_wrapper}
platform config -updatehw $platform_hardware_file
platform generate -domains 
platform active {system_wrapper}
platform config -updatehw $platform_hardware_file
platform generate
platform active {system_wrapper}
platform config -updatehw $platform_hardware_file
platform generate -domains 
platform active {system_wrapper}
platform config -updatehw $platform_hardware_file
platform generate -domains 
bsp reload
bsp setlib -name xilffs -ver 4.8
bsp write
bsp reload
catch {bsp regenerate}
bsp reload
bsp config use_lfn "1"
bsp write
bsp reload
catch {bsp regenerate}
platform generate -domains standalone_ps7_cortexa9_0 
platform clean
platform generate
platform active {system_wrapper}
platform config -updatehw $platform_hardware_file
platform clean
platform generate
platform clean
platform generate
platform clean
platform generate
