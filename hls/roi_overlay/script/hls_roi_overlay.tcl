# 独立 Overlay C 仿真。根据脚本位置定位源码，将工程生成到仓库的 build/hls/。
set script_dir [file dirname [file normalize [info script]]]
set overlay_dir [file dirname $script_dir]
set repo_dir [file normalize [file join $overlay_dir .. ..]]
set build_dir [file join $repo_dir build hls]

file mkdir $build_dir
cd $build_dir
open_project roi_overlay
set_top roi_overlay
add_files -cflags {-std=c++14} [file join $overlay_dir src roi_overlay.cpp]
add_files -tb -cflags {-std=c++14} [file join $overlay_dir testbench test_roi_overlay.cpp]
open_solution solution1 -flow_target vivado
set_part {xc7z020-clg400-2}
create_clock -period 8 -name default
csim_design -clean
exit
