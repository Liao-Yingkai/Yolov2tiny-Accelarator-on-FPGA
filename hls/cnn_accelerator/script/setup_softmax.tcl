# 仅建立工程、注册源码和约束，不调用任何仿真、综合或导出阶段。
set script_dir [file dirname [file normalize [info script]]]
set accelerator_dir [file dirname $script_dir]
set repo_dir [file normalize [file join $accelerator_dir .. ..]]
set build_dir [file join $repo_dir build hls]
file mkdir $build_dir
cd $build_dir
open_project softmax_test
set_top softmax
add_files -cflags {-std=c++14} [file join $accelerator_dir src postprocess softmax.cpp]
add_files -tb -cflags {-std=c++14} [file join $accelerator_dir testbench test_softmax.cpp]
open_solution solution1 -flow_target vivado
set_part {xc7z020-clg400-2}
create_clock -period 10 -name default
exit
