# 依据脚本位置解析路径，可从任意工作目录调用。
set script_dir [file dirname [file normalize [info script]]]
set accelerator_dir [file dirname $script_dir]
set src_dir [file join $accelerator_dir src]
set repo_dir [file normalize [file join $accelerator_dir .. ..]]
set build_dir [file join $repo_dir build hls]
file mkdir $build_dir
cd $build_dir
open_project cnn_ddr_csim
set_top cnn_accelerator_ddr
add_files [file join $src_dir core/cnn_core.cpp]
add_files [file join $src_dir im2row/im2row.cpp]
add_files [file join $src_dir systolic/systolic.cpp]
add_files [file join $src_dir postprocess/conv_postprocess.cpp]
add_files [file join $src_dir pool/pool.cpp]
add_files [file join $src_dir fc/fc.cpp]
add_files [file join $src_dir postprocess/fc_postprocess.cpp]
add_files [file join $src_dir axi/ddr_io.cpp]
add_files [file join $src_dir top/cnn_accel_ddr.cpp]
add_files [file join $src_dir postprocess/logits_argmax.cpp]
add_files -tb [file join $accelerator_dir testbench test_cnn_accel_ddr.cpp]
open_solution solution1 -flow_target vivado
set_part {xc7z020-clg400-2}
create_clock -period 10 -name default
csim_design
# 如需综合、协同仿真或导出，请自行追加命令，或在图形界面中设置。
exit
