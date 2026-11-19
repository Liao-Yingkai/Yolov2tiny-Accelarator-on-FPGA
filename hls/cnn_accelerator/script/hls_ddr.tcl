# 依据脚本位置解析路径，可从任意工作目录调用。
set script_dir [file dirname [file normalize [info script]]]
set accelerator_dir [file dirname $script_dir]
set src_dir [file join $accelerator_dir src]
set repo_dir [file normalize [file join $accelerator_dir .. ..]]
set build_dir [file join $repo_dir build hls]
set stage csim
if {[info exists ::env(YOLO_HLS_STAGE)] && $::env(YOLO_HLS_STAGE) ne ""} {
    set stage $::env(YOLO_HLS_STAGE)
}
if {$stage ni {csim csynth}} {
    error "YOLO_HLS_STAGE 只能是 csim 或 csynth；不自动运行 co-sim 或导出 IP。"
}
set run_synthesis [expr {$stage eq "csynth"}]
if {!$run_synthesis && (![info exists ::env(YOLO_DATA_DIR)] || $::env(YOLO_DATA_DIR) eq "")} {
    error "请先设置 YOLO_DATA_DIR 为包含 bins/ 和 trace/ 的 generated 根目录。"
}
set cpp_flags {-std=c++14}
if {!$run_synthesis} {
    # 九层仿真中 ap_int 运算量较大，-O3 只用于软件仿真，不改变综合选项。
    append cpp_flags " -O3"
}
file mkdir $build_dir
cd $build_dir
open_project cnn_yolo_ddr
set_top cnn_accelerator_ddr
add_files -cflags $cpp_flags [file join $src_dir im2row/im2row.cpp]
add_files -cflags $cpp_flags [file join $src_dir systolic/systolic_OS.cpp]
add_files -cflags $cpp_flags [file join $src_dir postprocess/conv_postprocess.cpp]
add_files -cflags $cpp_flags [file join $src_dir pool/pool.cpp]
add_files -cflags $cpp_flags [file join $src_dir axi/ddr_io.cpp]
add_files -cflags $cpp_flags [file join $src_dir top/cnn_accel_ddr.cpp]
add_files -cflags $cpp_flags -tb [file join $accelerator_dir testbench test_cnn_accel_ddr.cpp]
open_solution solution1 -flow_target vivado
set_part {xc7z020-clg400-2}
create_clock -period 10 -name default
# 默认只做优化编译的 C 仿真；可通过环境变量单独执行综合。
if {$run_synthesis} {
    csynth_design
} else {
    csim_design -O -clean
}
exit
