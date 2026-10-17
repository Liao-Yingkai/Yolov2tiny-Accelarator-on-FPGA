MNIST DDR 裸机验证程序（Zynq Cortex-A9 / Vitis）

构建与运行
1. 在 Vitis 中先 Build Platform/BSP，再 Build Application cnn_accel_ddr。
   当前平台若尚未生成 BSP 头文件和 libxil.a，必须先完成平台构建。
   不需 platform.h；UART0 标准输出由 BSP 配置，程序不修改 UART/MIO。
2. 下载与当前 DDR wrapper 匹配的 bitstream：结果必须为 16 字节，
   byte 0..9 为 INT8 logits、10..11 为 0、12..15 为 uint32 class。
   每层参数依次是完整 bias、multiplier、shift 三组 INT32 数组。
   旧版 24 字节结果 IP 不兼容；尾部哨兵检查会报告 ABI overwrite。
   本轮权重布局已升级为 v2：1133 个64位 word，而非旧1196个。
   需要重新导出 RTL/IP 并生成 bitstream；此处未修改 Vivado 工程或旧IP。
   结果哨兵不能保证识别旧权重布局，不得将本程序与旧权重布局 bitstream 混用。
3. JTAG 调试先执行本硬件平台的 ps7_init/PS 初始化，再运行 ELF。
   从启动介质运行则由 FSBL 完成 DDR、FCLK、UART 等 PS 初始化。
   main 不调用 ps7_init。确认 PL 已配置且时钟/复位正常。
4. 打开 BSP UART0 串口终端（波特率按平台设置），执行程序。
   程序依次运行 trace_data_0000、0001、0002 三张图，逐项核对
   10 个 logits、首次最大值对应的 class、保留字节和结果尾部哨兵。
   每图打印 class/errors/inference_us/PASS 或 FAIL，末尾打印 Overall。
   只有真实上板运行并全部通过，才能报告硬件 PASS；离线构建不等于上板通过。

程序实现流程
main 按“初始化 → 准备一次模型并设置地址 → 逐图推理 → 检查结果”组织。
prepare_model 将三个新版权重 header 原始字节直接 memcpy 到 DDR，再复制每层三块参数，
只执行一次；run_inference 负责单张图的输入打包、缓存维护、启动和等待；
check_result 负责所有 logits、类别、保留字节和尾部哨兵检查。
卷积 header 顺序为 Cout tile → kernel token → 4个 Cout lane → 有效 Cin。
Conv1 每 token 只有4个真实字节，一个 DDR word 装两个 token，不补七路零；
Conv2 每 token 为4×8字节。FC 为输出 tile → 输入 token → 5个输出 lane → 8个 Cin。
原权重数组已由 Python exporter 排好，PS 不再重排，不生成逻辑通道零填充。
图像按连续像素打包，低 byte 对应较早像素；先转 uint8_t 保留二补码，
再转 uint64_t 后移位，避免负数左移和符号扩展。
Loader 未使用字段透传已读 word 的字节；Im2Row 保留 Conv1 通道规范化，
Systolic 自行保证 Conv1 仅 channel0 参与点积，不依赖上游已清零。

四个 DDR 缓冲区
image_ddr：784 字节输入，补齐到 832 字节，每次更新一张图。
weights_ddr：1133 个 64-bit word，9064 有效字节，补齐到 9088 字节。
  Conv1/Conv2/FC 的 word 起点为0/9/153，长度为9/144/980。
params_ddr：102 个 INT32，408 有效字节，补齐到 448 字节；每层顺序为
  [完整 bias 数组][完整 multiplier 数组][完整 shift 数组]，不按通道交错。
result_ddr：64 字节缓存隔离区，PL 只写前 16 字节，后 48 字节保持 0xa5。
这些 static 数组由现有链接脚本分配到 DDR .bss；程序取得其真实地址后，
用 Set_image_r / Set_weights / Set_params / Set_result 写给 PL 地址寄存器。
不需要人工为 PS 和 PL 定死相同的四个缓冲区地址。

mnist_test_data.h 来源与数据更新
src/mnist_test_data.h 已包含本地模型与三张图，不依赖训练目录绝对 include。
它由 tools/export_test_data.py 生成：headers/ 中 conv1、conv2、fc 各自的
weight、bias、quant、shift 共 12 个头文件内的全部原始数组，类型保持
int8_t / int32_t；quant 文件中的数组名为 *_multiplier。
每张 trace 仅取 trace_input_int8_hwc 和 trace_fc_requant_int8，分别作为
784 个 INT8 输入和 10 个最终量化 logits；不包含庞大的中间 trace。
生成头逐字复制新版模型 header 的数组，权重已经是硬件紧凑顺序，main 不再
重排。提取脚本要求三个 *_WEIGHT_LAYOUT_VERSION 均为2，本地头发布
MNIST_WEIGHT_LAYOUT_VERSION=2，应用编译期检查，防止误用旧版模型头。
Python metadata 为5；NPY 保留数学矩阵，golden model 将硬件 header 解码回
该矩阵计算，量化公式不变。数组用 static const，三图使用不同名字避免冲突。
C 不支持 HLS 布局头里的 C++ namespace，因此 main 用注释清楚的 C 常量
派生相同布局，并保留必要的 ABI/数量断言，没有修改 HLS C++ 接口。
在本应用根目录运行：
  python3 tools/export_test_data.py --generated-dir /path/to/CNN_FPGA/generated
本次来源为：
  /home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated
该目录需有 headers/{conv1,conv2,fc}_{weight,bias,quant,shift}.h
及 trace/trace_data_0000.h、0001.h、0002.h。
脚本只使用 Python 标准库解析十进制 C 整数数组，不训练、不修改来源；
它不会监控训练目录，也不会自动同步新训练结果；训练更新后须主动执行
导出命令并重新 Build Application，当前已有头文件可脱离原始绝对路径构建。
缺文件、元素数量或类型范围不匹配会失败。更改网络形状/并行度时还须
同步 main.c 内标注的冻结布局并重新生成 IP，不能只替换数据头。

缓存、计时与故障
DDR 缓冲区位于静态 .bss，均按 64 字节对齐并向上补齐容量；无 malloc。
D-cache 保持开启。模型只打包/flush 一次；每图 flush 输入和哨兵结果，
轮询完成后 invalidate 完整结果缓存区域，再读取数据。
原因是当前 PL DDR AXI 访问不与 PS D-cache 自动保持一致：flush 使 CPU
写入真正到达 DDR 供 PL 读取，invalidate 则使 CPU 丢弃旧缓存并读取 PL
写回的结果；dsb 用于等待内存维护完成后再启动或读取，不能只用 volatile
替代这些操作。缓存维护始终覆盖包括补齐尾部在内的整个分配区。
程序禁用 autorestart、全局中断和 IP 中断使能，仅由 PS 轮询 ap_done。
每图开始前要求 IP idle，并清除旧 done；单次 5 秒超时打印 CTRL 后立即
停止，不复用可能仍被 PL 访问的缓冲区，不自动复位 IP 或继续下一图。
遇到此类失败先检查 bitstream、PS 初始化、FCLK/复位和 DDR AXI 连接。
inference_us 使用 PS CPU 全局计时器测量启动到轮询完成的墙钟时间，
包含软件开销，不是 PL cycles，也不包含模型打包和串口打印。
本应用不修改链接脚本、BSP、硬件或 HLS 实现。

离线检查边界
本轮检查结果以 build/hls/cin_layout_update_verify/ 中的日志为准。
三图数据头已用新版 exporter 重新生成；三层数学权重和量化参数与旧版一致。
全部15个 HLS testbench 和 Vitis DDR C Simulation 三图检查通过。
main 使用实际 BSP/driver 头文件以 ARM -Wall -Wextra -Werror 编译通过；
host mock 三图类别0/4/1、每图错误数0，缓存范围检查通过。
ARM 编译和接入 HLS C++ 顶层的 host mock 仅验证软件/功能，不代表上板通过；
mock 计时没有硬件性能意义。仍须构建实际 BSP、链接 PS ELF，并使用与
新版紧凑权重匹配的 RTL/bitstream 上板确认。当前文档不宣称本轮 II 或资源已验证。
