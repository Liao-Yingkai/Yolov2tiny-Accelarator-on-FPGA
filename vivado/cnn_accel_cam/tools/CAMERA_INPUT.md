# 摄像头取帧、预处理与CNN推理

PS取得稳定快照后，用独立camera_preprocess.c/.h算术生成单通道28×28模型输入，不用LUT，再由camera_cnn模块提交新CNN IP并读回类别。失败轮不预处理：camera_gray和camera_model_input保留上一成功轮结果（尚无成功轮时为静态初值），不能将其误认成失败轮新数据；camera_snapshot可能已被部分覆盖，内容无效、不得使用。图像和识别精度仍需板测。

## 模块与内存

main.c保留摄像头、VDMA及显示初始化，调度取帧、成功后的PS预处理及CNN推理；camera_input.c/.h只负责检查运行前提、请求READ停车、等待帧事件、整帧复制及比较，不负责CNN。camera_cnn.c/.h只封装CNN初始化与单次轮询推理。vdma_api.c保持原实现，不新增配置代码。BD、XDC、HLS不变；硬件像素时钟配置25.2MHz，未用于配置该时钟的.freq=25.12字段保持原样。

header中CAMERA_WIDTH/HEIGHT单位是像素/行；CAMERA_BYTES_PER_PIXEL=3表示BGR888；CAMERA_FRAMES=3表示VDMA物理帧槽数；CAMERA_STRIDE=640×3=1920字节/行；CAMERA_FRAME_BYTES=1920×480=921600字节。无行尾填充，像素及行均从左到右、从上到下连续存放。capture RTL的[23:0]={R,G,B}经视频输入AXIS原位传递到VDMA，因此DDR低地址三字节依次B、G、R；仍需后续板测确认实际颜色。

main提供64字节对齐的静态连续三帧数组，共2764800字节，VDMA只访问这三帧。另有64字节对齐的camera_snapshot普通DDR数组，大小921600字节，CPU独占，它不是第四个VDMA frame store，不会被DMA覆盖。camera_gray和camera_model_input也各64字节对齐，容量各832字节，前784字节有效、尾48字节仅隔离cacheline；这些数组只有CPU访问，不作调试用途的flush。总图像缓冲占3688064字节。链接脚本将静态数组放DDR，无动态分配。

## 预处理顺序与模型参数

输入必须是CPU自有稳定640×480 BGR888快照，无stride填充。编译期检查输入尺寸及ROI边界。中央ROI左上角(x=180,y=100)，宽高280，每个10×10块对应一个输出像素，按行优先存放：

1. 每个输入像素先计算整型灰度 `(77*R+150*G+29*B+128)>>8`。
2. 对100个已舍入的灰度求和，再用 `(sum+50)/100` 做整数均值；不是先求彩色均值再灰度。
3. 默认 `255-均值` 反色，对应白纸黑字转暗背景亮笔画；camera_gray保存此时的uint8灰度。宏CAMERA_PREPROCESS_INVERT=0可关闭反色，适用于已是黑底白字的输入。
4. 每个灰度严格分步float32：除255、减float32的0.1307、除float32的0.3081；对应dataset.py的ToTensor/Normalize。局部volatile float确保各步落到float32，禁止fast-math；不将运算合并为double或融合表达式。
5. 将归一化float32转double，除metadata导出的double scale，确定性nearest-ties-even后clamp至[-128,127]。不使用半整数远离零的roundf，不依赖libm，无像素映射表。zero_point必须为0。输出signed int8仅784字节，不扩展8通道。

当前scale=0.022129308432340622。白纸纯白经反色变0，量化背景约-19，不是0；纯黑笔画经反色变255，量化约127（上限）。没有阈值分割、居中、笔画提取或自动极性检测，纸面阴影/位置/大小仍会影响输入。

camera_model_input.h只是少量标量常量，来源写在头注释，不是LUT。mean/std固定与dataset.py一致；训练模型重新导出后，在仓库根执行：

```sh
python3 vivado/cnn_accel_cam/tools/export_camera_model.py --generated-dir /path/to/generated
```

此命令同一generated目录读取headers及metadata.json，生成camera_model_data.h与camera_model_input.h，不能只换权重不换scale。默认generated路径沿用当前训练工程。脚本仅用Python标准库，校验scale有限且正、zero_point为0或[0]；不会修改训练工程。若dataset.py的归一化常量改变，应同步修改脚本中的mean/std并重生成头。公开camera_gray_to_int8入口用于独立核对全部256个灰度值。

## 取帧每一步及其原因

1. 检查实际运行前提：三帧、内部动态Genlock（READ slave=3、WRITE master=2）、两通道运行及循环、内部Genlock源位、EnableFrameCounter=0、MM2S事件阈值1及通知使能。这里只读回检查，不偷偷重配；不满足返回config。main继续调用原run_vdma_frame_buffer(...,1,1,BOTH)，不连接GIC，不限定帧数停止。
2. CurrFrameStore读取当前READ槽0..2。它是DDR槽编号，不是新画面的递增序号。VDMA API读写控制/状态寄存器，不是通过API读取像素流；像素实际位于DDR。
3. StartParking请求READ停留在目标槽。请求需跨帧边界生效，不能立即复制。清掉旧MM2S frameflag后等两个新MM2S事件，给请求跨越边界并完成稳定显示周期的时间。事件次数本身不足以证明安全，仍须下一步所有权检查。
4. 两事件共用XTime计时的200ms超时，每次轮询检查READ/WRITE错误。确认READ仍在目标且WRITE不在目标；内部动态Genlock应使S2MM避开READ保护帧。异常返回timeout、dma_error或ownership等，不无限等待。
5. DCache保持启用，VDMA启动前flush三帧数组，之后CPU不写摄像头帧。源帧整帧invalidate后再读，避免CPU读到DMA写DDR前的旧缓存。
6. 保持READ停车，memcpy完整921600字节到CPU快照，再memcmp逐字节比较；不一致返回copy_mismatch。这只验证CPU看到的原始字节搬运一致，不证明传感器图像内容、颜色、采集端正确，也不证明新帧唯一。
7. 所有停车后出口请求StopParking恢复READ循环，不主动停止摄像头/S2MM。DMA故障可能需用户复位重启，不保证画面恢复，不自动清错/reset。失败输出可能未完成，禁止使用。

## CNN DDR接口与缓存方向

camera_model_data.h只有12个static const模型数组，不带测试图片或trace。导出脚本检查每个数组的类型、数量和整数范围，拒绝非v2权重布局。模型初始化只加载一次：Conv1 72B、Conv2 1152B、FC 7840B紧密拼接共9064B（1133个u64），不转置、不补通道。参数每层按[完整bias][完整multiplier][完整shift]拼接，三层8/16/10输出共102个int32。输入是784个连续signed INT8单通道像素，Zynq小端下memcpy到u64缓冲即可使每8像素低byte优先。

camera_cnn有一个静态驱动实例，使用当前BSP DeviceID和控制地址宏，初始化检查驱动实际基地址、完整DDR范围及64B对齐，关闭AutoRestart与中断，确认Idle后设置四个u64地址，不在初始化启动CNN。所有实际DMA缓冲起点与容量均按64B对齐，避免维护缓存时影响邻居。模型和每轮输入由CPU写、PL读，因此flush后dsb；每轮结果整缓存行先写哨兵并flush，避免旧脏缓存覆盖PL结果。完成后dsb、invalidate结果、dsb，再memcpy前16字节：10个signed INT8 logits、2个零保留字节、offset12的uint32类别。检查类别<10、等于首次最大logit索引且保留字节为零；不与测试图片golden比较。

每轮先检查Idle，再在Start之前读取一次IsDone清上轮COR（读后清零）完成位；Start之后只轮询IsDone，不额外读取控制寄存器。CNN超时为5秒。busy、timeout或其他CNN错误立即终止应用；超时不表示PL已经停止访问DDR，因此不自动reset、不复用缓冲重试，须板级复位后重启。

## 调度及串口

main仅用XTime从本轮开始计100ms目标周期，包含复制、预处理、推理和串口；超时直接下一轮，不额外等待100ms。正常每次成功仅输出`digit=<0..9>`，失败输出简短原因，无地址、计数、逐像素、logits或耗时日志。取帧失败不使用快照，跳过预处理和推理，在下一节拍重试。100ms是调度目标，不保证10fps、独立新帧或识别精度。HDMI预览保留，复制期间可能短暂定格。

## 上板步骤

1. 在训练端完成一次一致的模型导出，再用上面的export_camera_model命令同步两个模型头；无需trace，不修改训练工程。
2. 使用用户已更新的Vivado硬件/XSA和对应新平台BSP，确认包含cnn_accelerator_ddr驱动（DeviceID 0、控制基址0x40000000，Set_image_r/weights/params/result为u64），重建平台及相机应用。将camera_cnn.c纳入应用源码；不要手工改IDE生成Makefile。
3. 下载与此BSP一致的bitstream，初始化PS/DDR后加载新ELF；保留摄像头与HDMI连接，用白纸黑字放入中央280×280 ROI，观察串口digit与预览。
4. 确认实际BGR颜色、ROI/极性、快照稳定及类别变化；检查取帧错误会重试、CNN错误会停止。CNN busy/timeout后板级复位，不能直接继续运行旧缓冲。
5. 如需观察工作数组，在成功轮完成后通过符号定位，注意CPU缓存中的新值可能尚未写DDR；正常路径不为调试器flush，也不打印数组地址。不要把失败轮或半更新数组当有效输入。

## 验证边界

此前预处理阶段ARM GCC检查7个应用C文件通过；未修改的vdma_api.c仍有一条原有的有符号/无符号比较警告，新预处理文件没有新增警告或libm依赖。独立主机数值测试在O0/O2下覆盖全部256种灰度、全白/全黑、随机BGR、逐块颜色编码及ROI外扰动，灰度和INT8结果均与NumPy reference逐元素一致，输出保护区未被改写。参数脚本能复现当前标量头并拒绝13种非法参数；当前256种灰度不产生精确半整数，不能把这次枚举当作half-tie边界测试。

此前预处理检查脚本及日志放仓库build/hls/camera_preprocess_check，不放源码目录；host_validation.log保存主机数值结果，arm_syntax.log保存ARM语法检查结果。未综合、未下载或运行硬件。板测仍需确认停车保护、完整复制、异常路径、解除停车后的实际画面、纸面极性和稳态计时。100ms目标不保证10fps，更不保证每轮都是独立新帧或识别正确。

新增CNN接入通过8个应用C文件的ARM编译，并按现有Vitis链接选项生成ELF，无未解析符号；唯一编译警告仍为原vdma_api.c的有符号/无符号比较。12个模型数组与训练端导出逐一一致，输入scale和零点也一致。独立host mock检查通过，覆盖模型/参数/输入/结果布局、缓存维护及完成位调用顺序、连续两次推理、错误结果和busy/timeout分支；mock不执行真实CNN计算，也不代表DDR总线已经验证。

记录放在build/hls/camera_cnn_check：final_validation.log保存host mock结果，vitis_link.log保存按Vitis参数进行的ELF链接结果。未运行HLS综合/CSim或下载硬件，不能据此认定新IP连接、AXI数据搬运或识别精度已经板测通过。
