# PS原始取帧与模型输入预处理

PS取得稳定快照后，用独立camera_preprocess.c/.h算术生成单通道28×28模型输入，不用LUT，不连接CNN加速器。失败轮不预处理：camera_gray和camera_model_input保留上一成功轮结果（尚无成功轮时为静态初值），不能将其误认成失败轮新数据；camera_snapshot可能已被部分覆盖，内容无效、不得使用。图像和识别精度仍需板测。

## 模块与内存

main.c保留摄像头、VDMA及显示初始化，调度取帧和成功后的PS预处理；camera_input.c/.h只负责检查运行前提、请求READ停车、等待帧事件、整帧复制及比较，不接CNN。vdma_api.c保持原实现，不新增配置代码。BD、XDC、HLS不变；硬件像素时钟配置25.2MHz，未用于配置该时钟的.freq=25.12字段保持原样。

header中CAMERA_WIDTH/HEIGHT单位是像素/行；CAMERA_BYTES_PER_PIXEL=3表示BGR888；CAMERA_FRAMES=3表示VDMA物理帧槽数；CAMERA_STRIDE=640×3=1920字节/行；CAMERA_FRAME_BYTES=1920×480=921600字节。无行尾填充，像素及行均从左到右、从上到下连续存放。capture RTL的[23:0]={R,G,B}经视频输入AXIS原位传递到VDMA，因此DDR低地址三字节依次B、G、R；仍需后续板测确认实际颜色。

main提供64字节对齐的静态连续三帧数组，共2764800字节，VDMA只访问这三帧。另有64字节对齐的camera_snapshot普通DDR数组，大小921600字节，CPU独占，它不是第四个VDMA frame store，不会被DMA覆盖。camera_gray和camera_model_input也各64字节对齐，容量各832字节，前784字节有效、尾48字节仅隔离cacheline；flush整对象不会触及相邻数组。总图像缓冲占3688064字节。链接脚本将静态数组放DDR，无动态分配。

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
python3 vivado/cnn_accel_cam/tools/generate_camera_model_input.py --metadata /path/to/generated/metadata.json
```

默认metadata路径沿用当前训练工程。脚本仅用Python标准库，校验scale有限且正、zero_point为0或[0]；不会修改训练工程。若dataset.py的归一化常量改变，应同步修改脚本中的mean/std并重生成头。公开camera_gray_to_int8入口用于独立核对全部256个灰度值。

## 取帧每一步及其原因

1. 检查实际运行前提：三帧、内部动态Genlock（READ slave=3、WRITE master=2）、两通道运行及循环、内部Genlock源位、EnableFrameCounter=0、MM2S事件阈值1及通知使能。这里只读回检查，不偷偷重配；不满足返回config。main继续调用原run_vdma_frame_buffer(...,1,1,BOTH)，不连接GIC，不限定帧数停止。
2. CurrFrameStore读取当前READ槽0..2。它是DDR槽编号，不是新画面的递增序号。VDMA API读写控制/状态寄存器，不是通过API读取像素流；像素实际位于DDR。
3. StartParking请求READ停留在目标槽。请求需跨帧边界生效，不能立即复制。清掉旧MM2S frameflag后等两个新MM2S事件，给请求跨越边界并完成稳定显示周期的时间。事件次数本身不足以证明安全，仍须下一步所有权检查。
4. 两事件共用XTime计时的200ms超时，每次轮询检查READ/WRITE错误。确认READ仍在目标且WRITE不在目标；内部动态Genlock应使S2MM避开READ保护帧。异常返回timeout、dma_error或ownership等，不无限等待。
5. DCache保持启用，VDMA启动前flush三帧数组，之后CPU不写摄像头帧。源帧整帧invalidate后再读，避免CPU读到DMA写DDR前的旧缓存。
6. 保持READ停车，memcpy完整921600字节到CPU快照，再memcmp逐字节比较；不一致返回copy_mismatch。这只验证CPU看到的原始字节搬运一致，不证明传感器图像内容、颜色、采集端正确，也不证明新帧唯一。
7. 所有停车后出口请求StopParking恢复READ循环，不主动停止摄像头/S2MM。DMA故障可能需用户复位重启，不保证画面恢复，不自动清错/reset。失败输出可能未完成，禁止使用。

## 计时及正常/异常现象

main用XTime/COUNTS_PER_SECOND从本轮起点计100ms目标周期，复制、比较、预处理、调试flush和串口时间都包含在调度周期内；超时直接下一轮，不在工作后额外等待100ms。每约一秒输出一行：capture_success_count/capture_failure_count为本窗口成功/失败次数；last_capture_us为最近capture调用时间；last_preprocess_us为本轮纯预处理时间，失败轮为0；last_cycle_us为本轮取帧+预处理+debug flush工作耗时，不含串口统计输出和100ms等待；所有耗时单位微秒。last_failure为本窗口最后失败原因，窗口结束清零次数及失败原因，避免旧失败冒充新失败。没有逐像素输出。调试flush只为观察DDR，不计入纯预处理性能。

正常目标约10次成功/秒、失败0、无VDMA错误，复制期间显示可能短暂定格；成功意味着原始复制逐字节比较通过。100ms调度不保证10个独立新帧，不保证板上总能达到10fps。config表示当前运行配置不满足安全取帧前提，不会擅自修复；持续DMA错误或超时应检查硬件/配置，必要时由用户复位重启。

## 用Vitis查看完整raw snapshot

启动时串口只打印一次snapshot_address、frame_bytes、gray_address、int8_address、valid_pixels与buffer_bytes。每次成功预处理后main flush三个调试数组，目的仅为让调试器从DDR看到CPU写入的完整数据；CPU自身读取不需要这次flush。此flush与复制前源帧invalidate方向、目的不同。

以调试配置构建，在main中成功分支之后的XTime_GetTime(&cycle_end)行设条件断点capture_result == CAMERA_OK。断点命中时本轮capture和预处理已成功、三个数组已flush。打开Vitis Memory视图，用串口地址或符号查看camera_snapshot的921600字节（640×480 BGR888 stride1920）、camera_gray前784个uint8以及camera_model_input前784个signed int8（均28×28行优先）。不要把尾48字节填充当图像。不要随机暂停在memcpy/预处理中途后把未完成数据误作结果，也不要使用失败轮次输出。调试暂停会打乱100ms周期，恢复后的一秒统计不能当稳态性能数据。

## 验证边界

本轮ARM GCC检查全部7个应用C文件通过；未修改的vdma_api.c仍有一条原有的有符号/无符号比较警告，新预处理文件没有新增警告或libm依赖。独立主机数值测试在O0/O2下覆盖全部256种灰度、全白/全黑、随机BGR、逐块颜色编码及ROI外扰动，灰度和INT8结果均与NumPy reference逐元素一致，输出保护区未被改写。参数脚本能复现当前标量头并拒绝13种非法参数；当前256种灰度不产生精确半整数，不能把这次枚举当作half-tie边界测试。

检查脚本及日志放仓库build/hls/camera_preprocess_check，不放源码目录；host_validation.log保存主机数值结果，arm_syntax.log保存ARM语法检查结果。未综合、未下载或运行硬件。板测仍需确认停车保护、完整复制、异常路径、解除停车后的实际画面、纸面极性和稳态计时。100ms目标不保证10fps，更不保证每轮都是独立新帧或识别正确。
