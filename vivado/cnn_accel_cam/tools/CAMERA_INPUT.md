# PS原始取帧验证

本轮只验证PS能安全取得完整原始帧，并逐字节确认复制结果。不做图像精度验证，不实现任何预处理或模型输入转换。所有预处理留下一步，届时按用户要求在PS用算术实现，不用LUT。

## 模块与内存

main.c保留摄像头、VDMA及显示初始化，并调度取帧；camera_input.c/.h只负责检查运行前提、请求READ停车、等待帧事件、整帧复制及比较，不接CNN。vdma_api.c保持原实现，不新增配置代码。BD、XDC、HLS不变；硬件像素时钟配置25.2MHz，未用于配置该时钟的.freq=25.12字段保持原样。

header中CAMERA_WIDTH/HEIGHT单位是像素/行；CAMERA_BYTES_PER_PIXEL=3表示BGR888；CAMERA_FRAMES=3表示VDMA物理帧槽数；CAMERA_STRIDE=640×3=1920字节/行；CAMERA_FRAME_BYTES=1920×480=921600字节。无行尾填充，像素及行均从左到右、从上到下连续存放。capture RTL的[23:0]={R,G,B}经视频输入AXIS原位传递到VDMA，因此DDR低地址三字节依次B、G、R；仍需后续板测确认实际颜色。

main提供64字节对齐的静态连续三帧数组，共2764800字节，VDMA只访问这三帧。另有64字节对齐的camera_snapshot普通DDR数组，大小921600字节，CPU独占，它不是第四个VDMA frame store，不会被DMA覆盖。总图像缓冲占3686400字节。链接脚本将静态数组放DDR，无动态分配。

## 取帧每一步及其原因

1. 检查实际运行前提：三帧、内部动态Genlock（READ slave=3、WRITE master=2）、两通道运行及循环、内部Genlock源位、EnableFrameCounter=0、MM2S事件阈值1及通知使能。这里只读回检查，不偷偷重配；不满足返回config。main继续调用原run_vdma_frame_buffer(...,1,1,BOTH)，不连接GIC，不限定帧数停止。
2. CurrFrameStore读取当前READ槽0..2。它是DDR槽编号，不是新画面的递增序号。VDMA API读写控制/状态寄存器，不是通过API读取像素流；像素实际位于DDR。
3. StartParking请求READ停留在目标槽。请求需跨帧边界生效，不能立即复制。清掉旧MM2S frameflag后等两个新MM2S事件，给请求跨越边界并完成稳定显示周期的时间。事件次数本身不足以证明安全，仍须下一步所有权检查。
4. 两事件共用XTime计时的200ms超时，每次轮询检查READ/WRITE错误。确认READ仍在目标且WRITE不在目标；内部动态Genlock应使S2MM避开READ保护帧。异常返回timeout、dma_error或ownership等，不无限等待。
5. DCache保持启用，VDMA启动前flush三帧数组，之后CPU不写摄像头帧。源帧整帧invalidate后再读，避免CPU读到DMA写DDR前的旧缓存。
6. 保持READ停车，memcpy完整921600字节到CPU快照，再memcmp逐字节比较；不一致返回copy_mismatch。这只验证CPU看到的原始字节搬运一致，不证明传感器图像内容、颜色、采集端正确，也不证明新帧唯一。
7. 所有停车后出口请求StopParking恢复READ循环，不主动停止摄像头/S2MM。DMA故障可能需用户复位重启，不保证画面恢复，不自动清错/reset。失败输出可能未完成，禁止使用。

## 计时及正常/异常现象

main用XTime/COUNTS_PER_SECOND从本轮起点计100ms目标周期，复制、比较、快照flush和串口时间都包含在周期内；超时直接下一轮，不在工作后额外等待100ms。每约一秒输出一行capture_success_count、capture_failure_count、last_capture_us、last_failure；窗口结束清零次数及失败原因，避免旧失败冒充新失败。没有逐像素输出。

正常目标约10次成功/秒、失败0、无VDMA错误，复制期间显示可能短暂定格；成功意味着原始复制逐字节比较通过。100ms调度不保证10个独立新帧，不保证板上总能达到10fps。config表示当前运行配置不满足安全取帧前提，不会擅自修复；持续DMA错误或超时应检查硬件/配置，必要时由用户复位重启。

## 用Vitis查看完整raw snapshot

启动时串口只打印一次snapshot_address与frame_bytes。每次成功返回后main flush camera_snapshot，目的仅为让调试器从DDR看到CPU写入的完整快照；CPU自身随后读取快照不需要这次flush。此flush与复制前源帧invalidate方向、目的不同。

以调试配置构建，在main中成功分支之后的XTime_GetTime(&capture_end)行设条件断点capture_result == CAMERA_OK。断点命中时本轮capture已返回成功、snapshot已经flush。再打开Vitis Memory视图，使用串口地址或camera_snapshot符号查看921600字节；若导出为文件，应按640×480、BGR888、stride1920解释。不要随机暂停在memcpy半途后把未完成数据误作快照，也不要使用失败轮次输出。调试暂停会打乱100ms周期，恢复后的一秒统计不能当稳态性能数据。

## 验证边界

仅做ARM GCC全部应用C语法检查及静态差异检查，日志在仓库build/hls/camera_static_check。未综合、未跑仿真、未下载或运行硬件。本轮不进行图像精度验证；板测还需确认停车保护、完整复制、异常路径、解除停车请求后的实际画面以及稳态计时。
