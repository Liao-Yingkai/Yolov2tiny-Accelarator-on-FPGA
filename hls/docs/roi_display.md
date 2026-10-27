# ROI外框与PS控制的数字显示

本轮只修改源码并做静态检查，不运行testbench、C-sim、co-sim、综合、实现或上板。
原CNN运算和DSP映射不变；新显示逻辑的II、资源及AXI-Lite接口推断需后续确认。

## 源码与显示位置

HLS源码仍在用户原工程目录：

```text
/home/liaoyingkai/High_Level_Synthesis/ov5640_480p_280border/hls_src/
    roi_border.h
    roi_border.cpp
    test_roi_border.cpp
    README.md
```

接口新增一个32位标量：

```cpp
void roi_border(hls::stream<video_axis_t>& s_axis_video,
                hls::stream<video_axis_t>& m_axis_video,
                ap_uint<32> display_control);
```

保持640×480、逐像素AXIS和`ap_ctrl_none`自由运行，不需要PS启动视频处理。
每次有限函数调用处理一个像素事务，目标`PIPELINE II=1`，不缓存整帧。
所有AXIS旁带保持原样，坐标随有效事务推进，SOF重新对齐。

原绿色4像素外框及280×280 ROI保持不变。新增数字面板位于
`[476,532) × [100,172)`，在原外框右侧、不覆盖ROI；56×72黑底中显示
5×7点阵放大8倍的40×56白色数字，四边各8像素留白。
数字字模是固定字符图案，与模型、Softmax scale或指数表无关。
关闭显示时，整个面板区域透传本帧原像素，不画黑底，不保留旧数字。

## PS控制

`display_control`通过AXI-Lite写入，偏移由HLS自动分配，PS使用生成的驱动：

| 位 | 含义 |
|---|---|
| 3:0 | 数字0～9；10～15无效并隐藏 |
| 8 | 显示使能 |
| 其余 | 忽略，PS写0 |

使能和数字用一次32位写入，避免分别更新两寄存器造成组合错位。
HLS只在SOF采样整个控制字，一帧内保持不变。复位或首次SOF前隐藏。
帧中改数字或关闭请求从下一SOF生效，因此最多需等一个视频帧才清除旧显示，
不是清除VDMA帧缓存，也不要求停止视频。

PS源码为`vivado/cnn_accel_cam/vitis/cnn_accel_cam/src/camera_display.h/.c`。
阈值在该头文件中，默认9500，概率编码为0～10000：

- 初始化成功后写0，确保没有旧显示。
- 有效数字且`9500 < probability <= 10000`时写`(1U << 8) | digit`。
- 恰好95%、低于95%或非法值写0。
- 取帧失败或CNN错误也写0，不保留上次成功结果。
- 正常采集/推理开始不提前清屏，避免每100ms周期闪烁；本次结果返回后更新。

PS/PL各自维护控制常量，一致性要求见[仓库README](../../README.md)。
UART仍输出数字与概率，不受显示阈值过滤；阈值暂不通过UART交互调整。

## Vivado与Vitis更新步骤

1. 在原HLS工程重新综合、导出`roi_border`，仍使用`xc7z020-clg400-2`、10ns、
   C++14及低有效复位。设计源添加`roi_border.cpp`，TB仅添加到testbench。
2. 更新Vivado中的ROI IP并Upgrade。视频连线保持原有：
   `VDMA/M_AXIS_MM2S → roi_border/s_axis_video → roi_border/m_axis_video → v_axi4s_vid_out/video_in`。
   因为显示在VDMA读出之后叠加，DDR原帧和CNN输入不会被数字污染。
3. 把新增`s_axi_control`接到PS的`M_AXI_GP0`外设互联。
   现有`ps7_0_axi_periph`的M00/M01/M02用于原设备，可新增M03连接此口；
   或对ROI控制口运行Connection Automation自动扩展互联。
4. ROI的`ap_clk`继续接同域100MHz FCLK0；低有效`ap_rst_n`接该域
   `peripheral_aresetn`。控制口也使用该时钟域。原25.2/126MHz显示时钟不变。
5. Address Editor自动分配ROI控制地址，Validate Design，再重新生成bitstream并导出XSA。
   不新增中断、DDR端口、start/done或额外视频DMA。
6. 用新XSA更新Vitis平台/BSP，确认生成`xroi_border.h`、`XRoi_border`及
   `XRoi_border_Set_display_control()`，并提供`XPAR_ROI_BORDER_0_DEVICE_ID`。
   当前无控制口的旧ROI IP不会生成这些驱动，不能与新PS源码直接构建。
7. 重新构建应用，确认`camera_display.c`被纳入编译；不要手改自动生成的Makefile。
   Run/Debug配置须选择新ELF，不能再次下载旧Debug程序。

本轮不修改现有BD、导出的RTL/IP、XSA、BSP或bitstream。

## 待执行的验证

TB使用确定性像素和独立ASCII字模参考，检查全部10个数字、原绿框4544像素、
ROI与旁带不变、开关/非法数字、帧中改值延迟到SOF、连续帧和事务守恒。
仅提供测试源码，没有运行。PS边界需后续核对9500隐藏、9501显示、
低概率和错误路径清除，并上板确认字形、位置、更新时刻与实际视频反压。

本版HLS DUT/TB已通过G++和Vitis Clang 3.9的C++14 `-fsyntax-only`；
PS `main.c`已通过真实BSP头文件下的ARM GCC严格语法检查。
`camera_display.c`当前因真实BSP尚无`xroi_border.h`而不能完成检查或构建，
没有增加临时驱动、伪造头文件或兼容占位。静态检查日志位于被忽略的
`build/hls/roi_display_static_check/`，这些检查不验证pragma推断、II、资源或硬件行为。
