# DDR Loader / Writer Wrapper

本版只作静态检查，未执行 testbench、C-sim、co-sim、综合或 host DUT 计算。
未验证功能、精度、资源、RTL死锁、时序或板上行为。旧版16字节结果的历史报告不能验证本版。
旧IP/16字节结果协议不兼容，部署必须重新导出IP、XSA/BSP及bitstream。

## 接口与控制

`cnn_accelerator_ddr(image, weights, params, result, softmax_scale_q20)` 每次处理一图。
AXI-Lite寄存器偏移由HLS自动分配，保持ap_ctrl_hs。scale在调用期间稳定，当前模型264174。
PS必须使用本版RTL重新生成的驱动及寄存器定义，不使用旧BSP或手写固定offset。
image/result使用gmem_io，weights用gmem_weights，params用gmem_params。
有效元素数98/1133/102/3，有效字节784/9064/408/24。三个master共享HP0时带宽共同仲裁。
buffer须互不重叠、隔离cache line，建议64-byte对齐，按平台要求flush/invalidate及内存屏障。

## DDR布局

PL常量由config.h/ddr_layout.h维护；PS/PL常量维护说明见[根README](../../README.md)。
image/weights/params布局不变：低byte是较早INT8，图像为784个raster字节。
权重布局v2紧密拼接Conv1/Conv2/FC，word offset 0/9/153，总1133 words。
Conv1只保存真实Cin=1；FC为[2][98][5][8]硬件顺序。
参数每层按完整bias、multiplier、shift数组连续保存，层offset 0/24/72，
字段stride 8/16/10，总102个signed32元素。

### 结果：3个64-bit words / 24 bytes

| word | 内容（低位先） |
|---|---|
| 0 | uint16 p0、p1、p2、p3 |
| 1 | uint16 p4、p5、p6、p7 |
| 2 | uint16 p8、p9；高32位uint32 class |

p为0..10000，表示概率乘10000。class按原signed INT8 logits严格大于更新的first-max，
不是舍入后概率argmax。class byte offset=20，即word2 bit32，无保留字节。
writer显式打包，不依赖HLS struct对象布局。PS记录需检查sizeof=24、class offsetof=20。

## DATAFLOW与验证边界

原cnn_accelerator/FCPost仍输出一个10类signed INT8 logits pack，不改核心接口。
外层actor为image_loader、weight_loader、param_loader、compute_runner、softmax、result_writer。
Softmax单次写一个含10概率和class的inference_result_t，writer只消费此完整包。
不再分流logits或等待独立类别流；类别选择合并在Softmax中，不保留独立Argmax模块。
wrapper FIFO仍为2，core Pool1 FIFO、阶段barrier和通用权重loader均保持原实现。
weights/params保持独立loader，避免浅FIFO阶段依赖形成等待环。
Softmax数值与共享运算意图见[Softmax](softmax.md)。真正实例数、II、资源、AXI和有限FIFO反压待验证。

独立test_softmax覆盖极值、并列、不同scale、连续调用与单包守恒。
test_cnn_accel_ddr仍用原绝对路径trace/模型头构造三图输入；
expected logits仅供double Softmax参考与first-max，不当作实际DDR结果。
检查30概率（参考允差10单位）、概率和10000±5、类别、24字节区守卫及只读快照，
失败返回非零。scale头为本地model_softmax_params.h，不依赖PS应用路径。
CSV仍保存sample/output/expected/actual/match及输入raster。
本轮仅注册hls_ddr.tcl新源码，未执行脚本；脚本含csim命令，注册不表示测试通过。
本轮不启用HDMI数字或threshold。
