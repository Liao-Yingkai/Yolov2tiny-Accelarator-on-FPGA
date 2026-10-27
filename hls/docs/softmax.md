# 定点 Softmax

本轮只静态检查，未运行DUT、testbench、C-sim、co-sim或综合。
功能、精度、资源、RTL和时序未验证。参考允差10个万分单位仅为待执行TB标准。

接口：`softmax(stream<logits_pack_t>&, stream<inference_result_t>&, ap_uint<32> scale_q20)`。
每次输入/输出各一包，输出同包携带10个uint16概率与原signed logits的first-max类别。
核心FCPost仍输出INT8 logits。有效输入要求scale_q20>0，且调用期间稳定；当前264174来自
s_fc=0.251935750246048。非综合assert检查正值契约，不为硬件新增错误控制端口。

求最大值后以signed9位计算delta=qmax-q_i，再转unsigned8，范围0..255。
40位P=delta*scale_q20；delta=0强制E=2^20，P>=16*2^20强制E=0。
截断最多忽略9exp(-16)，小于输出一个万分单位。
其余P安全窄化signed28取负，通过range原始位码赋给ap_fixed<28,8>，
调用HLS库定点hls::exp。输出<=0置零、>=1置2^20，其余取Q20 raw bits。
硬件不使用float/double/std::exp，不使用自定义AP_RND指数模板重载。

10个E保存在complete-partition小数组；24位sum至少为2^20，不会除零。
34位N=E*10000，共享normalize helper计算q=N/sum（窄14位）；
余数由N-q*sum求得，不额外取模。比较2rem与sum执行nearest ties-even，
最后clamp0..10000。各通道独立舍入，概率和允许10000±5，不另分配残差。
class按原logits最大值，并列取最小索引；舍入形成新的概率并列不会改变class。

最大值与类别用完整分区、展开的10→5→3→2→1比较树求出，连续索引区间等值选左，保留first-max。
指数循环与sum解耦，先写10个21位E，再用22位两项、23位四项及24位总和的显式加法树，
总和包含前八项与末尾两项。logits、概率、E及树数组均complete partition。
exp和normalize helper保留INLINE off并尝试PIPELINE II=1，两条通道循环也请求II=1但不UNROLL，
各保留一个昂贵helper调用位置，目标是共享流水运算而非复制十套exp/div。
真正实例数、资源、pragma接受情况与实际II必须以后续综合报告确认，当前未作保证。
使用Vitis HLS 2022.2 hls_math.h定点重载。

独立TB用稳定double/std::exp参考，覆盖全相同（各1000）、并列、signed极值、
delta0/255、当前及合成scale、连续调用、概率范围/和与stream清空，
并补充所有索引依次唯一最大、跨比较树分组并列与指数截断阈值前/等于/后的用例。
DDR ABI及部署迁移见[DDR Wrapper](ddr_wrapper.md)。

`cnn_accelerator/script/setup_softmax.tcl`仅建立`build/hls/softmax_test`工程，
以C++14注册Softmax源码和独立TB，设置softmax top、xc7z020-clg400-2及10ns约束。
脚本不调用仿真、综合、协同仿真或导出阶段；本轮也未执行此setup脚本。
`hls_ddr.tcl`的所有DUT/TB源码同样显式使用C++14，但该脚本原有运行阶段仍保留，本轮未执行。
