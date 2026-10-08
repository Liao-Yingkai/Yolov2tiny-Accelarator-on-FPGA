# 独立 logits Argmax

## 接口与语义

`src/postprocess/logits_argmax.h` 暴露独立 `logits_argmax` 模块：每次调用从
`hls::stream<logits_pack_t>` 读取一个完整的10路 signed INT8 logits Pack，并向
`hls::stream<class_index_t>` 写入一个类别索引。`class_index_t` 为 `ap_uint<4>`，
固定网络的有效取值是0..9。相同最大值时选择较小索引，与从左向右扫描、仅在严格
更大时更新的 scalar argmax 一致。

本模块不改动现有 Top 或 logits 接口；未来可将它接在 FCPostProcess 的 logits 输出
之后。本阶段只增加独立的源文件和 testbench。

## 比较结构与综合结果

比较树显式按 `10 -> 5 -> 3 -> 2 -> 1` 组织，包含9个 signed INT8 两两比较，最长
比较路径为4层。每层的候选数组完整 partition，固定边界循环完整 unroll；pair helper
inline，以表达并行平衡树，而非串行逐项扫描。静态断言固定输入为10类。树中每个节点
的左、右候选始终来自递增且不重叠的索引区间，左区间索引更小；因此单次 signed `>=`
比较在相等时保留左候选，无须额外索引比较，也保证最终选择更小索引。

Vitis HLS 2022.2 对 `xc7z020-clg400-2`、10 ns 时钟约束的独立综合结果如下：

- Latency：2 cycles（20 ns），包含 stream 读写及控制开销；
- Interval：3 cycles；
- Estimated clock period：6.826 ns；
- 资源估算：59 FF、183 LUT、0 BRAM、0 DSP；
- 报告列出9个8-bit比较操作，无综合 warning。

独立 C Simulation 验证通过，0 errors。完整报告见
[logits_argmax_csynth.rpt](../../build/hls/logits_argmax_verify/project/solution1/syn/report/logits_argmax_csynth.rpt)。

## 独立验证

`testbench/test_logits_argmax.cpp` 是单 `main` 的轻量 C Simulation testbench，使用
独立的 signed scalar reference。确定性用例包括全负 logits、`-128/127` 端点、十个
位置逐一成为唯一最大值、非相邻并列时选择 first maximum，以及全相等时选择类别0。
每个 case 均检查一入一出和 stream 完全消费；不依赖完整网络数据或额外测试框架。
