# 独立 FC 设计

## Pool2 与 Exporter 映射

FC 固定实现 MNIST 的 `784 -> 10`。Pool2 输出为 `7 x 7 x 16` 个 signed INT8，
其紧密 NHWC 流顺序为 `pooled_y -> pooled_x -> channel_pack`：每个空间位置先是
channel 0..7，再是8..15。因此 FC 有98个输入 token，
`token=(y*7+x)*2+pack`，每个 token 含8个连续通道。

权重 exporter 顺序固定为 `out_tile(0..1) -> token(0..97) -> cout_lane(0..4)
-> cin_lane(0..7)`。一个 `fc_weight_pack_t` 是5x8个 signed INT8（逻辑320 bit）；
两个 tile 覆盖10类。参数也为两个 `fc_param_pack_t`，每组5路 signed INT32
bias、multiplier、shift。

当前 exporter 参数契约要求 multiplier 和 shift 非负；非综合态会断言该契约。

## 接口、缓存和吞吐目标

```cpp
void fully_connected(hls::stream<channel_pack_t>& input_stream,
                     hls::stream<fc_weight_pack_t>& weight_stream,
                     hls::stream<fc_acc_pack_t>& output_stream);
void fc_postprocess(hls::stream<fc_acc_pack_t>& input_stream,
                    hls::stream<fc_param_pack_t>& param_stream,
                    hls::stream<logits_pack_t>& output_stream);
```

第一个 output tile 直接消费98个 Pool2 token，并将每个 token 显式打包为64-bit
字写到 `activation_buffer[98]`。该数组请求 `ram_1p` BRAM；第二 tile 从它回放，
因此输入流只读取98个事务。权重流每次读取196个 Pack，输出两个5路 INT32
accumulator Pack；非综合态检查这些事务守恒。

每 token 的5个 cout lane 完全展开，8项 signed dot 使用明确
`INT16 -> INT17 -> INT18 -> INT19` 平衡树；5个 INT32 accumulator 完全分区。
token 循环目标为 `PIPELINE II=1`，token0 覆盖 accumulator，避免跨 tile/调用残留。

MAC 与后处理严格分开。后处理读取两个 accumulator/parameter Pack，按 tile 再按
lane 串行处理，每组5 lane 只结构化地复用一套32x32 signed requant datapath，最后
一次输出10路 logits。数值为 `bias add(33位检查) -> INT32xINT32=INT64 -> 直接
算术右移 -> clamp[-128,127]`；无 ReLU、无 rounding offset，`shift>=64` 与 NumPy
signed right_shift 一样符号扩展。lane loop 目标 `PIPELINE II=1`；实际DSP数量和II
须由后续综合确认。

## 与卷积 Systolic 的区别和状态

卷积 Systolic 面向3x3滑窗、4个空间 lane 和跨空间 tile 的权重回放；FC 没有滑窗，
只扁平化 Pool2 的 HWC 向量，并仅为第二输出 tile 缓存输入，不复用卷积阵列或其
权重缓存。本阶段只添加实现与 `test_fc.cpp` 源码，尚未运行 C-sim、co-sim 或综合。
测试源码覆盖完整784x10 MAC reference、HWC顺序、signed、两个输出 tile、负数
requant/shift40/41/饱和、事务守恒及连续两次调用。
