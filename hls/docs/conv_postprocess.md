# YOLOv2-tiny ConvPostProcess 设计

将 Systolic 的 INT32 累加结果转换为 Pool/Bypass 使用的 INT8 Tile。层级调用顺序见 [DDR Top](top.md)。

```cpp
void conv_postprocess(hls::stream<yolo_acc_tile_t>& input_stream,
                      hls::stream<postprocess_param_pack_t>& param_stream,
                      hls::stream<yolo_activation_tile_t>& output_stream,
                      const ConvGeometry& geometry);
```

## 1. 整层调用与 Cout 块调度

`conv_postprocess` 接收整层 `ConvGeometry`，输入为 `yolo_acc_tile_t`，输出为 `yolo_activation_tile_t`：含 `spatial_valid`、`cout_valid` 的 `4 spatial × 4 Cout` INT8 Tile。集成时 Top 按层调用 PostProcess，不传当前块的 `first_cout_tile` 或 `block_cout_tiles`；模块按与 Systolic 相同的块划分自行推导。PostProcess 不遍历 Cin Group，输入已完成整个卷积归约。

整个输入流顺序是 `Cout 块 → spatial_tile（y → tile_x）→ local_cout_tile`。32 Cout 块包含 **8 个** 4-Cout 计算 Tile；例子中 local tile `0…3` 对应 Conv1 的 16 Cout 小块，完整 32 Cout 块则为 `0…7`。先处理当前块在所有空间位置的结果，才切到下一组通道。

```text
for 每个 Cout 块：
    first = 本块的起始全局 Cout Tile
    B = min(geometry.resident_cout_tiles, geometry.cout_tiles - first)
    按 local_cout_tile = 0…B-1 读 B 个参数 Pack，准备本块缓存
    for spatial_tile = 0…geometry.spatial_tiles-1：
        for local_cout_tile = 0…B-1：
            读一个 INT32 Tile
            只用本地缓存[local_cout_tile]，生成一个 INT8 Tile
            按原顺序写出，传递两个 valid mask
```

每块读 `B` 个参数 Pack 和 `spatial_tiles*B` 个累加 Tile，写同样多的 INT8 Tile。整层读 `geometry.cout_tiles` 个参数 Pack，输入输出各 `geometry.spatial_tiles*geometry.cout_tiles` 个 Tile。块边界由计数确定，不额外发送块头或坐标。参数流按全局 Cout Tile 递增；BIN 里只有逻辑 Cout 个 bias/M/S，Loader 先按 `oc=(first+local_cout_tile)*4+lane` 判界再读，无效 lane 在 Pack 内填零，不读越界地址。Prediction 最后一个 Pack 只含 C124 一个有效 lane。

Systolic OS入口一次处理一个Cout缓存块。整网集成时，上游层级调度依次调用 OS，把各块接续到同一条输出流；PostProcess 同时按整层运行，不能让上游等待整层 PostProcess 返回后才供给下一块。Loader 须保证对应块参数可以到达，不能先因写满累加路径相关的浅 FIFO 而停在参数发送之前。Pool/Writer 按同一块顺序消费，层切换仍等待整层写回完成。

## 2. 参数缓存与一次 signed requant

配套模型在 Conv1–Conv8 的 LeakyReLU 之后量化，不在卷积与激活之间插入 activation
FakeQuant；Pool 沿用输入 scale。Prediction 无激活，采用卷积输出的量化 scale。
导出器为每个输出通道近似
`input_scale × weight_scale[channel] / output_scale ≈ M / 2^S`，
只导出一组 `M, S`。带激活前量化边界的 checkpoint 不能直接视为等价输入。

模块只缓存当前块的 `bias`、`M_pos`、`M_neg`、`S`，最多 32 通道，不保存整层参数或 Feature Map。块内按 4 Cout lane 并行读，local Cout Tile 为缓存地址。参数加载阶段用共享运算路径逐通道派生 `M_neg`，不复制 32 份常量乘法器：

```text
M_pos = 导出的 M
M_neg = (INT64(M_pos) * 1717986918 + 2^33) >> 34

a = accumulator + bias
m = geometry.use_leaky_relu && a < 0 ? M_neg : M_pos
out = clamp((INT64(a) * m) 算术右移 S, -128, 127)
```

负支系数只在装载时计算一次；`+2^33` 仅用于系数派生，逐元素乘移不舍入。正负支共用 S，且只做一次逐元素乘法。Prediction 装载时直接设置 `M_neg=M_pos`，不派生 LeakyReLU 系数，逐元素路径不再判断层模式。bias 相加须满足 INT32 契约；`S>=64` 按符号扩展。任一 mask 无效时，lane 不加 bias、不做乘移，输出零但不删掉 Tile。参数缓存仅在 Cout lane 维分区；每个 Tile 取一次本地参数，供四个空间 lane 复用。保留 4 路 Cout 并行、逐 spatial lane 处理；循环目标 `II=1`，实际 Tile 间隔、DSP 和移位器 LUT 留待综合确认。

## 3. 参数加载与缓存复用

模块先加载当前块的参数，再消费该块的首个累加 Tile。最后一个 INT8 Tile 写出后即可覆盖缓存，无需等待 DDR 写回。参数不预取、不使用双缓冲；等待期间由 stream 传递反压。

## 4. 验证

HLS Top设为`conv_postprocess`，设计源文件选`hls/cnn_accelerator/src/postprocess/conv_postprocess.cpp`，testbench只加入[`test_conv_postprocess_pytorch.cpp`](../cnn_accelerator/testbench/test_conv_postprocess_pytorch.cpp)。默认数据目标是仓库内`data/yolov2_tiny/`，样本为`0000`；路径定位条件见[HLS入口说明](../README.md#手动模块工程与路径限制)。数据准备见[数据说明](data.md)。每层读取同一套导出的四个文件：

- `trace/trace_<layer>_systolic_int32_hwc_0000.bin`：bias 前的 little-endian INT32 HWC 输入。
- `bins/<layer>_bias.bin`、`bins/<layer>_quant.bin`：参数按逻辑 Cout 排布；quant 先存全部 `M`，再存全部 `S`，均为 little-endian INT32。组包时先判 Cout 边界再读文件。
- `trace/trace_<layer>_postprocess_int8_hwc_0000.bin`：预期的 signed INT8 HWC 结果。

默认跑九层，一层调用一次。输入按 Cout 块、行、空间 Tile 和本地 Cout Tile 组包；输出按相同顺序恢复 HWC 坐标逐项核对，不重排完整输出。检查两种 valid、无效 lane 为零、BIN 字节数和流事务数。数值边界的定向测试与真实 BIN 对照分开，不修改导出文件。

**testbench 只读 BIN，不导出 CSV 或其他数据文件。** 终端输出每层事务数、验证元素数、错误数和 PASS/FAIL，错误详情限前 10 条，失败返回非零。Im2Row、Systolic、Pool和Top测试采用相同规则。

C 仿真不检查 RTL 时序和反压；修改流水或缓存后须独立验证这些行为。
