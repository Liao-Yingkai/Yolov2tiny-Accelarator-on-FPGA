# MaxPool 设计

## 范围与接口

`maxpool` 固定实现 `2×2`、`stride=2` 的有符号 INT8 MaxPool：

```cpp
void maxpool(hls::stream<activation_tile_t>& input_stream,
             hls::stream<channel_pack_t>& output_stream,
             layer_mode_t layer_mode);
```

接口只支持两种固定模式；具体形状、事务数、Line Buffer深度和channel-pack数由`src/config.h`的编译期表达式生成：

| 模式 | 输入 `(H,W,Cout)` | 输入 Tile | 输出尺寸 | 输出 pack |
|---|---:|---:|---:|---:|
| Pool1 | `LAYER1_MODE`，`28×28×8` | `28×7×2 = 392` | `14×14×8` | `196` |
| Pool2 | `LAYER2_MODE`，`14×14×16` | `14×4×4 = 224` | `7×7×16` | `98` |

综合态由调用级`layer_mode`选择Pool1或Pool2，模式在单次层调用中保持稳定且不进入stream事务。`channel_pack_t`是`pixel_t`的类型别名，一个输出事务表示一个空间位置的连续8个物理通道。

## 数据顺序与有效位

输入严格为：

```text
row -> tile_x -> cout_tile
```

每个 `activation_tile_t` 有4个连续空间 lane 与4条连续 Cout lane。输出不携带坐标或无效占位，紧密顺序为：

```text
pooled_H -> pooled_W -> channel_pack -> channel
```

对每个输入空间 tile，两个横向窗口的位置有效位固定为：

```text
pool_valid[0] = top[0] & top[1] & bottom[0] & bottom[1]
pool_valid[1] = top[2] & top[3] & bottom[2] & bottom[3]
```

Pool2每行最后一个 tile 的 `spatial_valid=0011`，因此只写出 position0；position1完全跳过。非综合态会检查同一 `(row,tile_x)` 的所有 Cout tile 具有相同 `spatial_valid`。

## Line Buffer 与计算

Line Buffer 是16个顺序寻址的128-bit字：

```text
line_address = 0 at the start of each row
line_buffer[line_address++] for every input Cout transaction
word[(spatial_lane * 4 + cout_lane) * 8 +: 8]
    = activation_tile.value[spatial_lane][cout_lane]
```

每个字保存INT8的二进制补码位型；读出后先恢复 `activation_t`，再进行有符号比较。Pool1使用地址`0..13`，Pool2使用`0..15`。偶数行只读输入流并写BRAM，紧随的奇数行只读BRAM并读取输入流，因此设计请求 `ram_1p` 单端口BRAM，且不会在同一控制分支提出读写同端口请求。

每个奇数行输入事务同时生成两个 `ap_uint<32>` half：每个half包含一个pooled位置、当前Cout tile的4个结果。`pooled_half[2][4]` 完全分区；所有Cout tile完成后，Cout tile `0+1` 合成第一个8-channel pack，Pool2的 `2+3` 合成第二个pack。输出使用一个按`position -> pack`映射的固定`output_slot`循环，目标`II=1`，每次迭代最多写出一个channel pack；无效position的slot只跳过写操作，不产生占位事务。每个pool结果采用平衡的两级有符号比较树。

## 层间连接

Pool1输出由顶层顺序写入深度196的 `hls::stream<channel_pack_t>` BRAM FIFO，之后按原顺序作为Conv2 Im2Row的输入。这是因为两层卷积共享同一套物理卷积流水硬件，两个阶段顺序执行。

Pool2默认直接连接独立FC模块。若FC为多个输出tile回放784个activation，其输入缓存属于FC内部；Pool不要求、更不强制顶层设置FC Input Buffer。FC不复用当前Systolic。

## 当前验证状态

本阶段仅新增静态语法检查与独立 `test_pool.cpp` 测试源码；未运行C simulation、RTL co-simulation或C synthesis。Line Buffer实际BRAM映射、输入循环II、时序、资源与反压行为均待后续Vitis HLS验证确认。
