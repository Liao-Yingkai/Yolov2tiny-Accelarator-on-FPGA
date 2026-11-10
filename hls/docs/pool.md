# YOLOv2-tiny MaxPool 设计

## 接口与事务顺序

`maxpool(input_stream, output_stream, geometry)` 一次处理一层，配置来自固定
`ConvGeometry` 层表。输入和输出均为 `yolo_activation_tile_t`：
`value[spatial_lane][cout_lane]` 是 4 个连续横坐标 × 4 个输出通道的 signed INT8。

流顺序为 **Cout 驻留块 → 行 → 空间 Tile → 块内 Cout Tile**。模块内部按层表
切换驻留块，不要求 Top 遍历块。同一 Tile 只含一行；输出按池化后的坐标重新紧密
排列，每个空间 Tile 最多 4 个连续结果。Writer 再把两个 4-Cout Tile 组装成
8 通道 Pack；Pool 不做通道拼接或 DDR 寻址。

| 卷积层 | 池化配置 | 输出宽度 |
|---|---|---:|
| Conv1–Conv5 | 2×2，stride=2，无补边 | 208 / 104 / 52 / 26 / 13 |
| Conv6 | 2×2，stride=1，只在右侧、下侧各补 1 | 13 |
| Conv7–Conv8、Prediction | Bypass，逐 Tile 原样转发 | 13 |

输入 Tile 数为 `H × ceil(W/4) × ceil(Cout/4)`，输出 Tile 数为
`Ho × ceil(Wo/4) × ceil(Cout/4)`。这不是层表里 `output_transactions`
表示的 8 通道 Pack 数。池化层的 Cout Tile 全有效；输出尾位置由 `spatial_valid`
标记，非法位置 payload 为 0。Bypass 保留两种输入 mask 和 payload，含
Prediction 的最后一个 Cout Tile（`0001`）。

## 一行原值缓存

地址按 `tile_x × 当前块的 Cout Tile 数 + local_cout_tile` 排列。一个 128-bit
字保存 4×4 个**原始** INT8，缓存深度由活跃 Pool 层表推导，为 416 字。没有全层
feature map 缓存，也不在行缓存中保存水平 max。字内顺序为 spatial lane 在外、
Cout lane 在内；比较前恢复 signed INT8。

共享行缓存绑定为 `ram_s2p/bram`，请求 1 拍读延迟。stride=2 隔行只写或只读；
stride=1 对同一地址先取旧行原值，再写入当前行。未使用 pragma 强行忽略该依赖。
内层 Cout Tile 循环请求 `II=1`；实际读写排程、映射、II 和外层 Tile 切换开销
需综合确认。流端阻塞时不承诺固定吞吐。

## Stride=2：隔行缓存、合并半 Tile

偶数行写入原值；奇数行读取对应的偶数行并分别计算输入 lane (0,1) 与 (2,3)
的两个 2×2 最大值。第一个输入空间 Tile 的结果按块内 Cout Tile 暂存，与第二个
输入空间 Tile 的结果合并成 4 个连续池化位置。Conv4/Conv5 的最后一个输入空间
Tile 没有配对的右邻，直接输出，分别仅 lane 0–1 / lane 0 有效。无效输入不会
影响任何合法窗口。

## Stride=1：读旧行、写新行、跨 Tile

先将输入 row0 写入行缓存。之后当前输入行为 B，缓存的上一行为 A，生成输出行 A。
以同一个 Cout lane 为例，输入 Tile0 含列 0–3，Tile1 含列 4–7：

1. 处理 Tile0 时锁存 A0–A3，再写入 B0–B3。先算输出 x=0、1、2，
   暂存这三个结果与**原值** A3、B3，暂不输出完整 Tile。
2. Tile1 到来后，用 `max(A3,A4,B3,B4)` 补齐 x=3，输出 x=0–3，
   再暂存 Tile1 的前三个结果与原值 A7、B7。x=7、11 同理。
3. 末输入 Tile 只有列 12 有效。输出前一 Tile 后，等该行全部 Cout Tile
   处理完毕，再按相同 Cout 顺序输出 x=12（mask `0001`）。其右邻取 -128。

这些暂存按块内 Cout Tile 独立寻址；新行 Tile0 不读取上一行的边界暂存。真实
row12 之后用一条值为 -128 的虚拟底行生成最后一行输出，不读输入流、不写行缓存。
右/下补边参与合法窗口的有符号最大值，不把输入无效 lane 的零当成补边。

## 当前验证状态

本阶段仅新增静态语法检查与独立 `test_pool.cpp` 测试源码；未运行C simulation、RTL co-simulation或C synthesis。Line Buffer实际BRAM映射、输入循环II、时序、资源与反压行为均待后续Vitis HLS验证确认。
