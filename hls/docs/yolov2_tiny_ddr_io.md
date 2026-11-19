# YOLOv2-tiny Loader、Writer 与层间 DDR 缓冲

三个 Loader 和 Writer 已接入 [YOLO DDR Top](top.md)，按张量或模型段基址
访问 DDR。Top 负责层级调度和 A/B 交换；源码顺序与 AXI 实际完成时序须分别验证。

## 1. 层与 Cout 块的控制

Top 遍历固定层表，每层只调用一次 `run_layer`。层内由并行进程运行：

```text
卷积生产端：逐 Cout 块调用 run_conv_block
  块内数据通路：Feature Loader → Im2Row → Systolic
整层权重通路：Weight Loader → weight_stream → Systolic
整层参数通路：Param Loader → param_stream → PostProcess
整层结果通路：Systolic → PostProcess → Pool / Bypass → Writer
```

每个 Cout 块从同一个输入基址调用一次 Feature Loader 和一次 Im2Row。两者与
Systolic 同处块内 DATAFLOW；上一块的三者完整结束后才启动下一块。Im2Row 的
重启是**重新调用**：3×3 从首行预填和起始 Strip 开始，1×1 从第一个像素开始
收集。不拉硬件 reset，不清空整块 BRAM，也不把多遍 Feature 串进一次调用。

Weight Loader、Param Loader、PostProcess、Pool 和 Writer 各运行一整层。
Systolic、PostProcess 按同一 Cout 块划分消费流；Pool 和 Writer 按同一块顺序
处理结果。卷积生产端可在上一个块的结果尚未完全写回时开始下一块，受 FIFO 反压
限制。不能等整层下游返回才发下一块，也不能先向浅 FIFO 送完整层权重再启动消费端。

层循环和卷积生产端的块循环不展开、不跨迭代流水化。复用同一处调用，后续用综合
层级确认实际只有一套 Im2Row、Systolic、PostProcess 和 Pool。

## 2. Feature Loader：一次扫描，负责通道尾补零

DDR 首图采用紧凑 RGB INT8 HWC：每像素连续三个字节 `R,G,B`，共
519,168 B。PyTorch、PS 不增添通道槽位；PS 可直接复制紧凑 BIN。中间层均为
8 的倍数，仍按普通 NHWC 存储，每像素的所有通道相邻，不在行尾插入额外数据。

Loader 一次读一遍真实输入，按 `y → x → cin_group` 发送
`H × W × ceil(Cin/8)` 个 `channel_pack_t`。一个 Pack 永远是**同一个像素**
的 8 个通道槽位，不跨像素取数据来填尾槽。有效通道放低 lane，最后一组超出逻辑
Cin 的槽位写零。DDR 输入的有效通道字节偏移为 `(y*W+x)*Cin+c`。对首版
固定层表，同一个 Loader 有两条读取路径：

- **紧凑 RGB**：连续读取 3 个 64-bit 字，拆成 8 个 RGB 像素，发出 8 个
  `[R,G,B,0,0,0,0,0]`。每行 416 像素可整分，无需跨行拼包或缓存整行。
- **8 通道对齐的中间层**：连续读取 64-bit 字，每字直接解成一个 Pack。
  字内第 `lane` 个 INT8 使用 bits `[8*lane+7:8*lane]`，保持原补码。

尾零只在发往 Im2Row 的流上生成，不写回 DDR。Im2Row 已移除
`normalized_word()`，直接接收已规范化的 Pack；仍用
`cin_groups` 控制读数和窗口 Token 数，不新增 `cin_valid`。3×3 空间补边
以及行尾 `spatial_valid` 仍由 Im2Row 处理；Padding 不使真实输出位置失效。

## 3. Weight Loader 和 Param Loader：整层独立供流

**权重。** 沿用导出 BIN 的
`global_cout_tile → kh → kw → cin_group → cout_lane → cin_lane` 布局。
每 32 B（4 个 64-bit 字）组一个 `weight_pack_t`，从层基址顺序读完
`ceil(Cout/4) × R` 个 Pack，`R = Kh × Kw × ceil(Cin/8)`。第 `k` 个
Pack 的起始字节偏移为 `32*k`，低地址的字节占低位。物理 Cin/Cout
尾权重已由导出器置零。Loader 不再补零、不重排、不按 32 Cout 重启，也不维护
整块权重副本。

Systolic 单次只消费当前块的 `block_tiles × R` 个 Pack，装入自己的驻留
缓存。计算期间不读权重；FIFO 填满后 Loader 由反压暂停，下一块开始装载时从
原位置继续。整层权重只读一遍。首版不引入整块权重双缓冲。

**参数。** DDR 保留每层连续的 `[bias 全部][M 全部][S 全部]` INT32，可分别
复制 `bias.bin` 和 `quant.bin` 得到；三个字段按逻辑 Cout 定长，不插入物理
尾槽。Loader 按全局 4-Cout Tile 递增发送 `ceil(Cout/4)` 个
`postprocess_param_pack_t`，从三个字段基址按通道索引取值。这是参数 Pack
顺序连续，不是整段 DDR 地址单调。对 Prediction 的 C125–C127，先判界再读，
对应槽位置零，不越界访问。PostProcess 自己缓存当前块使用的参数。

以层参数基址为起点，`bias[oc]`、`M[oc]`、`S[oc]` 的 INT32 字下标分别
是 `oc`、`Cout+oc`、`2*Cout+oc`。层之间可以对齐；同一层的字段之间不插空。

Feature、权重和参数须独立供流。不能让同一个顺序进程阻塞在发送 Feature，
却还没发出 Systolic 正等待的权重。

## 4. Writer：一个空间 Tile 的当前 Cout 块

Writer 一次处理整层，消费 `Cout 块 → y → tile_x → 块内 Cout Tile`。
`y`、`x` 均使用 **Pool/Bypass 后的输出坐标**。每笔
`yolo_activation_tile_t` 同时有四个空间 lane、每 lane 四个 Cout。

1. 对固定的 `(Cout 块, y, tile_x)`，按顺序收完当前块的 4-Cout Tile。
   相邻两笔的同一空间 lane 组合成一个 64-bit 字。暂存最多
   `4 空间 lane × 4 通道组` 个字，即 **128 B**；Conv1 的 16 Cout
   块只用其中两组。
2. 当前空间 Tile 的 Cout 块到齐后，按 `spatial_lane → 块内 8-Cout group`
   写 DDR。一个像素当前块最多连续写 4 个字，然后再写下一个像素。既不等整行，
   也不等所有 Cout 块到齐。
3. 写完再收下一组。暂存只由 Writer 访问，是一个小型私有缓冲；Pool 只能写
   FIFO。Writer 停止读流时让反压向上传递，不用共享 BRAM 或 ping-pong。

```text
C_groups   = ceil(Cout / 8)
global_g   = first_cout_tile / 2 + local_group
x          = 4 * tile_x + spatial_lane
word_index = dst_base_word + (y * output_w + x) * C_groups + global_g
```

`first_cout_tile` 以 4 通道为单位，固定层表中每个块起点均与 8 通道对齐；
`dst_base_word` 和 `word_index` 以 **64-bit 字**为单位。不同 Cout 块写同一
像素的不同完整字，不需读改写或提前清零目标区。无效空间 lane 不写；Prediction
按 128 通道的物理步长写，末组 C120–C124 保留，C125–C127 写零。

普通 NHWC 下，跨像素地址仍可能跳跃。例如 Conv8 每像素有 128 个字，本块仅写
连续 4 字。此处提供的是像素内短 burst 的机会，不保证 HLS 已推导出 burst。
Writer 每层消费 `output_h × ceil(output_w/4) × ceil(Cout/4)` 个 Tile，
写回 `output_h × output_w × C_groups` 个 64-bit 字。

## 5. 层间 DDR 两区与写回屏障

A/B 是同一个 Feature DDR 分配中互不重叠的两段地址，不是片上双缓冲。
PS 把初始 RGB 放 A，
Conv1 读 A 写 B，Conv2 读 B 写 A，依此交替。**一层内所有 Cout 块使用同一
对源/目标区域**。即使已写出一部分结果，也不能覆盖当前层会重新读取的源区。

两个等大区域各至少 **692,224 B**：首图占 519,168 B，最大写回 Tensor 为
Pool1 的 `208×208×16`。建议两区基址按 64 B 对齐；权重、参数和 Feature
地址范围独立。Prediction 完成后，结果在 B，逻辑形状为 `13×13×125`，
按 `13×13×128` 存 21,632 B。

只有**整层所有 Cout 块写回完成、AXI 写入完成**后才交换 A/B，并允许下一层
发读请求。最后一个 Pool Tile 产生、Writer 收到最后一个 Tile、甚至最后一个
写请求发出，都不能单独作为层间屏障。首版不用跨层预读或原地覆盖。

## 6. 检查入口与历史记录

当前整网已完成测试及上板功能验证，见[Top](top.md)。以下先记录独立实现阶段结果，再列出后续修改应重新核对的边界，不表示本轮运行了这些流程。

独立实现阶段通过 G++ 功能检查和 Vitis HLS 2022.2 Clang 静态检查。
九层均检查了 Loader 的顺序与尾槽、Writer 的 NHWC 地址和范围外哨兵；
首层另用 PyTorch BIN 验证紧凑 RGB 经 Feature Loader、Im2Row 后的逐元素结果。
检查不导出 CSV，该阶段未运行综合或 RTL co-sim。

1. 先独立验证三个 Loader 和 Writer：RGB 跨字拆包与尾零、参数末组不越界、
   多 Cout 块写同一像素、池化后行尾和 Prediction 末字。与逻辑 HWC golden
   逐元素比对，同时检查事务数和 DDR 目标区外的哨兵值。不导出 CSV。
2. 再接一层多 Cout 块，验证 Feature 每块从基址读一遍、Im2Row 每块重新启动，
   权重和参数却只各供一次整层；在 RTL co-sim 检查浅 FIFO 反压和排空。
3. 接连续两层，确认下一层首个 DDR 读不会越过上一层写回屏障。综合核对计算
   实例数、Writer 暂存映射和 AXI burst，再记录等窗口、等权重及写回停顿。

AXI bundle 分组、FIFO 深度、outstanding 数量和实际 burst 长度留到集成验证。
不把首版布局、独立模块 II 或顺序地址视作整机吞吐已达标。
