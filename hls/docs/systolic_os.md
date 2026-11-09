# 传统 Output-Stationary 脉动阵列

本文件对应 `cnn_accelerator/src/systolic/systolic_OS.cpp`。`systolic_OS` 只保留 YOLO 按块实现，已独立实现并通过静态语法检查，尚未接入整网或做功能仿真。旧 MNIST OS 不再保留；仍未迁移的 MNIST 调用点使用已有的[广播式接口](systolic.md)。

## YOLOv2-tiny 重构计划

2026-11-09 确认的首版范围：固定九层 YOLOv2-tiny VOC、单套 OS 阵列、一次调用处理
一个 Cout 驻留块。各层统一按**最多 32 个 Cout 驻留**，Conv8 也按 32 设计；实际综合后再决定是否缩小。
先实现和验证功能。权重加载重叠是后续明确要做的性能优化，首版不实现预取。
全网调度见 [YOLOv2-tiny 重构总案](yolov2_tiny_refactor_plan.md)。

### 阵列与归约

- 保留 `4 spatial × 4 Cout` OS 阵列、PE 内 Dot8、平衡加法树和 INT32 累加。满载时一个逻辑推进步执行 128 路 INT8 乘法。32 Cout 是存储容量，不是 PE 列数。Activation 向右传播、Weight 向下传播，两侧 lane 按 `0/1/2/3` 步错拍；PE(i,j) 在 `token+i+j` 步取得同一 token 的两个操作数。
- 以 `ConvGeometry` 替代 MNIST 的 `layer_mode`。令 `G=ceil(Cin/8)`、`R=geometry.reduction_steps=Kh*Kw*G`，token 顺序为 `kh → kw → cin_group`，即 `token=(kh*Kw+kw)*G+cin_group`。Activation 与 Weight 的第 `token` 个 Pack 必须对应同一归约位置。
- 每个 `spatial_tile × local_cout_tile` 开始时初始化累加器、错拍及传播状态；连续注入 R 个有效归约 token，再注入 6 步无效 token 排空。Group 或 kernel 位置切换不清零、不输出。排空后只输出一次完整 INT32 Tile。`R+6` 是逻辑步数，不含缓存首读、初始化、接口握手或调度开销。
- 3×3 与 1×1 共用同一条 MAC 路径。Conv1、Conv2、Conv8 的 R 分别为 9、18、1152；Prediction 为 1×1、Cin=1024，R 为 128。Padding 和 1×1 的空间组装由 Im2Row 完成。最大逻辑归约 `3*3*1024` 个乘积，绝对值保守上界 `9216*128²` 在 INT32 范围内。

### 调用粒度、循环和流守恒

唯一 OS 入口为 `systolic_OS(activation_stream, weight_stream, output_stream, geometry, first_cout_tile, block_cout_tiles)`，输出流使用 `yolo_acc_tile_t`。不提供 `layer_mode` 重载；旧广播式入口与 `acc_tile_t` 不变。

一次调用以 `ConvGeometry`、起始全局 4-Cout Tile 编号 `first_cout_tile` 和本块实际计算 Tile 数 `B` 为固定配置。`B` 最大为 8；Conv1 只需 4 个 Tile，不人为装满缓存。外层控制器负责遍历 Cout 块，每块重新供给一次完整 Feature Map，Im2Row 输出一次完整空间扫描。Systolic 内部不读 DDR，也不触发重扫。

```text
整层控制器：按全局 Cout 顺序遍历驻留块
    Systolic 单次调用：先加载本块权重（local_cout_tile → token）
    for spatial_tile，按 y → tile_x：
        for local_cout_tile = 0 .. B-1：
            本空间 Tile 的首个计算 Tile：直读 Im2Row，同时填 Replay
            其余计算 Tile：从 Replay 读，不再消费 Im2Row
            读取驻留权重，归约 R 步，排空 6 步
            写一个 INT32 结果 Tile
```

设 `S=geometry.spatial_tiles`，单次调用读取 **`B*R` 个权重 Pack**、**`S*R` 个 Activation Pack**，输出 **`S*B` 个结果 Tile**。整层权重各读一次，不随空间重扫重复加载；Activation 的整层输入量按 Cout 块数乘以 `S*R`。一块结束后下游自然排空，再切下一块；层间还要等 DDR 写回完成。

通过阻塞 stream 自然传递反压，不增设 pause/busy 信号。Replay 期间上游可在 FIFO 容量内推进，满时由 stream 阻塞。集成时要保证权重能先到达 Systolic；不能让同一 Loader 卡在向浅 Feature FIFO 写数据，还没发出 Systolic 正在等待的权重。

### 权重加载和缓存

沿用 Python 冻结的物理 BIN：`global_cout_tile → kh → kw → cin_group → cout_lane → cin_lane`。**一个权重 Pack 固定为 `4 Cout × 8 Cin` 的 32 字节**；Loader burst 读的是当前块的连续文件区间，向 Systolic 逐 Pack 发送，不一次发送 32 个 Cout。AXI beat 与 Pack 是不同粒度，例如 64-bit AXI 可用 4 个 beat 拼出一个 Pack。

```text
DDR 起始字节地址 = layer_weight_base + first_cout_tile * R * 32
本块有效字节数   = B * R * 32
本地权重地址     = local_cout_tile * R + token
```

- **Weight Buffer**：Systolic 独占一份缓存，按 4 个 64-bit Cout Lane Bank 组织，每路逻辑深度 `8*1152=9216`，实现为 `8192` 主段和 `1024` 尾段两个单口 BRAM。计算时每路取同一逻辑地址的一字，合计提供 256 bit/步。首版整块预加载后才计算，写入与读取分阶段；逻辑载荷仍为 288 KiB。Loader 不另留一份完整块缓存。
- **Activation Replay**：4 个 64-bit Spatial Lane Bank，每 Bank 最大深度 1152，逻辑载荷 36 KiB。只缓存一个 Spatial Tile，不保存完整 Feature Map。`spatial_valid` 从该 Tile 首 token 取出、单独保存。`B=1` 可跳过 Replay 写入，但支持多 Tile 的同一份硬件仍需预留该缓存。
- 深度使用 `config.h` 的 YOLO 派生容量，地址递增不要求内循环每步乘法；不将本地 8 个 Cout Tile 展开成 8 套 RAM 或 PE。实际 BRAM 数量、读延迟、地址对齐和 `II=1` 需综合确认，逻辑载荷不等于器件占用。

### 权重 Bank 分段与 BRAM 预算

旧实现每路只有一个 `9216×64` 数组。Vitis HLS 估算每路需要 64 个 BRAM18K；旧 Vivado 2022.2 checkpoint 中，每路确实有 32 个 `RAMB36E1`，均以 A 口 2-bit 读写、B 口不用，即横向拼成 `16384×64` 的资源容量。逻辑数组仍只访问 9216 个地址，余下深度是映射碎片，不是多缓存了一份权重。四个权重 Bank 共占 128 个 RAMB36；加上四个 Replay Bank 原本需要 16 个，超出器件的 140 个。因此旧 Vivado 综合把一个 Replay Bank 自动转为 LUTRAM，最终报告 140 个 RAMB36 加 1152 个分布式 RAM LUT。

当前实现保留逻辑地址 `0…9215`，用地址 bit 13 选择存储段：`0…8191` 取低 13 位访问主段；`8192…9215` 取低 10 位访问尾段 `0…1023`。装载和回放使用同一映射。分段与 `local_cout_tile` 边界无关，跨段不增加 token、跳过地址或改变顺序。

按每路 `8192×64` 约 16 个 RAMB36、`1024×64` 约 2 个 RAMB36 估算，四路权重目标约 **72 个 RAMB36**。Replay 源码保持不变；若四路 Replay 都回到 BRAM，Systolic 目标约 **88 个 RAMB36 等效容量**。以上是设计预算，拆分后只做了静态检查，尚未重新综合。复核时需确认主/尾段实际映射、Replay 不落 LUTRAM、跨段读返回对齐以及归约 II/时序。

### Valid 与结果交付

- **Cin 尾通道**是输入数值契约，不是空间 mask：PyTorch 权重 BIN 与 Im2Row 都已对组内无效 Cin 填零。YOLO Systolic 直接使用固定 Dot8，不重复清零；Loader 只恢复文件字节。TB 核对 RGB 的补零契约。
- **`spatial_valid`** 从 Im2Row 继承，适用于当前 Spatial Tile 的全部本地 Cout Tile。不存在的空间位置不更新累加器；卷积 Padding 的零值不使真实空间位置失效。
- **`cout_valid`** 是 4-bit 的计算 Tile mask。在每个本地 Cout Tile 开始时，以 `global_tile=first_cout_tile+local_cout_tile` 判断是否为整层最后一个 Tile；若是则取 `geometry.last_cout_valid_mask`，否则为 `1111`。注入时令权重 lane 的传播 valid 等于 `注入有效归约 token && cout_valid[lane]`，随权重一起错拍、向下传播。排空只停止注入新的有效 token，不丢弃已在传播中的数据。PE 保留激活、权重与空间有效条件；无效 Cout 不更新累加器，输出为零。不删减 token 或结果事务，也不因此减少物理乘法器数。
- Prediction 的最后一块仍有 8 个物理计算 Tile、29 个逻辑 Cout；仅整层最后一个计算 Tile 的 `cout_valid=0001`，只对应 C124。不能将这个 4-lane mask 与 Writer 的 8 通道末组（5 个有效通道）混用。

结果契约为 INT32 `value[spatial_lane][cout_lane]` 加 `spatial_valid` 和 `cout_valid`，块内顺序 `y → tile_x → local_cout_tile`。块起点与 Tile 数是调用级配置，坐标不必随每个事务发送。Systolic 不缓存整图输出、不重排、不访问 DDR；PostProcess 根据全局 Cout 索引处理参数，先检查 `cout_valid` 再读 bias/M/S；Pool 或 Bypass 后，由 **Writer** 按最终空间坐标和全局 Cout 写回。Feature DDR 物理布局、Writer 的 8 通道组装和 burst 策略留到集成阶段。

### 后续性能优化：重叠权重加载

**保留为后续实施项，不作为首版功能验收前置条件。** 每个 Cout 块开始时的整块装载等待会跨各层累积。此前约 **20 ms** 是九层合计约 **15.12 MiB/图** 的权重，在 **100 MHz、64-bit 接口每拍持续传一字、加载完全不与计算重叠**时的理想传输估算，不是单层耗时，也不是板测值。实际开销还取决于 burst 效率、块启动和 DDR 争用。基线应分层记录权重装载周期、计算周期、阵列等权重/Activation 及下游反压，再比较整图延迟。

1. **优先恢复首 Spatial Tile 边算边填。** 参考现有 MNIST OS：首遍的权重 Pack 旁路进入阵列，同时写入原 Weight Buffer，随后空间位置只读缓存。该方案不增加第二份块缓存，也不要求 Weight Buffer 同拍读写。必须验证权重和 Activation 两路在任意先后到达、暂停或反压时仍按 token 对齐，不漏读、不重复写、不提前推进。加载慢于阵列时，首遍仍会等待，只能隐藏可重叠的部分。
2. **再评估跨块预取。** 若上一项后仍有显著启动开销，再研究当前块计算时预取下一块。完整双缓冲在 Conv8 最坏情况下仅权重就需 576 KiB 逻辑容量，不作为 Z-7020 默认方案。分段预取或用完即覆盖必须证明旧地址已不再被使用，并测量 DDR 读写竞争；需要额外资源与调度评估。

### 实现与验收顺序

1. 独立重构 OS（已实现）：接入块配置、可变 R、单缓存预加载、Replay 和两种结果 mask。Im2Row 与 Python BIN 格式不变。旧调用点移除 OS 切换分支，不迁移其数据流或测试数据。已通过 G++ 和 Vitis HLS 2022.2 Clang 的 C++14 静态语法检查；未运行 testbench 或综合。
2. 用真实 Im2Row 和权重 BIN，按块对照 Python 的 **bias 前 INT32 Systolic** golden，检查数值、mask 与每块流守恒。重点覆盖 RGB、多个 Group、Conv8 多块、Prediction 1×1 和最后 Cout Tile、W=13 行尾、连续调用；加定向非零无效 lane 测试确认 mask 屏蔽，不把该测试误当成允许违反 Cin 补零契约。
3. 以 `xc7z020-clg400-2`、10 ns 为评估基线，综合确认单套 `4×4×8` 数据通路、四 Bank 读带宽、实际 BRAM 余量、归约 II 和时序；RTL co-sim 验证输入气泡、输出反压、BRAM 返回与波前排空。32 Cout 为首版目标，不是整机资源已通过的结论。若放不下，优先评估仅 Conv8 降为 16、其余层仍最多 32，保持导出格式不变。
4. 接入后续 PostProcess、Pool、Loader/Writer 后测量整网基线，再按上节实现加载重叠。Feature DDR 布局、FIFO 深度、板级带宽和最终 fps 指标不阻塞 Systolic 首版独立实现。

