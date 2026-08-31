# MNIST FPGA Accelerator

本项目使用 Vitis HLS 2022.2 开发面向 MNIST CNN 推理的 FPGA 加速器。设计采用流式接口、片上缓存与空间并行，当前正在完成 Conv 前端的 Im2Row 模块；Systolic、卷积输出组装、Pool、FC 和顶层控制将在后续阶段实现。

## 工程原则

- 模块间使用阻塞式 `hls::stream`；
- 激活采用对称八位有符号量化，零点为0；
- 特征图逻辑布局采用 NHWC；
- `pixel_t` 打包一个空间位置的全部物理通道；
- 物理并行度是编译期参数，H、W、Cin、Cout 是运行时参数；
- 注释、文档、断言和工程说明统一使用中文；
- 性能和资源结论必须以综合与协同仿真报告为准。

## 当前数据流

```text
输入加载器或片上输入缓冲
           │
           ▼
标量、通道打包 pixel_t 流
           │
           ▼
Im2Row ── P 路 activation_pack_t ──► 广播型 P×P Systolic
                                              │
                                              ▼
                                 输出通道组装 / Pool / 后续 Conv / FC
```

## Im2Row 当前状态

Im2Row 固定支持 `3×3`、`stride=1`、`same padding`，每行独立划分 `ceil(W/P)` 个 Tile，不允许 Tile 跨行。尾部不足 P 的 Lane 在9拍中输出全零并携带 `spatial_valid=0`；卷积边界 Padding 也输出零，但有效 Lane 保持 `spatial_valid=1`。

当前网络为两层卷积：Conv1 的 Im2Row 输入是 `28×28`，第一层 Pool 后 Conv2 的 Im2Row 输入是 `14×14`。Conv2 后的 Pool 虽得到 `7×7`，但该数据不会进入第三层卷积，因此 `7×7` 不属于本 Im2Row 的实际工作尺寸。`W<P` 等小宽度仅保留作运行时接口健壮性与边界检查，不参与当前网络的时序和资源决策。

内部采用两个行 Bank 与 `strip[2][3][P+2]` 双 Strip Ping-Pong：

- `current Strip` 连续9拍输出给 Systolic；
- `next Strip` 在这9拍中并行准备；
- `token=8` 后交换，因此 Tile 间和换行间的目标是不出现输出气泡；
- 全局 Token 循环每次迭代只写出一个 `activation_pack_t`，目标 `II=1`；
- Bottom 输入在当前 Tile 的 Token 0～5 中按需突发读取，峰值为每周期一个 `pixel_t`；
- 首行预装后开始输出，输入总量为 `H×W`，输出总量为 `H×ceil(W/P)×9`。

完整的周期时序、双 Strip 所有权、换行两 Beat 预取、反压语义和未确认风险见 [Im2Row 模块详细设计](im2row.md)。

当前仅完成静态检查，未执行 C Simulation、C Synthesis 或 RTL Co-simulation。实际 II、Line Buffer 的 BRAM 端口与读延迟、资源、频率、反压和数值正确性仍需在后续验证阶段确认。
