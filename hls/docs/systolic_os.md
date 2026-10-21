# 传统 Output-Stationary 脉动阵列

本文件对应 `cnn_accelerator/src/systolic/systolic_OS.cpp`；广播式实现与其历史
结果仍见 [Systolic / Broadcast PE Array](systolic.md)。两者接口一致，但内部
数据传播和 Tile 时序不同，不应将广播式的9-token reduction时序直接套用到本实现。

## 数据流与保持不变的契约

- 一个调用处理完整 Conv1 或 Conv2，通过 `layer_mode` 选择固定层配置。
- 物理阵列为4 spatial ×4 Cout；每个 PE 并行完成8个 INT8乘法及平衡加法树，
  累加器为 INT32。保留原有16/17/18/19-bit乘积与树级位宽。
- Activation 向右传播，Weight 向下传播；两侧第0/1/2/3条输入分别延迟0/1/2/3步。
  PE(i,j)在逻辑步 `token+i+j` 累加同一token的两个操作数。
- 每个 Cout Tile 推进15步：前9步注入token0～8，后6步注入无效零事务排空。
  每个 Tile 开始前显式初始化 PE、边界、错拍线和 accumulator。
- 错拍线先取旧尾端、再从尾向头搬移；PE传播按反向索引更新，避免本次循环的
  新值被相邻 PE 当作上一拍数据使用。
- Activation 在首个 Cout Tile 边读边计算并缓存，后续 Cout Tile 本地回放。
  Weight 只在首个 spatial Tile 读取整层，随后本地回放。
- Activation 为4个9×64-bit BRAM bank；Weight 为4个36×64-bit BRAM bank。
  bank地址、channel bit序、流事务和输出 `row → tile_x → cout_tile` 顺序不变。
- Conv2行尾继续继承 `spatial_valid=0011`；无效 spatial lane 不更新 accumulator。
  Bias、requant、ReLU仍由独立 ConvPostProcess 完成。

计算核心与两个 Systolic testbench 统一读取 `config.h` 中的
`CNN_SYSTOLIC_USE_OS`：默认 `1` 选择该函数，`0` 选择广播式实现。
需要切换时修改唯一默认值，或对全部相关编译单元统一传入
`-DCNN_SYSTOLIC_USE_OS=0`；不再在各个源文件分别定义选择宏。
该选择不改变 Top 的调用接口或 DATAFLOW 拓扑。OS 的15步循环边界常量命名为
`SYSTOLIC_WAVEFRONT_STEPS`，表示逻辑波前步数，而非整个 Tile 的实际时钟数。

