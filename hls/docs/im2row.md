# YOLOv2-tiny Im2Row 设计

更新日期：2026-11-02。状态：重构规划，尚未实现或完成综合验证。

本文只描述 YOLOv2-tiny VOC 的目标设计，不保留旧网络的设计章节。本轮仅修改文档，不修改源码、配置、接口或 testbench，不运行仿真、综合。总体背景见[YOLOv2-tiny重构方案](yolov2_tiny_refactor_plan.md)。

## 1. 已确定的范围

- 只支持总草案列出的416×416输入的YOLOv2-tiny层表，不承诺其他网络、分辨率或任意H/W/Cin组合。
- 单套物理Im2Row，空间并行度P=4，通道并行度8，signed INT8，量化零点0。
- 输入一个事务是一个空间位置的一个8通道组，即64 bit；输出一个事务是4个空间位置的同一8通道组，即256 bit数据与4-bit spatial_valid。
- 3×3模式：stride=1、四侧Padding=1；1×1模式：stride=1、无Padding。不扩展其他kernel、stride、dilation或Padding组合。
- 采用现有 `ConvGeometry` 结构承载调用级配置，扩展它的层表与必要字段，不另建平行配置体系，也不继续用二层bool模式选择配置。
- 每个空间Tile包含同一图像行中最多4个连续W位置，不跨图像行。3×3的吞吐目标仍是启动后、上下游可服务时稳态II=1，并争取Tile间及换行无额外准备气泡；BRAM实现能否达到须重新验证。
- 一次调用扫描一次输入Feature Map。Cout缓存块切换时由Top重新供给输入；Im2Row不负责Cout循环或Activation Replay。

空间Tile是否跨图像行，与Tile之间是否有时间气泡是两回事：Tile0可以覆盖第0～3列、Tile1覆盖第4～7列，二者的Token仍可以在时间上连续输出。

### 1.1 网络工作配置

| 阶段 | 输入H×W×Cin | Kernel | G=ceil(Cin/8) | 每Tile输出Token数K |
|---|---|---|---:|---:|
| Conv1 | 416×416×3 | 3×3 | 1 | 9 |
| Conv2 | 208×208×16 | 3×3 | 2 | 18 |
| Conv3 | 104×104×32 | 3×3 | 4 | 36 |
| Conv4 | 52×52×64 | 3×3 | 8 | 72 |
| Conv5 | 26×26×128 | 3×3 | 16 | 144 |
| Conv6 | 13×13×256 | 3×3 | 32 | 288 |
| Conv7 | 13×13×512 | 3×3 | 64 | 576 |
| Conv8 | 13×13×1024 | 3×3 | 128 | 1152 |
| Prediction | 13×13×1024 | 1×1 | 128 | 128 |

## 2. 接口和ConvGeometry

目标函数形式：

```cpp
void im2row(
    hls::stream<channel_pack_t>& input_stream,
    hls::stream<activation_pack_t>& output_stream,
    const ConvGeometry& geometry);
```

这是规划接口，尚未修改头文件。`pixel_t`如继续作为`channel_pack_t`的别名，不改变物理数据内容。

`ConvGeometry`提供H、W、逻辑Cin、kernel模式，以及校验一致的G、每行Tile数、K、调用输入/输出事务数和边界参数。派生值来自固定层表或公共构造逻辑，不同时手写互相矛盾的kernel、Padding和输出尺寸。它可以是全卷积流水共用的配置；其中Cout不参与Im2Row的循环和事务计数。

配置在一次完整调用期间不变。所有层经过同一个函数调用路径，不能按层实例化多套模板硬件。函数结束后逻辑状态回到下一次扫描的起点；不同层切换不能依赖RAM保留了什么旧值。

## 3. 输入、输出顺序和reference

定义：

```text
G = ceil(Cin / 8)
T = ceil(W / 4)
K = Kh × Kw × G
base_x = tile_x × 4
```

### 3.1 输入顺序

输入按 `y → x → group` 提供真实数据，不含空间Padding、不含不存在的空间Lane：

```text
input_id = ((y × W) + x) × G + group
pack.channel[c] 对应逻辑通道 group×8+c
```

消费输入时对 `group×8+c>=Cin` 的物理通道清零。Conv1仅C0～C2有效，Loader不必重复规范化其余5通道。DDR物理布局未冻结，但Loader必须提供这个stream顺序；不得把DDR块间填充当作图像像素送进来。

### 3.2 输出顺序

输出按 `output_y → tile_x → kh → kw → group`：

```text
q = kh × Kw + kw
reduction_token = q × G + group
output_id = ((output_y × T + tile_x) × K) + reduction_token
```

一个Activation Pack的4个Lane始终对应4个空间位置，且使用相同的q和group。对应关系为：

```text
output_x = base_x + spatial_lane
source_y = output_y + kh - padding_top
source_x = output_x + kw - padding_left
source_c = group×8+channel_lane
```

输出空间位置不存在、源位置属于Padding或源通道无效时，数据为0。`spatial_valid[s]`只判断 `base_x+s<W`，Padding不使有效输出位置失效。同一Tile的全部K个Token使用相同mask。

例：Cin=16，当前四个输出位置是 `(1,4)、(1,5)、(1,6)、(1,7)`，3×3模式的前四个事务如下：

| Token | (kh,kw) | 通道组 | Lane0源坐标 | Lane1源坐标 | Lane2源坐标 | Lane3源坐标 |
|---:|---|---|---|---|---|---|
| 0 | (0,0) | C0～C7 | (0,3) | (0,4) | (0,5) | (0,6) |
| 1 | (0,0) | C8～C15 | (0,3) | (0,4) | (0,5) | (0,6) |
| 2 | (0,1) | C0～C7 | (0,4) | (0,5) | (0,6) | (0,7) |
| 3 | (0,1) | C8～C15 | (0,4) | (0,5) | (0,6) | (0,7) |

每一行都是4×8个INT8数据，不是一个单独Pixel。

### 3.3 事务守恒

```text
每次调用输入Pack数 = H × W × G
每次调用输出Pack数 = H × T × K
```

上述数量不乘Cout Tile数。重复扫描的总量由Top按Cout缓存块数量计算。

软件reference直接使用源坐标公式；PyTorch unfold逻辑索引为 `k=((kh×Kw)+kw)×Cin+source_c`，不能把8通道物理尾部补齐计入逻辑Cin。

## 4. 3×3窗口和两列重叠

一个Tile覆盖输出列 `base_x ... base_x+3`，需要的输入列为 `base_x-1 ... base_x+4`，共6列。双Strip的逻辑内容为：

```text
strip[ping_pong][3行][6列][G]
3行依次为Top、Middle、Bottom
```

例如：

```text
当前Tile的输出列：4、5、6、7
当前Strip的源列：3、4、5、6、7、8

下一Tile的输出列：8、9、10、11
下一Strip的源列：7、8、9、10、11、12
```

两者共享源列7、8。重叠复制就是把当前Strip最后两列复制到下一Strip前两列，每列的全部Group都要处理。它不是复制输出Tile，更不涉及跨Cout比较。

预取指在当前Tile输出期间，提前从Line Buffer读出下一Strip需要的新Top/Middle列，例如上例源列9～12，再写入下一Strip。预取与重叠复制是不同操作。

## 5. BRAM组织和覆盖规则

### 5.1 Line Buffer

保留两行循环Bank、四路列Bank，每Word为一个8通道Pack：

```text
column_bank = x % 4
bank_address = (x / 4) × G + group
```

单Bank深度取支持层表中 `ceil(W/4)×G` 的最大值，当前为512。两行×四Bank×512×8 Byte的数据容量为32 KiB。按实际工作组合分配，不采用最大W与最大Cin的笛卡尔积。

处理输出行y时，一个Bank行保存输入行y，另一个原保存y-1并逐位置、逐Group覆盖成y+1。覆盖前必须确认旧值已进入仍需使用它的Strip。重叠列从当前Strip保留，不回头从已经覆盖的旧行地址取值。

### 5.2 Strip

优先评估18个空间Bank，即3个Strip行×6个Strip列。每Bank为64-bit Word，地址包含Ping-Pong和Group：

```text
depth = 2 × MAX_G
address = ping_pong × MAX_G + group
```

Group维不complete partition。当前MAX_G=128，双Strip数据容量为36 KiB。将Ping-Pong放入地址可避免机械拆成更多浅Bank，但当前读与下一Strip写会共享端口，因此优先评估每Bank一读一写RAM，而不是单端口。

一个输出Pack从同一Strip行上的4个不同列Bank各读一个64-bit Word。预取返回直接按Group写入下一Strip，暂存仅保留有限个在途Beat与地址标签，不把Group扩成大寄存器数组。

Line Buffer与Strip合计68 KiB是数据容量，不是实际BRAM消耗。实际占用还受宽深配置、端口与存储复制影响，必须纳入全系统预算。RAM类型参考[Vitis HLS 2022.2存储绑定说明](https://docs.amd.com/r/2022.2-English/ug1399-vitis-hls/pragma-HLS-bind_storage)。

## 6. 并行操作不等于端口冲突

寄存器化窗口可以同时读多个位置、复制旧列、写新列，因为这些位置有独立寄存器。BRAM也可以让部分操作重叠，但需要逐Bank计数，不能把“同一数组上有多个操作”直接等同于冲突。

以输出Bottom的kw=2、同时准备下一Strip为例：

| 对某个Bottom列Bank的操作 | 地址/来源 | 端口需求 |
|---|---|---|
| 输出当前窗口 | current区域的group地址 | 一次读 |
| 下一Strip该列写零或准备新值 | next区域的group地址 | 一次写 |

两次访问虽为不同Ping-Pong地址，仍在同一物理Bank。单口RAM不能同拍完成；一读一写RAM则有机会支持。这是需要改RAM类型/排程的具体情况，不是必然需要气泡。

当前Strip列4、5的值已经随kw=2窗口读出，可直接转发到下一Strip列0、1。若又为了复制独立读取相同源Bank，会额外要求读口；明确复用已读Word可避免这次额外访问。

Top重叠复制可放在输出Middle时，Middle重叠复制可放在输出Bottom时，使它们使用不同Strip行Bank。Line Buffer的预取和Strip输出本来就是不同存储，也不因同时发生而互相冲突。

真正需要排查的是：某个物理Bank同拍是否超过一读一写、是否同地址依赖，以及返回值是否在首次使用前到达。不能仅凭操作名称认定冲突，也不能仅凭双Strip就认定所有返回均已就绪。

## 7. 3×3候选调度

### 7.1 q和group分开计数

3×3仍只有9个kernel位置：q为0～8，不因为Cin增加而改变其含义。每个q内部遍历group=0～G-1。

如果保存扁平Token编号，则范围变成0～9G-1；例如G=128时需要表达0～1151，不能仍用4 bit。实现也可以不保存扁平Token，只使用q和group两个计数器。

对应事件必须处理全部Group。例如在q=3阶段复制Top重叠列，应按group复制G次，而不是只在扁平token=3时复制一次。q=3/6等逻辑释放点可以继续使用，但BRAM请求和写回的实际拍数要计入。

行计数需要覆盖416行、tile_x需要覆盖104个Tile，不能继续使用只容纳小尺寸的窄计数器。位宽变化来自尺寸和Group范围，而不是3×3 kernel本身发生变化；访问地址与one-past终点计数也要分别确定宽度。

### 7.2 启动及Bottom输入

启动按 `x → group` 预装首行，共W×G个Pack，同时建立首Strip的Middle并准备所需Padding。

对非最后输出行：

```text
first_x = tile_x==0 ? 0 : base_x+1
load_columns = max(0, min(W, base_x+5) - first_x)
新Bottom读取Pack数 = load_columns × G
```

Top/Middle输出共有6G个Token时隙，首Tile最多读5G个新Pack、同行内部最多4G个。输入按列遍历Group，填当前Strip Bottom并更新Line Buffer。该操作从q=0开始，按需持续到q=3/4，不仅限于Top的三个q阶段。

Bottom首次输出前，相应数据必须已写入且可读；真实RAM延迟需要排表确认。最后一个输出行的Bottom全是Padding，不读取下一输入行。

### 7.3 同行切Tile

下面是候选工作分配，不是已验证的逐拍RTL：

| 当前输出q | 当前使用的Strip行 | 允许评估的准备工作 |
|---:|---|---|
| 0～2 | Top | 开始填当前Bottom，按需持续到q=4 |
| 3 | Middle | 按Group复制Top最后两列 |
| 4 | Middle | 按Group预取下一Strip的4个新Top/Middle列 |
| 5 | Middle | 接收在途返回，确认Bottom可用 |
| 6 | Bottom | 按Group复制Middle最后两列 |
| 7 | Bottom | 完成必要的返回/写入 |
| 8 | Bottom | 复用窗口读出的列4、5，准备下一Strip Bottom重叠列及其余列初值 |

预取的4个新列为 `base_x+5 ... base_x+8`。其中真实位置读Line Buffer，越界位置写Padding零。两个Line Buffer行Bank每组各读4列，列模4互不冲突；若同拍有新Bottom写入，需确认相应行列Bank可支持一读一写。

不能只看最后一个Group已经发出请求就无条件交换角色。更准确的要求是：下一Tile将读取的数据已就绪，所有在途读写仍携带正确的Group、地址和Ping-Pong目标。

这不要求所有操作全局排空后才能继续。若目标地址已锁存、后继读取不会碰到未完成写入，上一Tile的尾部写回可以和下一Tile的其他Bank读取重叠。能否做到无额外准备气泡，以实际访问表和综合为准。

### 7.4 换行

下一图像行的首Strip需要真实源列0～4。列0和列4落在同一Line Buffer列Bank，因此用两个读请求阶段：

- q=6：按Group请求列0～3的Top/Middle。
- q=7：按Group请求列4的Top/Middle。
- q=8及相关返回流水级：完成写入、构造Padding，满足下一行首Token的读依赖。

下一行Top来自旧Middle行Bank；下一行Middle来自已填入新数据的旧Top行Bank。角色变化不得改变已发出请求的目的位置。

双Strip继续用于隐藏换行准备，不是取消它的功能；但寄存器改BRAM后，新增读延迟是否仍能隐藏，必须验证，尤其要覆盖G=1这个最紧的情况。

## 8. W=13每行末Tile的具体含义

“末Tile”在这里指每一条输出图像行的最后一个空间Tile，不只是整层最后一个Tile。W=13时每行分成：

| tile_x | 输出图像列 | 当前Strip覆盖的源列 | 从前一Strip保留的源列 | 新读取的Bottom真实列 |
|---:|---|---|---|---|
| 0 | 0～3 | -1～4 | 无 | 0～4 |
| 1 | 4～7 | 3～8 | 3、4 | 5～8 |
| 2 | 8～11 | 7～12 | 7、8 | 9～12 |
| 3 | 12，另3个Lane无效 | 11～16 | 11、12 | 无 |

末Tile的有效输出位置是第12列，3×3窗口横向只需要源列11、12、13。源列11、12已经从前一Strip留下，源列13在图像外，填0即可，不需要从输入stream再读一个新位置。

这里“不读取新Bottom”不表示Bottom行不存在，也不表示没有输出；该窗口Bottom的真实列11、12已经被前一个Tile读入。每行新Bottom读取列数是5、4、4、0，合计13，每列读取G个Pack。

不需要硬编码 `W==13` 特殊分支；普通边界公式应自然覆盖新读取数为0。最后一个输出行另按Bottom Padding处理。

## 9. Padding如何进入窗口

保持online Padding：在首Strip准备、预取返回写入下一Strip、以及构造Bottom初值时，将越界位置准备为零，不依赖Loader发送Padding，更不生成完整Padding Feature Map。

对每个Group都必须成立：

- 顶部第一输出行的Top为零，底部最后输出行的Bottom为零。
- 首Strip左边界、末Strip右边界为零。
- 预取源列越界时，不读取不存在的Line Buffer地址，向目标窗口提供零。
- 重叠复制搬运的是源位置已正确准备的原始Word；如果源位置是Padding，它的值也必须是零。
- 无效spatial Lane的最终Payload强制为零，与合法窗口Padding是两个不同条件。

“Padding覆盖预取和复制”不是说现有Padding只在输出处清零，也不是额外增加一种Padding。它要求在多Group/BRAM化时保留这些边界语义：不能为了省写操作直接删除原有零值准备，却仍读取或复制未初始化的RAM槽位。

写零也占RAM写口，必须进入端口表。可以在不冲突的启动或后台阶段按Group写所需窗口位置，不要求每次调用把整块Line Buffer和Strip RAM全部清零。

## 10. 1×1：收集与输出的关系

1×1不读3×3邻域、不减1、不加Padding。仍按图像行切空间Tile，输出按group将对应位置组合成4个Lane。

满Tile输入依次为：

```text
位置0：g0、g1、...、g(G-1)
位置1：g0、g1、...、g(G-1)
位置2：g0、g1、...、g(G-1)
位置3：g0、g1、...、g(G-1)
```

而一个输出Pack需要位置0～3的同一个group。输入4G个64-bit Pack、输出G个256-bit Pack完全正确，没有数据量错误。

### 10.1 推荐评估：最后位置到来时直接输出

不必先收齐4G个Pack，再单独花G拍输出。可以：

1. 将前3个位置的各Group写入3个64-bit Bank。
2. 读取第4个位置的group g时，从3个Bank读取同组数据。
3. 用缓存的前三个位置和当前输入的第4个位置组成输出Pack，读延迟通过少量寄存器对齐。

G=128时暂存前三个位置的数据量为3 KiB，可评估复用Strip的3个列Bank，不另建完整Feature缓存。实际RAM读请求、直接输入、mask与Group必须对齐。

尾Tile有r个真实位置时，只缓存前r-1个位置，最后一个位置直接参与输出；W=13的最后Tile有r=1，可直接将输入放在Lane0，其余Lane为零。

这属于待实现的优化方案，不是已经验证的吞吐结果。若要求输出不受最后位置输入气泡影响，则需完整Gather或额外预取缓冲，另算存储与延迟。

### 10.2 “收集间隙”不等于PE必然空闲

在单路输入每拍最多读取一个Pack的前提下，理想满Tile时间分配是：

```text
前G拍：缓存位置0，尚无完整四Lane输出
再G拍：缓存位置1，尚无完整四Lane输出
再G拍：缓存位置2，尚无完整四Lane输出
最后G拍：位置3边输入，边输出G个Pack
```

因此可争取每4G拍处理一个满Tile，而不是先收集再输出的5G拍。最后G拍可以连续输出；整层不能把4G拍内仅有G个输出误报为每拍持续输出。这是输入/输出宽度比例与输入顺序形成的速率上限，不是功能缺陷。

Systolic在Replay其他Cout计算Tile时不需要新Activation，Im2Row可以利用这段时间准备后续位置。间隙是否真的导致PE等待，需要结合Cout驻留数量、FIFO、DDR供给及阵列排空测算，不能只看Im2Row就断言系统停顿。

## 11. 反压、状态与初始化

- Line Buffer、Strip与相关状态优先由一个Im2Row过程拥有，不为读写拆分而制造跨DATAFLOW过程的共享数组所有权问题。
- 输入未成功读取时不推进输入坐标或group；Bottom未就绪时不能输出依赖它的数据。
- 输出阻塞时保留完整Pack和相关标签，不重复读输入，不释放仍需使用的数据。
- BRAM在途读写携带目的地址与Group/Ping-Pong标签，不能因为角色计数器变化而误写下一Tile。
- 1×1直接输入与缓存读返回一同受流控，不能丢失其中任一部分。
- 调用结束前完成本次必要的存储提交；下次调用重置逻辑状态，不依赖RAM复位内容。

阻塞stream保证事务握手，具体流水寄存器与RAM在途访问如何暂停仍须RTL验证，不能单凭 `write()` 的源码位置认定所有状态同拍冻结。

## 12. 实施前仍需确定的事项

已确定仅支持YOLO层表、采用ConvGeometry、保留8通道接口；这些不再作为待选问题。

仍需确定：

1. Im2Row的BRAM预算、目标时钟和RAM读延迟。68 KiB只是Line Buffer加Strip的数据容量，实际物理消耗需核算。
2. 18-Bank、Ping-Pong合并地址的一读一写组织是否满足完整端口表；不满足时先讨论排程或Bank调整，不自行增加LUTRAM或复制存储。
3. 3×3无额外准备气泡目标在G=1、行尾、最后Group及反压时能否保持，不能沿用寄存器实现结论。
4. 1×1是否采用上述前三位置缓存、最后位置直接输出，是否还要求与输入气泡隔离的整段G连续输出；后者需要额外预取保证。
5. ConvGeometry新增字段、派生值的统一构造与校验，以及新测试和Top的同步迁移细节；配置结构方向本身已确定。

DDR布局与Cout缓存容量尚未冻结，但Loader只要遵循本文件输入契约，Im2Row不应直接依赖DDR地址或Cout数量。

## 13. 后续验证与本次范围

先完成每Bank逐拍读请求、返回和写入排表，再编码。不要以 `DEPENDENCE false` 隐藏真实依赖，不将G维完全展开，也不新增完整Im2Row矩阵或整张Padding缓存。

按后续获准阶段进行：

- 静态检查：ConvGeometry字段一致、固定数组容量、地址与终点位宽、函数签名。
- 功能测试：只针对层表工作配置，覆盖G=1/2/128、RGB poison、九个kernel位置、1×1无Padding、W=13尾Tile、输入/输出事务守恒及连续模式切换。
- 综合：真实RAM映射、Bank端口、BRAM/LUT/FF、输出循环II、单套物理实例。
- co-sim：输入气泡、输出反压、预取返回跨角色切换、最后Group/行/调用排空；同时测量启动、Tile间隔与层内平均速率。

testbench使用简单确定性数据和源坐标reference，保留详细中文注释、CSV与必要终端统计，不让reference复刻Strip调度。

本轮仅更新本文件及总草案中的对应结论，不执行上述流程，不修改DUT或其他工程。
