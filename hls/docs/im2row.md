# YOLOv2-tiny Im2Row 设计

更新日期：2026-11-08。状态：固定YOLO配置补全与表内专门化已实现；新增读取真实YOLO BIN的Im2Row testbench，本次仅做导出数据核对及静态检查。2026-11-02的II=1、52个BRAM_18K等综合结果属于修改前版本，不覆盖后续新增字段或条件简化。新版testbench尚未运行C simulation或RTL co-simulation，不能据此声称DUT功能或RTL反压验证通过。

本文描述 YOLOv2-tiny VOC 的接口、存储契约和源码实现。独立模块阶段的检查与综合记录保留如下；后续整网集成结果见 [DDR Top](top.md)，不以静态检查代替仿真、综合或RTL反压验证。总体背景见[YOLOv2-tiny设计记录](yolov2_tiny_refactor_plan.md)。

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

已实现函数形式：

```cpp
void im2row(
    hls::stream<channel_pack_t>& input_stream,
    hls::stream<activation_pack_t>& output_stream,
    const ConvGeometry& geometry);
```

头文件采用此接口，输入数据类型为 `channel_pack_t`；Top 在每个 Cout 缓存块调用一次 Im2Row。

`ConvGeometry`提供H、W、逻辑Cin、kernel模式，以及校验一致的G、每行Tile数、K、调用输入/输出事务数和边界参数。派生值来自固定层表或公共构造逻辑，不同时手写互相矛盾的kernel、Padding和输出尺寸。它可以是全卷积流水共用的配置；其中Cout不参与Im2Row的循环和事务计数。

同一份九层表现已补齐下列阶段字段，不另建LayerGeometry，也不实例化逐层模板：

- `cin_group_shift=log2(G)`、`tail_spatial_lanes`；前者直接供Line地址移位。
- `pool_kernel/pool_stride`（均为0表示Bypass）、`pool_padding_right/bottom`；Conv1～5为2×2/2无补边，Conv6为2×2/1仅右/下各1，后续Bypass。
- `output_h/output_w`为Pool/Bypass后的尺寸，依次208、104、52、26、13、13、13、13、13；`h/w`仍同时表示输入与卷积输出尺寸，不再添加同义字段。
- `output_channel_groups=ceil(Cout/8)`、`output_transactions=output_h*output_w*output_channel_groups`表示完整阶段的逻辑8通道Pack数，不冻结DDR布局、对齐或缓存块写出顺序。
- `use_leaky_relu`仅前八层为真，Prediction为假；这里只保存模式，不实现后处理数值。
- `resident_cout_tiles=min(8,cout_tiles)`、`cache_blocks=ceil(cout_tiles/8)`、`last_cache_tiles`和整层末4-Cout Tile的`last_cout_valid_mask`。Conv1驻留4个Tile；Prediction末缓存块仍有8个Tile、29个逻辑Cout，末计算Tile仅Lane0有效。后续消费者须用全局Cout Tile编号判断有效性，不能把29误当Tile数。

YOLO专用容量从各层实际工作集取最大值：`YOLO_TINY_WEIGHT_BUFFER_DEPTH=max(resident_cout_tiles*R)=9216`，`YOLO_TINY_ACTIVATION_REPLAY_CAPACITY=max(R)=1152`。Systolic、PostProcess、Pool 和 Top 使用同一固定层表的阶段与缓存字段。

调用方必须传入表内配置：不再逐字段运行时匹配层表，不再为非法枚举返回空geometry。C++11兼容的编译期断言证明固定表前提；生产config仅保留Prediction专门化和容量等少量断言，逐层幂2 G、所有W≥5、唯一1×1、全部35字段、Pool尺寸及相邻层匹配等详尽校验放在build的静态检查TU，不建立运行时或生产递归校验框架。非综合构建仍保留真实读写事务守恒断言。

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

按 [DDR 搬运规格](yolov2_tiny_ddr_io.md)，Feature Loader 从紧凑 RGB 拆包并
补齐 `group×8+c>=Cin` 的无效 Cin；Im2Row 不再清零通道尾槽。
Im2Row 仍每笔接收固定 8-lane Pack，使用 `cin_groups` 控制事务数量；
`spatial_valid` 仅标记空间位置，不新增通道 mask。Loader 不发送空间 Padding。

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
bottom_end = tile_x==last_tile_x ? W : base_x+5
load_columns = bottom_end - first_x
新Bottom读取Pack数 = load_columns × G
```

Top/Middle输出共有6G个Token时隙，首Tile最多读5G个新Pack、同行内部最多4G个。输入按列遍历Group，填当前Strip Bottom并更新Line Buffer。该操作从q=0开始，按需持续到q=3/4，不仅限于Top的三个q阶段。

表内所有W≥5，首Tile终点固定为`P+padding=5`。后续仅在Tile切换时按更新后的`tile_x`选择真实行尾或候选终点，不为表外小尺寸生成通用min/max。真实行尾、底边、右Padding及未初始化旧Top读门控仍保留。

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

源码已采用此方案，以一个连续输入事务循环维护position/group/tile，跨Tile和行不重新启动流水；2026-11-02上一版本HLS报告该输入事务循环II=1，但不是每拍持续输出：满Tile仍须4G个输入事务才产生G个输出。本次简化未重新综合，实际流控吞吐尚未通过RTL验证。若要求输出不受最后位置输入气泡影响，则需完整Gather或额外预取缓冲，另算存储与延迟。

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

## 12. 已确认实现与后续待验证事项

已确定仅支持YOLO层表、采用ConvGeometry、保留8通道接口；这些不再作为待选问题。

仍需确定：

1. 本轮HLS确认BRAM组织及latency=1，资源为52个BRAM_18K；68 KiB是数据容量，物理资源预算仍须纳入全系统。目标周期10 ns、uncertainty 2.7 ns，HLS组合路径预算7.3 ns；实际布局布线时序尚未验证。
2. 18-Bank、Ping-Pong合并地址的一读一写组织在本轮HLS中达到所有流水循环II=1，未增加LUTRAM或复制存储；真实握手下的端口时序仍须RTL co-sim验证。
3. 3×3主循环HLS II=1已确认；G=1、行尾、最后Group及反压下的实际启动/Tile间隔仍须仿真测量，不能仅由II推断。
4. 1×1已采用前三位置缓存、最后位置直接输出；不保证与输入气泡隔离的整段G连续输出，后者需要额外预取保证。
5. ConvGeometry统一构造及固定表前提的编译期校验已实现；当前测试与Top均使用固定YOLO配置，整网验证边界见 [Top](top.md)。

首版 DDR 布局和重扫边界见 [DDR 搬运规格](yolov2_tiny_ddr_io.md)。Top 在每个
Cout 块重新调用 Im2Row；Im2Row 自身不依赖 DDR 地址或 Cout 数量。

## 13. 后续验证与本次范围

后续修改须先完成每Bank逐拍读请求、返回和写入排表。不要以 `DEPENDENCE false` 隐藏真实依赖，不将G维完全展开，也不新增完整Im2Row矩阵或整张Padding缓存。

按后续获准阶段进行：

- 静态检查：ConvGeometry字段一致、固定数组容量、地址与终点位宽、函数签名。
- 功能测试：只针对层表工作配置，覆盖G=1/2/128、RGB poison、九个kernel位置、1×1无Padding、W=13尾Tile、输入/输出事务守恒及连续模式切换。
- 综合：真实RAM映射、Bank端口、BRAM/LUT/FF、输出循环II、单套物理实例。
- co-sim：输入气泡、输出反压、预取返回跨角色切换、最后Group/行/调用排空；同时测量启动、Tile间隔与层内平均速率。

定向测试可使用简单确定性数据和源坐标reference，不让reference复刻Strip调度。当前真实BIN testbench见第14节。

### 13.1 本轮源码优化及待复核项

前一版综合报告显示主3×3循环II=2，Top overlap copy与预取返回的不同store在ST9/ST8形成WAW；输入使能路径还包含32-bit的`base+5 → min(width) → bottom_x比较`。旧报告共13 DSP，其中顶层三个32×32乘法各占3 DSP，主循环3 DSP、首行预装1 DSP。

本轮源码针对这些来源做以下改变，HLS报告结果见本节末表：

- 主循环全部Strip写（重叠copy、Line预取、Bottom输入、Padding零）先进入固定18 Bank的当前事务`WriteBeat`，每Bank独立enable/address/data；全部数据采集结束后，在本迭代末尾由每Bank唯一store入口提交，再推进状态。每迭代新建零默认值的`next`，无效Bank的数据不保留旧值，地址仍由统一逻辑译码。enable/address在数据分支之前按每Bank独立布尔公式生成，数据分支只填data，不让大分支汇合的phi拖后写使能。写命令不随G增长。实际BRAM返回寄存器及copy/prefetch的数据、标签对齐由HLS按latency=1和真实RAW生成，不人为增加持久`returned`及跨迭代提前消费约束；因此也不需要额外尾部flush。未添加`DEPENDENCE inter false`。
- 输出选择采用9个独立kernel位置比较、按行三个offset掩码OR、再三行OR，而非9级优先switch；每Bank在当前地址只读一次，overlap copy复用这些Bank返回，不增加第二读口。Top只在q0..3读、Middle只在q3..6读、Bottom只在q6..8读，不让Bottom准备阶段的无用读引入额外存储反馈。q8的Bottom overlap直接转发Bank4/5，不穿过输出选择树。
- `bottom_end`只在Tile切换时更新。输入握手只比较已寄存的窄`bottom_x`与终点；独立3-bit `bottom_column`维护目标Strip列，不在写使能路径计算`bottom_x-base+1`。坐标9 bit（含416及预取偏移）、Tile7 bit（含104）、Group8 bit（含128）、Line地址10 bit（含512 one-past）、Strip地址8 bit。输入/输出事务循环计数位宽由固定层表最大事务数及one-past推导。
- 上一版调用准备阶段把G=1/2/.../128译为shift；本次改为直接读取层表编译期生成的`cin_group_shift`。扫描中的Line地址仍由移位加Group生成，首行预装使用递增块地址。首行预装改单事务while，1×1改单连续输入事务for，避免多层flatten根据运行时边界生成通用乘法；未将乘法简单绑定到fabric。
- G=1关键逻辑顺序：最后Bottom输入q4采集并提交，q6首次读取；Top copy在q3采集并提交；换行q6准备列1..4及左Padding，q7准备列5；q8准备下一Tile的Bottom，下一Tile q0只读取Top。提交均位于所属迭代的数据采集之后，真实RAW必须由HLS满足；此为源码依赖排表，不是RTL逐拍验证结论。
- 仅3×3主循环声明`#pragma HLS DEPENDENCE variable=strip inter RAW distance=2 true`，保留真实RAW而明确其最短迭代距离，绝非`inter false`。证明覆盖所有Bank：G=1时最紧的是Bottom最后新输入q4写→q6首次读，以及换行Top列5 q7写→下一Tile q0全列读取，均相隔2次迭代；Top换行q6写→下一Tile q0为3，同列同行copy/prefetch与Middle/Bottom首次读更远。G>1的相同Group访问间隔不会缩短。此结论依赖固定层表、q事件位置及行读门控；改变它们须重新证明。`distance`用于`inter true`的语义见[AMD UG1399 2022.2](https://www.xilinx.com/support/documents/sw_manuals/xilinx2022_2/ug1399-vitis-hls.pdf)。
- Line写入对固定物理Bank做UNROLL使能译码：首行4列Bank、Bottom两行×4列Bank。动态值只进入写使能和窄RAM地址，不改变物理结构、写事务总数或每Bank一读一写需求。
- q4普通预取及换行q6/q7共用每个物理Line Bank的唯一读入口。对固定列Bank b，普通源列为`base+(b==0?8:4+b)`，q6为b，q7仅Bank0读取列4；普通越界及首行未初始化旧Top在读入口屏蔽。目标Top/Middle列c的Line Bank始终是`(c-1)&3`，读返回统一按行角色onehot OR路由，各目标Bank的数据仅赋值一次，再与互斥overlap来源合并。这样避免三个事件各自读RAM、各自路由后再汇合的大选择链；返回与标签时序仍由HLS负责。

存储仍为`line[2][4][512]`及`strip[3][6][256]`，均保持BRAM、latency=1、一读一写；kernel_pos→group顺序、在线Padding、W=13尾Lane及阻塞stream契约不变。

以下为2026-11-02上一版本的综合对比（Vitis HLS 2022.2，xc7z020-clg400-2），不是2026-11-03本次源码的综合结果：

| 指标 | 优化前 | 本轮优化后 |
|---|---:|---:|
| 3×3主循环II | 2 | 1 |
| Top HLS估计周期 | 7.508 ns | 7.212 ns |
| 7.3 ns预算余量 | -0.208 ns | +0.088 ns |
| BRAM_18K | 52 | 52 |
| DSP | 13 | 0 |
| FF | 12733 | 4065 |
| LUT | 13982 | 13718 |

优化后所有流水循环均II=1、循环约束满足，无WARNING/ERROR；3×3主循环路径约6.51 ns，预算余量约0.79 ns。上述数值来自HLS估计，不是Vivado布局布线后的routed timing；尤其Top仅0.088 ns估计余量，不代表最终硬件时序已收敛。1×1循环II=1只表示连续输入事务，满Tile的平均速率仍为4G输入/G输出，不是持续每拍输出一个Pack。

综合脚本、日志和工程位于源码树外`build/hls/im2row_yolo_optimize/`（项目根目录下）；最终源码的日志为`final_synth.log`，报告位于`project/optimized/syn/report/`。原`/home/liaoyingkai/High_Level_Synthesis/cnn_im2row_test`基线工程未被覆盖。系统G++与Vitis Clang 3.9静态语法检查、差异检查通过；本轮未实现或修改TB，未运行C simulation、RTL co-simulation或功能回归。跨Tile返回、G=1换行、W=13边界、输入气泡/输出反压及连续调用切换仍须功能/RTL验证。

2026-11-03本次额外简化：1×1仅Prediction使用，尾收集Lane固定为表内常量0；`spatial_lane<last_gather_lane`已保证索引至多2，删除重复`<3`判断。不改变收集循环、缓存、输出及任何真实边界保护。3×3的RAW distance=2 true、RAM尺寸/Bank、pragma和q调度不变。本次检查日志位于项目根目录`build/hls/yolo_fixed_geometry/`，仅G++及Vitis Clang 3.9的`-fsyntax-only`和编译期值断言，不运行程序、仿真或综合。

## 14. 真实PyTorch导出数据testbench

当前YOLO测试入口为`cnn_accelerator/testbench/test_im2row_pytorch.cpp`。HLS工程设置`im2row`为唯一Top，设计文件只需`src/im2row/im2row.cpp`及其头文件，testbench只加入这个文件，不加入其他含`main()`的测试。

本测试读取`/home/liaoyingkai/python_project/minist_train_fpga/YOLOV2_TINY/generated/trace/`中的真实样本`0000`。迁移机器或切换导出目录时修改testbench顶部路径；切换样本时修改样本后缀。先在Python工程运行`export.py`和`dump_trace.py`，同一份trace必须配套生成时的权重及量化参数；本测试本身不重新生成或修改它们。

Python的`dump_trace.py`以导出的整数权重和参数进行数学参考计算，使用标准PyTorch unfold生成窗口；后续层输入来自整数参考，不是QAT浮点激活。2026-11-08已核对导出脚本、文档及现有样本：九层权重BIN与逻辑NPY一致，bias/M/S的BIN与相应NPY一致；所有Im2Row BIN和unfold BIN均与独立源坐标参考逐元素一致。这是数据契约检查，不是HLS DUT仿真结果。

### 14.1 各层输入来源

| 当前Im2Row配置 | 上游BIN边界（省略`trace_`和`_0000.bin`） |
|---|---|
| Conv1 | `input_int8_hwc` |
| Conv2～Conv6 | `pool1_int8_hwc`～`pool5_int8_hwc` |
| Conv7 | `pool6_int8_hwc` |
| Conv8 | `conv7_postprocess_int8_hwc` |
| Prediction | `conv8_postprocess_int8_hwc` |

每层独立读取自己的上游HWC文件，不串接其他HLS模块。原始输入仅含`H*W*Cin`个signed INT8字节：按`y→x→group`组装输入事务，通道`c=group*8+cin_lane`，索引为`(y*W+x)*Cin+c`。Conv1的逻辑输入为RGB三通道，不是单通道；组内其余五通道填零，空间Padding不作为输入发送。

### 14.2 输出比较与事务检查

输出按`y→tile_x→kh→kw→group`直接消费，不缓存完整输出再重排。每个事务顺序读取`trace_<层名>_im2row_int8_0000.bin`的32个字节，其偏移为：

```text
T = ceil(W/4), G = ceil(Cin/8), R = Kh*Kw*G
token = (kh*Kw+kw)*G+group
byte_offset = 32*((y*T+tile_x)*R+token)+8*spatial_lane+cin_lane
```

同时用`trace_<层名>_unfold_int8_0000.bin`作逻辑位置交叉核对：有效空间位置`x=tile_x*4+spatial_lane`、有效通道`c=group*8+cin_lane`对应
`(y*W+x)*(Kh*Kw*Cin)+(kh*Kw+kw)*Cin+c`。无效空间Lane或通道不访问unfold，期望值为零；空间Padding已体现在unfold内容中。

BIN不保存`spatial_valid`或C++结构体。掩码独立根据`tile_x*4+lane<W`计算，W=26每行末Tile为`0011`、W=13为`0001`，卷积Padding不改变它。每层检查输入`H*W*G`和输出`H*T*R`的事务数、stream清空、BIN精确长度和读取状态。九层之后再次运行Conv1，检查Prediction/G128到3×3/G1的连续调用切换。

CSV写入当前仿真工作目录下`simulation_data/pytorch/im2row/`，完整保留实际输入及输出期望值、实际值、valid和match。每个Pack一行，避免每个标量一行造成过大文件；每层及重复调用使用不同文件名。终端仅输出事务数、验证元素数、错误数和PASS/FAIL；失败返回非零，错误详情限制前若干项。CSV列名中的Lane为同一行的空间位置，channel为组内Cin Lane。

本轮不运行C-sim、RTL co-sim或综合。后续仿真才验证DUT是否符合这些文件；真实接口气泡、反压、RAM读写调度、II和时序还需相应RTL测试或综合报告确认。
