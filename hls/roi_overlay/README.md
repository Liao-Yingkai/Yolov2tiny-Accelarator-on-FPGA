# 独立 YOLO 视频叠加核

本目录只提供独立HLS源码，不修改现有CNN、Vivado或Vitis工程。顶层为`roi_overlay`，输入输出均为`ap_axiu<24,1,0,0>`的AXI4-Stream，双向接口寄存器启用。核使用 `ap_ctrl_none`，复位释放后随视频流持续运行。RGB从高到低字节为R、G、B，所有旁带原样透传。

文件包括：`roi_overlay.h`（接口与配置）、`roi_overlay.cpp`（叠加核）、`font.h`（字模）、`test_roi_overlay.cpp`（独立测试台）和本说明。综合脚本与生成物保存在源码目录之外。

系统视频接线预期为`MM2S → roi_overlay → video_out`，PS控制口接AXI4-Lite；插入叠加核不改变摄像头和DDR的数据处理职责。

## 像素事务与启动

当前源码使用 `ap_ctrl_none` 和显式 `PIPELINE II=1 style=flp`：C++一次调用处理一个像素，RTL复位释放后持续接收视频，不需要PS调用Start或开启AutoRestart。`II=1`是流水目标，实际II、延迟和时序以本版综合结果为准。

2026-11-18源码修订显式选择FLP（可排空流水线），不再交由工具自动选择流水风格。按照本次源码中的修复说明，该选择针对Vitis HLS 2022.2生成的FRP输出FIFO在反压下覆盖待发送像素的问题；输入间歇时，FLP允许已进入流水线的像素继续排出。这里记录修改意图，不将其推广为所有FRP设计的问题，也不以源码修改代替RTL反压测试或板上验证。

PS 初始化 AXI-Lite 驱动后等待视频同步，再按 sequence ACK 下发命令。静态坐标、上传状态及双页缓存跨迭代保留，SOF 负责同步坐标及切换已提交的配置。

## 视频与显示

固定640×480，`user`标记一帧第一个像素，`last`标记行尾。ROI为半开区间`[112,528) × [32,448)`，大小416×416。ROI外以及首次SOF前输出黑色，ROI内保留原图，不画ROI轮廓。坐标只随成功的视频事务推进，输入空闲或输出反压不会按空时钟递增。假设合法640像素行；异常行长不额外纠正，下次SOF重同步。

最多16个检测框，全视频坐标，右下边界半开，必须完整位于ROI且宽高严格为正。绿色3像素框线绘于框内。标签在框内左下方，距框坐标左边和底边各2像素，白字黑底；类别第一行、概率第二行，行间2像素。字体单元8×8，5×7字形左上对齐；优先2倍、否则1倍，均用移位缩放。若最大行长为L，2倍要求框宽至少`16L+4`、高至少38，1倍要求宽至少`8L+4`、高至少22；仍不够则整个标签隐藏，框线照画。两行长度均0时不画标签；单行长度0时该行保留黑底。

支持英文字母（小写显示为大写）、数字、空格、`%`、`.`、`-`，其他字节显示`?`。PS预先按score降序排序；重叠标签低索引优先，所有标签优先于所有边框。文本由PS直接提供，HLS不计算置信度、不查类别名称。

## 控制寄存器

`command`、`command_data`、`status`均属于`control` AXI4-Lite接口。具体地址由重新生成的寄存器头文件确定，不在此硬编码，也不假定与旧IP地址相同。复位时ACK、sequence跟踪、框数量及控制状态均为0。

`status`显式组合`s_axilite`和`ap_none`：前者将状态映射到AXI-Lite，后者不请求核侧的`ap_vld`端口握手。PS读取状态并通过`status[31:16]`的sequence ACK确认命令完成，包括COMMIT的帧边界确认；不能用Start、Done或数据有效标记代替该ACK。以接口pragma为准，源码末尾仍提到`ap_vld`的注释与当前配置不一致，本轮只更新文档，不修改源码。

`command`字段：

| 位 | 内容 |
| --- | --- |
|31:16|16位sequence，首命令1，此后每条加1并模65536回绕|
|15:12|box index，0..15|
|11:8|word index，0..8|
|7:0|opcode：BEGIN=1，WRITE_WORD=2，COMMIT=3|

每框9个32位word：

| word | 内容 |
| --- | --- |
|0|left低16位，top高16位|
|1|right低16位，bottom高16位|
|2|class_len在4:0（0..16），score_len在8:5（0..8），31:9必须0|
|3..6|类别ASCII，共16字节，每word低byte在先|
|7..8|概率ASCII，共8字节，每word低byte在先|

长度之外的字符不显示，不要求NUL；PS可零填。BEGIN和COMMIT的box/word字段不使用；BEGIN忽略data。COMMIT的完整32位data为count（0..16），count0合法，用于清除全部框。

`status`字段：

| 位 | 内容 |
| --- | --- |
|31:16|最近已ACK的sequence|
|15:8|当前实际显示的框数量|
|0|pending：COMMIT正在检查，或检查通过后等待SOF；不代表已通过检查|
|1|upload_open：BEGIN后可写入|
|2|sticky error：检测到非法命令或批次|
|3|synced：已见过SOF|
|7:4|保留为0|

## 整批上传与ACK

```text
seq = 0                         // 仅硬件复位后这样初始化
send(op, box, word, data):
    seq = (seq + 1) & 0xffff
    write(command_data, data)   // 保证写入顺序，必要时使用平台内存屏障
    write(command, seq<<16 | box<<12 | word<<8 | op)
    等待 read(status)>>16 == seq，必须有超时
    返回 read(status)

send(BEGIN, 0, 0, 0)
for box in [0, count):
    for word in [0, 9):          // 可改为任意顺序，但所有9word均必须写
        send(WRITE_WORD, box, word, packed[box][word])
s = send(COMMIT, 0, 0, count)
if s.error: 报告错误，从BEGIN重新上传整批
```

PS一次只发一条命令：先写data、再写command、等ACK后才发下一条。相同sequence不会重复执行，独立保存最近看过的sequence和已ACK的sequence；调用者必须遵守递增协议。文字WRITE和无需准备的普通命令处理后直接ACK。

word0/1/2的有效WRITE统一进入准备状态，下一像素事务只检查本框的几何mask。本框三个几何word尚未齐时直接ACK；已经齐备时，使用后续5个像素事务依次完成原始字段读取、合法性与尺寸校验、字号计算、布局缓存写入和ACK。WRITE当拍不再检查全部框的ready位，也不读取未齐备的原始字段。之后重写任意几何word也重新准备；任意9word上传顺序均合法。准备期间不接收新sequence，pending仍为0，upload_open和error不因准备而改变，非法几何仍只在COMMIT时报错。

COMMIT不再当拍检查16框并切页，而是锁存count/sequence后逐项检查：

```text
接收COMMIT → 每个像素事务检查1个框 → 扫描结束 → 登记检查结果 → 等待SOF
```

有效count=N时，第1..N个后续像素事务检查框0..N-1，第N+1个事务结束扫描，第N+2个事务登记结果。count=0仍需2个后续事务；完整32位count超16时将内部扫描数量钳为0并记录非法，2个后续事务后错误ACK，不能按截断后的低位接受。扫描和等待SOF期间pending均为1，并冻结上传命令；扫描期间upload_open保留原值，不能将其理解为允许并发上传。

检查通过才关闭upload_open；在登记结果时若恰好收到SOF，可在该事务切页，否则等待下一个SOF。**命令接收当拍或检查中遇到的SOF不会激活新页**，因此提交靠近帧头时可能顺延一帧。检查失败在登记结果时置error、ACK并退出pending，当前显示及原upload_open不变。等待ACK期间不得修改command或data，新命令不会被消费。合法COMMIT的ACK仍表示新批次已在帧边界生效，不是仅表示收到或检查通过。上述等待量指后续像素事务，不是PS可硬编码的AXI时钟延迟；PS始终轮询ACK。

BEGIN清除inactive页的written masks、上传完整标记和几何有效标记、开启上传并清除sticky error，不影响当前显示。WRITE_WORD必须在BEGIN后且word<9，每次WRITE同时更新本框的9word完整标记。准备器只在三个几何word已齐后读取字段，以完整16位坐标验证边界和严格正宽高，并验证长度与保留位，然后才截窄显示坐标。COMMIT检查已BEGIN、完整32位count范围、前count个框的9word完整性及准备器保存的几何有效标记，不重新计算布局。无效或不完整批次在检查完成时ACK并置sticky error；原上传若已开启仍可继续修正，也可重新BEGIN。成功提交关闭上传；下一批必须重新BEGIN。旧sticky error只由BEGIN清除。

因此错误后原地修字并成功COMMIT，历史error仍为1，不能仅凭该位判断修正后的提交是否成功。推荐出现error就重新BEGIN并重传整批，使本批错误状态明确。

初始化驱动后等视频稳定、检查synced并发送BEGIN。顶层阻塞读取视频，**没有输入像素时不处理命令，也不会更新ACK**；没有后续SOF，合法COMMIT也不会完成。PS应使用有限超时，检查复位、摄像头、视频时钟、SOF及下游是否反压，不能无限忙等。若PS进程重启但核未复位，应先确保像素事务仍可持续运行，再从当前ACK恢复sequence，并确保没有遗留pending命令；无法确认状态时协调复位后重新上传。

## 资源边界与系统关系

原始几何为start/end/control三块32×32位LUTRAM，地址`page*16+box`；它们只供上传和准备器使用，不需要16框并行读。文字独立保存于256×32位单读单写LUTRAM（`ram_s2p`，读延迟1），地址为`page*128+box*8+payload_word`。每框8word步长中仅0..5使用，分别对应上传word3..8；类别读取0..3，概率读取4..5。每个像素至多读取一个文字word，可同时向非活动页写入一个word，不把文字完全分区为寄存器。

双页显示缓存保存窄位宽外框、内框、标签和第二行边界，以及字号、显示使能和行长。一个共享准备器在上传时预计算这些字段；像素路径仅进行16框并行边界比较、低索引优先选择、局部坐标计算、一次文字读取和一份字模查询。优先选择使用并行前缀零检测产生one-hot，再通过平衡OR树汇总字段，避免串联最小值比较器。上传控制使用单一4位状态机：空闲、几何准备的5个阶段、提交扫描、提交结果登记、等待SOF；逐框检查取代16框归约，并与换页/ACK分拍。视频坐标计数独立于命令处理。字体ROM请求LUTRAM实现，不复制16份字模。无图像FIFO、行缓存、帧缓存、浮点、动态内存或STL；SOF只切页号，不串行复制整表。

HLS资源与时序属于工具估计，不等同于Vivado布局布线结果；实际视频反压、AXI4-Lite访问和最终时序仍需集成验证。

PS仍独立完成裁剪、坐标换算、量化、推理及文本格式化；相机与DDR路径不作修改。检测结果对应较早采集的图像，本核只保证在视频帧边界整批更新，不建立推理帧与实时显示帧的时间匹配。因此物体运动时框与实时画面可有推理及上传延迟，不应把帧边界原子更新误解为同源帧同步。

## 测试入口与验证边界

2026-11-18本轮仅对照用户修改后的源码更新文档，没有运行静态编译、C-sim、RTL co-sim或综合，也没有更新导出IP、BSP或PS程序。当前FLP版的验证结果须单独记录，不能直接沿用旧版的资源、II、时序或PASS结论。

C++测试台直接调用`roi_overlay()`，一次调用对应一个像素事务，检查绘图、旁带、命令ACK和错误处理；它不能证明流水风格及RTL反压行为。完整测试约215万个像素事务，不输出CSV。

历史上，自由运行接口的自动co-sim曾在生成的`roi_overlay.autotb.v`中出现`ap_done`未声明及`AESL_done`多驱动，未进入RTL仿真；之后临时改为带Start的版本，用户反馈该版本co-sim通过。**该PASS不属于当前恢复`ap_ctrl_none`并指定FLP的版本。** 当前接口没有块级启动握手；若自动测试台仍依赖`ap_done`，应检查生成测试台的兼容性，不能把编译失败视为DUT功能失败，也不能为仿真悄悄替换控制接口。

重新验证时先从当前源码执行C Synthesis，再生成co-sim或系统RTL测试，不复用旧`sim/`。重点覆盖输入停顿、下游持续及间歇反压，检查像素和SOF/TLAST不丢失、不重复、不乱序；上板还需检查Video Out锁定、连续显示及提交/清框。重新导出后，XSA、bitstream和BSP必须配套更新。

### 历史检查记录（以下数据不代表当前显式FLP版本）

2026-11-12：DUT与独立testbench均通过系统G++和Vitis HLS 2022.2附带Clang 3.9的C++14 `-fsyntax-only`检查；Tcl文本括号/引号完整性及文件空白检查通过。日志位于仓库根`build/hls/roi_overlay_static_check/`，没有生成或执行仿真程序，也没有执行建工程脚本或综合。

HLS Top选择`roi_overlay`，DUT只加入`roi_overlay.cpp`及头文件，testbench只加入`test_roi_overlay.cpp`。测试源码包含独立像素公式和字模参考，覆盖ROI遮罩、字号回退、标签重叠、帧中上传/帧边界生效、重复sequence、非法或不完整提交、16框上限、零框清屏及像素/旁带守恒，不输出CSV；失败返回非零。普通命令在有限像素事务内等待ACK；COMMIT用发送count建立独立倒计时、用SOF确定切页，不从DUT输出复制期望状态。还覆盖接收/扫描期间的SOF不得提前切页、登记结果与SOF同事务切页、高位非法count、任意word顺序、重写几何字段和最长标签。

### 2026-11-14上一轮优化结果（iter9，尚未采用多周期COMMIT）

使用Vitis HLS 2022.2、`xc7z020-clg400-2`，保持原工程的8 ns时钟、2.16 ns时钟不确定度，没有通过放宽约束取得以下改善。下表保留上一轮`iter9`作为本轮对比基线，不代表当前多周期状态机的结果：

| 项目 | 原实现 | 优化后 |
| --- | ---: | ---: |
| LUT | 13,680 | 9,566 |
| FF | 21,579 | 8,636 |
| BRAM_18K / DSP | 0 / 0 | 0 / 0 |
| HLS估计周期 | 12.282 ns | 6.977 ns |
| II | 1 | 1 |
| 报告流水延迟 | 7周期 | 10周期 |

LUT减少约30.1%，FF减少约60.0%。文字RAM、三组原始几何RAM和共享字体ROM均生成LUTRAM，而非BRAM。几何和文字仍使用原命令字段；几何WRITE的ACK延后1或5个像素事务，PS须按原协议等ACK，不得假设当拍完成。

C-sim完成，`input_pixels=1843673 output_pixels=1843673 checked=1843673 errors=0`。更新后的testbench也曾对优化前DUT运行通过，避免以改变期望绘图结果掩盖行为变化。最终DUT与testbench通过系统G++及Vitis Clang的C++14静态检查，`git diff --check`通过。

**上一轮未通过原时序裕量检查。** 有效逻辑预算是`8-2.16=5.84 ns`，不是8 ns；6.977 ns虽然小于名义时钟周期，仍触发`HLS 200-871`。控制路径上的提交资格、ACK与几何WRITE完整性归约仍存在长组合逻辑。不得将该历史结果描述为125 MHz时序完全收敛，也不能替代Vivado布局布线时序。

其他保留warning包括：变量位选择可能影响QoR（`SYNCHK 200-23`）、静态寄存器上电初始化，以及`status ap_none`可能缺少自动co-sim数据有效握手的通用提示。控制入口状态显式复位；未复位的显示缓存必须由BEGIN、全部字段上传、准备完成及合法COMMIT后才能启用。AXI4-Lite真实访问、复位和视频反压仍需系统级RTL验证。本轮没有运行RTL co-simulation、Vivado综合或布局布线。

本机脚本和全部生成物在仓库根的忽略目录中，不污染源码：

- [综合脚本](../../build/hls/roi_overlay_opt_20261114/run_hls.tcl)
- [上一轮综合报告](../../build/hls/roi_overlay_opt_20261114/roi_overlay_opt/iter9/syn/report/roi_overlay_csynth.rpt)
- [上一轮C-sim与综合日志](../../build/hls/roi_overlay_opt_20261114/iter9_console.log)

用户原有`~/High_Level_Synthesis/yolov2_roi_overlay`工程未被覆盖；上述生成物不随Git提交。

### 历史多周期控制优化结果（multicycle1，旧自由运行接口）

该历史版本采用上文的逐框COMMIT状态机；几何WRITE当拍也不再归约全部ready位，而在下一准备事务只检查目标框。绘图公式、命令字段、字模、视频尺寸和ROI不变；提交检查完成前的SOF不再允许激活新页，pending覆盖检查期。当前版本保留相同命令格式，并仍采用无需Start的自由运行接口；与该历史版本的流水风格区别见上文。

器件、8 ns时钟及2.16 ns不确定度均保持不变，未放宽约束：

| 项目 | 历史iter9 | 历史multicycle1 |
| --- | ---: | ---: |
| LUT | 9,566 | 9,131 |
| FF | 8,636 | 8,227 |
| BRAM_18K / DSP | 0 / 0 | 0 / 0 |
| HLS估计周期 | 6.977 ns | 5.775 ns |
| II | 1 | 1 |
| 报告流水延迟 | 10周期 | 12周期 |

当前估计周期小于5.84 ns有效预算，`HLS 200-871/200-1016`时序warning消失，未出现II violation或综合错误。HLS估计裕量仅约`5.84-5.775=0.065 ns`，不能等同于Vivado布线后WNS，必须继续做实际系统时序检查。

C-sim通过：`input_pixels=2150923 output_pixels=2150923 checked=2150923 errors=0`。新增检查覆盖COMMIT接收和扫描期间的SOF不得切页、FINISH与SOF同事务切页，以及带非零高位的非法count；原有最大16框、count0清屏、错误提交和逐像素检查保留。当前DUT和testbench通过系统G++与Vitis Clang的C++14静态检查，`git diff --check`通过。

仍有两处动态数组索引的`HLS 214-358` QoR提示（准备器选择本框mask及扫描器选择当前框标记），另有变量位选择、静态寄存器上电初始化和`status ap_none`通用提示。这些不等于已发现功能错误，当前C-sim与II检查通过；复位、真实AXI4-Lite访问及反压仍需RTL系统验证。本轮未运行RTL co-simulation或Vivado。

- [历史multicycle1综合报告](../../build/hls/roi_overlay_opt_20261114/roi_overlay_opt/multicycle1/syn/report/roi_overlay_csynth.rpt)
- [历史multicycle1 C-sim与综合日志](../../build/hls/roi_overlay_opt_20261114/multicycle1_console.log)
