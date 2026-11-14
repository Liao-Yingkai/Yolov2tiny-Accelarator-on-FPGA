# 独立 YOLO 视频叠加核

本目录只提供独立HLS源码，不修改现有CNN、Vivado或Vitis工程。顶层为`roi_overlay`，输入输出均为`ap_axiu<24,1,0,0>`的AXI4-Stream，双向接口寄存器启用；自由运行`ap_ctrl_none`，不需要启动。RGB从高到低字节为R、G、B，所有旁带原样透传。

文件包括：`roi_overlay.h`（接口与配置）、`roi_overlay.cpp`（叠加核）、`font.h`（字模）、`test_roi_overlay.cpp`（独立测试台）、`setup_hls.tcl`（建工程脚本）和本说明。

系统视频接线预期为`MM2S → roi_overlay → video_out`，PS控制口接AXI4-Lite；插入叠加核不改变摄像头和DDR的数据处理职责。

## 视频与显示

固定640×480，`user`标记一帧第一个像素，`last`标记行尾。ROI为半开区间`[112,528) × [32,448)`，大小416×416。ROI外以及首次SOF前输出黑色，ROI内保留原图，不画ROI轮廓。坐标只随成功的视频事务推进，输入空闲或输出反压不会按空时钟递增。假设合法640像素行；异常行长不额外纠正，下次SOF重同步。

最多16个检测框，全视频坐标，右下边界半开，必须完整位于ROI且宽高严格为正。绿色3像素框线绘于框内。标签在框内左下方，距框坐标左边和底边各2像素，白字黑底；类别第一行、概率第二行，行间2像素。字体单元8×8，5×7字形左上对齐；优先2倍、否则1倍，均用移位缩放。若最大行长为L，2倍要求框宽至少`16L+4`、高至少38，1倍要求宽至少`8L+4`、高至少22；仍不够则整个标签隐藏，框线照画。两行长度均0时不画标签；单行长度0时该行保留黑底。

支持英文字母（小写显示为大写）、数字、空格、`%`、`.`、`-`，其他字节显示`?`。PS预先按score降序排序；重叠标签低索引优先，所有标签优先于所有边框。文本由PS直接提供，HLS不计算置信度、不查类别名称。

## 控制寄存器

`command`、`command_data`、`status`均属于`control` AXI4-Lite接口，`status`无额外valid握手。具体地址由HLS生成的寄存器头文件确定，不在此硬编码。复位时ACK、sequence跟踪、框数量及控制状态均为0。

`status`同时指定`s_axilite`与`ap_none`是有意组合：前者选择AXI4-Lite适配器，后者选择适配器与核之间的端口级协议，不请求`ap_vld`状态握手。AMD的[S_AXILITE and Port-Level Protocols](https://docs.amd.com/r/2024.1-English/ug1399-vitis-hls/S_AXILITE-and-Port-Level-Protocols)说明了这两个层次；实际2022.2接口生成尚未通过综合验证，应以生成寄存器头文件与RTL为准。

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
|0|pending：合法COMMIT等待SOF|
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

PS一次只发一条命令：先写data、再写command、等ACK后才发下一条。相同sequence不会重复执行，独立保存最近看过的sequence和已ACK的sequence；调用者必须遵守递增协议。普通命令处理后立即ACK。合法COMMIT先置pending，收到该提交之后首个可见SOF时才切换页号与count并ACK；若提交处理所在像素本身为SOF，就在该像素生效。等待期间不得修改command或data，新命令不会被消费。ACK意味着新批次已经在帧边界生效，不是仅表示收到提交。

BEGIN清除inactive页的written masks、开启上传并清除sticky error，不影响当前显示。WRITE_WORD必须在BEGIN后且word<9。COMMIT先检查已BEGIN、count范围以及前count个框的9word完整性，然后才读描述验证坐标、长度与保留位。无效或不完整的批次立即ACK并置sticky error，当前显示不变，upload仍可继续修正；也可重新BEGIN。成功提交关闭上传；下一批必须重新BEGIN。旧sticky error只由BEGIN清除。

因此错误后原地修字并成功COMMIT，历史error仍为1，不能仅凭该位判断修正后的提交是否成功。推荐出现error就重新BEGIN并重传整批，使本批错误状态明确。

初始化时先复位，再等视频稳定、检查synced并发送BEGIN。顶层阻塞读取视频，**没有输入像素就不处理命令，也不会更新ACK**；没有后续SOF，合法COMMIT也不会完成。PS应使用有限超时，检查摄像头、视频时钟、SOF及下游是否反压，不能无限忙等。若PS进程重启但核未复位，应从当前ACK恢复sequence，并确保没有遗留pending命令；无法确认状态时协调复位后重新上传。

## 资源边界与系统关系

只有两份16×9个word的描述表、written masks、坐标和控制状态，无图像FIFO、行缓存、帧缓存、浮点、动态内存或STL。描述表完全分区为寄存器，16框比较展开；标签先经过平衡优先选择，再共享一份字体查询。字体ROM请求LUTRAM实现，不复制16份字模。SOF只切页号，不串行复制整表。

本阶段未运行新仿真或综合：II=1、无DSP和无BRAM是设计目标，不是已验证的时序或资源承诺；实际结果须以目标器件与工具报告为准。

PS仍独立完成裁剪、坐标换算、量化、推理及文本格式化；相机与DDR路径不作修改。检测结果对应较早采集的图像，本核只保证在视频帧边界整批更新，不建立推理帧与实时显示帧的时间匹配。因此物体运动时框与实时画面可有推理及上传延迟，不应把帧边界原子更新误解为同源帧同步。

## 本次检查与测试入口

2026-11-12：DUT与独立testbench均通过系统G++和Vitis HLS 2022.2附带Clang 3.9的C++14 `-fsyntax-only`检查；Tcl文本括号/引号完整性及文件空白检查通过。日志位于仓库根`build/hls/roi_overlay_static_check/`，没有生成或执行仿真程序，也没有执行建工程脚本或综合。

HLS Top选择`roi_overlay`，DUT只加入`roi_overlay.cpp`及头文件，testbench只加入`test_roi_overlay.cpp`。测试源码包含独立像素公式和字模参考，覆盖ROI遮罩、字号回退、标签重叠、帧中上传/下帧生效、重复sequence、非法或不完整提交、16框上限、零框清屏及像素/旁带守恒，不输出CSV；失败返回非零。本次尚未运行这些测试，语法通过不等于功能、真实AXI-Lite寄存器行为、反压或RTL时序通过。
