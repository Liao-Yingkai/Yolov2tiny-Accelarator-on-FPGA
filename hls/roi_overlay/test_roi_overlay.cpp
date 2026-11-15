#include "roi_overlay.h"
#include <cstdint>
#include <cstring>
#include <iostream>

// 独立逐像素参考：不用DUT的绘图函数、字模函数或内部寄存器。
// 独立检查命令ACK、SOF切页、逐像素结果与AXIS旁带。
struct TestBox {
    int left, top, right, bottom;
    const char* name;
    const char* score;
};

static unsigned font_row(char c, int row) {
    // 每个数值表示一行，从bit4到bit0对应从左到右的5个点。
    static const unsigned digits[10][7] = {
        {14,17,19,21,25,17,14}, {4,12,4,4,4,4,14},
        {14,17,1,2,4,8,31}, {30,1,1,14,1,1,30},
        {2,6,10,18,31,2,2}, {31,16,16,30,1,1,30},
        {14,16,16,30,17,17,14}, {31,1,2,4,8,8,8},
        {14,17,17,14,17,17,14}, {14,17,17,15,1,1,14}
    };
    static const unsigned a[7] = {14,17,17,31,17,17,17};
    static const unsigned b[7] = {30,17,17,30,17,17,30};
    static const unsigned percent[7] = {25,25,2,4,8,19,19};
    static const unsigned dot[7] = {0,0,0,0,0,12,12};
    static const unsigned unknown[7] = {14,17,1,2,4,0,4};
    if (c >= '0' && c <= '9') return digits[c-'0'][row];
    if (c == 'a') c = 'A';
    if (c == 'A') return a[row];
    if (c == 'B') return b[row];
    if (c == '%') return percent[row];
    if (c == '.') return dot[row];
    if (c == ' ') return 0;
    return unknown[row];
}

static unsigned reference(int x, int y, unsigned source,
                          const TestBox* boxes, int count, bool synced) {
    if (!synced || x < 112 || x >= 528 || y < 32 || y >= 448) return 0;
    // 标签先查低index，所以重叠时低index连同黑底一起优先。
    for (int i=0; i<count; ++i) {
        const TestBox& b = boxes[i];
        const int n = static_cast<int>(std::strlen(b.name));
        const int m = static_cast<int>(std::strlen(b.score));
        const int longest = n > m ? n : m;
        int scale = 0;
        for (int s=2; s>=1; --s)
            if (longest*8*s+4 <= b.right-b.left &&
                16*s+6 <= b.bottom-b.top) { scale=s; break; }
        if (!scale) continue;
        const int lx = b.left+2;
        const int ly = b.bottom-2-(16*scale+2);
        if (x < lx || x >= lx+longest*8*scale ||
            y < ly || y >= ly+16*scale+2) continue;
        const int dy = y-ly;
        if (dy >= 8*scale && dy < 8*scale+2) return 0;
        const bool second = dy >= 8*scale+2;
        const char* str = second ? b.score : b.name;
        const int length = second ? m : n;
        const int cell = (x-lx)/(8*scale);
        const int gx = ((x-lx)/scale)%8;
        const int gy = (dy-(second ? 8*scale+2 : 0))/scale;
        if (cell >= length || gx >= 5 || gy >= 7) return 0;
        return (font_row(str[cell], gy) & (1U << (4-gx))) ? 0xffffff : 0;
    }
    for (int i=0; i<count; ++i) {
        const TestBox& b = boxes[i];
        if (x >= b.left && x < b.right && y >= b.top && y < b.bottom &&
            (x < b.left+3 || x >= b.right-3 || y < b.top+3 || y >= b.bottom-3))
            return 0x00ff00;
    }
    return source;
}

static std::uint32_t box_word(const TestBox& b, int word) {
    if (word == 0) return unsigned(b.left) | (unsigned(b.top)<<16);
    if (word == 1) return unsigned(b.right) | (unsigned(b.bottom)<<16);
    if (word == 2) return unsigned(std::strlen(b.name)) |
                          (unsigned(std::strlen(b.score))<<5);
    const char* text = word < 7 ? b.name : b.score;
    const int offset = word < 7 ? (word-3)*4 : (word-7)*4;
    const int length = static_cast<int>(std::strlen(text));
    std::uint32_t result = 0;
    for (int j=0; j<4; ++j)
        if (offset+j < length) result |= std::uint32_t(static_cast<unsigned char>(text[offset+j]))<<(8*j);
    return result;
}

int main() {
    hls::stream<video_axis_t> input, output;
    ap_uint<32> status = 0;
    std::uint32_t command = 0, data = 0;
    unsigned sequence = 0, ack = 0, pending_sequence = 0;
    unsigned waiting_sequence = 0;
    bool waiting_ack = false;
    unsigned long inputs = 0, outputs = 0, checked = 0;
    int errors = 0, x = 0, y = 0;
    bool synced = false, uploading = false, pending = false, error = false;
    int commit_ticks = 0;
    bool commit_valid = false;
    TestBox active[16] = {}, staged[16] = {};
    int active_count = 0, staged_count = 0;
    auto check = [&](bool ok, const char* why) {
        if (ok) return;
        if (errors < 10) std::cerr << "pixel=" << inputs << " x=" << x
                                  << " y=" << y << " " << why << '\n';
        ++errors;
    };
    // 每次严格先写一个输入，再调用一次DUT，再取一个输出。
    auto pixel = [&](bool sof) {
        // 收命令像素也计一拍：接收、逐框扫描、扫描结束、FINISH。
        // 该倒计时只依赖发送内容，不读取DUT状态决定提交结果。
        if (commit_ticks && --commit_ticks==0) {
            if (commit_valid) uploading=false;
            else { error=true; pending=false; ack=pending_sequence; }
        }
        if (sof) {
            x=0; y=0; synced=true;
            if (pending && commit_ticks==0) {
                for (int i=0; i<staged_count; ++i) active[i]=staged[i];
                active_count=staged_count; pending=false; uploading=false;
                ack=pending_sequence;
            }
        }
        video_axis_t in{};
        const unsigned source = unsigned(((inputs*37+19)&0x7fffff)<<1)|1U;
        in.data=source; in.keep=(inputs&1) ? 7 : 5;
        in.strb=(inputs&2) ? 7 : 3; in.user=sof; in.last=(x==639);
        input.write(in); ++inputs;
        roi_overlay(input, output, command, data, status);
        check(!output.empty(), "缺少输出");
        if (!output.empty()) {
            const video_axis_t out=output.read(); ++outputs;
            check(out.data == reference(x,y,source,active,active_count,synced), "像素不一致");
            check(out.user==in.user && out.last==in.last && out.keep==in.keep &&
                  out.strb==in.strb, "AXIS旁带不一致");
            ++checked;
        }
        check(input.empty() && output.empty(), "一入一出不守恒");
        // 普通命令允许若干像素才完成；只接受原ACK或当前在等的序号，
        // 绝不能把任意DUT返回序号直接当作参考状态。COMMIT不走此通道，
        // 它的ACK和图像切页均由上面的独立SOF参考决定。
        const unsigned observed_ack = status.to_uint() >> 16;
        const bool allowed_ack = observed_ack == ack ||
            (waiting_ack && observed_ack == waiting_sequence);
        check(allowed_ack, "ACK序号不属于当前命令");
        if (waiting_ack && observed_ack == waiting_sequence) ack=waiting_sequence;
        const unsigned expected = (ack<<16) | (unsigned(active_count)<<8) |
            (pending ? 1U : 0U) | (uploading ? 2U : 0U) |
            (error ? 4U : 0U) | (synced ? 8U : 0U);
        check(status.to_uint()==expected, "状态/ACK不一致");
        if (synced && ++x==640) { x=0; if (++y==480) y=0; }
    };
    // 仅建立协议参考并锁定邮箱；允许测试自行安排接收/扫描/FINISH的SOF。
    auto start_commit = [&](unsigned value, bool valid) {
        sequence=(sequence+1)&65535U;
        data=value;
        command=(sequence<<16)|3U;
        pending=true;
        pending_sequence=sequence;
        commit_valid=valid;
        commit_ticks=(value<=16 ? int(value) : 0)+3;
    };
    // valid_commit仅表示协议参考预期，不参与DUT执行。
    auto send = [&](unsigned op, unsigned index, unsigned word,
                    unsigned value, bool invalid, bool valid_commit) {
        if (op==3) {
            start_commit(value,valid_commit);
            // 包括非法提交也严格等待FINISH；普通ACK的宽松窗口不适用。
            while (commit_ticks) pixel(false);
            return;
        }
        sequence=(sequence+1)&65535U;
        data=value;
        command=(sequence<<16)|(index<<12)|(word<<8)|op;
        if (op==1) { uploading=true; error=false; }
        if (invalid) error=true;
        {
            waiting_sequence=sequence;
            waiting_ack=true;
            // data先于command写入，轮询期间二者均保持不变。
            // 允许缓存计算延迟，不把普通命令ACK固定为同一像素返回。
            for (int ticks=0; ticks<64 && ack!=sequence; ++ticks) pixel(false);
            check(ack==sequence, "普通命令等待ACK超时");
            waiting_ack=false;
        }
    };
    auto begin = [&]() { send(1,0,0,0,false,false); };
    auto write_box = [&](int index, const TestBox& box) {
        for (int word=0; word<9; ++word)
            send(2,unsigned(index),unsigned(word),box_word(box,word),false,false);
    };
    auto commit = [&](const TestBox* boxes, int count) {
        staged_count=count;
        for (int i=0; i<count; ++i) staged[i]=boxes[i];
        send(3,0,0,unsigned(count),false,true);
    };
    auto frame = [&]() {
        for (int i=0; i<640*480; ++i) pixel(i==0);
    };

    for (int i=0; i<7; ++i) pixel(false); // 初见SOF前所有输出必须黑。
    const TestBox first[2] = {
        {140,80,420,250,"Aa B@", "98.5%"}, // @使用未知字符的问号字形。
        {144,84,424,254,"B", "12%"}
    };
    begin(); write_box(0,first[0]);
    // seq不变时即便低位变成BEGIN也不能清掉已写mask；随后COMMIT检验。
    // 这是重复序号的防御性检查，正常PS发送时低位和data会一起保持稳定。
    command=(sequence<<16)|1U; pixel(false); pixel(false);
    write_box(1,first[1]); commit(first,2);
    for (int i=0; i<11; ++i) pixel(false); // pending期间不得提前ACK。

    const TestBox second[3] = {
        {120,50,190,100,"A B", "12.3%"}, // 只放得下1倍字。
        {240,80,254,94,"A", "9%"},      // 两种倍率均放不下，仅边框。
        {280,200,500,330,"0123456789", "45.6%"}
    };
    // 第一帧中途上传第二页，当前帧必须一直显示旧页。
    pixel(true);
    bool uploaded = false;
    while (x!=0 || y!=0) {
        if (!uploaded && x==0 && y==10) {
            uploaded=true;
            begin();
            write_box(0,second[0]);
            write_box(1,second[1]); write_box(2,second[2]); commit(second,3);
        }
        // send本身消耗的像素已推进x/y，不假定每条命令固定花一拍。
        if (x!=0 || y!=0) pixel(false);
    }
    check(uploaded, "帧中上传触发点未到达");
    frame(); // SOF换到1倍/隐藏/数字标签新页。

    // 16字符的2倍标签恰有256像素宽，第二行也写满8字符。
    // 几何word按任意顺序写，且文本全写完后再修改几何及长度，
    // 检查渲染缓存由最新字段重新计算，而不是沿用首次完整时的值。
    const TestBox boundary = {120,120,380,200,"AB0123456789AB01","12.3456%"};
    const TestBox original = {120,120,260,180,"A","1"};
    begin();
    const int word_order[9] = {8,4,1,6,0,7,5,3,2};
    for (int i=0; i<9; ++i) {
        const int word=word_order[i];
        send(2,0,unsigned(word),box_word(original,word),false,false);
    }
    for (int word=3; word<9; ++word)
        send(2,0,unsigned(word),box_word(boundary,word),false,false);
    send(2,0,2,box_word(boundary,2),false,false);
    send(2,0,1,box_word(boundary,1),false,false);
    send(2,0,0,box_word(boundary,0),false,false);
    // SOF在接收及扫描期间只能重置视频坐标，不能提前激活新页。
    // count=1共四拍，最后FINISH所在像素的SOF则必须立即切页。
    staged_count=1; staged[0]=boundary;
    start_commit(1,true);
    pixel(true);  // 接收
    pixel(true);  // 扫描第0框
    pixel(false); // 扫描结束
    pixel(true);  // FINISH与SOF同事务
    frame();

    // 各错误仅花少量前缀像素；之后的新SOF会恢复坐标。
    begin(); pixel(false); // BEGIN命令同seq保持，不能额外改变状态。
    for (int word=0; word<8; ++word)
        send(2,0,unsigned(word),box_word(second[0],word),false,false);
    send(3,0,0,1,true,false); // 少第8个word。
    begin(); send(2,0,9,0,true,false); // word越界。
    begin();
    TestBox bad={111,50,220,100,"A","1"}; write_box(0,bad);
    send(3,0,0,1,true,false); // 坐标不完全落在ROI。
    begin(); write_box(0,second[0]); send(2,0,2,17,false,false);
    // len/control内容在COMMIT校验；WRITE本身只检查word与上传状态。
    send(3,0,0,1,true,false);
    begin(); write_box(0,second[0]); send(2,0,2,9U<<5,false,false);
    send(3,0,0,1,true,false); // score_len上限为8。
    begin(); write_box(0,second[0]); send(2,0,2,1U<<9,false,false);
    send(3,0,0,1,true,false); // control的保留位必须为0。
    begin(); send(3,0,0,17,true,false); // count超16。
    begin(); send(3,0,0,0x10000000U,true,false);
    frame(); // 非法完整32位count不能按低位0清屏，仍显示boundary。
    begin(); commit(nullptr,0); frame(); // 清框后ROI内恢复源像素，ROI外仍黑。
    send(2,0,0,0,true,false); // 未BEGIN，WRITE必须报错。

    TestBox maximum[16];
    for (int i=0; i<16; ++i)
        maximum[i]=TestBox{120+(i%4)*100,40+(i/4)*96,
                          200+(i%4)*100,110+(i/4)*96,"A","9%"};
    begin();
    // 保持BEGIN数拍，再逐项写满16框，检查索引上限和全部新写word。
    pixel(false); pixel(false);
    for (int i=0; i<16; ++i) write_box(i,maximum[i]);
    commit(maximum,16); frame();
    begin(); commit(nullptr,0); frame(); // 最大批次清空也不能残留旧框。
    check(inputs==outputs && outputs==checked, "最终事务计数不守恒");
    std::cout << (errors ? "FAIL" : "PASS") << " input_pixels=" << inputs
              << " output_pixels=" << outputs << " checked=" << checked
              << " errors=" << errors << '\n';
    return errors ? 1 : 0;
}
