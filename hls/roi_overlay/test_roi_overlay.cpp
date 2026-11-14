#include "roi_overlay.h"
#include <cstdint>
#include <cstring>
#include <iostream>

// 独立逐像素参考：不用DUT的绘图函数、字模函数或内部寄存器。
// 只编写测试；本轮不执行仿真或综合。
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
    unsigned long inputs = 0, outputs = 0, checked = 0;
    int errors = 0, x = 0, y = 0;
    bool synced = false, uploading = false, pending = false, error = false;
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
        if (sof) {
            x=0; y=0; synced=true;
            if (pending) {
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
        const unsigned expected = (ack<<16) | (unsigned(active_count)<<8) |
            (pending ? 1U : 0U) | (uploading ? 2U : 0U) |
            (error ? 4U : 0U) | (synced ? 8U : 0U);
        check(status.to_uint()==expected, "状态/ACK不一致");
        if (synced && ++x==640) { x=0; if (++y==480) y=0; }
    };
    // valid_commit仅表示协议参考预期，不参与DUT执行。
    auto send = [&](unsigned op, unsigned index, unsigned word,
                    unsigned value, bool invalid, bool valid_commit) {
        sequence=(sequence+1)&65535U;
        data=value;
        command=(sequence<<16)|(index<<12)|(word<<8)|op;
        if (op==1) { uploading=true; error=false; }
        if (invalid) error=true;
        if (valid_commit) { pending=true; uploading=false; pending_sequence=sequence; }
        else ack=sequence;
        pixel(false);
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
    for (int i=0; i<640*480; ++i) {
        if (i==640*10) {
            begin();
            write_box(0,second[0]);
            write_box(1,second[1]); write_box(2,second[2]); commit(second,3);
            // 上述命令也消耗像素，循环相应跳过，保证完整帧长度不变。
            i+=29;
        }
        pixel(i==0);
    }
    frame(); // SOF换到1倍/隐藏/数字标签新页。

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
