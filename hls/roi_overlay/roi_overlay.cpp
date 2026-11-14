#include "roi_overlay.h"
#include "font.h"
using namespace roi_overlay_config;

void roi_overlay(hls::stream<video_axis_t>& s_axis_video,
                 hls::stream<video_axis_t>& m_axis_video,
                 ap_uint<32> command, ap_uint<32> command_data,
                 ap_uint<32>& status) {
#pragma HLS INTERFACE axis register both port=s_axis_video
#pragma HLS INTERFACE axis register both port=m_axis_video
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INTERFACE s_axilite port=command bundle=control
#pragma HLS INTERFACE s_axilite port=command_data bundle=control
#pragma HLS INTERFACE s_axilite port=status bundle=control
#pragma HLS INTERFACE ap_none port=status
#pragma HLS PIPELINE II=1
    // 双页仅保存描述，帧边界切页，不复制整表。
    static ap_uint<32> words[2][MAX_BOXES][BOX_WORDS];
    static ap_uint<9> written[MAX_BOXES]; // 本次BEGIN后，各框哪些word已重新写入。
#pragma HLS ARRAY_PARTITION variable=words complete dim=0
#pragma HLS ARRAY_PARTITION variable=written complete dim=1
    // page为当前显示页，!page为上传页；pending期间冻结上传，等SOF交换角色。
    static bool page = false, pending = false, upload_open = false;
    static bool error = false, synced = false;
    static ap_uint<5> active_count = 0, pending_count = 0;
    // last_seen防止同一命令重复执行；提交命令的ack须等到SOF才追上pending_seq。
    static ap_uint<16> last_seen = 0, ack = 0, pending_seq = 0;
    static ap_uint<10> x = 0;
    static ap_uint<9> y = 0;
#pragma HLS RESET variable=page
#pragma HLS RESET variable=pending
#pragma HLS RESET variable=upload_open
#pragma HLS RESET variable=error
#pragma HLS RESET variable=synced
#pragma HLS RESET variable=active_count
#pragma HLS RESET variable=pending_count
#pragma HLS RESET variable=last_seen
#pragma HLS RESET variable=ack
#pragma HLS RESET variable=pending_seq
#pragma HLS RESET variable=x
#pragma HLS RESET variable=y
    // 命令消费：阻塞读绑定像素事务，无视频时PS必须设置超时。
    const video_axis_t in = s_axis_video.read();
    video_axis_t out = in;
    const ap_uint<16> seq = command.range(31,16);
    if (!pending && seq != last_seen) {
        const ap_uint<8> opcode = command.range(7,0);
        const ap_uint<4> box = command.range(15,12);
        const ap_uint<4> word = command.range(11,8);
        last_seen = seq;
        bool defer_ack = false;
        if (opcode == BEGIN) {
            for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
                written[i] = 0;
            }
            upload_open = true;
            error = false;
        } else if (opcode == WRITE_WORD) {
            if (upload_open && word < BOX_WORDS) {
                words[!page][box][word] = command_data;
                written[box][word] = 1;
            } else error = true;
        } else if (opcode == COMMIT) {
            bool valid = upload_open && command_data <= MAX_BOXES;
            // 提交完整性验证：先检查mask，再读字段；空批不访问mask。
            if (valid) {
                for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
                    if (i < command_data && written[i] != 0x1ff) valid = false;
                }
            }
            if (valid) {
                for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
                    if (i < command_data) {
                        const ap_uint<32> start_word = words[!page][i][0];
                        const ap_uint<32> end_word = words[!page][i][1];
                        const ap_uint<32> control_word = words[!page][i][2];
                        // 校验保留完整16位坐标，禁止截断后误接收非法几何。
                        const ap_uint<16> left = start_word.range(15,0), top = start_word.range(31,16);
                        const ap_uint<16> right = end_word.range(15,0), bottom = end_word.range(31,16);
                        if (left < ROI_LEFT || right > ROI_RIGHT || top < ROI_TOP || bottom > ROI_BOTTOM ||
                            left >= right || top >= bottom || control_word.range(4,0) > CLASS_CHARS ||
                            control_word.range(8,5) > SCORE_CHARS || control_word.range(31,9) != 0) valid = false;
                    }
                }
            }
            if (valid) {
                pending = true;
                pending_count = command_data;
                pending_seq = seq;
                upload_open = false;
                defer_ack = true;
            } else error = true;
        } else error = true;
        if (!defer_ack) ack = seq;
    }
    // SOF切页：同事务刚接收的合法提交也可在此生效。
    if (in.user) {
        x = 0;
        y = 0;
        synced = true;
        if (pending) {
            page = !page;
            active_count = pending_count;
            pending = false;
            ack = pending_seq;
        }
    }
    const bool roi = synced && x >= ROI_LEFT && x < ROI_RIGHT && y >= ROI_TOP && y < ROI_BOTTOM;
    if (!roi) out.data = BLACK;
    // 框与标签命中：已校验的活动页使用视频所需窄位宽。
    ap_uint<5> choice[31];
    bool borders[MAX_BOXES];
#pragma HLS ARRAY_PARTITION variable=choice complete
#pragma HLS ARRAY_PARTITION variable=borders complete
    for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
        choice[15+i] = 16;
        borders[i] = false;
        if (roi && i < active_count) {
            const ap_uint<10> left = words[page][i][0].range(15,0);
            const ap_uint<9> top = words[page][i][0].range(31,16);
            const ap_uint<10> right = words[page][i][1].range(15,0);
            const ap_uint<9> bottom = words[page][i][1].range(31,16);
            const ap_uint<5> class_length = words[page][i][2].range(4,0);
            const ap_uint<5> score_length = words[page][i][2].range(8,5);
            const ap_uint<5> max_length = class_length > score_length ? class_length : score_length;
            const ap_uint<10> box_width = right-left;
            const ap_uint<9> box_height = bottom-top;
            // 移位前扩位，16字符的2倍宽256不得在5位长度中截断。
            const ap_uint<10> text_width_1 = ap_uint<10>(max_length) << FONT_CELL_SHIFT;
            const ap_uint<10> text_width_2 = text_width_1 << 1;
            const bool large = box_width >= text_width_2 + 2*MARGIN &&
                               box_height >= LABEL_HEIGHT_2 + 2*MARGIN;
            const bool small = box_width >= text_width_1 + 2*MARGIN &&
                               box_height >= LABEL_HEIGHT_1 + 2*MARGIN;
            const ap_uint<10> text_width = large ? text_width_2 : text_width_1;
            const ap_uint<6> text_height = large ? LABEL_HEIGHT_2 : LABEL_HEIGHT_1;
            borders[i] = x >= left && x < right && y >= top && y < bottom &&
                (x < left+BORDER || x+BORDER >= right || y < top+BORDER || y+BORDER >= bottom);
            if (max_length != 0 && (large || small) && x >= left+MARGIN && x < left+MARGIN+text_width &&
                y >= bottom-MARGIN-text_height && y < bottom-MARGIN) choice[15+i] = i;
        }
    }
    for (int i = 14; i >= 0; --i) {
#pragma HLS UNROLL
        choice[i] = choice[2*i+1] < choice[2*i+2] ? choice[2*i+1] : choice[2*i+2];
    }
    bool border = false;
    for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
        border = border || borders[i];
    }
    if (border) out.data = GREEN;
    // 共享字模：优先树已选出唯一标签，后面只查询一次ROM。
    if (choice[0] < MAX_BOXES) {
        const ap_uint<4> i = choice[0];
        const ap_uint<10> left = words[page][i][0].range(15,0);
        const ap_uint<9> top = words[page][i][0].range(31,16);
        const ap_uint<10> right = words[page][i][1].range(15,0);
        const ap_uint<9> bottom = words[page][i][1].range(31,16);
        const ap_uint<5> class_length = words[page][i][2].range(4,0);
        const ap_uint<5> score_length = words[page][i][2].range(8,5);
        const ap_uint<5> max_length = class_length > score_length ? class_length : score_length;
        const ap_uint<10> box_width = right-left;
        const ap_uint<9> box_height = bottom-top;
        const ap_uint<10> large_width = ap_uint<10>(max_length) << (FONT_CELL_SHIFT+1);
        const bool large = box_width >= large_width+2*MARGIN && box_height >= LABEL_HEIGHT_2+2*MARGIN;
        const ap_uint<6> text_height = large ? LABEL_HEIGHT_2 : LABEL_HEIGHT_1;
        const ap_uint<8> relative_x = x-left-MARGIN; // 最多255，标签内局部坐标
        const ap_uint<6> relative_y = y-(bottom-MARGIN-text_height);
        const ap_uint<5> cell_height = large ? FONT_CELL*2 : FONT_CELL;
        const bool second_line = relative_y >= cell_height+LINE_GAP;
        const ap_uint<6> row_y = second_line ? ap_uint<6>(relative_y-cell_height-LINE_GAP) : relative_y;
        const ap_uint<4> column = large ? relative_x >> (FONT_CELL_SHIFT+1) : relative_x >> FONT_CELL_SHIFT;
        const ap_uint<3> glyph_x = large ? ap_uint<3>(relative_x >> 1) : ap_uint<3>(relative_x);
        const ap_uint<6> glyph_y = large ? ap_uint<6>(row_y >> 1) : row_y;
        ap_uint<8> character = ' ';
        if (column < (second_line ? score_length : class_length)) {
            const ap_uint<4> word_index = (second_line ? 7 : 3)+(column >> 2);
            const ap_uint<5> byte_shift = ap_uint<5>(column.range(1,0)) << 3;
            character = words[page][i][word_index] >> byte_shift;
        }
        const ap_uint<5> glyph = roi_overlay_detail::overlay_font_row(character, ap_uint<3>(glyph_y));
        out.data = BLACK;
        if (row_y < cell_height && glyph_x < 5 && glyph_y < 7 && glyph[4-glyph_x]) out.data = WHITE;
    }
    // 输出与状态：成功传输后推进坐标，每调用写回无valid握手的状态。
    m_axis_video.write(out);
    if (synced) {
        if (in.last) {
            x = 0;
            y = y == HEIGHT-1 ? ap_uint<9>(0) : ap_uint<9>(y+1);
        } else x = x+1;
    }
    // 无ap_vld握手：每个已完成像素事务更新可读状态。
    status = (ap_uint<32>(ack) << 16) | (ap_uint<32>(active_count) << 8) |
             (ap_uint<32>(synced) << 3) | (ap_uint<32>(error) << 2) |
             (ap_uint<32>(upload_open) << 1) | ap_uint<32>(pending);
}
