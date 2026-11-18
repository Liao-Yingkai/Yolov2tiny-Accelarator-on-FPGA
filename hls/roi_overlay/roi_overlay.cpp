#include "roi_overlay.h"
#include "font.h"
using namespace roi_overlay_config;

namespace {
// 仅像素比较需要的窄字段；布局和字号只在上传时计算。
struct RenderBox {
    ap_uint<10> left, right, inner_left, inner_right, label_left, label_right;
    ap_uint<9> top, bottom, inner_top, inner_bottom, label_top, label_bottom, second_top;
    ap_uint<5> class_length, score_length;
    bool large, label_enable;
};
}

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
    // 连续运行并在输入暂时无数据时排空流水线。
    // 显式选用 FLP，避免 2022.2 自动推断的 FRP 输出 FIFO 在满时覆盖待发送像素。
#pragma HLS PIPELINE II=1 style=flp
    // 文字独立为单写单读LUTRAM；每框预留8word，实际使用6word。
    static ap_uint<32> text_memory[256];
#pragma HLS BIND_STORAGE variable=text_memory type=ram_s2p impl=lutram latency=1
    // 原始几何只由共享准备器读取，无需展开成并行寄存器表。
    static ap_uint<32> starts[32], ends[32], controls[32];
#pragma HLS BIND_STORAGE variable=starts type=ram_s2p impl=lutram latency=1
#pragma HLS BIND_STORAGE variable=ends type=ram_s2p impl=lutram latency=1
#pragma HLS BIND_STORAGE variable=controls type=ram_s2p impl=lutram latency=1
    static RenderBox render[2][MAX_BOXES];
    static bool geometry_valid[MAX_BOXES];
    static bool words_complete[MAX_BOXES];
    static ap_uint<9> written[MAX_BOXES]; // 本次BEGIN后，各框哪些word已重新写入。
#pragma HLS ARRAY_PARTITION variable=render complete dim=0
#pragma HLS ARRAY_PARTITION variable=geometry_valid complete
#pragma HLS ARRAY_PARTITION variable=words_complete complete
#pragma HLS ARRAY_PARTITION variable=written complete dim=1
    // 单命令邮箱允许几何WRITE延迟ACK，准备器由后续像素事务推进。
    enum ControlState { IDLE=0, RAW_LOAD=1, GEOM_VALIDATE=2, FIT=3, CACHE_WRITE=4,
                        WRITE_ACK=5, COMMIT_SCAN=6, COMMIT_FINISH=7, WAIT_SOF=8 };
    static ap_uint<4> control_state = IDLE;
    static ap_uint<5> scan_index = 0;
    static bool commit_bad = false;
    static ap_uint<4> prepare_box = 0;
    static bool prepare_page = false;
    static ap_uint<16> prepare_seq = 0;
    static ap_uint<16> prep_left, prep_right, prep_top, prep_bottom;
    static ap_uint<23> prep_reserved;
    static ap_uint<10> prep_width, prep_text_width;
    static ap_uint<9> prep_height;
    static ap_uint<5> prep_class, prep_score, prep_max;
    static bool prep_valid, prep_large, prep_small;
#pragma HLS RESET variable=control_state
#pragma HLS RESET variable=scan_index
#pragma HLS RESET variable=commit_bad
    // page为当前显示页，!page为上传页；pending期间冻结上传，等SOF交换角色。
    static bool page = false, upload_open = false;
    static bool error = false, synced = false;
    static ap_uint<5> active_count = 0, pending_count = 0;
    // last_seen防止同一命令重复执行；提交命令的ack须等到SOF才追上pending_seq。
    static ap_uint<16> last_seen = 0, ack = 0, pending_seq = 0;
    static ap_uint<10> x = 0;
    static ap_uint<9> y = 0;
#pragma HLS RESET variable=page
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
    // 坐标递推只依赖视频事务，与命令解析、prepare和切页选择完全分离。
    const ap_uint<10> pixel_x = in.user ? ap_uint<10>(0) : x;
    const ap_uint<9> pixel_y = in.user ? ap_uint<9>(0) : y;
    const bool pixel_synced = synced || bool(in.user);
    if (pixel_synced) {
        x = in.last ? ap_uint<10>(0) : ap_uint<10>(pixel_x+1);
        y = in.last ? (pixel_y == HEIGHT-1 ? ap_uint<9>(0) : ap_uint<9>(pixel_y+1)) : pixel_y;
    }
    synced = pixel_synced;
    const ap_uint<16> seq = command.range(31,16);
    const ap_uint<8> opcode = command.range(7,0);
    // 提交逐框校验；SOF仅接受已完成校验的批次，命令接收当拍不切换。
    const bool switch_now = bool(in.user) &&
        (control_state == WAIT_SOF || (control_state == COMMIT_FINISH && !commit_bad));
    if (control_state == RAW_LOAD) {
        // 到准备事务才检查已写字段，WRITE当拍不串接mask归约和状态选择。
        if ((written[prepare_box.to_uint()] & 7) == 7) {
            const unsigned address = (unsigned(prepare_page) << 4) | prepare_box.to_uint();
            const ap_uint<32> a = starts[address], b = ends[address], c = controls[address];
            prep_left = a.range(15,0); prep_top = a.range(31,16);
            prep_right = b.range(15,0); prep_bottom = b.range(31,16);
            prep_class = c.range(4,0); prep_score = c.range(8,5);
            prep_reserved = c.range(31,9);
            control_state = GEOM_VALIDATE;
        } else {
            ack = prepare_seq;
            control_state = IDLE;
        }
    } else if (control_state == GEOM_VALIDATE) {
            prep_valid = prep_left >= ROI_LEFT && prep_right <= ROI_RIGHT &&
                prep_top >= ROI_TOP && prep_bottom <= ROI_BOTTOM &&
                prep_left < prep_right && prep_top < prep_bottom &&
                prep_class <= CLASS_CHARS && prep_score <= SCORE_CHARS && prep_reserved == 0;
            prep_width = prep_right-prep_left; prep_height = prep_bottom-prep_top;
            prep_max = prep_class > prep_score ? prep_class : prep_score;
            control_state = FIT;
    } else if (control_state == FIT) {
            const ap_uint<10> small_width = ap_uint<10>(prep_max) << FONT_CELL_SHIFT;
            const ap_uint<10> large_width = small_width << 1;
            prep_large = prep_width >= large_width+2*MARGIN && prep_height >= LABEL_HEIGHT_2+2*MARGIN;
            prep_small = prep_width >= small_width+2*MARGIN && prep_height >= LABEL_HEIGHT_1+2*MARGIN;
            prep_text_width = prep_large ? large_width : small_width;
            control_state = CACHE_WRITE;
    } else if (control_state == CACHE_WRITE) {
            RenderBox r;
            r.left = prep_left; r.right = prep_right; r.top = prep_top; r.bottom = prep_bottom;
            r.inner_left = prep_left+BORDER; r.inner_right = prep_right-BORDER;
            r.inner_top = prep_top+BORDER; r.inner_bottom = prep_bottom-BORDER;
            r.label_left = prep_left+MARGIN; r.label_right = prep_left+MARGIN+prep_text_width;
            r.label_bottom = prep_bottom-MARGIN;
            r.label_top = prep_bottom-MARGIN-(prep_large ? LABEL_HEIGHT_2 : LABEL_HEIGHT_1);
            r.second_top = r.label_top+(prep_large ? FONT_CELL*2 : FONT_CELL)+LINE_GAP;
            r.class_length = prep_class; r.score_length = prep_score;
            r.large = prep_large; r.label_enable = prep_max != 0 && (prep_large || prep_small);
            for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
                if (prepare_box == ap_uint<4>(i)) {
                    if (prepare_page) render[1][i] = r;
                    else render[0][i] = r;
                    geometry_valid[i] = prep_valid;
                }
            }
            control_state = WRITE_ACK;
    } else if (control_state == WRITE_ACK) {
        ack = prepare_seq;
        control_state = IDLE;
    } else if (control_state == COMMIT_SCAN) {
        if (scan_index < pending_count) {
            const unsigned index = scan_index.to_uint();
            commit_bad = commit_bad || !words_complete[index] || !geometry_valid[index];
            scan_index = scan_index+1;
        } else {
            control_state = COMMIT_FINISH;
        }
    } else if (control_state == COMMIT_FINISH) {
        if (commit_bad) {
            error = true;
            ack = pending_seq;
            control_state = IDLE;
        } else {
            upload_open = false;
            control_state = WAIT_SOF;
        }
    } else if (control_state == IDLE && seq != last_seen) {
        const ap_uint<4> box = command.range(15,12);
        const ap_uint<4> word = command.range(11,8);
        last_seen = seq;
        bool defer_ack = false;
        if (opcode == BEGIN) {
            for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
                written[i] = 0;
                geometry_valid[i] = false;
                words_complete[i] = false;
            }
            upload_open = true;
            error = false;
        } else if (opcode == WRITE_WORD) {
            if (upload_open && word < BOX_WORDS) {
                const unsigned upload_page = page ? 0 : 1;
                const unsigned box_index = box.to_uint();
                const unsigned word_index = word.to_uint();
                ap_uint<9> word_bit = 0;
                for (int bit = 0; bit < BOX_WORDS; ++bit) {
#pragma HLS UNROLL
                    word_bit[bit] = word == ap_uint<4>(bit);
                }
                for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
                    const ap_uint<9> updated_mask = written[i] | word_bit;
                    if (box_index == unsigned(i)) {
                        written[i] = updated_mask;
                        words_complete[i] = updated_mask == 0x1ff;
                    }
                }
                const unsigned geometry_address = (upload_page << 4) | box_index;
                if (word_index == 0) starts[geometry_address] = command_data;
                else if (word_index == 1) ends[geometry_address] = command_data;
                else if (word_index == 2) controls[geometry_address] = command_data;
                if (word_index >= 3)
                    text_memory[(upload_page << 7) | (box_index << 3) | (word_index-3)] = command_data;
                // 几何WRITE统一进入准备入口，完整性判断留给后续事务。
                if (word_index < 3) {
                    prepare_box = box; prepare_page = !page; prepare_seq = seq;
                    control_state = RAW_LOAD;
                    defer_ack = true;
                }
            } else error = true;
        } else if (opcode == COMMIT) {
            pending_count = command_data <= MAX_BOXES ? ap_uint<5>(command_data) : ap_uint<5>(0);
            pending_seq = seq;
            commit_bad = !upload_open || command_data > MAX_BOXES;
            scan_index = 0;
            control_state = COMMIT_SCAN;
            defer_ack = true;
        } else error = true;
        if (!defer_ack) ack = seq;
    }
    // 校验完成后的首个SOF切页，也允许完成校验所在事务恰好为SOF。
    if (switch_now) {
        page = !page;
        active_count = pending_count;
        ack = pending_seq;
        control_state = IDLE;
    }
    const bool roi = pixel_synced && pixel_x >= ROI_LEFT && pixel_x < ROI_RIGHT && pixel_y >= ROI_TOP && pixel_y < ROI_BOTTOM;
    if (!roi) out.data = BLACK;
    // 框与标签命中：已校验的活动页使用视频所需窄位宽。
    ap_uint<16> hits = 0, borders = 0, grant = 0;
    RenderBox active_render[MAX_BOXES];
#pragma HLS ARRAY_PARTITION variable=active_render complete
    for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
        const RenderBox r = page ? render[1][i] : render[0][i];
        active_render[i] = r;
        if (i < active_count) {
            borders[i] = pixel_x >= r.left && pixel_x < r.right && pixel_y >= r.top && pixel_y < r.bottom &&
                (pixel_x < r.inner_left || pixel_x >= r.inner_right || pixel_y < r.inner_top || pixel_y >= r.inner_bottom);
            if (r.label_enable && pixel_x >= r.label_left && pixel_x < r.label_right &&
                pixel_y >= r.label_top && pixel_y < r.label_bottom) hits[i] = 1;
        }
    }
    // 并行前缀零检测产生低索引优先one-hot，不串接最小值选择器。
    ap_uint<39> selected[31];
#pragma HLS ARRAY_PARTITION variable=selected complete
    for (int i = 0; i < MAX_BOXES; ++i) {
#pragma HLS UNROLL
        grant[i] = hits[i] && (hits & ap_uint<16>((1u << i)-1u)) == 0;
        const RenderBox r = active_render[i];
        ap_uint<39> fields = 0;
        fields.range(9,0) = r.label_left;
        fields.range(18,10) = r.label_top;
        fields.range(27,19) = r.second_top;
        fields.range(32,28) = r.class_length;
        fields.range(37,33) = r.score_length;
        fields[38] = r.large;
        selected[15+i] = fields & (ap_uint<39>(0)-ap_uint<39>(grant[i]));
    }
    for (int i = 14; i >= 0; --i) {
#pragma HLS UNROLL
        selected[i] = selected[2*i+1] | selected[2*i+2];
    }
    if (roi && borders != 0) out.data = GREEN;
    // 共享字模：优先树已选出唯一标签，后面只查询一次ROM。
    if (roi && hits != 0) {
        ap_uint<4> index = 0;
        index[0] = (grant & 0xaaaa) != 0;
        index[1] = (grant & 0xcccc) != 0;
        index[2] = (grant & 0xf0f0) != 0;
        index[3] = (grant & 0xff00) != 0;
        const unsigned i = index.to_uint();
        const unsigned active_page = page ? 1 : 0;
        const ap_uint<10> label_left = selected[0].range(9,0);
        const ap_uint<9> label_top = selected[0].range(18,10), second_top = selected[0].range(27,19);
        const ap_uint<5> class_length = selected[0].range(32,28), score_length = selected[0].range(37,33);
        const bool large = selected[0][38];
        const ap_uint<8> relative_x = pixel_x-label_left; // 最多255，标签内局部坐标
        const ap_uint<5> cell_height = large ? FONT_CELL*2 : FONT_CELL;
        const bool second_line = pixel_y >= second_top;
        const ap_uint<6> row_y = second_line ? ap_uint<6>(pixel_y-second_top) : ap_uint<6>(pixel_y-label_top);
        const ap_uint<4> column = large ? relative_x >> (FONT_CELL_SHIFT+1) : relative_x >> FONT_CELL_SHIFT;
        const ap_uint<3> glyph_x = large ? ap_uint<3>(relative_x >> 1) : ap_uint<3>(relative_x);
        const ap_uint<6> glyph_y = large ? ap_uint<6>(row_y >> 1) : row_y;
        ap_uint<8> character = ' ';
        if (column < (second_line ? score_length : class_length)) {
            const unsigned word_index = (second_line ? 4 : 0)+(column.to_uint() >> 2);
            const ap_uint<5> byte_shift = ap_uint<5>(column.range(1,0)) << 3;
            character = text_memory[(active_page << 7) | (i << 3) | word_index] >> byte_shift;
        }
        const ap_uint<5> glyph = roi_overlay_detail::overlay_font_row(character, ap_uint<3>(glyph_y));
        out.data = BLACK;
        if (row_y < cell_height && glyph_x < 5 && glyph_y < 7 && glyph[4-glyph_x.to_uint()]) out.data = WHITE;
    }
    // 输出与状态：每个完成的像素事务输出一个像素并写回status。
    m_axis_video.write(out);
    // ap_vld通知AXI-Lite适配器更新可读状态，不代表命令ACK。
    status = (ap_uint<32>(ack) << 16) | (ap_uint<32>(active_count) << 8) |
             (ap_uint<32>(synced) << 3) | (ap_uint<32>(error) << 2) |
             (ap_uint<32>(upload_open) << 1) |
             ap_uint<32>(control_state == COMMIT_SCAN || control_state == COMMIT_FINISH || control_state == WAIT_SOF);
}
