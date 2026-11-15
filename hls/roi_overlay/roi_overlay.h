#ifndef ROI_OVERLAY_H
#define ROI_OVERLAY_H
#include <ap_axi_sdata.h>
#include <hls_stream.h>

namespace roi_overlay_config {
constexpr int WIDTH = 640, HEIGHT = 480;
constexpr int ROI_LEFT = 112, ROI_RIGHT = 528, ROI_TOP = 32, ROI_BOTTOM = 448;
constexpr int MAX_BOXES = 16, BOX_WORDS = 9, CLASS_CHARS = 16, SCORE_CHARS = 8;
constexpr unsigned BEGIN = 1, WRITE_WORD = 2, COMMIT = 3;
constexpr unsigned GREEN = 0x00ff00, WHITE = 0xffffff, BLACK = 0;
constexpr int BORDER = 3, MARGIN = 2, LINE_GAP = 2;
constexpr int FONT_CELL = 8, FONT_CELL_SHIFT = 3;
constexpr int LABEL_HEIGHT_1 = 2*FONT_CELL + LINE_GAP;
constexpr int LABEL_HEIGHT_2 = 4*FONT_CELL + LINE_GAP;
static_assert(MAX_BOXES == 16 && BOX_WORDS == 9, "fixed command layout and priority tree");
static_assert(CLASS_CHARS == 16 && SCORE_CHARS == 8, "fixed ASCII payload layout");
static_assert(ROI_LEFT >= 0 && ROI_TOP >= 0 && ROI_LEFT < ROI_RIGHT &&
              ROI_TOP < ROI_BOTTOM && ROI_RIGHT <= WIDTH && ROI_BOTTOM <= HEIGHT,
              "ROI must fit video");
static_assert(WIDTH <= (1 << 10) && HEIGHT <= (1 << 9), "coordinate width overflow");
static_assert(FONT_CELL == (1 << FONT_CELL_SHIFT) && FONT_CELL == 8,
              "font addressing uses 8-pixel cells");
static_assert(LABEL_HEIGHT_2 < (1 << 6) && CLASS_CHARS*FONT_CELL*2 <= (1 << 8) &&
              SCORE_CHARS <= CLASS_CHARS, "label local-coordinate width overflow");
}
using video_axis_t = ap_axiu<24, 1, 0, 0>;
// 每次调用传递一个像素；视频自由运行，PS不需要Start。
// command/data为单命令邮箱，status返回ACK及整批切换状态，位定义见README。
// PS先BEGIN、上传前count框的全部9word，再COMMIT；逐框检查完才可在SOF确认。
// COMMIT检查需count+2个后续像素事务，检查期间的SOF不会激活新页。
// status.pending覆盖检查及等待SOF；ACK仍表示新批次已在帧边界生效。
// 几何WRITE延后1或5个像素事务ACK（字段未齐/准备布局），PS始终逐条等ACK。
void roi_overlay(hls::stream<video_axis_t>& s_axis_video,
                 hls::stream<video_axis_t>& m_axis_video,
                 ap_uint<32> command, ap_uint<32> command_data,
                 ap_uint<32>& status);
#endif
