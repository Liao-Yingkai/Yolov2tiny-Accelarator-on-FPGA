#include "camera_display.h"
#include "camera_geometry.h"
#include "xparameters.h"
#include "xroi_overlay.h"
#include "xstatus.h"
#include "xtime_l.h"
#include "xpseudo_asm.h"
#include <math.h>
#include <stdint.h>

enum { MAX_BOXES = 16, BOX_WORDS = 9, OP_BEGIN = 1, OP_WRITE = 2,
       OP_COMMIT = 3, PENDING = 1, OPEN = 2, ERROR = 4, SYNCED = 8 };
static XRoi_overlay display;
static uint16_t sequence;
static int initialized;
static int fault;

static int expired(XTime started)
{
    XTime now;
    XTime_GetTime(&now);
    return now - started >= (XTime)COUNTS_PER_SECOND * CAMERA_DISPLAY_TIMEOUT_MS / 1000U;
}

static camera_display_status_t available(void)
{
    if (fault) return CAMERA_DISPLAY_FAULT;
    if (!initialized) return CAMERA_DISPLAY_NOT_INITIALIZED;
    return CAMERA_DISPLAY_OK;
}

/* 邮箱只容纳一条命令，ACK前绝不触碰下一条payload或command。
 * 几何WRITE也可能延迟；COMMIT ACK表示已经在SOF激活整批列表。 */
static camera_display_status_t send_command(unsigned opcode, unsigned box,
                                           unsigned word, uint32_t data)
{
    XTime started;
    uint32_t status;
    uint16_t next = (uint16_t)(sequence + 1U);
    XTime_GetTime(&started);
    XRoi_overlay_Set_command_data(&display, data);
    dsb();
    XRoi_overlay_Set_command(&display, ((uint32_t)next << 16) |
                             (box << 12) | (word << 8) | opcode);
    dsb();
    for (;;) {
        status = XRoi_overlay_Get_status(&display);
        if ((uint16_t)(status >> 16) == next) break;
        if (expired(started)) {
            fault = 1;
            return CAMERA_DISPLAY_TIMEOUT;
        }
    }
    sequence = next;
    /* 只用同一个ACK快照检查状态；pending下降不能替代ACK。
     * BEGIN清sticky error，每批都重新BEGIN以明确本批错误状态。 */
    if ((status & (PENDING | ERROR)) != 0U || (status & SYNCED) == 0U ||
        (opcode == OP_COMMIT ?
         ((status & OPEN) != 0U || ((status >> 8) & 255U) != data) :
         (status & OPEN) == 0U)) {
        fault = 1;
        return CAMERA_DISPLAY_PROTOCOL;
    }
    return CAMERA_DISPLAY_OK;
}

static camera_display_status_t upload(uint32_t packed[MAX_BOXES][BOX_WORDS], unsigned count)
{
    unsigned box, word;
    camera_display_status_t result = send_command(OP_BEGIN, 0, 0, 0);
    if (result != CAMERA_DISPLAY_OK) return result;
    for (box = 0; box < count; ++box) {
        for (word = 0; word < BOX_WORDS; ++word) {
            result = send_command(OP_WRITE, box, word, packed[box][word]);
            if (result != CAMERA_DISPLAY_OK) return result;
        }
    }
    return send_command(OP_COMMIT, 0, 0, count);
}

camera_display_status_t camera_display_init(void)
{
    XTime started;
    uint16_t previous;
    uint32_t status;
    camera_display_status_t result;
    if (fault) return CAMERA_DISPLAY_FAULT;
    if (initialized) return CAMERA_DISPLAY_OK;
    if (XRoi_overlay_Initialize(&display, XPAR_ROI_OVERLAY_0_DEVICE_ID) != XST_SUCCESS)
        return CAMERA_DISPLAY_DRIVER;
    /* PS暖重启不代表PL复位。只读等待旧命令自然完成；几何准备时
     * pending也可能为0，所以还须ACK追上command里的旧sequence。
     * 历史error允许保留到后续BEGIN，不能在等待阶段抢写邮箱。 */
    previous = (uint16_t)(XRoi_overlay_Get_command(&display) >> 16);
    XTime_GetTime(&started);
    for (;;) {
        status = XRoi_overlay_Get_status(&display);
        if ((status & SYNCED) != 0U && (status & PENDING) == 0U &&
            (uint16_t)(status >> 16) == previous) break;
        if (expired(started)) {
            fault = 1;
            return CAMERA_DISPLAY_TIMEOUT;
        }
    }
    sequence = (uint16_t)(status >> 16);
    /* ap_ctrl_none无需Start；不自动重试或复位硬件。 */
    initialized = 1;
    result = camera_display_clear();
    if (result != CAMERA_DISPLAY_OK) initialized = 0;
    return result;
}

camera_display_status_t camera_display_clear(void)
{
    camera_display_status_t result = available();
    if (result != CAMERA_DISPLAY_OK) return result;
    /* count=0仍完整执行BEGIN+COMMIT，ACK时旧框才真正消失。 */
    return upload(0, 0);
}

static unsigned clipped_edge(float value, int upper)
{
    /* 先clip再取整结果相同，且避免超大浮点数直接转换为整数。 */
    if (value <= 0.0f) return 0;
    if (value >= (float)CAMERA_ROI_SIDE) return CAMERA_ROI_SIDE;
    return (unsigned)(upper ? ceilf(value) : floorf(value));
}

/* score已是objectness乘类别概率。转为千分比后手写十进制，
 * 固定一位小数且不依赖嵌入式printf浮点支持。 */
static unsigned score_text(float score, char text[8])
{
    unsigned value = (unsigned)(score * 1000.0f + 0.5f);
    unsigned whole = value / 10U;
    unsigned length = 0;
    if (whole >= 100U) text[length++] = '1';
    if (whole >= 10U) text[length++] = (char)('0' + (whole / 10U) % 10U);
    text[length++] = (char)('0' + whole % 10U);
    text[length++] = '.';
    text[length++] = (char)('0' + value % 10U);
    text[length++] = '%';
    return length;
}

camera_display_status_t camera_display_update(const camera_detections_t *detections)
{
    uint32_t packed[MAX_BOXES][BOX_WORDS] = {{0}};
    unsigned order[MAX_BOXES];
    unsigned i, j, index, left, top, right, bottom, class_length, score_length;
    char score[8];
    const char *name;
    const camera_detection_t *box;
    camera_display_status_t result = available();
    if (result != CAMERA_DISPLAY_OK) return result;
    if (!detections || detections->count > MAX_BOXES) return CAMERA_DISPLAY_ARGUMENT;
    /* 校验所有输入并组装完整9word后才能开始上传。非法参数保留旧
     * 显示，不产生半批写入，也不锁存硬件故障。 */
    for (i = 0; i < detections->count; ++i) {
        box = &detections->boxes[i];
        if (!isfinite(box->left) || !isfinite(box->top) ||
            !isfinite(box->right) || !isfinite(box->bottom) ||
            !isfinite(box->score) || box->score < 0.0f || box->score > 1.0f ||
            box->left >= box->right || box->top >= box->bottom ||
            camera_class_name(box->class_id) == 0) return CAMERA_DISPLAY_ARGUMENT;
        /* 稳定降序确保高分框的重叠标签优先，不依赖调用方排序。 */
        j = i;
        while (j > 0 && detections->boxes[order[j-1]].score < box->score) {
            order[j] = order[j-1];
            --j;
        }
        order[j] = i;
    }
    for (i = 0; i < detections->count; ++i) {
        box = &detections->boxes[order[i]];
        left = clipped_edge(box->left, 0);
        top = clipped_edge(box->top, 0);
        right = clipped_edge(box->right, 1);
        bottom = clipped_edge(box->bottom, 1);
        if (left >= right || top >= bottom) return CAMERA_DISPLAY_ARGUMENT;
        name = camera_class_name(box->class_id);
        for (class_length = 0; class_length < 16U && name[class_length]; ++class_length) {}
        if (name[class_length] != '\0') return CAMERA_DISPLAY_ARGUMENT;
        score_length = score_text(box->score, score);
        packed[i][0] = (left + CAMERA_ROI_X) | ((top + CAMERA_ROI_Y) << 16);
        packed[i][1] = (right + CAMERA_ROI_X) | ((bottom + CAMERA_ROI_Y) << 16);
        packed[i][2] = class_length | (score_length << 5);
        /* 最前字符放在word最低byte，其余未使用字节为零。 */
        for (index = 0; index < class_length; ++index)
            packed[i][3U + index / 4U] |= (uint32_t)(unsigned char)name[index] << ((index % 4U) * 8U);
        for (index = 0; index < score_length; ++index)
            packed[i][7U + index / 4U] |= (uint32_t)(unsigned char)score[index] << ((index % 4U) * 8U);
    }
    return upload(packed, detections->count);
}

const char *camera_display_reason(camera_display_status_t status)
{
    switch (status) {
    case CAMERA_DISPLAY_OK: return "ok";
    case CAMERA_DISPLAY_DRIVER: return "overlay driver initialization failed";
    case CAMERA_DISPLAY_ARGUMENT: return "invalid overlay detections";
    case CAMERA_DISPLAY_NOT_INITIALIZED: return "overlay is not initialized";
    case CAMERA_DISPLAY_TIMEOUT: return "overlay ACK/video synchronization timeout; fault latched";
    case CAMERA_DISPLAY_PROTOCOL: return "overlay ACK status mismatch; fault latched";
    case CAMERA_DISPLAY_FAULT: return "overlay fault latched; coordinated restart/reset required";
    default: return "unknown overlay status";
    }
}
