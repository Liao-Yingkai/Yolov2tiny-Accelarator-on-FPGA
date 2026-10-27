#include "camera_display.h"
#include "xparameters.h"
#include "xroi_border.h"
#include "xstatus.h"

static XRoi_border display;

int camera_display_init(void)
{
    int status = XRoi_border_Initialize(&display, XPAR_ROI_BORDER_0_DEVICE_ID);
    if (status != XST_SUCCESS) return status;
    camera_display_clear();
    return XST_SUCCESS;
}

void camera_display_clear(void)
{
    XRoi_border_Set_display_control(&display, 0U);
}

void camera_display_update(uint32_t digit, uint32_t probability)
{
    uint32_t control = 0U;
    /* 严格大于95%才显示，低概率或非法结果立即请求隐藏旧数字。 */
    if (digit < CAMERA_DISPLAY_CLASSES && probability > CAMERA_DISPLAY_THRESHOLD &&
        probability <= CAMERA_DISPLAY_PROBABILITY_SCALE)
        control = CAMERA_DISPLAY_ENABLE | digit;
    /* 自由运行的视频IP不需要Start；整字在SOF采样，下一帧生效。 */
    XRoi_border_Set_display_control(&display, control);
}
