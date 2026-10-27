#ifndef CAMERA_DISPLAY_H
#define CAMERA_DISPLAY_H
#include <stdint.h>
#include "camera_cnn.h"

/* PS独立维护控制字：bit8使能，bits3:0数字，其余位写0。 */
enum { CAMERA_DISPLAY_CLASSES = CAMERA_CNN_CLASSES, CAMERA_DISPLAY_THRESHOLD = 9500,
    CAMERA_DISPLAY_PROBABILITY_SCALE = CAMERA_CNN_PROBABILITY_SCALE,
    CAMERA_DISPLAY_ENABLE = 1U << 8 };
/* 概率单位1/10000，严格>9500才显示；改此头阈值只需重编译PS应用。 */
/* 返回XST状态码，成功时清屏；update/clear仅在init成功后调用。 */
int camera_display_init(void);
/* 低概率或非法结果关闭显示；控制字更新（包括关闭）在下一SOF生效。 */
void camera_display_update(uint32_t digit, uint32_t probability);
void camera_display_clear(void);
#endif
