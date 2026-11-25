#ifndef CAMERA_DISPLAY_H
#define CAMERA_DISPLAY_H
#include "camera_postprocess.h"
/* 每条命令以及启动同步分别限时，视频停止也不能无限等待。 */
#define CAMERA_DISPLAY_TIMEOUT_MS 500U
typedef enum {
    CAMERA_DISPLAY_OK = 0,
    CAMERA_DISPLAY_DRIVER,
    CAMERA_DISPLAY_ARGUMENT,
    CAMERA_DISPLAY_NOT_INITIALIZED,
    CAMERA_DISPLAY_TIMEOUT,
    CAMERA_DISPLAY_PROTOCOL,
    CAMERA_DISPLAY_FAULT
} camera_display_status_t;
/* 单线程顺序调用。超时/协议错误锁存故障，重复init不会解除，
 * 需协调软件重启及硬件复位；参数错误不发命令，也不锁存故障。 */
/* 调用前视频须已运行；先同步已有命令，成功返回时已清空旧列表。 */
camera_display_status_t camera_display_init(void);
/* 输入为416×416 ROI内浮点坐标及objectness×类别概率score。
 * 成功返回表示整批已在SOF切换生效；count=0提交空列表清屏。 */
camera_display_status_t camera_display_update(const camera_detections_t *detections);
/* 显式BEGIN+COMMIT空列表，成功返回表示旧框已在SOF消失。 */
camera_display_status_t camera_display_clear(void);
const char *camera_display_reason(camera_display_status_t status);
#endif
