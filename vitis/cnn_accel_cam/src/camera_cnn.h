#ifndef CAMERA_CNN_H
#define CAMERA_CNN_H
#include <stdint.h>

#define CAMERA_CNN_TIMEOUT_SECONDS 10U
typedef enum {
    CAMERA_CNN_OK = 0, CAMERA_CNN_DDR, CAMERA_CNN_DRIVER,
    CAMERA_CNN_BASE, CAMERA_CNN_BUSY, CAMERA_CNN_MODEL,
    CAMERA_CNN_ARGUMENT, CAMERA_CNN_NOT_INITIALIZED,
    CAMERA_CNN_TIMEOUT, CAMERA_CNN_FAULT
} camera_cnn_status_t;
/* 初始化只加载模型和配置 DDR 地址，不启动推理；重复调用不重复加载。 */
camera_cnn_status_t camera_cnn_init(void);
/* 只有初始化成功、无终止故障且硬件空闲时才允许 PS 写 A 输入。
 * NULL 表示不可写；busy/timeout 后必须复位开发板，不能继续复用 DDR。 */
int8_t *camera_cnn_input_buffer(void);
/* 同步单次推理。output 必须非空；失败时 *output 为 NULL。
 * 成功视图为 B+0 的 13×13×128 int8，125 个有效通道，不复制结果。
 * 视图只在下一次推理前有效；这里不做 NMS 或检测框解释。 */
camera_cnn_status_t camera_cnn_run(const int8_t **output);
const char *camera_cnn_reason(camera_cnn_status_t status);
#endif
