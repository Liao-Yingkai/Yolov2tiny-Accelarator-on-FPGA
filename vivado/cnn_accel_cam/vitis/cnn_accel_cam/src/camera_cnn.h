#ifndef CAMERA_CNN_H
#define CAMERA_CNN_H
typedef enum {
    CAMERA_CNN_OK = 0, CAMERA_CNN_DDR, CAMERA_CNN_DRIVER,
    CAMERA_CNN_BASE, CAMERA_CNN_BUSY, CAMERA_CNN_MODEL
} camera_cnn_status_t;
/* 本阶段只加载模型和配置 DDR 地址，绝不启动推理。 */
camera_cnn_status_t camera_cnn_init(void);
const char *camera_cnn_reason(camera_cnn_status_t status);
#endif
