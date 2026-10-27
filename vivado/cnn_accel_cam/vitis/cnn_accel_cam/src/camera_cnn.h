#ifndef CAMERA_CNN_H
#define CAMERA_CNN_H
#include <stdint.h>

/* PS-owned constants: keep in sync with the independently maintained PL layout. */
enum { CAMERA_CNN_INPUT_BYTES = 784, CAMERA_CNN_CLASSES = 10,
    CAMERA_CNN_PROBABILITY_SCALE = 10000, CAMERA_CNN_RESULT_BYTES = 24,
    CAMERA_CNN_CLASS_BYTE_OFFSET = 20, CAMERA_CNN_SOFTMAX_SCALE_FRAC_BITS = 20 };
/* ABI v2: ten uint16 probabilities, followed by uint32 class. */
typedef struct {
    uint16_t probability[CAMERA_CNN_CLASSES];
    uint32_t class_index;
} camera_cnn_result_t;
typedef enum {
    CAMERA_CNN_OK = 0, CAMERA_CNN_ARGUMENT, CAMERA_CNN_DDR,
    CAMERA_CNN_DRIVER, CAMERA_CNN_BASE, CAMERA_CNN_BUSY,
    CAMERA_CNN_TIMEOUT, CAMERA_CNN_RESULT, CAMERA_CNN_NOT_INITIALIZED
} camera_cnn_status_t;
/* 启动时初始化一次、准备并flush模型；只配置地址，不启动CNN。 */
camera_cnn_status_t camera_cnn_init(void);
/* 输入为784个单通道signed byte；仅返回OK时result有效。
 * busy/timeout时必须结束程序，板级复位后重启；不能重试复用DMA缓冲。 */
camera_cnn_status_t camera_cnn_predict(const int8_t input[CAMERA_CNN_INPUT_BYTES],
                                     camera_cnn_result_t *result);
const char *camera_cnn_reason(camera_cnn_status_t status);
#endif
