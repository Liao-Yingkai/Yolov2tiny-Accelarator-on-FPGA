#ifndef CAMERA_PREPROCESS_H
#define CAMERA_PREPROCESS_H
#include <stdint.h>

/* 输入固定640×480、紧密排列BGR888；ROI坐标/边长单位均为像素。 */
#define CAMERA_PREPROCESS_WIDTH 640U
#define CAMERA_PREPROCESS_HEIGHT 480U
#define CAMERA_ROI_X 180U
#define CAMERA_ROI_Y 100U
#define CAMERA_ROI_SIZE 280U
/* 每个输出像素覆盖10×10输入像素；输出28×28，行优先、单通道。 */
#define CAMERA_BLOCK_SIZE 10U
#define CAMERA_BLOCK_PIXELS (CAMERA_BLOCK_SIZE * CAMERA_BLOCK_SIZE)
#define CAMERA_MODEL_SIDE 28U
#define CAMERA_MODEL_PIXELS (CAMERA_MODEL_SIDE * CAMERA_MODEL_SIDE)
/* 1：白纸黑字转为暗背景亮笔画；0：保留原灰度极性。 */
#define CAMERA_PREPROCESS_INVERT 1
/* 调试缓冲容量按64字节向上取整；有效数据仅前784字节。 */
#define CAMERA_DEBUG_BYTES (((CAMERA_MODEL_PIXELS + 63U) / 64U) * 64U)

/* 输入为CPU独占且稳定的整帧，输出不得与输入或彼此重叠。
 * gray保存反色后的28×28 uint8灰度；output是模型单通道int8输入。
 * 所有配置在编译期确定，不做8通道补零、不访问DMA。 */
void camera_preprocess(const uint8_t *snapshot,
                       uint8_t gray[CAMERA_MODEL_PIXELS],
                       int8_t output[CAMERA_MODEL_PIXELS]);
/* 独立入口方便对全部256种灰度进行数值核对；无LUT。 */
int8_t camera_gray_to_int8(uint8_t gray);
#endif
