#include "camera_preprocess.h"
#include "camera_model_input.h"

/* 固定几何约束在编译时检查，不增加逐帧分支。 */
_Static_assert(CAMERA_ROI_X + CAMERA_ROI_SIZE <= CAMERA_PREPROCESS_WIDTH &&
    CAMERA_ROI_Y + CAMERA_ROI_SIZE <= CAMERA_PREPROCESS_HEIGHT, "ROI must fit in frame");
_Static_assert(CAMERA_ROI_SIZE == CAMERA_BLOCK_SIZE * CAMERA_MODEL_SIDE, "ROI must tile exactly");
_Static_assert(CAMERA_PREPROCESS_WIDTH == 640U &&
    CAMERA_PREPROCESS_HEIGHT == 480U && CAMERA_MODEL_PIXELS == 784U &&
    CAMERA_BLOCK_SIZE == 10U, "fixed camera/model geometry required");
#ifdef __FAST_MATH__
#error "camera_preprocess requires strict floating-point operations (no fast-math)"
#endif

int8_t camera_gray_to_int8(uint8_t gray)
{
    /* volatile强制每一步写回float32，保留ToTensor、Normalize的运算顺序，
     * 不允许把减法和除法合并或提前用double计算。模型scale则为导出的double。 */
    volatile float tensor = (float)gray / 255.0f;
    volatile float centered = tensor - CAMERA_MODEL_MEAN;
    volatile float normalized = centered / CAMERA_MODEL_STD;
    double scaled = (double)normalized / CAMERA_MODEL_INPUT_SCALE;
    int rounded;
    double fraction;

    /* 先截向零，再比较余数；正负半整数都向偶数舍入，不依赖roundf
     * (它在半整数处远离零)，也不依赖当前浮点舍入模式的整数转换函数。 */
    if (scaled <= -128.0) return -128;
    if (scaled >= 127.0) return 127;
    rounded = (int)scaled;
    fraction = scaled - (double)rounded;
    if (fraction > 0.5 || (fraction == 0.5 && rounded % 2 != 0)) ++rounded;
    if (fraction < -0.5 || (fraction == -0.5 && rounded % 2 != 0)) --rounded;
    return (int8_t)rounded;
}

void camera_preprocess(const uint8_t *snapshot,
                       uint8_t gray[CAMERA_MODEL_PIXELS],
                       int8_t output[CAMERA_MODEL_PIXELS])
{
    unsigned int y, x, dy, dx;
    for (y = 0; y < CAMERA_MODEL_SIDE; ++y) {
        for (x = 0; x < CAMERA_MODEL_SIDE; ++x) {
            unsigned int sum = 0;
            unsigned int index = y * CAMERA_MODEL_SIDE + x;
            for (dy = 0; dy < CAMERA_BLOCK_SIZE; ++dy) {
                for (dx = 0; dx < CAMERA_BLOCK_SIZE; ++dx) {
                    unsigned int offset = ((CAMERA_ROI_Y + y * CAMERA_BLOCK_SIZE + dy) *
                        CAMERA_PREPROCESS_WIDTH + CAMERA_ROI_X + x * CAMERA_BLOCK_SIZE + dx) * 3U;
                    /* BGR低地址顺序；先对每个像素做整型灰度及舍入，再累加。 */
                    sum += (77U * snapshot[offset + 2U] + 150U * snapshot[offset + 1U] +
                            29U * snapshot[offset] + 128U) >> 8;
                }
            }
            /* 100点均值四舍五入；反色在均值之后，保留此灰度便于观察。 */
            gray[index] = (uint8_t)((sum + CAMERA_BLOCK_PIXELS / 2U) / CAMERA_BLOCK_PIXELS);
#if CAMERA_PREPROCESS_INVERT
            gray[index] = (uint8_t)(255U - gray[index]);
#endif
            output[index] = camera_gray_to_int8(gray[index]);
        }
    }
}
