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
#error "camera_preprocess must not use fast-math"
#endif
_Static_assert(CAMERA_MODEL_INPUT_SHIFT >= 1 && CAMERA_MODEL_INPUT_SHIFT <= 30,
    "quantization shift must be 1..30");
_Static_assert(CAMERA_MODEL_INPUT_ZERO_POINT == 0 && CAMERA_MODEL_INPUT_MULTIPLIER >= 0,
    "symmetric quantization and nonnegative multiplier required");
_Static_assert((int64_t)CAMERA_MODEL_INPUT_MULTIPLIER <= INT32_MAX &&
    255LL * CAMERA_MODEL_INPUT_MULTIPLIER <= INT32_MAX &&
    (int64_t)CAMERA_MODEL_INPUT_OFFSET >= INT32_MIN &&
    (int64_t)CAMERA_MODEL_INPUT_OFFSET <= INT32_MAX &&
    255LL * CAMERA_MODEL_INPUT_MULTIPLIER + CAMERA_MODEL_INPUT_OFFSET >= INT32_MIN &&
    255LL * CAMERA_MODEL_INPUT_MULTIPLIER + CAMERA_MODEL_INPUT_OFFSET <= INT32_MAX,
    "quantization intermediate must fit int32");

/* 先取无符号幅值，避免负数右移；半整数向偶数舍入。 */
static inline int32_t round_shift_even(int32_t value)
{
    uint32_t magnitude = value < 0 ? 0U - (uint32_t)value : (uint32_t)value;
    uint32_t integer = magnitude >> CAMERA_MODEL_INPUT_SHIFT;
    uint32_t remainder = magnitude & ((1U << CAMERA_MODEL_INPUT_SHIFT) - 1U);
    uint32_t half = 1U << (CAMERA_MODEL_INPUT_SHIFT - 1);
    if (remainder > half || (remainder == half && (integer & 1U))) ++integer;
    /* SHIFT>=1，舍入后的幅值可安全转换为int32，包括输入INT32_MIN。 */
    return value < 0 ? -(int32_t)integer : (int32_t)integer;
}

int8_t camera_gray_to_int8(uint8_t gray)
{
    int32_t value = (int32_t)gray * (int32_t)CAMERA_MODEL_INPUT_MULTIPLIER +
        (int32_t)CAMERA_MODEL_INPUT_OFFSET;
    int32_t rounded = round_shift_even(value);
    if (rounded < -128) return -128;
    if (rounded > 127) return 127;
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
