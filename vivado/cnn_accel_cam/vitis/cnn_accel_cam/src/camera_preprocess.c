#include "camera_preprocess.h"
#include "camera_model_input.h"

_Static_assert(CAMERA_ROI_X + CAMERA_ROI_SIDE <= CAMERA_WIDTH &&
    CAMERA_ROI_Y + CAMERA_ROI_SIDE <= CAMERA_HEIGHT, "ROI must fit in frame");
_Static_assert(CAMERA_MODEL_INPUT_SHIFT >= 1 && CAMERA_MODEL_INPUT_SHIFT <= 30,
    "quantization shift must be 1..30");
_Static_assert(CAMERA_MODEL_INPUT_ZERO_POINT == 0 && CAMERA_MODEL_INPUT_MULTIPLIER >= 0 &&
    255LL * CAMERA_MODEL_INPUT_MULTIPLIER <= INT32_MAX,
    "symmetric quantization product must fit int32");

int8_t camera_byte_to_int8(uint8_t pixel)
{
    /* 离线参数已证明与FP32 byte/255及reciprocal量化的256种输入一致。
     * 乘积非负且不超过INT32_MAX；余数恰好一半时向偶数舍入。 */
    uint32_t product = (uint32_t)pixel * CAMERA_MODEL_INPUT_MULTIPLIER;
    uint32_t integer = product >> CAMERA_MODEL_INPUT_SHIFT;
    uint32_t remainder = product & ((1U << CAMERA_MODEL_INPUT_SHIFT) - 1U);
    uint32_t half = 1U << (CAMERA_MODEL_INPUT_SHIFT - 1);
    if (remainder > half || (remainder == half && (integer & 1U))) ++integer;
    /* byte输入和zero_point=0使结果非负，故下限-128天然满足。 */
    return integer > 127U ? 127 : (int8_t)integer;
}

void camera_preprocess(const uint8_t snapshot[CAMERA_ROI_BYTES],
                       int8_t output[CAMERA_ROI_BYTES])
{
    unsigned int offset;
    /* 逐行逐像素的三字节顺序保持不变，仅交换B/R位置并分别量化。 */
    for (offset = 0; offset < CAMERA_ROI_BYTES; offset += 3U) {
        output[offset] = camera_byte_to_int8(snapshot[offset + 2U]);
        output[offset + 1U] = camera_byte_to_int8(snapshot[offset + 1U]);
        output[offset + 2U] = camera_byte_to_int8(snapshot[offset]);
    }
}
