#ifndef CAMERA_PREPROCESS_H
#define CAMERA_PREPROCESS_H
#include <stdint.h>
#include "camera_geometry.h"
/* CPU独占的紧密416×416 BGR快照转换为HWC RGB INT8；两缓冲不得重叠。
 * 裁剪已由采集完成，这里不缩放、不灰度化、不反色、不补齐8通道。 */
void camera_preprocess(const uint8_t snapshot[CAMERA_ROI_BYTES],
                       int8_t output[CAMERA_ROI_BYTES]);
/* 独立字节量化入口便于穷举核对；运行时仅使用整数，无LUT。 */
int8_t camera_byte_to_int8(uint8_t pixel);
#endif
