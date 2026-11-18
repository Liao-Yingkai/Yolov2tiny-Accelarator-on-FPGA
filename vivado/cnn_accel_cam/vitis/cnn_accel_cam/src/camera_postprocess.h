#ifndef CAMERA_POSTPROCESS_H
#define CAMERA_POSTPROCESS_H

#include <stdint.h>
#include "camera_model_output.h"

#define CAMERA_POSTPROCESS_SCORE_THRESHOLD 0.55f
#define CAMERA_POSTPROCESS_IOU_THRESHOLD 0.55f
#define CAMERA_POSTPROCESS_MAX_DETECTIONS 8U

/* 坐标为416×416 ROI内的浮点像素坐标，不含原视频的裁剪偏移。
 * right/bottom是连续坐标边界，计算面积时不能加1。 */
typedef struct {
    float left;
    float top;
    float right;
    float bottom;
    float score;
    unsigned class_id;
} camera_detection_t;

typedef struct {
    unsigned count;
    camera_detection_t boxes[CAMERA_POSTPROCESS_MAX_DETECTIONS];
} camera_detections_t;

enum {
    CAMERA_POSTPROCESS_OK = 0,
    CAMERA_POSTPROCESS_ERROR_ARGUMENT = 1
};

/* prediction指向完整13×13×128的有符号int8结果；不读取125..127填充通道。
 * 返回0表示成功（无目标也成功）。results非空时总会先清空count。
 * 内部复用静态DDR工作区，仅供单线程顺序调用；核心不打印串口日志。 */
int camera_postprocess(const int8_t *prediction, camera_detections_t *results);

/* 类别名称来自模型元数据，非法类别返回空指针。 */
const char *camera_class_name(unsigned class_id);

#endif
