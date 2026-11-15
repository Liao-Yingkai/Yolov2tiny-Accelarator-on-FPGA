#include "camera_postprocess.h"
#include "camera_geometry.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#define VALUES_PER_ANCHOR (5U + CAMERA_MODEL_OUTPUT_CLASSES)
#define CANDIDATE_CAPACITY (CAMERA_MODEL_OUTPUT_GRID * CAMERA_MODEL_OUTPUT_GRID * \
                            CAMERA_MODEL_OUTPUT_ANCHORS * CAMERA_MODEL_OUTPUT_CLASSES)

/* 编译期检查，避免生成的模型布局与解码器悄悄不一致。 */
_Static_assert(
    CAMERA_MODEL_OUTPUT_LOGICAL_CHANNELS ==
    CAMERA_MODEL_OUTPUT_ANCHORS * VALUES_PER_ANCHOR,
    "Logical output channels must match the anchor/class layout");
_Static_assert(
    CAMERA_MODEL_OUTPUT_PHYSICAL_CHANNELS >= CAMERA_MODEL_OUTPUT_LOGICAL_CHANNELS,
    "Physical output channels must contain all logical channels");
_Static_assert(CAMERA_MODEL_OUTPUT_ZERO_POINT == 0,
               "Postprocess requires symmetric zero-point-zero output");
_Static_assert(CANDIDATE_CAPACITY <= 65536U,
               "Candidate order must fit in uint16_t");

static const float anchors[CAMERA_MODEL_OUTPUT_ANCHORS][2] =
    CAMERA_MODEL_OUTPUT_ANCHOR_VALUES;
static const char *const class_names[CAMERA_MODEL_OUTPUT_CLASSES] =
    CAMERA_MODEL_OUTPUT_CLASS_NAMES;

typedef struct {
    /* NMS之前始终使用[0,1]归一化连续坐标。 */
    camera_detection_t box;
    uint16_t order;
} candidate_t;

/* 最坏情况为13×13×5×20=16900个候选，不做预先Top-K截断。
 * 普通PS程序的静态BSS工作区位于DDR，不占大栈、不申请堆，也不是PL BRAM。 */
static candidate_t candidates[CANDIDATE_CAPACITY];

static float sigmoid(float value)
{
    /* exp的输入非正，避免直接计算exp(-value)时溢出。 */
    float e = expf(-fabsf(value));
    return value >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

static float clip_unit(float value)
{
    if (value < 0.0f) return 0.0f;
    if (value > 1.0f) return 1.0f;
    return value;
}

static int compare_candidates(const void *lhs, const void *rhs)
{
    const candidate_t *a = (const candidate_t *)lhs;
    const candidate_t *b = (const candidate_t *)rhs;
    if (a->box.score > b->box.score) return -1;
    if (a->box.score < b->box.score) return 1;
    /* 同分按Python展平顺序anchor、y、x、class决定，保证每次结果一致。 */
    return (a->order > b->order) - (a->order < b->order);
}

static float box_iou(const camera_detection_t *a, const camera_detection_t *b)
{
    float left = fmaxf(a->left, b->left);
    float top = fmaxf(a->top, b->top);
    float right = fminf(a->right, b->right);
    float bottom = fminf(a->bottom, b->bottom);
    float intersection = fmaxf(0.0f, right - left) * fmaxf(0.0f, bottom - top);
    float area_a = (a->right - a->left) * (a->bottom - a->top);
    float area_b = (b->right - b->left) * (b->bottom - b->top);
    float union_area = area_a + area_b - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

int camera_postprocess(const int8_t *prediction, camera_detections_t *results)
{
    const float scale = CAMERA_MODEL_OUTPUT_SCALE;
    const float grid = (float)CAMERA_MODEL_OUTPUT_GRID;
    unsigned count = 0U;
    unsigned kept[CAMERA_POSTPROCESS_MAX_DETECTIONS];
    unsigned kept_count = 0U;
    unsigned a, y, x, c, i, j;

    if (results != NULL) results->count = 0U;
    if (prediction == NULL || results == NULL) return CAMERA_POSTPROCESS_ERROR_ARGUMENT;

    /* PS只对最终输出进行float解码；不会改变PL中的定点卷积或量化规则。
     * 每像素物理跨度128，anchor的逻辑通道基址为25*a；填充通道不参与计算。 */
    for (a = 0U; a < CAMERA_MODEL_OUTPUT_ANCHORS; ++a) {
        for (y = 0U; y < CAMERA_MODEL_OUTPUT_GRID; ++y) {
            for (x = 0U; x < CAMERA_MODEL_OUTPUT_GRID; ++x) {
                const int8_t *q = prediction +
                    (y * CAMERA_MODEL_OUTPUT_GRID + x) * CAMERA_MODEL_OUTPUT_PHYSICAL_CHANNELS +
                    a * VALUES_PER_ANCHOR;
                float objectness = sigmoid((float)q[4] * scale);
                float class_exp[CAMERA_MODEL_OUTPUT_CLASSES];
                float scores[CAMERA_MODEL_OUTPUT_CLASSES];
                float sum = 0.0f;
                int maximum;
                unsigned passing = 0U;
                camera_detection_t box;
                float center_x, center_y, width, height;

                /* 任意类别概率<=1，因此这个提前退出不会丢失过阈值候选。 */
                if (objectness <= CAMERA_POSTPROCESS_SCORE_THRESHOLD) continue;
                maximum = q[5];
                for (c = 1U; c < CAMERA_MODEL_OUTPUT_CLASSES; ++c) {
                    if (q[5U + c] > maximum) maximum = q[5U + c];
                }
                /* 在int域先减最大值再反量化；最大项exp(0)=1，sum至少为1。 */
                for (c = 0U; c < CAMERA_MODEL_OUTPUT_CLASSES; ++c) {
                    class_exp[c] = expf((float)((int)q[5U + c] - maximum) * scale);
                    sum += class_exp[c];
                }
                for (c = 0U; c < CAMERA_MODEL_OUTPUT_CLASSES; ++c) {
                    scores[c] = objectness * (class_exp[c] / sum);
                    if (scores[c] > CAMERA_POSTPROCESS_SCORE_THRESHOLD) ++passing;
                }
                /* 仅在至少一个类别过阈值时计算框；保留所有过阈值类别，不取argmax。 */
                if (passing == 0U) continue;
                center_x = (sigmoid((float)q[0] * scale) + (float)x) / grid;
                center_y = (sigmoid((float)q[1] * scale) + (float)y) / grid;
                width = expf((float)q[2] * scale) * anchors[a][0] / grid;
                height = expf((float)q[3] * scale) * anchors[a][1] / grid;
                box.left = clip_unit(center_x - width * 0.5f);
                box.top = clip_unit(center_y - height * 0.5f);
                box.right = clip_unit(center_x + width * 0.5f);
                box.bottom = clip_unit(center_y + height * 0.5f);
                /* 先裁剪再NMS；浮点舍入后零面积的框明确丢弃。 */
                if (box.right <= box.left || box.bottom <= box.top) continue;
                for (c = 0U; c < CAMERA_MODEL_OUTPUT_CLASSES; ++c) {
                    if (scores[c] <= CAMERA_POSTPROCESS_SCORE_THRESHOLD) continue;
                    box.score = scores[c];
                    box.class_id = c;
                    candidates[count].box = box;
                    candidates[count].order = (uint16_t)(
                        ((a * CAMERA_MODEL_OUTPUT_GRID + y) * CAMERA_MODEL_OUTPUT_GRID + x) *
                        CAMERA_MODEL_OUTPUT_CLASSES + c);
                    ++count;
                }
            }
        }
    }

    qsort(candidates, count, sizeof(candidates[0]), compare_candidates);
    /* 全局按分数降序，仅同类别互相抑制。只需与已经保留的至多16个框比较；
     * 第16个幸存者之后停止，与完整class-aware NMS后取前16个等价。 */
    for (i = 0U; i < count && kept_count < CAMERA_POSTPROCESS_MAX_DETECTIONS; ++i) {
        const camera_detection_t *box = &candidates[i].box;
        for (j = 0U; j < kept_count; ++j) {
            const camera_detection_t *previous = &candidates[kept[j]].box;
            if (box->class_id == previous->class_id &&
                box_iou(box, previous) > CAMERA_POSTPROCESS_IOU_THRESHOLD) break;
        }
        if (j == kept_count) kept[kept_count++] = i;
    }
    /* 所有IoU比较结束后才转成ROI浮点像素，绝不加视频偏移或取整。 */
    for (i = 0U; i < kept_count; ++i) {
        camera_detection_t *box = &results->boxes[i];
        *box = candidates[kept[i]].box;
        box->left *= (float)CAMERA_ROI_SIDE;
        box->top *= (float)CAMERA_ROI_SIDE;
        box->right *= (float)CAMERA_ROI_SIDE;
        box->bottom *= (float)CAMERA_ROI_SIDE;
    }
    results->count = kept_count;
    return CAMERA_POSTPROCESS_OK;
}

const char *camera_class_name(unsigned class_id)
{
    return class_id < CAMERA_MODEL_OUTPUT_CLASSES ? class_names[class_id] : NULL;
}
