#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H
#include <stddef.h>
#include <stdint.h>

/* PS 独立维护的 DDR 契约，不引用 HLS C++ 头文件。
 * 仅数组尾部补齐缓存行，层与字段之间不插入 padding。 */
enum {
    MODEL_CACHE_ALIGN = 64, MODEL_LAYER_COUNT = 9,
    MODEL_WEIGHT_BYTES = 15859328,
    MODEL_WEIGHT_WORDS = MODEL_WEIGHT_BYTES / sizeof(uint64_t),
    MODEL_PARAM_BYTES = 38172,
    MODEL_PARAM_COUNT = MODEL_PARAM_BYTES / sizeof(int32_t),
    MODEL_PARAM_CAPACITY_BYTES =
        ((MODEL_PARAM_BYTES + MODEL_CACHE_ALIGN - 1) / MODEL_CACHE_ALIGN) * MODEL_CACHE_ALIGN,
    MODEL_PARAM_CAPACITY_COUNT = MODEL_PARAM_CAPACITY_BYTES / sizeof(int32_t),
    MODEL_INPUT_BYTES = 519168, MODEL_FEATURE_BANK_BYTES = 692224,
    MODEL_FEATURE_BYTES = 2 * MODEL_FEATURE_BANK_BYTES,
    MODEL_FEATURE_WORDS = MODEL_FEATURE_BYTES / sizeof(uint64_t),
    MODEL_RESULT_OFFSET_BYTES = MODEL_FEATURE_BANK_BYTES,
    MODEL_RESULT_BYTES = 21632
};
/* 成功返回 0；失败打印具体文件和原因，调用者不得宣布 ready。 */
int model_loader_load(void *weights, size_t weight_capacity,
                      void *params, size_t param_capacity);
#endif
