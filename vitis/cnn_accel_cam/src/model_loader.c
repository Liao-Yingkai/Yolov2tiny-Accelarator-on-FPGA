#include "model_loader.h"
#include <stdio.h>
#include <string.h>
#include "ff.h"
#include "xil_printf.h"

typedef struct {
    const char *name;
    unsigned cin, cout, kernel;
} model_layer_t;
static const model_layer_t layers[MODEL_LAYER_COUNT] = {
    {"conv1",3,16,3}, {"conv2",16,32,3}, {"conv3",32,64,3},
    {"conv4",64,128,3}, {"conv5",128,256,3}, {"conv6",256,512,3},
    {"conv7",512,1024,3}, {"conv8",1024,1024,3}, {"prediction",1024,125,1}
};
static FATFS sd_fs;
static int mount_attempted, mounted;
static uint8_t staging[65536] __attribute__((aligned(64)));

static unsigned weight_bytes(const model_layer_t *layer)
{
    /* 每个 Cout tile 为 4 lane，每个 Cin group 为 8 lane。
     * conv1 的通道 padding 已包含在 BIN 中，这里不能再次补零。 */
    return ((layer->cout + 3U) / 4U) * layer->kernel * layer->kernel *
           ((layer->cin + 7U) / 8U) * 4U * 8U;
}

static int read_exact(const char *path, uint8_t *destination, unsigned expected)
{
    FIL file;
    FRESULT result;
    unsigned offset = 0;
    int failed = 0;
    result = f_open(&file, path, FA_READ);
    if (result != FR_OK) {
        xil_printf("Model %s: open failed FRESULT=%u\r\n", path, (unsigned)result);
        return -1;
    }
    if (f_size(&file) != expected) {
        xil_printf("Model %s: size mismatch expected=%u actual=%u\r\n",
                   path, expected, (unsigned)f_size(&file));
        failed = 1;
    }
    while (!failed && offset < expected) {
        UINT actual = 0;
        UINT requested = expected - offset > sizeof(staging) ?
                         (UINT)sizeof(staging) : (UINT)(expected - offset);
        result = f_read(&file, staging, requested, &actual);
        if (result != FR_OK || actual != requested) {
            xil_printf("Model %s: read failed FRESULT=%u expected=%u actual=%u offset=%u\r\n",
                       path, (unsigned)result, (unsigned)requested, (unsigned)actual, offset);
            failed = 1;
        } else {
            /* FatFs 可能由 CPU 复制非整扇区数据，因此 f_read 后不能统一
             * invalidate。只复制原始字节，不转置、不换端序、不重排。 */
            memcpy(destination + offset, staging, actual);
            offset += actual;
        }
    }
    result = f_close(&file);
    if (result != FR_OK) {
        xil_printf("Model %s: close failed FRESULT=%u\r\n", path, (unsigned)result);
        failed = 1;
    }
    return failed ? -1 : 0;
}

int model_loader_load(void *weights, size_t weight_capacity,
                      void *params, size_t param_capacity)
{
    unsigned i, weight_offset = 0, param_offset = 0;
    char path[64];
    if (!weights || !params || weight_capacity < MODEL_WEIGHT_BYTES ||
        param_capacity < MODEL_PARAM_BYTES) {
        xil_printf("Model: invalid destination capacity\r\n");
        return -1;
    }
    /* 启动期间只尝试挂载一次；失败后需要排障并重新启动板卡。 */
    if (!mount_attempted) {
        FRESULT result;
        mount_attempted = 1;
        xil_printf("SD mount 0: start\r\n");
        result = f_mount(&sd_fs, "0:", 1);
        if (result != FR_OK) {
            xil_printf("SD mount 0: failed FRESULT=%u\r\n", (unsigned)result);
            return -1;
        }
        mounted = 1;
        xil_printf("SD mount 0: success\r\n");
    }
    if (!mounted) return -1;
    for (i = 0; i < MODEL_LAYER_COUNT; ++i) {
        const model_layer_t *layer = &layers[i];
        unsigned bytes = weight_bytes(layer);
        unsigned bias_bytes = layer->cout * 4U;
        unsigned param_bytes = 3U * bias_bytes;
        if (bytes > MODEL_WEIGHT_BYTES - weight_offset ||
            param_bytes > MODEL_PARAM_BYTES - param_offset) {
            xil_printf("Model %s: schema exceeds DDR contract\r\n", layer->name);
            return -1;
        }
        snprintf(path, sizeof(path), "0:/yolo/%s_weight.bin", layer->name);
        if (read_exact(path, (uint8_t *)weights + weight_offset, bytes)) return -1;
        snprintf(path, sizeof(path), "0:/yolo/%s_bias.bin", layer->name);
        if (read_exact(path, (uint8_t *)params + param_offset, bias_bytes)) return -1;
        snprintf(path, sizeof(path), "0:/yolo/%s_quant.bin", layer->name);
        /* quant.bin 已是 [全部 M][全部 S]，紧接当前层的完整 bias。 */
        if (read_exact(path, (uint8_t *)params + param_offset + bias_bytes,
                       2U * bias_bytes)) return -1;
        weight_offset += bytes;
        param_offset += param_bytes;
        xil_printf("Model %s: weights=%u params=%u loaded\r\n",
                   layer->name, bytes, param_bytes);
    }
    if (weight_offset != MODEL_WEIGHT_BYTES || param_offset != MODEL_PARAM_BYTES) {
        xil_printf("Model: total mismatch weights=%u params=%u\r\n", weight_offset, param_offset);
        return -1;
    }
    xil_printf("Model: all 27 files loaded weights=%u params=%u\r\n", weight_offset, param_offset);
    return 0;
}
