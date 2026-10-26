#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "camera_cnn.h"
#include "camera_model_data.h"
#include "softmax_params.h"
#include "xparameters.h"
#include "xcnn_accelerator_ddr.h"
#include "xil_cache.h"
#include "xpseudo_asm.h"
#include "xtime_l.h"

enum { CACHE_ALIGN = 64, WORD_BYTES = 8, KERNEL_PIXELS = 3 * 3,
    CONV1_COUT = 8, CONV2_COUT = 16, FC_INPUTS = 7 * 7 * CONV2_COUT,
    PARAM_FIELDS = 3, CONV1_WORDS = KERNEL_PIXELS * CONV1_COUT / WORD_BYTES,
    CONV2_WORDS = KERNEL_PIXELS * CONV1_COUT * CONV2_COUT / WORD_BYTES,
    FC_WORDS = FC_INPUTS * CAMERA_CNN_CLASSES / WORD_BYTES,
    WEIGHT_WORDS = CONV1_WORDS + CONV2_WORDS + FC_WORDS,
    CONV2_PARAM_BASE = PARAM_FIELDS * CONV1_COUT,
    FC_PARAM_BASE = CONV2_PARAM_BASE + PARAM_FIELDS * CONV2_COUT,
    PARAM_COUNT = PARAM_FIELDS * (CONV1_COUT + CONV2_COUT + CAMERA_CNN_CLASSES) };
#define ROUND_CACHE(n) (((n) + CACHE_ALIGN - 1) / CACHE_ALIGN * CACHE_ALIGN)
_Static_assert(sizeof(camera_cnn_result_t) == CAMERA_CNN_RESULT_BYTES, "result size ABI");
_Static_assert(offsetof(camera_cnn_result_t, class_index) == CAMERA_CNN_CLASS_BYTE_OFFSET, "class offset ABI");
_Static_assert(MODEL_SOFTMAX_SCALE_FRAC_BITS == CAMERA_CNN_SOFTMAX_SCALE_FRAC_BITS, "scale ABI");
_Static_assert(CAMERA_CNN_INPUT_BYTES == 98 * WORD_BYTES, "image size ABI");
_Static_assert(WEIGHT_WORDS == 1133 && PARAM_COUNT == 102, "frozen DDR ABI");
_Static_assert(CAMERA_WEIGHT_LAYOUT_VERSION == 2, "blocked weights v2 required");
_Static_assert(sizeof(conv1_weight) == CONV1_WORDS * WORD_BYTES &&
    sizeof(conv2_weight) == CONV2_WORDS * WORD_BYTES &&
    sizeof(fc_weight) == FC_WORDS * WORD_BYTES, "weight size ABI");
/* 单一驱动实例。所有实际DMA区域独占完整64B缓存行，链接到DDR .bss。
 * 尾部容量补齐仅用于缓存维护，模型的有效权重之间没有补零或转置。 */
static XCnn_accelerator_ddr accelerator;
static int initialized;
static uint64_t image_ddr[ROUND_CACHE(CAMERA_CNN_INPUT_BYTES) / WORD_BYTES] __attribute__((aligned(64)));
static uint64_t weights_ddr[ROUND_CACHE(WEIGHT_WORDS * WORD_BYTES) / WORD_BYTES] __attribute__((aligned(64)));
static int32_t params_ddr[ROUND_CACHE(PARAM_COUNT * sizeof(int32_t)) / sizeof(int32_t)] __attribute__((aligned(64)));
static uint8_t result_ddr[ROUND_CACHE(sizeof(camera_cnn_result_t))] __attribute__((aligned(64)));

static int valid_ddr(const void *buffer, size_t size)
{
    uint64_t address = (uint64_t)(uintptr_t)buffer;
    return address % CACHE_ALIGN == 0 &&
        address >= XPAR_PS7_DDR_0_S_AXI_BASEADDR &&
        address + size - 1 <= XPAR_PS7_DDR_0_S_AXI_HIGHADDR;
}

static void pack_params(unsigned base, unsigned count, const int32_t *bias,
                        const int32_t *multiplier, const int32_t *shift)
{
    /* 每层是[完整bias][完整multiplier][完整shift]，不是逐通道交错。 */
    memcpy(params_ddr + base, bias, count * sizeof(int32_t));
    memcpy(params_ddr + base + count, multiplier, count * sizeof(int32_t));
    memcpy(params_ddr + base + 2 * count, shift, count * sizeof(int32_t));
}

camera_cnn_status_t camera_cnn_init(void)
{
    if (initialized) return CAMERA_CNN_OK;
    if (!valid_ddr(image_ddr, sizeof(image_ddr)) ||
        !valid_ddr(weights_ddr, sizeof(weights_ddr)) ||
        !valid_ddr(params_ddr, sizeof(params_ddr)) ||
        !valid_ddr(result_ddr, sizeof(result_ddr))) return CAMERA_CNN_DDR;
    if (XCnn_accelerator_ddr_Initialize(&accelerator,
        XPAR_CNN_ACCELERATOR_DDR_0_DEVICE_ID) != XST_SUCCESS) return CAMERA_CNN_DRIVER;
    if (accelerator.Control_BaseAddress != XPAR_CNN_ACCELERATOR_DDR_0_S_AXI_CONTROL_BASEADDR)
        return CAMERA_CNN_BASE;
    XCnn_accelerator_ddr_DisableAutoRestart(&accelerator);
    XCnn_accelerator_ddr_InterruptGlobalDisable(&accelerator);
    XCnn_accelerator_ddr_InterruptDisable(&accelerator, 0xffffffffU);
    if (!XCnn_accelerator_ddr_IsIdle(&accelerator)) return CAMERA_CNN_BUSY;
    /* 模型只加载一次：72+1152+7840=9064字节，1133个u64 word。 */
    memcpy(weights_ddr, conv1_weight, sizeof(conv1_weight));
    memcpy(weights_ddr + CONV1_WORDS, conv2_weight, sizeof(conv2_weight));
    memcpy(weights_ddr + CONV1_WORDS + CONV2_WORDS, fc_weight, sizeof(fc_weight));
    pack_params(0, CONV1_COUT, conv1_bias, conv1_multiplier, conv1_shift);
    pack_params(CONV2_PARAM_BASE, CONV2_COUT, conv2_bias, conv2_multiplier, conv2_shift);
    pack_params(FC_PARAM_BASE, CAMERA_CNN_CLASSES, fc_bias, fc_multiplier, fc_shift);
    /* PS写、PL读：flush让DDR可见，dsb等待维护完成。 */
    Xil_DCacheFlushRange((INTPTR)weights_ddr, sizeof(weights_ddr));
    Xil_DCacheFlushRange((INTPTR)params_ddr, sizeof(params_ddr));
    dsb();
    XCnn_accelerator_ddr_Set_image_r(&accelerator, (u64)(uintptr_t)image_ddr);
    XCnn_accelerator_ddr_Set_weights(&accelerator, (u64)(uintptr_t)weights_ddr);
    XCnn_accelerator_ddr_Set_params(&accelerator, (u64)(uintptr_t)params_ddr);
    XCnn_accelerator_ddr_Set_result(&accelerator, (u64)(uintptr_t)result_ddr);
    XCnn_accelerator_ddr_Set_softmax_scale_q20(&accelerator, MODEL_SOFTMAX_SCALE_Q20);
    initialized = 1; /* 初始化不启动CNN，无中断连接，逐次轮询。 */
    return CAMERA_CNN_OK;
}

camera_cnn_status_t camera_cnn_predict(const int8_t input[CAMERA_CNN_INPUT_BYTES],
                                     camera_cnn_result_t *result)
{
    XTime start, now;
    unsigned i, sum = 0;
    if (!input || !result) return CAMERA_CNN_ARGUMENT;
    if (!initialized) return CAMERA_CNN_NOT_INITIALIZED;
    if (!XCnn_accelerator_ddr_IsIdle(&accelerator)) return CAMERA_CNN_BUSY;
    /* ap_done为COR（读后清零）；在Start前清上轮完成位。Start之后
     * 只用IsDone轮询，不再额外读取CTRL以免偷清本轮完成事件。 */
    (void)XCnn_accelerator_ddr_IsDone(&accelerator);
    /* 单通道signed byte保持二补码。Zynq为little-endian，memcpy后
     * 每个u64恰有连续8像素，早像素处于低byte，无需逐像素移位。 */
    memcpy(image_ddr, input, CAMERA_CNN_INPUT_BYTES);
    memset(result_ddr, 0xa5, sizeof(result_ddr));
    Xil_DCacheFlushRange((INTPTR)image_ddr, sizeof(image_ddr));
    /* 先写回整条结果缓存行，避免CPU旧脏行以后覆盖PL新结果。 */
    Xil_DCacheFlushRange((INTPTR)result_ddr, sizeof(result_ddr));
    dsb();
    XTime_GetTime(&start);
    XCnn_accelerator_ddr_Start(&accelerator);
    while (!XCnn_accelerator_ddr_IsDone(&accelerator)) {
        XTime_GetTime(&now);
        if ((uint64_t)(now - start) >= (uint64_t)COUNTS_PER_SECOND * 5U) {
            /* 超时不归还buffer ownership：PL可能仍访问DDR。调用方必须
             * 结束程序，板级复位后重启；禁止复用缓冲或自动reset后重试。 */
            return CAMERA_CNN_TIMEOUT;
        }
    }
    /* PL写、PS读：完成之后invalidate丢弃CPU旧值，再取ABI结果。 */
    dsb();
    Xil_DCacheInvalidateRange((INTPTR)result_ddr, sizeof(result_ddr));
    dsb();
    memcpy(result, result_ddr, sizeof(*result));
    if (result->class_index >= CAMERA_CNN_CLASSES) return CAMERA_CNN_RESULT;
    for (i = 0; i < CAMERA_CNN_CLASSES; ++i) {
        if (result->probability[i] > CAMERA_CNN_PROBABILITY_SCALE ||
            result->probability[i] > result->probability[result->class_index])
            return CAMERA_CNN_RESULT;
        sum += result->probability[i];
    }
    if (sum < CAMERA_CNN_PROBABILITY_SCALE - 5 || sum > CAMERA_CNN_PROBABILITY_SCALE + 5)
        return CAMERA_CNN_RESULT;
    return CAMERA_CNN_OK;
}

const char *camera_cnn_reason(camera_cnn_status_t status)
{
    switch (status) {
    case CAMERA_CNN_OK: return "ok";
    case CAMERA_CNN_ARGUMENT: return "argument";
    case CAMERA_CNN_DDR: return "DDR range/alignment";
    case CAMERA_CNN_DRIVER: return "driver init";
    case CAMERA_CNN_BASE: return "control base";
    case CAMERA_CNN_BUSY: return "busy; board reset required";
    case CAMERA_CNN_TIMEOUT: return "timeout; board reset required";
    case CAMERA_CNN_RESULT: return "result ABI";
    default: return "not initialized";
    }
}
