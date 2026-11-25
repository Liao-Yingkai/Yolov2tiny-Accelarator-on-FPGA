#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "camera_cnn.h"
#include "model_loader.h"
#include "xparameters.h"
#include "xcnn_accelerator_ddr.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xpseudo_asm.h"
#include "xtime_l.h"

/* 三个区域独占完整 64B 缓存行，链接到 DDR .bss。
 * feature 是连续 A+B；输入在 A+0，最终结果在 B+0。 */
static uint64_t weights_ddr[MODEL_WEIGHT_WORDS] __attribute__((aligned(64)));
static int32_t params_ddr[MODEL_PARAM_CAPACITY_COUNT] __attribute__((aligned(64)));
static uint64_t feature_ddr[MODEL_FEATURE_WORDS] __attribute__((aligned(64)));
static XCnn_accelerator_ddr accelerator;
static int initialized;
static int terminal_fault;

_Static_assert(sizeof(weights_ddr) == MODEL_WEIGHT_BYTES, "weight DDR size");
_Static_assert(sizeof(params_ddr) == MODEL_PARAM_CAPACITY_BYTES, "parameter DDR capacity");
_Static_assert(sizeof(feature_ddr) == MODEL_FEATURE_BYTES, "feature A+B size");
_Static_assert(MODEL_RESULT_OFFSET_BYTES + MODEL_RESULT_BYTES <= MODEL_FEATURE_BYTES,
               "result fits bank B");

static int valid_ddr(const void *buffer, size_t size)
{
    uint64_t address = (uint64_t)(uintptr_t)buffer;
    uint64_t high = (uint64_t)XPAR_PS7_DDR_0_S_AXI_HIGHADDR;
    /* 先检查起点，再用减法检查末尾，避免 address+size 整数溢出。 */
    return size != 0 && address % MODEL_CACHE_ALIGN == 0 &&
           address >= (uint64_t)XPAR_PS7_DDR_0_S_AXI_BASEADDR &&
           address <= high && (uint64_t)(size - 1U) <= high - address;
}

camera_cnn_status_t camera_cnn_init(void)
{
    if (terminal_fault) return CAMERA_CNN_FAULT;
    if (initialized) return CAMERA_CNN_OK;
    if (!valid_ddr(weights_ddr, sizeof(weights_ddr)) ||
        !valid_ddr(params_ddr, sizeof(params_ddr)) ||
        !valid_ddr(feature_ddr, sizeof(feature_ddr))) return CAMERA_CNN_DDR;
    if (XCnn_accelerator_ddr_Initialize(&accelerator,
        XPAR_CNN_ACCELERATOR_DDR_0_DEVICE_ID) != XST_SUCCESS) return CAMERA_CNN_DRIVER;
    if (accelerator.Control_BaseAddress != XPAR_CNN_ACCELERATOR_DDR_0_S_AXI_CONTROL_BASEADDR)
        return CAMERA_CNN_BASE;
    XCnn_accelerator_ddr_DisableAutoRestart(&accelerator);
    XCnn_accelerator_ddr_InterruptGlobalDisable(&accelerator);
    XCnn_accelerator_ddr_InterruptDisable(&accelerator, 0xffffffffU);
    if (!XCnn_accelerator_ddr_IsIdle(&accelerator)) {
        terminal_fault = 1;
        return CAMERA_CNN_BUSY;
    }

    xil_printf("DDR weights=0x%08x params=0x%08x feature A=0x%08x B=0x%08x\r\n",
               (unsigned)(uintptr_t)weights_ddr, (unsigned)(uintptr_t)params_ddr,
               (unsigned)(uintptr_t)feature_ddr,
               (unsigned)((uintptr_t)feature_ddr + MODEL_FEATURE_BANK_BYTES));
    if (model_loader_load(weights_ddr, sizeof(weights_ddr),
                          params_ddr, sizeof(params_ddr))) return CAMERA_CNN_MODEL;
    /* params 尾部容量不属于模型；feature 尚无摄像头输入，先整体清零，
     * 让加载验收阶段的 A+B 具有确定初值，但绝不启动加速器。 */
    memset((uint8_t *)params_ddr + MODEL_PARAM_BYTES, 0,
           sizeof(params_ddr) - MODEL_PARAM_BYTES);
    memset(feature_ddr, 0, sizeof(feature_ddr));
    /* PS 写、PL 读：最终三个区域各 flush 一次，dsb 等待写回完成。
     * FlushRange 没有返回值，不虚构成功/失败检查。 */
    Xil_DCacheFlushRange((INTPTR)weights_ddr, sizeof(weights_ddr));
    Xil_DCacheFlushRange((INTPTR)params_ddr, sizeof(params_ddr));
    Xil_DCacheFlushRange((INTPTR)feature_ddr, sizeof(feature_ddr));
    dsb();
    XCnn_accelerator_ddr_Set_feature(&accelerator, (u64)(uintptr_t)feature_ddr);
    XCnn_accelerator_ddr_Set_weights(&accelerator, (u64)(uintptr_t)weights_ddr);
    XCnn_accelerator_ddr_Set_params(&accelerator, (u64)(uintptr_t)params_ddr);
    xil_printf("YOLO model ready; accelerator not started\r\n");
    initialized = 1;
    return CAMERA_CNN_OK;
}

int8_t *camera_cnn_input_buffer(void)
{
    if (!initialized || terminal_fault) return NULL;
    /* 必须先确认 PL 不再访问 A，之后调用者才可在 A 中直接预处理。 */
    if (!XCnn_accelerator_ddr_IsIdle(&accelerator)) {
        terminal_fault = 1;
        return NULL;
    }
    return (int8_t *)feature_ddr;
}

camera_cnn_status_t camera_cnn_run(const int8_t **output)
{
    XTime started, now;
    if (output == NULL) return CAMERA_CNN_ARGUMENT;
    *output = NULL;
    if (terminal_fault) return CAMERA_CNN_FAULT;
    if (!initialized) return CAMERA_CNN_NOT_INITIALIZED;
    if (!XCnn_accelerator_ddr_IsIdle(&accelerator)) {
        terminal_fault = 1;
        return CAMERA_CNN_BUSY;
    }
    /* ap_done 为读清除位：启动前清掉旧完成状态。启动后只用 IsDone
     * 读取控制寄存器，避免 IsIdle 等其他读取把本次完成位提前清掉。 */
    (void)XCnn_accelerator_ddr_IsDone(&accelerator);
    Xil_DCacheFlushRange((INTPTR)feature_ddr, MODEL_INPUT_BYTES);
    dsb();
    XTime_GetTime(&started);
    XCnn_accelerator_ddr_Start(&accelerator);
    while (!XCnn_accelerator_ddr_IsDone(&accelerator)) {
        XTime_GetTime(&now);
        if (now - started >= (XTime)COUNTS_PER_SECOND * CAMERA_CNN_TIMEOUT_SECONDS) {
            /* 超时后不知道 PL 是否仍写 DDR，只能停止应用并要求板级复位。 */
            terminal_fault = 1;
            return CAMERA_CNN_TIMEOUT;
        }
    }
    dsb();
    Xil_DCacheInvalidateRange((INTPTR)((uint8_t *)feature_ddr +
                             MODEL_RESULT_OFFSET_BYTES), MODEL_RESULT_BYTES);
    dsb();
    *output = (const int8_t *)((uint8_t *)feature_ddr + MODEL_RESULT_OFFSET_BYTES);
    return CAMERA_CNN_OK;
}

const char *camera_cnn_reason(camera_cnn_status_t status)
{
    switch (status) {
    case CAMERA_CNN_OK: return "ok";
    case CAMERA_CNN_DDR: return "DDR range/alignment";
    case CAMERA_CNN_DRIVER: return "driver init";
    case CAMERA_CNN_BASE: return "control base";
    case CAMERA_CNN_BUSY: return "busy; board reset required";
    case CAMERA_CNN_MODEL: return "SD model load; check previous log";
    case CAMERA_CNN_ARGUMENT: return "null output argument";
    case CAMERA_CNN_NOT_INITIALIZED: return "not initialized";
    case CAMERA_CNN_TIMEOUT: return "inference timeout; board reset required";
    case CAMERA_CNN_FAULT: return "terminal fault; board reset required";
    default: return "unknown status";
    }
}
