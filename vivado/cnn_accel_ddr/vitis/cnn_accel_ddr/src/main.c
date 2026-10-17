#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "xparameters.h"
#include "xcnn_accelerator_ddr.h"
#include "xil_cache.h"
#include "xil_printf.h"
#include "xpseudo_asm.h"
#include "xtime_l.h"
#include "mnist_test_data.h"

/* 这里是 HLS config.h / axi/ddr_layout.h 的冻结 C 版布局。
 * HLS 头使用 C++ namespace，不能直接包含进本 C 程序。
 * mnist_test_data.h 来自导出脚本：12 个模型数组和三图 input/golden。
 * 权重头 v2 已按硬件消费顺序导出，PS 只连续复制，不重排或补齐通道。 */
enum {
    IMAGE_H = 28, IMAGE_W = 28, KERNEL_TOKENS = 3 * 3,
    CONV1_COUT = 8, CONV2_COUT = 16, FC_INPUTS = 7 * 7 * CONV2_COUT,
    FC_OUTPUTS = 10, CIN_LANES = 8, CONV_OUT_LANES = 4, FC_OUT_LANES = 5,
    WORD_BYTES = 8, CACHE_ALIGN = 64, PARAM_FIELDS = 3,
    IMAGE_BYTES = IMAGE_H * IMAGE_W, IMAGE_WORDS = IMAGE_BYTES / WORD_BYTES,
    CONV1_WORDS = KERNEL_TOKENS * CONV1_COUT / WORD_BYTES,
    CONV2_WORDS = KERNEL_TOKENS * CONV2_COUT,
    FC_WORDS = (FC_OUTPUTS / FC_OUT_LANES) * (FC_INPUTS / CIN_LANES) * FC_OUT_LANES,
    WEIGHT_WORDS = CONV1_WORDS + CONV2_WORDS + FC_WORDS,
    CONV2_PARAM_BASE = PARAM_FIELDS * CONV1_COUT,
    FC_PARAM_BASE = CONV2_PARAM_BASE + PARAM_FIELDS * CONV2_COUT,
    PARAM_WORDS = FC_PARAM_BASE + PARAM_FIELDS * FC_OUTPUTS
};
#define ROUND_CACHE(n) (((n) + CACHE_ALIGN - 1) / CACHE_ALIGN * CACHE_ALIGN)
/* PL 只写前 16 字节：10 个有符号 logit、2 个零保留字节、4 字节类别。
 * memcpy 解码到结构体，避免把 byte 数组强转成结构体指针的别名问题。 */
typedef struct {
    int8_t logits[FC_OUTPUTS];
    uint8_t reserved[2];
    uint32_t class_index;
} ResultRecord;
_Static_assert(sizeof(ResultRecord) == 16, "16-byte result ABI required");
_Static_assert(offsetof(ResultRecord, class_index) == 12, "class offset ABI");
_Static_assert(MNIST_WEIGHT_LAYOUT_VERSION == 2, "requires blocked weight headers v2");
_Static_assert(sizeof(conv1_weight) == CONV1_WORDS * WORD_BYTES &&
               sizeof(conv2_weight) == CONV2_WORDS * WORD_BYTES &&
               sizeof(fc_weight) == FC_WORDS * WORD_BYTES, "weight byte counts mismatch");
_Static_assert(IMAGE_WORDS == 98 && WEIGHT_WORDS == 1133 && PARAM_WORDS == 102,
               "frozen DDR ABI mismatch");
_Static_assert((int)MNIST_INPUT_COUNT == (int)IMAGE_BYTES &&
               (int)MNIST_LOGIT_COUNT == (int)FC_OUTPUTS,
               "test shape mismatch");
_Static_assert(MNIST_TEST_COUNT == 3, "three test images required");
/* 各模型数组的元素数量/整数范围由 export_test_data.py 导出时逐个检查。
 * 四个静态缓冲区由链接脚本放入 DDR .bss，不占用 8 KiB 栈，也不需 malloc。
 * PS 将实际地址写入 Set_* 寄存器，PL 从寄存器取地址；两端无需硬编码
 * 相同的缓冲区地址。起点和容量都按 64 字节对齐，尾部补齐不与别的数据
 * 共用缓存行，因此 flush/invalidate 不会误伤相邻对象。 */
static uint64_t image_ddr[ROUND_CACHE(IMAGE_BYTES) / WORD_BYTES] __attribute__((aligned(CACHE_ALIGN)));
static uint64_t weights_ddr[ROUND_CACHE(WEIGHT_WORDS * WORD_BYTES) / WORD_BYTES] __attribute__((aligned(CACHE_ALIGN)));
static int32_t params_ddr[ROUND_CACHE(PARAM_WORDS * sizeof(int32_t)) / sizeof(int32_t)] __attribute__((aligned(CACHE_ALIGN)));
static uint8_t result_ddr[ROUND_CACHE(sizeof(ResultRecord))] __attribute__((aligned(CACHE_ALIGN)));

static int valid_ddr(const void *buffer, size_t size)
{
    const uint64_t address = (uint64_t)(uintptr_t)buffer;
    return (address % CACHE_ALIGN == 0) &&
        address >= XPAR_PS7_DDR_0_S_AXI_BASEADDR &&
        address + size - 1 <= XPAR_PS7_DDR_0_S_AXI_HIGHADDR;
}

static void pack_params(unsigned base, unsigned count, const int32_t *bias,
                        const int32_t *multiplier, const int32_t *shift)
{
    /* 每层 DDR 参数是 [全部 bias][全部 multiplier][全部 shift] 三整块，
     * 不是每输出通道交错放 bias/mul/shift。base 的单位是 int32_t 元素。 */
    memcpy(params_ddr + base, bias, count * sizeof(int32_t));
    memcpy(params_ddr + base + count, multiplier, count * sizeof(int32_t));
    memcpy(params_ddr + base + 2 * count, shift, count * sizeof(int32_t));
}

static void prepare_model(void)
{
    /* 三份 v2 权重头原字节紧密拼接：Conv1 72B、Conv2 1152B、FC 7840B。
     * DDR word 偏移为 0/9/153，总1133 word；不再用 PS 做权重布局转换。
     * Conv1 的逻辑 Cin=1 由 Systolic 保证，DDR 不保存七个补零通道。 */
    memcpy(weights_ddr, conv1_weight, sizeof(conv1_weight));
    memcpy(weights_ddr + CONV1_WORDS, conv2_weight, sizeof(conv2_weight));
    memcpy(weights_ddr + CONV1_WORDS + CONV2_WORDS, fc_weight, sizeof(fc_weight));
    pack_params(0, CONV1_COUT, conv1_bias, conv1_multiplier, conv1_shift);
    pack_params(CONV2_PARAM_BASE, CONV2_COUT, conv2_bias, conv2_multiplier, conv2_shift);
    pack_params(FC_PARAM_BASE, FC_OUTPUTS, fc_bias, fc_multiplier, fc_shift);
    /* PS 缓存和当前 PL DDR AXI 访问不是硬件一致的：CPU 写入可能只在缓存，
     * flush 才让 PL 能读到 DDR 新值。模型三图共用，所以只做一次。
     * dsb 等待这些内存维护完成，再允许后面的硬件启动访问。 */
    Xil_DCacheFlushRange((INTPTR)weights_ddr, sizeof(weights_ddr));
    Xil_DCacheFlushRange((INTPTR)params_ddr, sizeof(params_ddr));
    dsb();
}

static unsigned first_argmax(const int8_t *logits)
{
    /* 只在严格更大时更新，因此并列最大值取第一个，与 PL 约定一致。 */
    unsigned i, best = 0;
    for (i = 1; i < FC_OUTPUTS; ++i)
        if (logits[i] > logits[best]) best = i;
    return best;
}

static unsigned check_result(unsigned test, const ResultRecord *result)
{
    unsigned i, errors = 0;
    const int8_t *expected = mnist_expected_logits[test];
    unsigned expected_class = first_argmax(expected);
    /* 不只看预测类别：逐 logit 比较，且检查 PL 类别同时等于 golden
     * 和实际 logits 的首次最大值。保留字节和尾部哨兵用于检查输出 ABI。 */
    for (i = 0; i < FC_OUTPUTS; ++i)
        if (result->logits[i] != expected[i]) {
            xil_printf("test %u logit[%u]: expected %d actual %d\r\n", test, i, (int)expected[i], (int)result->logits[i]);
            ++errors;
        }
    if (result->class_index != expected_class || result->class_index != first_argmax(result->logits)) {
        xil_printf("test %u class: expected %u actual %u (actual logits argmax %u)\r\n",
                   test, expected_class, (unsigned)result->class_index, first_argmax(result->logits));
        ++errors;
    }
    for (i = 0; i < sizeof(result->reserved); ++i)
        if (result->reserved[i] != 0) {
            xil_printf("test %u reserved[%u]: expected 0 actual %u\r\n", test, i, (unsigned)result->reserved[i]);
            ++errors;
        }
    for (i = sizeof(*result); i < sizeof(result_ddr); ++i)
        if (result_ddr[i] != 0xa5) {
            xil_printf("test %u result byte[%u]: expected 165 actual %u (ABI overwrite)\r\n", test, i, (unsigned)result_ddr[i]);
            ++errors;
        }
    if (errors) {
        xil_printf("actual logits:");
        for (i = 0; i < FC_OUTPUTS; ++i) xil_printf(" %d", (int)result->logits[i]);
        xil_printf("\r\n");
    }
    return errors;
}

/* 完成一张图的“准备输入 → 启动等待 → 取回结果”。
 * 返回 0 表示完成；非零表示硬件状态异常，调用方必须停止而不能继续下一图。 */
static int run_inference(XCnn_accelerator_ddr *accelerator, unsigned test,
                         ResultRecord *result, u32 *elapsed_us)
{
    unsigned word, c;
    XTime start, now;
    if (!XCnn_accelerator_ddr_IsIdle(accelerator)) {
        xil_printf("FAIL: test %u accelerator not idle, CTRL=0x%08x; stopped\r\n", test,
                   XCnn_accelerator_ddr_ReadReg(accelerator->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_AP_CTRL));
        return 1;
    }
    /* ap_done 读后清零，先读一次，避免把上一次完成状态当成本次结果。 */
    (void)XCnn_accelerator_ddr_IsDone(accelerator);
    memset(image_ddr, 0, sizeof(image_ddr));
    /* 图像每 word 放连续 8 个像素，较早像素在低 byte；这里同样先转
     * uint8_t 再转 uint64_t，保留负像素的二补码而不发生符号扩展。
     * 图像仅存逻辑单通道；loader 的其余通道为确定 don't-care，
     * Im2Row 按既有输入契约清除 Conv1 无效通道。 */
    for (word = 0; word < IMAGE_WORDS; ++word)
        for (c = 0; c < WORD_BYTES; ++c)
            image_ddr[word] |= (uint64_t)(uint8_t)mnist_inputs[test][word * WORD_BYTES + c] << (c * 8);
    /* PL 应只覆盖前 16 字节。先把完整 64 字节写为哨兵并 flush，
     * 完成后就能发现旧 24 字节 ABI 等越界写入，而非读到旧缓存。 */
    memset(result_ddr, 0xa5, sizeof(result_ddr));
    Xil_DCacheFlushRange((INTPTR)image_ddr, sizeof(image_ddr));
    Xil_DCacheFlushRange((INTPTR)result_ddr, sizeof(result_ddr));
    dsb();
    /* PS 全局计时器测启动至轮询完成，单位换成微秒；不是 PL 周期数。 */
    XTime_GetTime(&start);
    XCnn_accelerator_ddr_Start(accelerator);
    for (;;) {
        if (XCnn_accelerator_ddr_IsDone(accelerator)) break;
        XTime_GetTime(&now);
        if ((uint64_t)(now - start) >= (uint64_t)COUNTS_PER_SECOND * 5U) {
            xil_printf("FAIL: test %u timeout (5 s), CTRL=0x%08x; stopped\r\n", test,
                       XCnn_accelerator_ddr_ReadReg(accelerator->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_AP_CTRL));
            /* 超时不代表 PL 已停止。它可能仍在访问 DDR，不能清空/复用
             * 这些缓冲区跑下一张图，因此立即向 main 报错结束。 */
            return 1;
        }
    }
    XTime_GetTime(&now);
    *elapsed_us = (u32)(((uint64_t)(now - start) * 1000000U) / COUNTS_PER_SECOND);
    /* PL 已把结果写入 DDR，但 CPU 可能仍缓存着之前的 0xa5。
     * invalidate 丢弃完整结果区域的旧缓存，之后 CPU 才从 DDR 取新值。
     * dsb 保证完成状态之后的内存维护和读取按要求完成。 */
    dsb();
    Xil_DCacheInvalidateRange((INTPTR)result_ddr, sizeof(result_ddr));
    dsb();
    memcpy(result, result_ddr, sizeof(*result));
    return 0;
}

int main(void)
{
    XCnn_accelerator_ddr accelerator;
    unsigned test, total_errors = 0;
    /* 1. 初始化：DDR/FCLK/UART 已由调试 ps7_init 或启动 FSBL 初始化。
     * BSP stdout 使用 UART0，这里不重配 MIO，也不关闭数据缓存。 */
    Xil_DCacheEnable();
    xil_printf("MNIST DDR test: 3 images, polling, CPU timer (not PL cycles)\r\n");
    if (!valid_ddr(image_ddr, sizeof(image_ddr)) ||
        !valid_ddr(weights_ddr, sizeof(weights_ddr)) ||
        !valid_ddr(params_ddr, sizeof(params_ddr)) ||
        !valid_ddr(result_ddr, sizeof(result_ddr))) {
        xil_printf("FAIL: buffers must be aligned and entirely in BSP DDR range\r\n");
        return 1;
    }
    if (XCnn_accelerator_ddr_Initialize(&accelerator, XPAR_CNN_ACCELERATOR_DDR_0_DEVICE_ID) != XST_SUCCESS) {
        xil_printf("FAIL: accelerator initialization\r\n");
        return 1;
    }
    if (accelerator.Control_BaseAddress != XPAR_CNN_ACCELERATOR_DDR_0_S_AXI_CONTROL_BASEADDR) {
        xil_printf("FAIL: unexpected accelerator control base\r\n");
        return 1;
    }
    XCnn_accelerator_ddr_DisableAutoRestart(&accelerator);
    /* 本程序由 PS 轮询完成位，不连接中断，也不允许自动连续启动。 */
    XCnn_accelerator_ddr_InterruptGlobalDisable(&accelerator);
    XCnn_accelerator_ddr_InterruptDisable(&accelerator, 0xffffffffU);
    /* 2. 准备一次模型，并把四个静态缓冲区的实际 DDR 地址交给 PL。
     * 接下来三张图都复用这些地址，只有输入和结果内容逐次更新。 */
    prepare_model();
    XCnn_accelerator_ddr_Set_image_r(&accelerator, (u64)(uintptr_t)image_ddr);
    XCnn_accelerator_ddr_Set_weights(&accelerator, (u64)(uintptr_t)weights_ddr);
    XCnn_accelerator_ddr_Set_params(&accelerator, (u64)(uintptr_t)params_ddr);
    XCnn_accelerator_ddr_Set_result(&accelerator, (u64)(uintptr_t)result_ddr);
    for (test = 0; test < MNIST_TEST_COUNT; ++test) {
        unsigned errors;
        ResultRecord result;
        u32 elapsed_us;
        /* 3. 逐图准备输入、启动等待并读取；4. 核对所有结果字段。 */
        if (run_inference(&accelerator, test, &result, &elapsed_us) != 0)
            return 1;
        errors = check_result(test, &result);
        total_errors += errors;
        xil_printf("test %u: class=%u errors=%u inference_us=%u %s\r\n", test,
                   (unsigned)result.class_index, errors, (unsigned)elapsed_us, errors ? "FAIL" : "PASS");
    }
    xil_printf("Overall %s: %u errors\r\n", total_errors ? "FAIL" : "PASS", total_errors);
    return total_errors ? 1 : 0;
}
