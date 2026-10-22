#include "camera_input.h"
#include <string.h>
#include "xil_cache.h"
#include "xtime_l.h"

static int camera_dma_error(XAxiVdma *vdma)
{
    /* 这里读的是两通道状态寄存器，不是摄像头像素流。 */
    return XAxiVdma_GetDmaChannelErrors(vdma, XAXIVDMA_READ) ||
           XAxiVdma_GetDmaChannelErrors(vdma, XAXIVDMA_WRITE);
}

int camera_input_check(XAxiVdma *vdma)
{
    unsigned int channel_index;
    XAxiVdma_FrameCounter counters;
    const u32 required_control = XAXIVDMA_CR_RUNSTOP_MASK |
        XAXIVDMA_CR_TAIL_EN_MASK | XAXIVDMA_CR_SYNC_EN_MASK |
        XAXIVDMA_CR_GENLCK_SRC_MASK;

    /* 停车保护依赖现有内部动态Genlock；只检查，不偷偷重配硬件。 */
    if (vdma->MaxNumFrames != CAMERA_FRAMES || !vdma->InternalGenLock ||
        vdma->ReadChannel.GenLock != XAXIVDMA_DYN_GENLOCK_SLAVE ||
        vdma->WriteChannel.GenLock != XAXIVDMA_DYN_GENLOCK_MASTER) {
        return CAMERA_CONFIG;
    }
    XAxiVdma_GetFrameCounter(vdma, &counters);
    if (counters.ReadFrameCount != 1) {
        return CAMERA_CONFIG;
    }
    for (channel_index = 0; channel_index < 2; ++channel_index) {
        u32 channel_base = vdma->BaseAddr +
            (channel_index ? XAXIVDMA_RX_OFFSET : XAXIVDMA_TX_OFFSET);
        u32 control = XAxiVdma_ReadReg(channel_base, XAXIVDMA_CR_OFFSET);
        if ((control & required_control) != required_control ||
            (control & XAXIVDMA_CR_FRMCNT_EN_MASK)) {
            return CAMERA_CONFIG;
        }
        /* 帧事件通知位与帧数限定停止位不同；这里只轮询，不接GIC。 */
        if (!channel_index && !(control & XAXIVDMA_IXR_FRMCNT_MASK)) {
            return CAMERA_CONFIG;
        }
    }
    return camera_dma_error(vdma) ? CAMERA_DMA_ERROR : CAMERA_OK;
}

int camera_input_capture(XAxiVdma *vdma, const uint8_t *frames,
                         uint8_t snapshot[CAMERA_FRAME_BYTES])
{
    XTime wait_start, current_time;
    unsigned int frame_index, event_index;
    const uint8_t *source_frame;
    int result = camera_input_check(vdma);
    if (result != CAMERA_OK) {
        return result;
    }

    /* CurrFrameStore读取停车指针寄存器：编号是DDR帧槽，不是新帧序号。 */
    frame_index = XAxiVdma_CurrFrameStore(vdma, XAXIVDMA_READ);
    if (frame_index >= CAMERA_FRAMES) {
        return CAMERA_SLOT;
    }
    /* StartParking写控制寄存器提出请求，不搬运像素，且在帧边界生效。 */
    if (XAxiVdma_StartParking(vdma, frame_index, XAXIVDMA_READ) != XST_SUCCESS) {
        result = CAMERA_PARK;
        goto release_parking;
    }

    /* 清除请求前的旧通知。等待两个新MM2S帧事件，给停车请求跨过帧边界
     * 并完成一个稳定显示周期的时间；之后还必须核实READ/WRITE槽所有权，
     * 不能仅凭事件次数假定源帧已安全。两事件共用200ms截止时间。 */
    XAxiVdma_IntrClear(vdma, XAXIVDMA_IXR_FRMCNT_MASK, XAXIVDMA_READ);
    XTime_GetTime(&wait_start);
    for (event_index = 0; event_index < 2; ++event_index) {
        for (;;) {
            if (camera_dma_error(vdma)) {
                result = CAMERA_DMA_ERROR;
                goto release_parking;
            }
            XTime_GetTime(&current_time);
            if (current_time - wait_start >= COUNTS_PER_SECOND / 5) {
                result = CAMERA_TIMEOUT;
                goto release_parking;
            }
            if (XAxiVdma_IntrGetPending(vdma, XAXIVDMA_READ) &
                XAXIVDMA_IXR_FRMCNT_MASK) {
                XAxiVdma_IntrClear(vdma, XAXIVDMA_IXR_FRMCNT_MASK, XAXIVDMA_READ);
                break;
            }
        }
    }

    /* 内部动态Genlock应使S2MM避开被READ停车保护的目标帧。 */
    if (XAxiVdma_CurrFrameStore(vdma, XAXIVDMA_READ) != frame_index ||
        XAxiVdma_CurrFrameStore(vdma, XAXIVDMA_WRITE) == frame_index) {
        result = CAMERA_OWNERSHIP;
        goto release_parking;
    }
    source_frame = frames + frame_index * CAMERA_FRAME_BYTES;
    /* DMA已经写DDR，CPU旧缓存可能过期；整帧对齐失效后才能读到DMA结果。 */
    Xil_DCacheInvalidateRange((INTPTR)source_frame, CAMERA_FRAME_BYTES);
    memcpy(snapshot, source_frame, CAMERA_FRAME_BYTES);
    /* 保持停车保护，逐字节比较CPU快照与源帧；不是传感器图像精度验证。 */
    if (memcmp(snapshot, source_frame, CAMERA_FRAME_BYTES) != 0) {
        result = CAMERA_COPY_MISMATCH;
        goto release_parking;
    }
    result = camera_dma_error(vdma) ? CAMERA_DMA_ERROR : CAMERA_OK;

release_parking:
    /* 每个停车后的出口都请求恢复READ循环；S2MM不主动停。
     * DMA故障可能需用户复位重启，不保证画面恢复，不自动清错/reset。
     * 只有返回CAMERA_OK时，调用方才能使用snapshot。 */
    XAxiVdma_StopParking(vdma, XAXIVDMA_READ);
    return result;
}

const char *camera_input_reason(int result)
{
    static const char * const names[] = {
        "ok", "config", "slot", "park", "dma_error", "timeout",
        "ownership", "copy_mismatch"
    };
    return result >= 0 && result <= CAMERA_COPY_MISMATCH ? names[result] : "unknown";
}
