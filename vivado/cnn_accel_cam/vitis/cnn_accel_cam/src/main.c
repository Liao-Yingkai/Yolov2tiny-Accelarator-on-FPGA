#include <stdlib.h>
#include "xil_types.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xaxivdma.h"
#include "xaxivdma_i.h"
#include "display_ctrl_hdmi/display_ctrl_hdmi.h"
#include "vdma_api/vdma_api.h"
#include "emio_sccb_cfg/emio_sccb_cfg.h"
#include "ov5640/ov5640_init.h"
#include "camera_input.h"
#include "camera_cnn.h"
#include "camera_geometry.h"
#include "camera_preprocess.h"
#include "camera_postprocess.h"
#include "camera_display.h"
#include "model_loader.h"

#define VDMA_ID            XPAR_AXIVDMA_0_DEVICE_ID    //VDMA器件ID
#define DISP_VTC_ID        XPAR_VTC_0_DEVICE_ID        //VTC器件ID

/* 三个连续DDR帧槽由VDMA循环写入和读取，维持摄像头到HDMI的预览。
 * 数组及各帧均按64字节对齐；启动前CPU执行flush，运行时只在抓拍模块
 * 确认槽所有权后读取 ROI。模型加载缓冲由 camera_cnn 模块独立管理。 */
static u8 camera_frames[CAMERA_FRAMES*CAMERA_FRAME_BYTES] __attribute__((aligned(64)));
/* 抓拍输出为紧密 BGR888 ROI；CPU 独占，不交给 VDMA。 */
static uint8_t roi_snapshot[CAMERA_ROI_BYTES] __attribute__((aligned(64)));
/* 保存完整解码列表供叠加消费；不持有A/B工作区指针。 */
static camera_detections_t detections;
_Static_assert(CAMERA_ROI_BYTES == MODEL_INPUT_BYTES, "ROI matches model input");
_Static_assert(CAMERA_WIDTH == 640U && CAMERA_HEIGHT == 480U &&
    CAMERA_BYTES_PER_PIXEL == 3U && CAMERA_FRAME_BYTES % 64U == 0U,
    "preview geometry and alignment must match");
//全局变量
XAxiVdma     vdma;
DisplayCtrl  dispCtrl;
VideoMode    vd_mode;

/* 非叠加故障退出时尽力清框；清除失败只报告，不再发命令。 */
static camera_display_status_t clear_overlay_on_error(void)
{
    const camera_display_status_t status = camera_display_clear();
    if (status != CAMERA_DISPLAY_OK)
        xil_printf("[OVERLAY] Clear failed: %s\r\n", camera_display_reason(status));
    return status;
}

int main(void)
{
	u32 status;
	u16 cmos_h_pixel;   //ov5640 DVP 输出水平像素点数
	u16 cmos_v_pixel;   //ov5640 DVP 输出垂直像素点数
	u16 total_h_pixel;  //ov5640 水平总像素大小
	u16 total_v_pixel;  //ov5640 垂直总像素大小

	cmos_h_pixel = CAMERA_WIDTH;
	cmos_v_pixel = CAMERA_HEIGHT;
	total_h_pixel = 2570;
	total_v_pixel = 980;

	emio_init();                         //初始化EMIO
	status = ov5640_init( cmos_h_pixel,  //初始化ov5640
						  cmos_v_pixel,
						 total_h_pixel,
						 total_v_pixel);
	if(status != 0) {
		xil_printf("ERROR: OV5640 initialization failed!\r\n");
		return EXIT_FAILURE;
	}

	vd_mode = VMODE_640x480;

	//配置VDMA
	Xil_DCacheEnable();
	Xil_DCacheFlushRange((INTPTR)camera_frames, sizeof(camera_frames));
	status = run_vdma_frame_buffer(&vdma, VDMA_ID, vd_mode.width, vd_mode.height,
							(unsigned int)(UINTPTR)camera_frames,1,1,BOTH);
	if (status != XST_SUCCESS) {
		xil_printf("ERROR: VDMA frame buffer initialization failed!\r\n");
		return EXIT_FAILURE;
	}

	//初始化Display controller
	status = DisplayInitialize(&dispCtrl, DISP_VTC_ID);
	if (status != XST_SUCCESS) {
		xil_printf("ERROR: Display controller initialization failed!\r\n");
		return EXIT_FAILURE;
	}
    //设置VideoMode
	status = DisplaySetMode(&dispCtrl, &vd_mode);
	if (status != XST_SUCCESS) {
		xil_printf("ERROR: Display video mode setup failed!\r\n");
		return EXIT_FAILURE;
	}
	status = DisplayStart(&dispCtrl);
	if (status != XST_SUCCESS) {
		xil_printf("ERROR: Display start failed!\r\n");
		return EXIT_FAILURE;
	}

    xil_printf("[VIDEO] 640x480 preview ready\r\n");
    {
        /* HDMI 已启动：先清旧框并在有限时间内确认视频 SOF。 */
        const camera_display_status_t overlay_status = camera_display_init();
        if (overlay_status != CAMERA_DISPLAY_OK) {
            xil_printf("[OVERLAY] Initialization failed: %s\r\n",
                       camera_display_reason(overlay_status));
            return EXIT_FAILURE;
        }
    }
    xil_printf("[OVERLAY] ready\r\n");
    {
        const camera_cnn_status_t cnn_status = camera_cnn_init();
        if (cnn_status != CAMERA_CNN_OK) {
            xil_printf("[APP] Model initialization failed: %s\r\n",
                       camera_cnn_reason(cnn_status));
            clear_overlay_on_error();
            return EXIT_FAILURE;
        }
    }
    xil_printf("[APP] Continuous YOLO detection/overlay ready\r\n");
    {
        int previous_capture_error = CAMERA_OK;
        for (;;) {
            const int8_t *prediction;
            camera_cnn_status_t cnn_status;
            int capture_status;
            /* 取得 A 前先查硬件 idle；绝不在 PL 仍运行时覆盖输入。 */
            int8_t *input = camera_cnn_input_buffer();
            if (input == NULL) {
                xil_printf("[APP] CNN input unavailable; board reset required\r\n");
                clear_overlay_on_error();
                return EXIT_FAILURE;
            }
            capture_status = camera_input_capture(&vdma, camera_frames, roi_snapshot);
            if (capture_status != CAMERA_OK) {
                /* 正常转为暂时失败时只清一次，连续失败不反复清框。 */
                if (previous_capture_error == CAMERA_OK &&
                    (capture_status == CAMERA_TIMEOUT || capture_status == CAMERA_OWNERSHIP)) {
                    if (clear_overlay_on_error() != CAMERA_DISPLAY_OK)
                        return EXIT_FAILURE;
                }
                if (capture_status != previous_capture_error)
                    xil_printf("[CAPTURE] %s\r\n", camera_input_reason(capture_status));
                previous_capture_error = capture_status;
                /* 仅暂时没有新帧/槽所有权竞争可重试；失败快照不可预处理。
                 * 配置、停车、槽号或 DMA 故障须停机检查，不自动复位。 */
                if (capture_status == CAMERA_TIMEOUT || capture_status == CAMERA_OWNERSHIP)
                    continue;
                clear_overlay_on_error();
                return EXIT_FAILURE;
            }
            previous_capture_error = CAMERA_OK;
            /* 在 A 中直接生成完整 416×416×3 int8 输入，不额外复制。 */
            camera_preprocess(roi_snapshot, input);
            cnn_status = camera_cnn_run(&prediction);
            if (cnn_status != CAMERA_CNN_OK) {
                xil_printf("[APP] CNN stopped: %s\r\n", camera_cnn_reason(cnn_status));
                clear_overlay_on_error();
                return EXIT_FAILURE;
            }
            /* 同步消费B中的输出，必须在下一轮复用A/B前完成解码和NMS。 */
            if (camera_postprocess(prediction, &detections) != CAMERA_POSTPROCESS_OK) {
                xil_printf("[APP] YOLO postprocess failed\r\n");
                clear_overlay_on_error();
                return EXIT_FAILURE;
            }
            {
                /* 空列表也提交，确保无目标时清除上一帧检测框。 */
                const camera_display_status_t overlay_status = camera_display_update(&detections);
                if (overlay_status != CAMERA_DISPLAY_OK) {
                    xil_printf("[OVERLAY] Update failed: %s\r\n",
                               camera_display_reason(overlay_status));
                    return EXIT_FAILURE;
                }
            }
        }
    }
}
