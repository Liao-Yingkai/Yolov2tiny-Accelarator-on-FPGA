#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#include "camera_preprocess.h"
#include "camera_cnn.h"
#include "xtime_l.h"

#define VDMA_ID            XPAR_AXIVDMA_0_DEVICE_ID    //VDMA器件ID
#define DISP_VTC_ID        XPAR_VTC_0_DEVICE_ID        //VTC器件ID

/* 静态DDR分配，整帧及数组都以cacheline对齐；启动后CPU只读。 */
static u8 camera_frames[CAMERA_FRAMES*CAMERA_FRAME_BYTES] __attribute__((aligned(64)));
/* CPU独占的普通DDR数组，不是VDMA第四帧槽；VDMA仍只访问上方三帧，
 * 不会覆盖此快照。成功后可供调试器查看。 */
static u8 camera_snapshot[CAMERA_FRAME_BYTES] __attribute__((aligned(64)));
/* 前784字节有效，容量832字节；这些工作数组仅由CPU访问。 */
static uint8_t camera_gray[CAMERA_DEBUG_BYTES] __attribute__((aligned(64)));
static int8_t camera_model_input[CAMERA_DEBUG_BYTES] __attribute__((aligned(64)));
_Static_assert(CAMERA_WIDTH == CAMERA_PREPROCESS_WIDTH &&
    CAMERA_HEIGHT == CAMERA_PREPROCESS_HEIGHT && CAMERA_BYTES_PER_PIXEL == 3U &&
    CAMERA_FRAME_BYTES % 64U == 0U && CAMERA_MODEL_PIXELS == CAMERA_CNN_INPUT_BYTES, "capture/preprocess geometry and alignment must match");
//全局变量
XAxiVdma     vdma;
DisplayCtrl  dispCtrl;
VideoMode    vd_mode;

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

    {
        camera_cnn_status_t cnn_status = camera_cnn_init();
        camera_cnn_result_t result;
        XTime cycle_start, now;
        if (cnn_status != CAMERA_CNN_OK) {
            xil_printf("CNN: %s\r\n", camera_cnn_reason(cnn_status));
            return EXIT_FAILURE;
        }
        for (;;) {
            int capture_status;
            XTime_GetTime(&cycle_start);
            capture_status = camera_input_capture(&vdma, camera_frames, camera_snapshot);
            if (capture_status == CAMERA_OK) {
                camera_preprocess(camera_snapshot, camera_gray, camera_model_input);
                cnn_status = camera_cnn_predict(camera_model_input, &result);
                if (cnn_status != CAMERA_CNN_OK) {
                    xil_printf("CNN: %s\r\n", camera_cnn_reason(cnn_status));
                    /* PL可能仍访问DDR，立即结束且不复用buffer；需板级复位重启，
                     * 不能仅自动reset CNN后继续，以免未完成AXI访问破坏数据。 */
                    return EXIT_FAILURE;
                }
                xil_printf("digit=%u probability=%u.%02u%%\r\n", (unsigned)result.class_index,
                    (unsigned)result.probability[result.class_index] / 100U,
                    (unsigned)result.probability[result.class_index] % 100U);
            } else {
                /* 失败快照无效，跳过预处理和CNN，下一节拍再取帧。 */
                xil_printf("capture: %s\r\n", camera_input_reason(capture_status));
            }
            /* 从本轮开始计100ms，含capture/预处理/CNN/串口；超时则立即
             * 进入下一轮。这是调度目标，不保证10fps或每次都是独立新帧。
             * snapshot/gray/model_input仅CPU读写，不做调试器专用flush；
             * CNN模块独立输入buffer的flush负责CPU→PL的数据可见性。 */
            do {
                XTime_GetTime(&now);
            } while (now - cycle_start < COUNTS_PER_SECOND / 10);
        }
    }
}
