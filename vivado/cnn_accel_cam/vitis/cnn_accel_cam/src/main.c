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
#include "xtime_l.h"

#define VDMA_ID            XPAR_AXIVDMA_0_DEVICE_ID    //VDMA器件ID
#define DISP_VTC_ID        XPAR_VTC_0_DEVICE_ID        //VTC器件ID

/* 静态DDR分配，整帧及数组都以cacheline对齐；启动后CPU只读。 */
static u8 camera_frames[CAMERA_FRAMES*CAMERA_FRAME_BYTES] __attribute__((aligned(64)));
/* CPU独占的普通DDR数组，不是VDMA第四帧槽；VDMA仍只访问上方三帧，
 * 不会覆盖此快照。成功后可供调试器查看。 */
static u8 camera_snapshot[CAMERA_FRAME_BYTES] __attribute__((aligned(64)));
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
	xil_printf("OV5640 detected successful!\r\n");

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
        XTime report_window_start; /* 当前一秒统计窗口的起点，单位为计时器tick。 */
        XTime capture_start;       /* 本轮起点，包括取帧及快照flush工作时间。 */
        XTime capture_end;         /* 本轮工作结束时间，用于计算耗时。 */
        XTime current_time;        /* 周期等待时读取的当前时间。 */
        unsigned int capture_success_count = 0; /* 当前窗口成功次数。 */
        unsigned int capture_failure_count = 0; /* 当前窗口失败次数。 */
        unsigned int last_capture_us = 0;       /* 最近一轮工作耗时，微秒。 */
        int last_failure = CAMERA_OK;           /* 当前窗口最后失败原因。 */
        int capture_result;                    /* 本轮返回码，失败输出不能用。 */

        xil_printf("snapshot_address=0x%x frame_bytes=%u\r\n",
            (unsigned int)(UINTPTR)camera_snapshot, CAMERA_FRAME_BYTES);
        XTime_GetTime(&report_window_start);
        for (;;) {
            XTime_GetTime(&capture_start);
            capture_result = camera_input_capture(&vdma, camera_frames, camera_snapshot);
            if (capture_result == CAMERA_OK) {
                ++capture_success_count;
                /* CPU写入snapshot后可能只在Cache中；flush仅为让调试器从DDR
                 * 看到完整快照；CPU自己读取snapshot不需要这次flush。
                 * 这不同于读取DMA源帧前的invalidate。
                 * 在下一行XTime_GetTime处设条件断点(capture_result==CAMERA_OK)，
                 * 此时capture已完成且成功快照已flush，勿在memcpy中途查看。 */
                Xil_DCacheFlushRange((INTPTR)camera_snapshot, sizeof(camera_snapshot));
            } else {
                ++capture_failure_count;
                last_failure = capture_result;
            }
            XTime_GetTime(&capture_end);
            last_capture_us = (unsigned int)((capture_end-capture_start) *
                1000000ULL / COUNTS_PER_SECOND);
            if (capture_end-report_window_start >= COUNTS_PER_SECOND) {
                /* 正常目标约10次成功/秒、失败0、无VDMA错误；复制期间预览
                 * 可能短暂定格。成功代表原始字节比较一致，不证明相机图像
                 * 内容/颜色正确，也不保证每次成功都是独立新帧。 */
                xil_printf("capture_success_count=%u capture_failure_count=%u last_capture_us=%u last_failure=%s\r\n",
                    capture_success_count, capture_failure_count, last_capture_us,
                    camera_input_reason(last_failure));
                /* 输出约一秒窗口后清零，只统计下一窗口；last_capture_us保留
                 * 最近一轮耗时，last_failure重置以免旧故障冒充新故障。 */
                capture_success_count = 0;
                capture_failure_count = 0;
                last_failure = CAMERA_OK;
                report_window_start = capture_end;
            }
            /* 从本轮起点到下一轮目标100ms，包含复制、flush及串口工作；
             * 若已经超时则直接进入下一轮，不在工作后额外再等100ms。 */
            do {
                XTime_GetTime(&current_time);
            } while (current_time-capture_start < COUNTS_PER_SECOND/10);
        }
    }
}
