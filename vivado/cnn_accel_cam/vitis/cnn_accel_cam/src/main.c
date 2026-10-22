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

#define VDMA_ID            XPAR_AXIVDMA_0_DEVICE_ID    //VDMA器件ID
#define DISP_VTC_ID        XPAR_VTC_0_DEVICE_ID        //VTC器件ID

//frame buffer的起始地址
unsigned int const frame_buffer_addr = (XPAR_PS7_DDR_0_S_AXI_BASEADDR
										+ 0x1000000);
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

	cmos_h_pixel = 640;
	cmos_v_pixel = 480;
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
	status = run_vdma_frame_buffer(&vdma, VDMA_ID, vd_mode.width, vd_mode.height,
							frame_buffer_addr,0,0,BOTH);
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

    return 0;
}
