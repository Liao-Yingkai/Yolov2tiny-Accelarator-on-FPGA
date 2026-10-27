// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xroi_border.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XRoi_border_CfgInitialize(XRoi_border *InstancePtr, XRoi_border_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XRoi_border_Set_display_control(XRoi_border *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoi_border_WriteReg(InstancePtr->Control_BaseAddress, XROI_BORDER_CONTROL_ADDR_DISPLAY_CONTROL_DATA, Data);
}

u32 XRoi_border_Get_display_control(XRoi_border *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoi_border_ReadReg(InstancePtr->Control_BaseAddress, XROI_BORDER_CONTROL_ADDR_DISPLAY_CONTROL_DATA);
    return Data;
}

