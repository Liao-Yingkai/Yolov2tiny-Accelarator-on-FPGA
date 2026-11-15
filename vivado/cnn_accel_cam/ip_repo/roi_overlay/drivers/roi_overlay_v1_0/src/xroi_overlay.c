// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xroi_overlay.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XRoi_overlay_CfgInitialize(XRoi_overlay *InstancePtr, XRoi_overlay_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XRoi_overlay_Set_command(XRoi_overlay *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoi_overlay_WriteReg(InstancePtr->Control_BaseAddress, XROI_OVERLAY_CONTROL_ADDR_COMMAND_DATA, Data);
}

u32 XRoi_overlay_Get_command(XRoi_overlay *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoi_overlay_ReadReg(InstancePtr->Control_BaseAddress, XROI_OVERLAY_CONTROL_ADDR_COMMAND_DATA);
    return Data;
}

void XRoi_overlay_Set_command_data(XRoi_overlay *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoi_overlay_WriteReg(InstancePtr->Control_BaseAddress, XROI_OVERLAY_CONTROL_ADDR_COMMAND_DATA_DATA, Data);
}

u32 XRoi_overlay_Get_command_data(XRoi_overlay *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoi_overlay_ReadReg(InstancePtr->Control_BaseAddress, XROI_OVERLAY_CONTROL_ADDR_COMMAND_DATA_DATA);
    return Data;
}

u32 XRoi_overlay_Get_status(XRoi_overlay *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoi_overlay_ReadReg(InstancePtr->Control_BaseAddress, XROI_OVERLAY_CONTROL_ADDR_STATUS_DATA);
    return Data;
}

