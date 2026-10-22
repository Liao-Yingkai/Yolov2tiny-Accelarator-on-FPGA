// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xcnn_accelerator_ddr.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XCnn_accelerator_ddr_CfgInitialize(XCnn_accelerator_ddr *InstancePtr, XCnn_accelerator_ddr_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XCnn_accelerator_ddr_Start(XCnn_accelerator_ddr *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_AP_CTRL) & 0x80;
    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XCnn_accelerator_ddr_IsDone(XCnn_accelerator_ddr *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XCnn_accelerator_ddr_IsIdle(XCnn_accelerator_ddr *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XCnn_accelerator_ddr_IsReady(XCnn_accelerator_ddr *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XCnn_accelerator_ddr_EnableAutoRestart(XCnn_accelerator_ddr *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XCnn_accelerator_ddr_DisableAutoRestart(XCnn_accelerator_ddr *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_AP_CTRL, 0);
}

void XCnn_accelerator_ddr_Set_image_r(XCnn_accelerator_ddr *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_IMAGE_R_DATA, (u32)(Data));
    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_IMAGE_R_DATA + 4, (u32)(Data >> 32));
}

u64 XCnn_accelerator_ddr_Get_image_r(XCnn_accelerator_ddr *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_IMAGE_R_DATA);
    Data += (u64)XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_IMAGE_R_DATA + 4) << 32;
    return Data;
}

void XCnn_accelerator_ddr_Set_weights(XCnn_accelerator_ddr *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_WEIGHTS_DATA, (u32)(Data));
    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_WEIGHTS_DATA + 4, (u32)(Data >> 32));
}

u64 XCnn_accelerator_ddr_Get_weights(XCnn_accelerator_ddr *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_WEIGHTS_DATA);
    Data += (u64)XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_WEIGHTS_DATA + 4) << 32;
    return Data;
}

void XCnn_accelerator_ddr_Set_params(XCnn_accelerator_ddr *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_PARAMS_DATA, (u32)(Data));
    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_PARAMS_DATA + 4, (u32)(Data >> 32));
}

u64 XCnn_accelerator_ddr_Get_params(XCnn_accelerator_ddr *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_PARAMS_DATA);
    Data += (u64)XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_PARAMS_DATA + 4) << 32;
    return Data;
}

void XCnn_accelerator_ddr_Set_result(XCnn_accelerator_ddr *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_RESULT_DATA, (u32)(Data));
    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_RESULT_DATA + 4, (u32)(Data >> 32));
}

u64 XCnn_accelerator_ddr_Get_result(XCnn_accelerator_ddr *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_RESULT_DATA);
    Data += (u64)XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_RESULT_DATA + 4) << 32;
    return Data;
}

void XCnn_accelerator_ddr_InterruptGlobalEnable(XCnn_accelerator_ddr *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_GIE, 1);
}

void XCnn_accelerator_ddr_InterruptGlobalDisable(XCnn_accelerator_ddr *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_GIE, 0);
}

void XCnn_accelerator_ddr_InterruptEnable(XCnn_accelerator_ddr *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_IER);
    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_IER, Register | Mask);
}

void XCnn_accelerator_ddr_InterruptDisable(XCnn_accelerator_ddr *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_IER);
    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_IER, Register & (~Mask));
}

void XCnn_accelerator_ddr_InterruptClear(XCnn_accelerator_ddr *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XCnn_accelerator_ddr_WriteReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_ISR, Mask);
}

u32 XCnn_accelerator_ddr_InterruptGetEnabled(XCnn_accelerator_ddr *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_IER);
}

u32 XCnn_accelerator_ddr_InterruptGetStatus(XCnn_accelerator_ddr *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XCnn_accelerator_ddr_ReadReg(InstancePtr->Control_BaseAddress, XCNN_ACCELERATOR_DDR_CONTROL_ADDR_ISR);
}

