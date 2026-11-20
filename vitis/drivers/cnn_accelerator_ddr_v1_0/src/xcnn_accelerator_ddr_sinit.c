// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xcnn_accelerator_ddr.h"

extern XCnn_accelerator_ddr_Config XCnn_accelerator_ddr_ConfigTable[];

XCnn_accelerator_ddr_Config *XCnn_accelerator_ddr_LookupConfig(u16 DeviceId) {
	XCnn_accelerator_ddr_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XCNN_ACCELERATOR_DDR_NUM_INSTANCES; Index++) {
		if (XCnn_accelerator_ddr_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XCnn_accelerator_ddr_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XCnn_accelerator_ddr_Initialize(XCnn_accelerator_ddr *InstancePtr, u16 DeviceId) {
	XCnn_accelerator_ddr_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XCnn_accelerator_ddr_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XCnn_accelerator_ddr_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

