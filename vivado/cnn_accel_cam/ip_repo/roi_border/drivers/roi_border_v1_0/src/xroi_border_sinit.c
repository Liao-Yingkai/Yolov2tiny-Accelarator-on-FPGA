// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xroi_border.h"

extern XRoi_border_Config XRoi_border_ConfigTable[];

XRoi_border_Config *XRoi_border_LookupConfig(u16 DeviceId) {
	XRoi_border_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XROI_BORDER_NUM_INSTANCES; Index++) {
		if (XRoi_border_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XRoi_border_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XRoi_border_Initialize(XRoi_border *InstancePtr, u16 DeviceId) {
	XRoi_border_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XRoi_border_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XRoi_border_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

