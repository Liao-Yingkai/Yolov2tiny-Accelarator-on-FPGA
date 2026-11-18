// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xroi_overlay.h"

extern XRoi_overlay_Config XRoi_overlay_ConfigTable[];

XRoi_overlay_Config *XRoi_overlay_LookupConfig(u16 DeviceId) {
	XRoi_overlay_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XROI_OVERLAY_NUM_INSTANCES; Index++) {
		if (XRoi_overlay_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XRoi_overlay_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XRoi_overlay_Initialize(XRoi_overlay *InstancePtr, u16 DeviceId) {
	XRoi_overlay_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XRoi_overlay_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XRoi_overlay_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

