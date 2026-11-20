// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XCNN_ACCELERATOR_DDR_H
#define XCNN_ACCELERATOR_DDR_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xcnn_accelerator_ddr_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Control_BaseAddress;
} XCnn_accelerator_ddr_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XCnn_accelerator_ddr;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XCnn_accelerator_ddr_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XCnn_accelerator_ddr_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XCnn_accelerator_ddr_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XCnn_accelerator_ddr_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XCnn_accelerator_ddr_Initialize(XCnn_accelerator_ddr *InstancePtr, u16 DeviceId);
XCnn_accelerator_ddr_Config* XCnn_accelerator_ddr_LookupConfig(u16 DeviceId);
int XCnn_accelerator_ddr_CfgInitialize(XCnn_accelerator_ddr *InstancePtr, XCnn_accelerator_ddr_Config *ConfigPtr);
#else
int XCnn_accelerator_ddr_Initialize(XCnn_accelerator_ddr *InstancePtr, const char* InstanceName);
int XCnn_accelerator_ddr_Release(XCnn_accelerator_ddr *InstancePtr);
#endif

void XCnn_accelerator_ddr_Start(XCnn_accelerator_ddr *InstancePtr);
u32 XCnn_accelerator_ddr_IsDone(XCnn_accelerator_ddr *InstancePtr);
u32 XCnn_accelerator_ddr_IsIdle(XCnn_accelerator_ddr *InstancePtr);
u32 XCnn_accelerator_ddr_IsReady(XCnn_accelerator_ddr *InstancePtr);
void XCnn_accelerator_ddr_EnableAutoRestart(XCnn_accelerator_ddr *InstancePtr);
void XCnn_accelerator_ddr_DisableAutoRestart(XCnn_accelerator_ddr *InstancePtr);

void XCnn_accelerator_ddr_Set_image_r(XCnn_accelerator_ddr *InstancePtr, u64 Data);
u64 XCnn_accelerator_ddr_Get_image_r(XCnn_accelerator_ddr *InstancePtr);
void XCnn_accelerator_ddr_Set_weights(XCnn_accelerator_ddr *InstancePtr, u64 Data);
u64 XCnn_accelerator_ddr_Get_weights(XCnn_accelerator_ddr *InstancePtr);
void XCnn_accelerator_ddr_Set_params(XCnn_accelerator_ddr *InstancePtr, u64 Data);
u64 XCnn_accelerator_ddr_Get_params(XCnn_accelerator_ddr *InstancePtr);
void XCnn_accelerator_ddr_Set_result(XCnn_accelerator_ddr *InstancePtr, u64 Data);
u64 XCnn_accelerator_ddr_Get_result(XCnn_accelerator_ddr *InstancePtr);
void XCnn_accelerator_ddr_Set_softmax_scale_q20(XCnn_accelerator_ddr *InstancePtr, u32 Data);
u32 XCnn_accelerator_ddr_Get_softmax_scale_q20(XCnn_accelerator_ddr *InstancePtr);

void XCnn_accelerator_ddr_InterruptGlobalEnable(XCnn_accelerator_ddr *InstancePtr);
void XCnn_accelerator_ddr_InterruptGlobalDisable(XCnn_accelerator_ddr *InstancePtr);
void XCnn_accelerator_ddr_InterruptEnable(XCnn_accelerator_ddr *InstancePtr, u32 Mask);
void XCnn_accelerator_ddr_InterruptDisable(XCnn_accelerator_ddr *InstancePtr, u32 Mask);
void XCnn_accelerator_ddr_InterruptClear(XCnn_accelerator_ddr *InstancePtr, u32 Mask);
u32 XCnn_accelerator_ddr_InterruptGetEnabled(XCnn_accelerator_ddr *InstancePtr);
u32 XCnn_accelerator_ddr_InterruptGetStatus(XCnn_accelerator_ddr *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
