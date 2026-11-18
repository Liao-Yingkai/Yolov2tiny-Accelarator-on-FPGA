// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.2 (64-bit)
// Tool Version Limit: 2019.12
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : reserved
// 0x04 : reserved
// 0x08 : reserved
// 0x0c : reserved
// 0x10 : Data signal of command
//        bit 31~0 - command[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of command_data
//        bit 31~0 - command_data[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of status
//        bit 31~0 - status[31:0] (Read)
// 0x24 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XROI_OVERLAY_CONTROL_ADDR_COMMAND_DATA      0x10
#define XROI_OVERLAY_CONTROL_BITS_COMMAND_DATA      32
#define XROI_OVERLAY_CONTROL_ADDR_COMMAND_DATA_DATA 0x18
#define XROI_OVERLAY_CONTROL_BITS_COMMAND_DATA_DATA 32
#define XROI_OVERLAY_CONTROL_ADDR_STATUS_DATA       0x20
#define XROI_OVERLAY_CONTROL_BITS_STATUS_DATA       32

