/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef INSPECTOR_HGGCWRAP_H_
#define INSPECTOR_HGGCWRAP_H_

#include <hggc.h>
#include <hggc_runtime.h>

// Inspector-specific HGGC wrapper for standalone compilation

// Include inspector.h for proper type definitions
#include "inspector.h"

// Function pointer types for HGGC driver API functions
typedef HGresult (*PFN_hgGetErrorString)(HGresult error, const char **pStr);
typedef HGresult (*PFN_hgDeviceGet)(HGdevice *device, int ordinal);
typedef HGresult (*PFN_hgDeviceGetUuid)(HGuuid *uuid, HGdevice dev);

// Function pointers - externally defined
extern PFN_hgGetErrorString pfn_hgGetErrorString;
extern PFN_hgDeviceGet pfn_hgDeviceGet;
extern PFN_hgDeviceGetUuid pfn_hgDeviceGetUuid;

// Convenience macro for calling function pointers
#define INSPECTOR_HGPFN(symbol) pfn_##symbol

// Initialize the HGGC wrapper (load function pointers)
inspectorResult_t inspectorHggcWrapInit(void);

// Cleanup the HGGC wrapper
void inspectorHggcWrapCleanup(void);

#endif // INSPECTOR_HGGCWRAP_H_
