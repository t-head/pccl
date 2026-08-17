/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "inspector_hggcwrap.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// Function pointer storage
PFN_hgGetErrorString pfn_hgGetErrorString = nullptr;
PFN_hgDeviceGet pfn_hgDeviceGet = nullptr;
PFN_hgDeviceGetUuid pfn_hgDeviceGetUuid = nullptr;

// Handle to the HGGC driver library
static void* hggcDriverLib = nullptr;

/*
 * Description:
 *
 *   Loads a HGGC driver function symbol from the library.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during initialization).
 *
 * Input:
 *   const char* symbol_name - name of the symbol to load.
 *
 * Output:
 *   None.
 *
 * Return:
 *   void* - pointer to the loaded symbol, or nullptr on failure.
 */
static void* loadHggcSymbol(const char* symbol_name) {
  if (!hggcDriverLib) {
    return nullptr;
  }

  void* symbol = dlsym(hggcDriverLib, symbol_name);
  if (!symbol) {
    INFO_INSPECTOR("Inspector: Failed to load HGGC symbol '%s': %s",
         symbol_name, dlerror());
    return nullptr;
  }

  return symbol;
}

/*
 * Description:
 *
 *   Initializes the HGGC wrapper by loading the HGGC driver library
 *   and resolving required function pointers.
 *
 * Thread Safety:
 *   Not thread-safe (should be called once during initialization).
 *
 * Input:
 *   None.
 *
 * Output:
 *   None.
 *
 * Return:
 *   inspectorResult_t - success or error code.
 */
inspectorResult_t inspectorHggcWrapInit(void) {
  // Clear any previous dlopen errors
  dlerror();

  // Try to load HGGC driver library
  hggcDriverLib = dlopen("libhggc.so", RTLD_LAZY);
  if (!hggcDriverLib) {
    // Try alternative name
    hggcDriverLib = dlopen("libhggc.so.1", RTLD_LAZY);
    if (!hggcDriverLib) {
      INFO_INSPECTOR("Inspector: Failed to load HGGC driver library: %s", dlerror());
      return inspectorHggcError;
    }
  }

  // Load required HGGC driver functions
  pfn_hgGetErrorString = (PFN_hgGetErrorString)loadHggcSymbol("hgGetErrorString");
  if (!pfn_hgGetErrorString) {
    INFO_INSPECTOR("Inspector: Failed to load hgGetErrorString");
    inspectorHggcWrapCleanup();
    return inspectorHggcError;
  }

  pfn_hgDeviceGet = (PFN_hgDeviceGet)loadHggcSymbol("hgDeviceGet");
  if (!pfn_hgDeviceGet) {
    INFO_INSPECTOR("Inspector: Failed to load hgDeviceGet");
    inspectorHggcWrapCleanup();
    return inspectorHggcError;
  }

  pfn_hgDeviceGetUuid = (PFN_hgDeviceGetUuid)loadHggcSymbol("hgDeviceGetUuid");
  if (!pfn_hgDeviceGetUuid) {
    INFO_INSPECTOR("Inspector: Failed to load hgDeviceGetUuid");
    inspectorHggcWrapCleanup();
    return inspectorHggcError;
  }

  INFO(NCCL_INIT, "Inspector: HGGC wrapper initialized successfully");
  return inspectorSuccess;
}

/*
 * Description:
 *
 *   Cleans up the HGGC wrapper by closing the driver library.
 *
 * Thread Safety:
 *   Not thread-safe (should be called during cleanup).
 *
 * Input:
 *   None.
 *
 * Output:
 *   None.
 *
 * Return:
 *   None.
 */
void inspectorHggcWrapCleanup(void) {
  // Clear function pointers
  pfn_hgGetErrorString = nullptr;
  pfn_hgDeviceGet = nullptr;
  pfn_hgDeviceGetUuid = nullptr;

  // Close library handle
  if (hggcDriverLib) {
    dlclose(hggcDriverLib);
    hggcDriverLib = nullptr;
  }
}
