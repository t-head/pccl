/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CUDAWRAP_H_
#define NCCL_CUDAWRAP_H_

#include <hggc.h>
#include <hggc_runtime.h>
#include "checks.h"
#include "nvmlwrap.h"
#include "drv_extension.h"
#include "drv_version.h"
extern bool ncclCudaLaunchBlocking; // initilized by ncclCudaLibraryInit()
// Is cuMem API usage enabled
extern int ncclCuMemEnable();
// Handle type used for cuMemCreate()
extern HGmemAllocationHandleType ncclHgMemHandleType;

extern int ncclLegacyCudaRegister();
// Check CUDA PFN driver calls
#define HGCHECK(cmd) do { \
    HGresult err = cmd; \
    if( err != HGGC_SUCCESS ) { \
      const char *errStr = "unknown"; \
      (void) hgGetErrorString(err, &errStr); \
      WARN("Hg failure: cmd=%s '%s'", #cmd, errStr); \
      return ncclUnhandledHggcError; \
    } \
} while(false)

#define HGCHECKGOTO(cmd, res, label) do { \
    HGresult err = cmd; \
    if( err != HGGC_SUCCESS ) { \
      const char *errStr = "unknown"; \
      (void) hgGetErrorString(err, &errStr); \
      WARN("Hg failure: cmd=%s '%s'", #cmd, errStr); \
      res = ncclUnhandledHggcError; \
      goto label; \
    } \
} while(false)

// Log-and-continue variant of HGCHECK; used in cleanup paths.
#define HGCHECKIGNORE(cmd) do { \
    HGresult err = cmd; \
    if( err != HGGC_SUCCESS ) { \
      const char *errStr; \
      (void) hgGetErrorString(err, &errStr); \
      INFO(NCCL_ALL, "%s:%d Hggc failure %d '%s'", \
           __FILE__, __LINE__, err, errStr); \
    } \
} while(false)

ncclResult_t ncclHggcLibraryInit(void);

extern uint ncclSystemDriverVersionCache;
extern uint ncclHggcDriverVersionCache;

inline ncclResult_t ncclHggcDriverVersion(uint* driver) {
  uint version = __atomic_load_n(&ncclHggcDriverVersionCache, __ATOMIC_RELAXED);
  if (version == PCCL_UNDEF_DRIVER_VERSION) {
    uint32_t major, minor;
    HGdriverVersionInfo versionInfo;
    HGresult res = hggcEitInterface::getInstance().wrapHggcDriverGetVersion(&versionInfo);
    if (res == HGGC_SUCCESS) {
      major = versionInfo.driver.major;
      minor = versionInfo.driver.minor;
      version = PCCL_HGGC_DRIVER_VERSION(major, minor, 0);
    } else {
      ERROR("hgDriverGetVersion faild(%d)", res);
      *driver = PCCL_UNDEF_DRIVER_VERSION;
      return ncclInternalError;
    }
    __atomic_store_n(&ncclHggcDriverVersionCache, version, __ATOMIC_RELAXED);
  }
  *driver = version;
  return ncclSuccess;
}

inline ncclResult_t ncclSystemDriverVersion(uint* driver) {
  uint version = __atomic_load_n(&ncclSystemDriverVersionCache, __ATOMIC_RELAXED);
  if (version == PCCL_UNDEF_DRIVER_VERSION) {
    char systemDriverVersionStr[128];
    if (pcclHgmlSystemGetDriverVersion(systemDriverVersionStr, 128) == ncclSuccess) {
      uint major, minor, patch;
      if (sscanf(systemDriverVersionStr, "%u.%u.%u", &major, &minor, &patch) == 3) {
        version = PCCL_SYSTEM_DRIVER_VERSION(major, minor, patch);
      }
    }
    __atomic_store_n(&ncclSystemDriverVersionCache, version, __ATOMIC_RELAXED);
  }
  *driver = version;
  return ncclSuccess;
}

#endif
