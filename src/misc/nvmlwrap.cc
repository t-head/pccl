/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nvmlwrap.h"
#include "checks.h"
#include "debug.h"

#include <initializer_list>
#include <memory>
#include <mutex>

int pcclHgmlDeviceCount = 0;
pcclHgmlDeviceInfo pcclHgmlDevices[pcclHgmlMaxDevices];
pcclHgmlDevicePairInfo pcclHgmlDevicePairs[pcclHgmlMaxDevices][pcclHgmlMaxDevices];

#if PCCL_HGML_DIRECT
  #define PCCL_HGML_FN(name, rettype, arglist) constexpr rettype(*pfn_##name)arglist = name;
#else
  #include <dlfcn.h>
  #define PCCL_HGML_FN(name, rettype, arglist) rettype(*pfn_##name)arglist = nullptr;
#endif

namespace {
  PCCL_HGML_FN(hgmlInit, hgmlReturn_t, ())
  PCCL_HGML_FN(hgmlInit_v2, hgmlReturn_t, ())
  PCCL_HGML_FN(hgmlShutdown, hgmlReturn_t, ())
  PCCL_HGML_FN(hgmlDeviceGetCount, hgmlReturn_t, (int*))
  PCCL_HGML_FN(hgmlDeviceGetCount_v2, hgmlReturn_t, (int*))
  PCCL_HGML_FN(hgmlDeviceGetHandleByPciBusId, hgmlReturn_t, (const char* pciBusId, hgmlDevice_t* device))
  PCCL_HGML_FN(hgmlDeviceGetHandleByIndex, hgmlReturn_t, (unsigned int index, hgmlDevice_t *device))
  PCCL_HGML_FN(hgmlDeviceGetIndex, hgmlReturn_t, (hgmlDevice_t device, unsigned* index))
  PCCL_HGML_FN(hgmlErrorString, char const*, (hgmlReturn_t r))
  PCCL_HGML_FN(hgmlDeviceGetIcnLinkState, hgmlReturn_t, (hgmlDevice_t device, unsigned int link, hgmlEnableState_t *isActive))
  PCCL_HGML_FN(hgmlDeviceGetPciInfo, hgmlReturn_t, (hgmlDevice_t device, hgmlPciInfo_t *pci))
  PCCL_HGML_FN(hgmlDeviceGetIcnLinkRemotePciInfo, hgmlReturn_t, (hgmlDevice_t device, unsigned int link, hgmlPciInfo_t *pci))
  PCCL_HGML_FN(hgmlDeviceGetIcnLinkRemoteDeviceType, hgmlReturn_t, (hgmlDevice_t device, unsigned int link, hgmlIntIcnLinkDeviceType_t *pIcnLinkDeviceType))
  PCCL_HGML_FN(hgmlDeviceGetIcnLinkCapability, hgmlReturn_t, (hgmlDevice_t device, unsigned int link, hgmlIcnLinkCapability_t capability, unsigned int *capResult))
  PCCL_HGML_FN(hgmlDeviceGetHggcComputeCapability, hgmlReturn_t, (hgmlDevice_t device, int* major, int* minor))
  PCCL_HGML_FN(hgmlDeviceGetP2PStatus, hgmlReturn_t, (hgmlDevice_t device1, hgmlDevice_t device2, hgmlGpuP2PCapsIndex_t p2pIndex, hgmlGpuP2PStatus_t* p2pStatus))
  PCCL_HGML_FN(hgmlDeviceGetFieldValues, hgmlReturn_t, (hgmlDevice_t device, int valuesCount, hgmlFieldValue_t *values))
  PCCL_HGML_FN(hgmlDeviceGetName, hgmlReturn_t, (hgmlDevice_t device, char *name, unsigned int length))
  PCCL_HGML_FN(hgmlSystemGetDriverVersion, hgmlReturn_t, (char* version, unsigned int length))
  // MNNVL support
  PCCL_HGML_FN(hgmlDeviceGetGpuFabricInfo, hgmlReturn_t, (hgmlDevice_t device, hgmlGpuFabricInfoV_t *gpuFabricInfo))
  PCCL_HGML_FN(hgmlDeviceGetGpuFabricInfoV, hgmlReturn_t, (hgmlDevice_t device, hgmlGpuFabricInfoV_t *gpuFabricInfo))

  std::mutex lock; // NVML has had some thread safety bugs
  bool initialized = false;
  thread_local bool threadInitialized = false;
  ncclResult_t initResult;
}

ncclResult_t pcclHgmlEnsureInitialized() {
  // Optimization to avoid repeatedly grabbing the lock when we only want to
  // read from the global tables.
  if (threadInitialized) return initResult;
  threadInitialized = true;

  std::lock_guard<std::mutex> locked(lock);

  if (initialized) return initResult;
  initialized = true;

#if !PCCL_HGML_DIRECT
  if (pfn_hgmlInit == nullptr) {
    void *libhandle = dlopen("libhgml.so", RTLD_NOW);
    if (libhandle == nullptr) {
      WARN("Failed to open libhgml.so");
      initResult = ncclSystemError;
      return initResult;
    }

    struct Symbol { void **ppfn; char const *name; };
    std::initializer_list<Symbol> symbols = {
      {(void**)&pfn_hgmlInit, "hgmlInit"},
      {(void**)&pfn_hgmlInit_v2, "hgmlInit_v2"},
      {(void**)&pfn_hgmlShutdown, "hgmlShutdown"},
      {(void**)&pfn_hgmlDeviceGetCount, "hgmlDeviceGetCount"},
      {(void**)&pfn_hgmlDeviceGetCount_v2, "hgmlDeviceGetCount_v2"},
      {(void**)&pfn_hgmlDeviceGetHandleByPciBusId, "hgmlDeviceGetHandleByPciBusId"},
      {(void**)&pfn_hgmlDeviceGetHandleByIndex, "hgmlDeviceGetHandleByIndex"},
      {(void**)&pfn_hgmlDeviceGetIndex, "hgmlDeviceGetIndex"},
      {(void**)&pfn_hgmlErrorString, "hgmlErrorString"},
      {(void**)&pfn_hgmlDeviceGetIcnLinkState, "hgmlDeviceGetIcnLinkState"},
      {(void**)&pfn_hgmlDeviceGetPciInfo, "hgmlDeviceGetPciInfo"},
      {(void**)&pfn_hgmlDeviceGetIcnLinkRemotePciInfo, "hgmlDeviceGetIcnLinkRemotePciInfo"},
      {(void**)&pfn_hgmlDeviceGetIcnLinkRemoteDeviceType, "hgmlDeviceGetIcnLinkRemoteDeviceType"},
      {(void**)&pfn_hgmlDeviceGetIcnLinkCapability, "hgmlDeviceGetIcnLinkCapability"},
      {(void**)&pfn_hgmlDeviceGetHggcComputeCapability, "hgmlDeviceGetHggcComputeCapability"},
      {(void**)&pfn_hgmlDeviceGetP2PStatus, "hgmlDeviceGetP2PStatus"},
      {(void**)&pfn_hgmlDeviceGetFieldValues, "hgmlDeviceGetFieldValues"},
      {(void**)&pfn_hgmlDeviceGetName, "hgmlDeviceGetName"},
      {(void**)&pfn_hgmlSystemGetDriverVersion, "hgmlSystemGetDriverVersion"},
      // MNNVL support
      {(void**)&pfn_hgmlDeviceGetGpuFabricInfo, "hgmlDeviceGetGpuFabricInfo"},
      {(void**)&pfn_hgmlDeviceGetGpuFabricInfoV, "hgmlDeviceGetGpuFabricInfoV"},
    };
    for(Symbol sym: symbols) {
      *sym.ppfn = dlsym(libhandle, sym.name);
    }
    // Coverity complains that we never dlclose this object, but that's
    // deliberate, since we want the loaded object to remain in memory until
    // the process terminates, so that we can use its code.
    // coverity[leaked_storage]
  }
#endif

#if PCCL_HGML_DIRECT
    bool have_v2 = true;
#else
    bool have_v2 = pfn_hgmlInit_v2 != nullptr; // if this compare is done in the PCCL_HGML_DIRECT=1 case then GCC warns about it never being null
#endif
  hgmlReturn_t res1 = (have_v2 ? pfn_hgmlInit_v2 : pfn_hgmlInit)();
  if (res1 != HGML_SUCCESS) {
    WARN("hgmlInit%s() failed: %s", have_v2 ? "_v2" : "", pfn_hgmlErrorString(res1));
    initResult = ncclSystemError;
    return initResult;
  }

  int ndev;
  res1 = (have_v2 ? pfn_hgmlDeviceGetCount_v2 : pfn_hgmlDeviceGetCount)(&ndev);
  if (res1 != HGML_SUCCESS) {
    WARN("hgmlDeviceGetCount%s() failed: %s", have_v2 ? "_v2" :"", pfn_hgmlErrorString(res1));
    initResult = ncclSystemError;
    return initResult;
  }

  pcclHgmlDeviceCount = int(ndev);
  if (pcclHgmlMaxDevices < pcclHgmlDeviceCount) {
    WARN("hgmlDeviceGetCount() reported more devices (%d) than the internal maximum (pcclHgmlMaxDevices=%d)", pcclHgmlDeviceCount, pcclHgmlMaxDevices);
    initResult = ncclInternalError;
    return initResult;
  }

  for(int a=0; a < pcclHgmlDeviceCount; a++) {
    res1 = pfn_hgmlDeviceGetHandleByIndex(a, &pcclHgmlDevices[a].handle);
    if (res1 != HGML_SUCCESS) {
      WARN("hgmlDeviceGetHandleByIndex(%d) failed: %s", int(a), pfn_hgmlErrorString(res1));
      initResult = ncclSystemError;
      return initResult;
    }

    res1 = pfn_hgmlDeviceGetHggcComputeCapability(pcclHgmlDevices[a].handle, &pcclHgmlDevices[a].computeCapabilityMajor, &pcclHgmlDevices[a].computeCapabilityMinor);
    if (res1 != HGML_SUCCESS) {
      WARN("hgmlDeviceGetHggcComputeCapability(%d) failed: %s", int(a), pfn_hgmlErrorString(res1));
      initResult = ncclSystemError;
      return initResult;
    }
  }

  for(int a=0; a < pcclHgmlDeviceCount; a++) {
    for(int b=0; b < pcclHgmlDeviceCount; b++) {
      hgmlDevice_t da = pcclHgmlDevices[a].handle;
      hgmlDevice_t db = pcclHgmlDevices[b].handle;

      res1 = pfn_hgmlDeviceGetP2PStatus(da, db, HGML_P2P_CAPS_INDEX_READ, &pcclHgmlDevicePairs[a][b].p2pStatusRead);
      if (res1 != HGML_SUCCESS) {
        WARN("hgmlDeviceGetP2PStatus(%d,%d,HGML_P2P_CAPS_INDEX_READ) failed: %s", a, b, pfn_hgmlErrorString(res1));
        initResult = ncclSystemError;
        return initResult;
      }

      res1 = pfn_hgmlDeviceGetP2PStatus(da, db, HGML_P2P_CAPS_INDEX_WRITE, &pcclHgmlDevicePairs[a][b].p2pStatusWrite);
      if (res1 != HGML_SUCCESS) {
        WARN("hgmlDeviceGetP2PStatus(%d,%d,HGML_P2P_CAPS_INDEX_READ) failed: %s", a, b, pfn_hgmlErrorString(res1));
        initResult = ncclSystemError;
        return initResult;
      }
    }
  }

  initResult = ncclSuccess;
  return initResult;
}


#define NVMLCHECK(name, ...) do { \
  hgmlReturn_t e44241808 = pfn_##name(__VA_ARGS__); \
  if (e44241808 != HGML_SUCCESS) { \
    WARN(#name "() failed: '%s'", pfn_hgmlErrorString(e44241808)); \
    return ncclSystemError; \
  } \
} while(0)

#define NVMLTRY(name, ...) do { \
  if (!PCCL_HGML_DIRECT && pfn_##name == nullptr) \
    return ncclInternalError; /* missing symbol is not a warned error */ \
  hgmlReturn_t e44241808 = pfn_##name(__VA_ARGS__); \
  if (e44241808 != HGML_SUCCESS) { \
    if (e44241808 != HGML_ERROR_NOT_SUPPORTED) \
      INFO(NCCL_INIT, #name "() failed: '%s'", pfn_hgmlErrorString(e44241808)); \
    return ncclSystemError; \
  } \
} while(0)

ncclResult_t pcclHgmlShutdown(void) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlShutdown);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetHandleByPciBusId(const char* pciBusId, hgmlDevice_t* device) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLCHECK(hgmlDeviceGetHandleByPciBusId, pciBusId, device);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetHandleByIndex(unsigned int index, hgmlDevice_t *device) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  *device = pcclHgmlDevices[index].handle;
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetIndex(hgmlDevice_t device, unsigned* index) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  for (int d=0; d < pcclHgmlDeviceCount; d++) {
    if (pcclHgmlDevices[d].handle == device) {
      *index = d;
      return ncclSuccess;
    }
  }
  return ncclInvalidArgument;
}

ncclResult_t pcclHgmlDeviceGetIcnLinkState(hgmlDevice_t device, unsigned int link, hgmlEnableState_t *isActive) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetIcnLinkState, device, link, isActive);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetPciInfo(hgmlDevice_t device, hgmlPciInfo_t *pci) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetPciInfo, device, pci);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetIcnLinkRemotePciInfo(hgmlDevice_t device, unsigned int link, hgmlPciInfo_t *pci) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetIcnLinkRemotePciInfo, device, link, pci);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetIcnLinkRemoteDeviceType(hgmlDevice_t device, unsigned int link, hgmlIntIcnLinkDeviceType_t *pIcnLinkDeviceType) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetIcnLinkRemoteDeviceType, device, link, pIcnLinkDeviceType);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetIcnLinkCapability(
    hgmlDevice_t device, unsigned int link, hgmlIcnLinkCapability_t capability,
    unsigned int *capResult
  ) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetIcnLinkCapability, device, link, capability, capResult);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetHggcComputeCapability(hgmlDevice_t device, int* major, int* minor) {
  NCCLCHECK(pcclHgmlEnsureInitialized());

  for(int d=0; d < pcclHgmlDeviceCount; d++) {
    if(device == pcclHgmlDevices[d].handle) {
      *major = pcclHgmlDevices[d].computeCapabilityMajor;
      *minor = pcclHgmlDevices[d].computeCapabilityMinor;
      return ncclSuccess;
    }
  }
  return ncclInvalidArgument;
}

ncclResult_t pcclHgmlDeviceGetFieldValues(hgmlDevice_t device, int valuesCount, hgmlFieldValue_t *values) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetFieldValues, device, valuesCount, values);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetCount(int* count) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetCount, count);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetName(hgmlDevice_t device, char *name, unsigned int length) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetName, device, name, length);
  return ncclSuccess;
}

ncclResult_t pcclHgmlSystemGetDriverVersion(char *version, unsigned int length) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlSystemGetDriverVersion, version, length);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetP2PStatus(
    hgmlDevice_t device1, hgmlDevice_t device2, hgmlGpuP2PCapsIndex_t p2pIndex,
    hgmlGpuP2PStatus_t* p2pStatus
  ) {
  NCCLCHECK(pcclHgmlEnsureInitialized());

  if (p2pIndex == HGML_P2P_CAPS_INDEX_READ || p2pIndex == HGML_P2P_CAPS_INDEX_WRITE) {
    int a = -1, b = -1;
    for(int d=0; d < pcclHgmlDeviceCount; d++) {
      if(device1 == pcclHgmlDevices[d].handle) a = d;
      if(device2 == pcclHgmlDevices[d].handle) b = d;
    }
    if (a == -1 || b == -1) return ncclInvalidArgument;
    if (p2pIndex == HGML_P2P_CAPS_INDEX_READ)
      *p2pStatus = pcclHgmlDevicePairs[a][b].p2pStatusRead;
    else
      *p2pStatus = pcclHgmlDevicePairs[a][b].p2pStatusWrite;
  }
  else {
    std::lock_guard<std::mutex> locked(lock);
    NVMLCHECK(hgmlDeviceGetP2PStatus, device1, device2, p2pIndex, p2pStatus);
  }
  return ncclSuccess;
}

// MNNVL support
ncclResult_t pcclHgmlDeviceGetGpuFabricInfo(hgmlDevice_t device, hgmlGpuFabricInfoV_t *gpuFabricInfo) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetGpuFabricInfo, device, gpuFabricInfo);
  return ncclSuccess;
}

ncclResult_t pcclHgmlDeviceGetGpuFabricInfoV(hgmlDevice_t device, hgmlGpuFabricInfoV_t *gpuFabricInfo) {
  NCCLCHECK(pcclHgmlEnsureInitialized());
  std::lock_guard<std::mutex> locked(lock);
  NVMLTRY(hgmlDeviceGetGpuFabricInfoV, device, gpuFabricInfo);
  return ncclSuccess;
}
