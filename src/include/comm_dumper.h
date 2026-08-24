/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_COMM_DUMPER_H_
#define PCCL_COMM_DUMPER_H_
#include "pccl_internal.h"
#include "drv_extension.h"
#include "hggc_runtime.h"

#define COMM_DUMP_MASK_BITS 4
typedef enum {
  DUMP_NONE = 0x0,
  DUMP_ROUGH = 0x1,
  DUMP_RAW = 0x2,
  DUMP_BUFFER = 0x4,
  DUMP_HWINFO = 0x8
} pcclCommDumpMaskType;

// Optional sink for diag-tier hot values produced as a byproduct of the dump
// path.  Lets RAS hang-check piggyback on the dump's already-issued device
// reads (per-channel kernel indices and per-peer head/tail/devStep) instead
// of running a second device-IO pass.  Any callback may be null.  Invocation
// order per dumped comm: initRankDiagHeader -> for each channel { collectChannelState -> for
// each connected (peer, connector, direction) { collectPeerConn } } -> finalizeRankDiagHeader.
//
// collectChannelState arguments:
//  - devHangReportedKernelIdx / devHangReportedKernelIdxValid: value of
//    hostDevIndexLevel1, the kernelIdx the kernel itself wrote when checkAbort
//    detected a suspected hang.  Sticky; never written = 0 (and Valid=true);
//    Valid=false means the device read failed.
//  - devCompletedKernelIdx / devCompletedKernelIdxValid: value of
//    devIndexLevel1 (stateMonitorLevel==1) or hostDevIndexLevel2 (==2), the
//    kernelIdx of the most recently completed kernel.  0 with Valid=true means
//    no kernel has FINI'd yet on this channel; Valid=false means the read
//    failed.  Each *Valid flag tracks success of its own read independently.
struct pcclCommDiagCollector {
  void* ctx;
  void (*initRankDiagHeader)(void* ctx, ncclComm_t comm);
  void (*collectChannelState)(void* ctx, ncclComm_t comm, int channelId,
                    uint64_t devHangReportedKernelIdx, bool devHangReportedKernelIdxValid,
                    uint64_t devCompletedKernelIdx,    bool devCompletedKernelIdxValid);
  void (*collectPeerConn)(void* ctx, ncclComm_t comm, int channelId, int peerRank,
                 uint8_t connector, uint8_t direction,
                 uint64_t head, uint64_t tail, uint64_t devStep);
  void (*finalizeRankDiagHeader)(void* ctx, ncclComm_t comm);
};

ncclResult_t setCommDumpMaskByStr(const char* usrMaskStr, pcclCommDumpMaskType* dumpMask);
ncclResult_t setCommDumpMaskByVal(const uint32_t usrMaskStr, pcclCommDumpMaskType* dumpMask);

///////////////////////////////////////////////////////////////////////////////
// Device-memory read helpers (shared by comm_dumper.cc and diag.cc).        //
///////////////////////////////////////////////////////////////////////////////

// Check pointer accessibility: *lockablePtr = 0 (unlockable), 1 (lockable device), 2 (host).
static inline ncclResult_t checkPtrLockable(const void* devPtr, int* lockablePtr) {
  *lockablePtr = 0;
  if (devPtr == NULL) return ncclSuccess;
  hggcPointerAttributes ptrAttr;
  if (hggcPointerGetAttributes(&ptrAttr, devPtr) != hggcSuccess) return ncclSuccess;
  if (ptrAttr.type == hggcMemoryTypeHost) { *lockablePtr = 2; return ncclSuccess; }
  if (ptrAttr.type != hggcMemoryTypeDevice) return ncclSuccess;

  hggcEitInterface& hggcEit = hggcEitInterface::getInstance();
  HGpointerIpc_attribute ipcAttr;
  if (hggcEit.wrapHggcPointerGetIpcAttribute((HGdeviceptr)devPtr, &ipcAttr) != hggcSuccess)
    return ncclUnhandledHggcError;
  if (ipcAttr == HG_POINTER_IPC_ATTRIBUTE_IMPORTED) return ncclSuccess;
  *lockablePtr = 1;
  return ncclSuccess;
}

// Best-effort single-u64 device read via MemLock; returns 0 on any failure.
static inline uint64_t safeReadDevU64(uint64_t* devPtr) {
  if (devPtr == nullptr) return 0;
  hggcEitInterface& hggcEit = hggcEitInterface::getInstance();
  if (!hggcEit.checkExtTable()) return 0;

  int lockable = 0;
  if (checkPtrLockable(devPtr, &lockable) != ncclSuccess || lockable == 0) return 0;
  if (lockable == 2) return *devPtr;

  void* cpuPtr = nullptr;
  if (hggcEit.wrapHggcMemLock((HGdeviceptr)devPtr, &cpuPtr) != hggcSuccess) return 0;
  uint64_t val = (cpuPtr ? *(uint64_t*)cpuPtr : 0);
  hggcEit.wrapHggcMemUnlock((HGdeviceptr)devPtr);
  return val;
}

// Device-memory u64 read via MemDump with validity flag (full u64 range is legitimate).
static inline uint64_t readDevU64(uint64_t* devPtr, bool* outValid) {
  *outValid = false;
  if (devPtr == nullptr) return 0;
  hggcEitInterface& hggcEit = hggcEitInterface::getInstance();
  if (!hggcEit.checkExtTable()) return 0;
  uint64_t idx = 0;
  if (hggcEit.wrapHggcMemDump(&idx, (HGdeviceptr)devPtr, sizeof(uint64_t)) != hggcSuccess) return 0;
  *outValid = true;
  return idx;
}

#endif  // PCCL_COMM_DUMPER_H_
