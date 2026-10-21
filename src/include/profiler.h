/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2023-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PROFILER_H_
#define PROFILER_H_

#include <hggc_runtime.h>
#include "pccl_profiler.h"

struct ncclProxyArgs;
struct ncclKernelPlan;
struct ncclTaskColl;
struct ncclTaskP2p;
struct ncclInfo;
struct pcclComm;
struct ncclProxyOp;
struct ncclProxyConnector;

/* Below is for Asys pccl profiler */

enum ncclProxyProfileStateAsys {
  ncclProxyProfileBegin = 0,

  ncclProxyProfileSendGPUWait = 1,
  ncclProxyProfileSendWait = 2,

  ncclProxyProfileRecvWait = 1,
  ncclProxyProfileRecvFlushWait = 2,
  ncclProxyProfileRecvGPUWait = 3,

  ncclProxyProfileEnd = 4,

  ncclProxyProfileSleep = 8,
  ncclProxyProfileWakeup = 9,

  ncclProxyProfileIdle = 16,
  ncclProxyProfileActive = 17,

  ncclProxyProfileAppend = 24,
  ncclProxyProfileAppendEnd = 25,

  ncclProxyProfileStateNum = 32
};

enum profilerType {
  PROFILER_NONE = 0,
  PROFILER_ASYS = 2
};

extern ncclResult_t (*pcclAsysProxyProfilingRecord)(struct ncclProxyArgs* args, int sub, int step, int state);

// init profiler type
void profilerInit(int type);

/* End of Asys pccl proxyprofiler */

struct ncclProfilerProxy {
  bool initialized;
  struct ncclDevProfiler* workStarted/*[MAXCHANNELS]*/;
  struct ncclDevProfiler* workCompleted/*[MAXCHANNELS]*/;
  uint64_t workCounter[MAXCHANNELS]; // host work counter
  struct ncclProxyConnector sendProxyConn[MAXCHANNELS];
  struct ncclProxyConnector recvProxyConn[MAXCHANNELS];
};

enum groupApiState {
  ncclProfilerGroupApiStartStateReset   = 0,
  ncclProfilerGroupApiStartStateStarted = 1,
  ncclProfilerGroupApiStartStateStopped = 2,
};

// Used by the profiler to track state for API events
typedef struct ncclProfilerApiState {
  int profilerGroupDepth;
  int eActivationMask;
  groupApiState state;
  void *groupApiEventHandle;
  // Tracks the latest API event handles for p2p/collectives/one-sided RMA
  void* p2pApiEventHandle;
  void *collApiEventHandle;
  void *rmaApiEventHandle;
} ncclProfilerApiState_t;

extern thread_local ncclProfilerApiState_t ncclProfilerApiState;

extern int ncclProfilerEventMask;

// To be moved to collectives.h (refer to NCCL 2.22.3).
const char* ncclFuncToString(ncclFunc_t op);
const char* ncclDatatypeToString(ncclDataType_t type);
const char* ncclAlgoToString(int algo);
const char* ncclProtoToString(int proto);

// Plugin Init/Finalize Wrappers
ncclResult_t ncclProfilerPluginInit(ncclComm_t comm);
ncclResult_t ncclProfilerPluginFinalize(ncclComm_t comm);

// Profiler Start/Stop/Record wrappers for ncclGroupStart and ncclGroupEnd API calls
ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo *info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopGroupApiEvent();
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState);

//Profiler Start/Stop wrappers for P2p API calls
ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo *info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopP2pApiEvent();

//Profiler Start/Stop wrappers for Collective API calls
ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo *info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopCollApiEvent();

//Profiler Start/Stop wrappers for One-sided RMA API calls
ncclResult_t ncclProfilerStartRmaApiEvent(struct ncclInfo *info, bool isGraphCaptured);
ncclResult_t ncclProfilerStopRmaApiEvent();

// Kernel Launch Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, hggcStream_t stream);
ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan);

// Profiler Start/Stop Group Wrappers
ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan);
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan);

// Profiler Start/Stop Task Events Wrappers
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan);
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan);

// Proxy Op Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartProxyOpEvent(int sub, struct ncclProxyArgs* args);
ncclResult_t ncclProfilerStopProxyOpEvent(int sub, struct ncclProxyArgs* args);

// Proxy Step Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartSendProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStartRecvProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);
ncclResult_t ncclProfilerStopProxyStepEvent(int sub, struct ncclProxyArgs* args, int stepId);

// Proxy Control Start/Stop Events Wrappers
ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle);
ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle);

// Kernel Channel Start/Stop Event Wrappers
ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t start);
ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t stop);

// Record Event Wrappers
ncclResult_t ncclProfilerRecordProxyOpEventState(int sub, struct ncclProxyArgs* args, ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyStepEventState(int sub, struct ncclProxyArgs* args, int stepId, ncclProfilerEventState_t eState);
ncclResult_t ncclProfilerRecordProxyCtrlEventState(void* eHandle, int appended, ncclProfilerEventState_t eState);

// Profiler utility functions
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op);
bool ncclProfilerNeedsProxy(ncclComm_t comm, struct ncclProxyOp* op);
bool ncclProfilerPluginLoaded(void);


// Profiler callback for network plugin
ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData);

// Profiler print events wrapper
typedef ncclResult_t (*ncclProfilerPrintEventsFn_t)(void* context, const char* path);
ncclResult_t ncclProfilerPrintEvent(ncclComm_t comm, const char* path);

// Forward declarations for CE types
struct ncclCeCollArgs;
struct ncclCeBatchOpsParams;

// CE profiler event start/stop functions (simple wrappers that call plugin callbacks)
ncclResult_t ncclProfilerStartCeCollEvent(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);
ncclResult_t ncclProfilerStopCeCollEvent(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream);
ncclResult_t ncclProfilerStartCeSyncEvent(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream, void** ceSyncHandle);
ncclResult_t ncclProfilerStopCeSyncEvent(ncclComm_t comm, void* ceSyncHandle, hggcStream_t stream);
ncclResult_t ncclProfilerStartCeBatchEvent(ncclComm_t comm, struct ncclCeCollArgs* args, const struct ncclCeBatchOpsParams* params, hggcStream_t stream, void** ceBatchHandle);
ncclResult_t ncclProfilerStopCeBatchEvent(ncclComm_t comm, void* ceBatchHandle, hggcStream_t stream);


#endif
