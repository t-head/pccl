/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "param.h"
#include "checks.h"
#include "comm.h"
#include "enqueue.h"
#include "utils.h"
#include "proxy.h"
#include "profiler.h"
#include "transport.h"
#include "plugin.h"
#include <mutex>

extern ncclProfiler_t* getNcclProfiler_v5(void* lib);
extern ncclProfiler_t* getNcclProfiler_v6(void* lib);

static std::mutex profilerMutex;
static int profilerPluginRefCount;
static void* profilerPluginLib;
static ncclProfiler_t* ncclProfiler;
static ncclProfilerPrintEventsFn_t printEventFn = nullptr;

extern thread_local int ncclGroupDepth;
thread_local ncclProfilerApiState_t ncclProfilerApiState;

#define MAX_STR_LEN 256

enum {
  profilerPluginLoadFailed = -1,
  profilerPluginLoadReady = 0,
  profilerPluginLoadSuccess = 1,
};
static int profilerPluginStatus = profilerPluginLoadReady;
static pid_t pid;

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncAlltoAll: return "AlltoAll";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncGather: return "Gather";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncScatter: return "Scatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  default: return "Invalid";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: return "pcclInt8";
  case ncclInt32: return "pcclInt32";
  case ncclUint32: return "pcclUint32";
  case ncclInt64: return "pcclInt64";
  case ncclUint64: return "pcclUint64";
  case ncclFloat16: return "pcclFloat16";
  case ncclFloat32: return "pcclFloat32";
  case ncclFloat64: return "pcclFloat64";
  case ncclBfloat16: return "pcclBfloat16";
  case ncclFloat8e4m3: return "pcclFloat8e4m3";
  case ncclFloat8e5m2: return "pcclFloat8e5m2";
  default: return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET: return "COLLNET";
  case NCCL_ALGO_FC: return "FC";
  default: return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  case NCCL_PROTO_SIGNAL: return "SIGNAL";
  case NCCL_PROTO_SIMPLE_BULK: return "SIMPLE_BULK";
  case NCCL_PROTO_SIGNAL_BULK: return "SIGNAL_BULK";
  case NCCL_PROTO_SIMPLE_BULK_AR: return "SIMPLE_BULK_AR";
  default: return "Unknown";
  }
}

static ncclResult_t ncclProfilerPluginLoad(void) {
  const char* profilerName;
  if (profilerPluginLoadFailed == profilerPluginStatus) {
    return ncclSuccess;
  }

  std::lock_guard<std::mutex> lock(profilerMutex);
  if (profilerPluginLoadSuccess == profilerPluginStatus) {
    ++profilerPluginRefCount;
    goto exit;
  }

  if ((profilerName = ncclGetEnv("PCCL_PROFILER_PLUGIN")) != nullptr) {
    INFO(NCCL_ENV, "PCCL_PROFILER_PLUGIN set by environment to %s", profilerName);
    if (strcasecmp(profilerName, "none") == 0)
      goto fail;
  }
  profilerPluginLib = ncclOpenProfilerPluginLib(profilerName);
  if (profilerPluginLib == nullptr) {
    profilerPluginLib = ncclGetNetPluginLib(ncclPluginTypeProfiler);
    if (nullptr == profilerPluginLib) {
      goto fail;
    }
    profilerName = nullptr;
  } else if (ncclPluginLibPaths[ncclPluginTypeProfiler]) {
    profilerName = ncclPluginLibPaths[ncclPluginTypeProfiler];
  }

  ncclProfiler = getNcclProfiler_v6(profilerPluginLib);
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v5(profilerPluginLib);
  }
  if (ncclProfiler == NULL) {
    if (profilerName) INFO(NCCL_INIT, "External profiler plugin %s is unsupported", profilerName);
    goto fail;
  }
  if (profilerName) INFO(NCCL_INIT, "Successfully loaded external profiler plugin %s", profilerName);

  printEventFn = (ncclProfilerPrintEventsFn_t)dlsym(profilerPluginLib, "exampleProfilerPrintEvent");

  ++profilerPluginRefCount;
  profilerPluginStatus = profilerPluginLoadSuccess;

  // Store the pid of the process loading the profiler.
  // This is attached to the proxyOp event descriptor
  // so the plugin can figure out if the parent event
  // is in the same address space or not
  pid = getpid();

exit:
  return ncclSuccess;
fail:
  if (profilerPluginLib) NCCLCHECK(ncclClosePluginLib(profilerPluginLib, ncclPluginTypeProfiler));
  profilerPluginLib = nullptr;
  profilerPluginStatus = profilerPluginLoadFailed;
  goto exit;
}

static ncclResult_t ncclProfilerPluginUnload(void) {
  std::lock_guard<std::mutex> lock(profilerMutex);
  if (0 == (--profilerPluginRefCount)) {
    if (__builtin_expect(ncclProfiler != NULL, 0)) {
      INFO(NCCL_INIT, "PROFILER/Plugin: Closing profiler plugin %s", ncclProfiler->name);
    }
    NCCLCHECK(ncclClosePluginLib(profilerPluginLib, ncclPluginTypeProfiler));
    profilerPluginLib = nullptr;
    ncclProfiler = nullptr;
    profilerPluginStatus = profilerPluginLoadReady;
  }
  return ncclSuccess;
}

int ncclProfilerEventMask;       // Set by profiler

// Print enabled profiler event types
static void printProfilerEventMask(int mask) {
  if (!mask) return;

  char enabled[512] = {0};
  int pos = 0;
  if (mask & ncclProfileGroup)        pos += sprintf(enabled + pos, "Group ");
  if (mask & ncclProfileColl)         pos += sprintf(enabled + pos, "Coll ");
  if (mask & ncclProfileP2p)          pos += sprintf(enabled + pos, "P2p ");
  if (mask & ncclProfileProxyOp)      pos += sprintf(enabled + pos, "ProxyOp ");
  if (mask & ncclProfileProxyStep)    pos += sprintf(enabled + pos, "ProxyStep ");
  if (mask & ncclProfileProxyCtrl)    pos += sprintf(enabled + pos, "ProxyCtrl ");
  if (mask & ncclProfileKernelCh)     pos += sprintf(enabled + pos, "KernelCh ");
  if (mask & ncclProfileNetPlugin)    pos += sprintf(enabled + pos, "NetPlugin ");
  if (mask & ncclProfileGroupApi)     pos += sprintf(enabled + pos, "GroupApi ");
  if (mask & ncclProfileCollApi)      pos += sprintf(enabled + pos, "CollApi ");
  if (mask & ncclProfileP2pApi)       pos += sprintf(enabled + pos, "P2pApi ");
  if (mask & ncclProfileKernelLaunch) pos += sprintf(enabled + pos, "KernelLaunch ");
  if (mask & ncclProfileCeColl)       pos += sprintf(enabled + pos, "CeColl ");
  if (mask & ncclProfileCeSync)       pos += sprintf(enabled + pos, "CeSync ");
  if (mask & ncclProfileCeBatch)      pos += sprintf(enabled + pos, "CeBatch ");
  INFO(NCCL_INIT, "Profiler event mask: 0x%x (%d) - Enabled: %s", mask, mask, enabled);
}

ncclResult_t ncclProfilerPluginInit(ncclComm_t comm) {
  ncclProfilerPluginLoad();
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    int err = ncclProfiler->init(&comm->profilerContext, comm->commHash, &ncclProfilerEventMask, comm->config.commName, comm->nNodes, comm->nRanks, comm->rank, ncclDebugLogForPlugin);
    if (err) {
      ncclProfilerPluginUnload();
      INFO(NCCL_INIT, "Profiler init failed with error '%d': %s. Continue without profiler.", err, strerror(errno));
    }

    printProfilerEventMask(ncclProfilerEventMask);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerPluginFinalize(ncclComm_t comm) {
  if (__builtin_expect(ncclProfiler != NULL, 0) && comm->profilerContext) {
    ncclProfiler->finalize(comm->profilerContext);
  }
  ncclProfilerPluginUnload();
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = { 0 };
  eDescr.type = ncclProfileGroupApi;
  eDescr.groupApi.graphCaptured = isGraphCaptured;

  ncclProfilerApiState.eActivationMask = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED);
  int groupApiMask = ncclProfileGroupApi | ncclProfileP2pApi | ncclProfileCollApi | ncclProfileKernelLaunch
          | ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep
          | ncclProfileKernelCh | ncclProfileNetPlugin | ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
  // Only count outermost groups when emitting group API events
  if (__builtin_expect(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & groupApiMask)) {
    if (ncclProfilerApiState.profilerGroupDepth == 0) {
      eDescr.groupApi.groupDepth = ncclGroupDepth;
      ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.groupApiEventHandle, &eDescr);
      ncclProfilerApiState.profilerGroupDepth = ncclGroupDepth;
      ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStarted;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupApiEvent() {
  void* groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  if (__builtin_expect(ncclProfiler != NULL, 0) && groupApiEventHandle && ncclProfilerApiState.profilerGroupDepth == 0) {
    ncclProfiler->stopEvent(groupApiEventHandle);
    ncclProfilerApiState.groupApiEventHandle = nullptr;
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState) {
  void* groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  bool shouldRecord = false;
  if (eState == ncclProfilerGroupStartApiStop && ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStarted) {
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStopped;
    shouldRecord = true;
  } else if (eState == ncclProfilerGroupEndApiStart && ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStopped) {
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateReset;
    shouldRecord = true;
  }

  if (__builtin_expect(ncclProfiler != NULL, 0) && groupApiEventHandle && shouldRecord) {
    ncclProfiler->recordEventState(groupApiEventHandle, eState, NULL);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo *info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = { 0 };
  eDescr.type = ncclProfileP2pApi;
  eDescr.parentObj = ncclProfilerApiState.groupApiEventHandle;
  eDescr.p2pApi.func = ncclFuncToString(info->coll);
  eDescr.p2pApi.count = info->count;
  eDescr.p2pApi.datatype = ncclDatatypeToString(info->datatype);
  eDescr.p2pApi.stream = (void *) info->stream;
  eDescr.p2pApi.graphCaptured = isGraphCaptured;
  int p2pApiMask = ncclProfileP2pApi | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin;
  if (__builtin_expect(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & p2pApiMask)) {
    ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.p2pApiEventHandle, &eDescr);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopP2pApiEvent() {
  if (__builtin_expect(ncclProfiler != NULL, 0) && ncclProfilerApiState.p2pApiEventHandle) {
    ncclProfiler->stopEvent(ncclProfilerApiState.p2pApiEventHandle);
    ncclProfilerApiState.p2pApiEventHandle = nullptr;
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo *info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = { 0 };
  eDescr.type = ncclProfileCollApi;
  eDescr.parentObj = ncclProfilerApiState.groupApiEventHandle;
  eDescr.collApi.func = ncclFuncToString(info->coll);
  eDescr.collApi.count = info->count;
  eDescr.collApi.datatype = ncclDatatypeToString(info->datatype);
  eDescr.collApi.stream = (void *) info->stream;
  eDescr.collApi.root = info->root;
  eDescr.collApi.graphCaptured = isGraphCaptured;
  int collApiMask = ncclProfileCollApi | ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin
          | ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
  if (__builtin_expect(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & collApiMask)) {
    ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.collApiEventHandle, &eDescr);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCollApiEvent() {
  if (__builtin_expect(ncclProfiler != NULL, 0) && ncclProfilerApiState.collApiEventHandle) {
    ncclProfiler->stopEvent(ncclProfilerApiState.collApiEventHandle);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, hggcStream_t stream) {
  ncclProfilerEventDescr_t eDescr = { 0 };
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    void* groupApiEventHandle = NULL;
    // Check if any collective in the plan has a set event activation mask
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    int eActivationMask_ = 0;
    while (ct) {
      if (ct->eActivationMask) {
        eActivationMask_ = ct->eActivationMask;
        groupApiEventHandle = ct->groupApiEventHandle;
        goto startKernelLaunchEvent;
      }
      ct = ct->next;
    }
    // Check if any pt2pt in the plan has a set event activation mask
    while (pt) {
      if (pt->eActivationMask) {
        eActivationMask_ = pt->eActivationMask;
        groupApiEventHandle = pt->groupApiEventHandle;
        goto startKernelLaunchEvent;
      }
      pt = pt->next;
    }

  startKernelLaunchEvent:
    if (eActivationMask_ & ncclProfileKernelLaunch) {
      eDescr.type = ncclProfileKernelLaunch;
      eDescr.parentObj = groupApiEventHandle;
      eDescr.kernelLaunch.stream = (void *) stream;
      ncclProfiler->startEvent(plan->comm->profilerContext, &plan->kernelLaunchEventHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan) {
  if (__builtin_expect(ncclProfiler != NULL, 0) && plan->kernelLaunchEventHandle) {
    ncclProfiler->stopEvent(plan->kernelLaunchEventHandle);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    // Check if any collective in the plan has a set event activation mask
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    int eActivationMask_ = 0;
    while (ct) {
      if (ct->eActivationMask) {
        eActivationMask_ = ct->eActivationMask;
        goto startGroup;
      }
      ct = ct->next;
    }
    // Check if any pt2pt in the plan has a set event activation mask
    while (pt) {
      if (pt->eActivationMask) {
        eActivationMask_ = pt->eActivationMask;
        goto startGroup;
      }
      pt = pt->next;
    }

  startGroup:
    if (eActivationMask_ & (ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileGroup;
      ncclProfiler->startEvent(plan->comm->profilerContext, &plan->groupEventHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  if (__builtin_expect(ncclProfiler != NULL, 0) && plan->groupEventHandle) {
    ncclProfiler->stopEvent(plan->groupEventHandle);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct) {
    if (__builtin_expect(ncclProfiler != NULL, 0)) {
      int enable = ct->eActivationMask & (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin);
      if (enable) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileColl;
        eDescr.coll.parentGroup = plan->groupEventHandle;
        eDescr.parentObj = ct->collApiEventHandle;
        eDescr.rank = plan->comm->rank;
        eDescr.coll.seqNumber = plan->comm->seqNumber[ct->func];
        eDescr.coll.func = ncclFuncToString(ct->func);
        eDescr.coll.sendBuff = ct->sendbuff;
        eDescr.coll.recvBuff = ct->recvbuff;
        eDescr.coll.count = ct->count;
        eDescr.coll.root = ct->root;
        eDescr.coll.datatype = ncclDatatypeToString(ct->datatype);
        eDescr.coll.nChannels = ct->nChannels;
        eDescr.coll.algo = ncclAlgoToString(ct->algorithm);
        eDescr.coll.proto = ncclProtoToString(ct->protocol);
        ncclProfiler->startEvent(plan->comm->profilerContext, &ct->eventHandle, &eDescr);
      }
    }
    // comm->seqNumber values are updated even if the plugin is not active, since they are used by RAS as well.
    // The test for "persistent" is a workaround for graph-captured collectives.  In their case this function may not be
    // consistently invoked on all the ranks, which would lead to mismatched counter values and thus false-positive
    // reports from RAS.  Instead, we choose not to include graph-captured collectives in our counts.  An exception is
    // made if ncclProfileKernelCh profiler events are active, as they result in proxy events always being added, which
    // gives the consistency.
    if (!plan->persistent || (__builtin_expect(ncclProfiler != NULL, 0) && (plan->groupEventHandle || ct->collApiEventHandle) &&
                              (ct->eActivationMask & ncclProfileKernelCh)))
      __atomic_fetch_add(&plan->comm->seqNumber[ct->func], 1, __ATOMIC_RELAXED);
    ct = ct->next;
  }
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    while (pt) {
      int enable = pt->eActivationMask & (ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin);
      if (enable) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileP2p;
        eDescr.p2p.parentGroup = plan->groupEventHandle;
        eDescr.parentObj = pt->p2pApiEventHandle;
        eDescr.rank = plan->comm->rank;
        eDescr.p2p.func = ncclFuncToString(pt->func);
        eDescr.p2p.buff = pt->buff;
        eDescr.p2p.count = pt->count;
        eDescr.p2p.datatype = ncclDatatypeToString(pt->datatype);
        eDescr.p2p.peer = pt->root;
        eDescr.p2p.nChannels = pt->nChannels;
        ncclProfiler->startEvent(plan->comm->profilerContext, &pt->eventHandle, &eDescr);
      }
      pt = pt->next;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    while (ct) {
      if (ct->eventHandle) ncclProfiler->stopEvent(ct->eventHandle);
      ct = ct->next;
    }
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    while (pt) {
      if (pt->eventHandle) ncclProfiler->stopEvent(pt->eventHandle);
      pt = pt->next;
    }
  }
  return ncclSuccess;
}

// Bellow we set the proxy descriptor step number to DIVUP(step, args->sliceSteps).
// The reason is that for some ncclOp (e.g. AllReduce) one network transfer is
// made of sliceSteps steps rather than one step. In the profiler we are still
// interested in whole network transfers though, so we account for this when
// computing the actual network step number.
ncclResult_t ncclProfilerStartProxyOpEvent(int s, struct ncclProxyArgs* args) {
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (sub->eActivationMask & (ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyOp;
      eDescr.parentObj = sub->taskEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyOp.pid = sub->pid;
      eDescr.proxyOp.channelId = sub->channelId;
      eDescr.proxyOp.peer = sub->peer;
      eDescr.proxyOp.nSteps = DIVUP(sub->nsteps, args->sliceSteps);
      eDescr.proxyOp.chunkSize = args->chunkSize * args->sliceSteps;
      eDescr.proxyOp.isSend = args->progress == ncclTransports[TRANSPORT_NET]->send.proxyProgress ? 1 : 0;
      ncclProfiler->startEvent(sub->profilerContext, &sub->opEventHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyOpEvent(int s, struct ncclProxyArgs* args) {
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    ncclProfiler->stopEvent(sub->opEventHandle);
    sub->opEventHandle = NULL;
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartSendProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  struct ncclProxySubArgs* sub = &args->subs[s];
  int step_ = DIVUP(stepId, args->sliceSteps);
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyStep;
      eDescr.parentObj = sub->opEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyStep.step = step_;
      ncclProfiler->startEvent(sub->profilerContext, &sub->pHandles[step_%NCCL_STEPS].stepEventHandle, &eDescr);
    }
  }
  sub->pHandles[step_%NCCL_STEPS].subArgPtr = sub;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartRecvProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  struct ncclProxySubArgs* sub = &args->subs[s];
  int step_ = DIVUP(stepId, args->sliceSteps);
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyStep;
      eDescr.parentObj = sub->opEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyStep.step = step_;
      ncclProfiler->startEvent(sub->profilerContext, &sub->pHandles[step_%NCCL_STEPS].stepEventHandle, &eDescr);
    }
  }
  sub->pHandles[step_%NCCL_STEPS].subArgPtr = sub;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    int step_ = DIVUP(stepId, args->sliceSteps);
    if (sub->pHandles[step_%NCCL_STEPS].stepEventHandle) {
      ncclProfiler->stopEvent(sub->pHandles[step_%NCCL_STEPS].stepEventHandle);
      sub->pHandles[step_%NCCL_STEPS].stepEventHandle = NULL;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    // for proxy control events we allow profiling mode to change on a per event basis
    int eActivationMaskProxy = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED);
    if (eActivationMaskProxy & ncclProfileProxyCtrl) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileProxyCtrl;
      ncclProfiler->startEvent(profilerContext, eHandle, &eDescr);
      return ncclSuccess;
    }
  }
  *eHandle = NULL;
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle) {
  if (__builtin_expect(ncclProfiler != NULL, 0) && eHandle) {
    ncclProfiler->stopEvent(eHandle);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t start) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    struct ncclProxySubArgs* sub = &args->subs[s];
    if (sub->eActivationMask & ncclProfileKernelCh) {
      ncclProfilerEventDescr_t eDescr = { };
      eDescr.type = ncclProfileKernelCh;
      eDescr.parentObj = sub->taskEventHandle;
      eDescr.kernelCh.channelId = sub->channelId;
      eDescr.kernelCh.pTimer = start;
      ncclProfiler->startEvent(sub->profilerContext, &sub->kernelEventHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs* args, int s, uint64_t stop) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    struct ncclProxySubArgs* sub = &args->subs[s];
    if (sub->kernelEventHandle) {
      ncclProfilerEventStateArgs_t a = { };
      a.kernelCh.pTimer = stop;
      ncclProfiler->recordEventState(sub->kernelEventHandle, ncclProfilerKernelChStop, &a);
      ncclProfiler->stopEvent(sub->kernelEventHandle);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyOpEventState(int s, struct ncclProxyArgs* args, ncclProfilerEventState_t eState) {
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    ncclProfilerEventStateArgs_t a = { };
    ncclProfiler->recordEventState(sub->opEventHandle, eState, &a);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyStepEventState(int s, struct ncclProxyArgs* args, int stepId, ncclProfilerEventState_t eState) {
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (__builtin_expect(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    int step_ = DIVUP(stepId, args->sliceSteps);
    if (sub->pHandles[step_%NCCL_STEPS].stepEventHandle) {
      ncclProfilerEventStateArgs_t a = { };
      a.proxyStep.transSize = sub->transSize;
      ncclProfiler->recordEventState(sub->pHandles[step_%NCCL_STEPS].stepEventHandle, eState, &a);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyCtrlEventState(void* eHandle, int appended, ncclProfilerEventState_t eState) {
  if (__builtin_expect(ncclProfiler != NULL, 0) && eHandle && __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ncclProfileProxyCtrl) {
    ncclProfilerEventStateArgs_t args = { };
    args.proxyCtrl.appendedProxyOps = appended;
    ncclProfiler->recordEventState(eHandle, eState, &args);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op) {
  op->pid = pid;
  return ncclSuccess;
}

static std::mutex proxyProfilerConnectMutex;

static ncclResult_t proxyProfilerConnect(ncclComm_t comm, struct ncclProxyOp* op) {
  ncclResult_t ret = ncclSuccess;
  std::lock_guard<std::mutex> lock(proxyProfilerConnectMutex);
  if (comm->profiler.initialized) goto exit;
  for (int c = 0; c < MAXCHANNELS; c++) {
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_PROFILER, 0, comm->rank, &comm->profiler.sendProxyConn[c]), ret, exit);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &comm->profiler.sendProxyConn[c], ncclProxyMsgConnect, NULL, 0, NULL, 0), ret, exit);
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_PROFILER, 0, comm->rank, &comm->profiler.recvProxyConn[c]), ret, exit);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &comm->profiler.recvProxyConn[c], ncclProxyMsgConnect, NULL, 0, NULL, 0), ret, exit);
  }
  comm->profiler.initialized = true;
exit:
  return ret;
}

bool ncclProfilerNeedsProxy(ncclComm_t comm, struct ncclProxyOp* op) {
  bool enabled = ncclProfilerPluginLoaded() && (op->eActivationMask & ncclProfileKernelCh);
  if (enabled && !comm->profiler.initialized) (void)proxyProfilerConnect(comm, op);
  return enabled;
}

bool ncclProfilerPluginLoaded(void) {
  return (__builtin_expect(ncclProfiler != NULL, 0));
}

ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (type == ncclProfilerNetEventStart) { // start
      struct ncclProxyEventHandle* p = (struct ncclProxyEventHandle*)pHandle;
      struct ncclProxySubArgs* sub = p->subArgPtr;
      if (sub->eActivationMask & ncclProfileNetPlugin) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileNetPlugin;
        eDescr.parentObj = p->stepEventHandle;
        eDescr.rank = sub->rank;
        eDescr.netPlugin.id = pluginId;
        eDescr.netPlugin.data = extData;
        ncclProfiler->startEvent(sub->profilerContext, eHandle, &eDescr);
      }
    } else if (type == ncclProfilerNetEventStop) { // stop
      ncclProfiler->stopEvent(*eHandle);
    } else if (type == ncclProfilerNetEventUpdate) { // update
      ncclProfilerEventStateArgs_t args = { };
      args.netPlugin.data = extData;
      ncclProfiler->recordEventState(*eHandle, ncclProfilerNetPluginUpdate, &args);
    } else { // update and stop
      ncclProfilerEventStateArgs_t args = { };
      args.netPlugin.data = extData;
      ncclProfiler->recordEventState(*eHandle, ncclProfilerNetPluginUpdate, &args);
      ncclProfiler->stopEvent(*eHandle);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerPrintEvent(ncclComm_t comm, const char* dumpFilePath) {
  if (__builtin_expect(ncclProfiler != NULL, 0) && comm->profilerContext && printEventFn) {
    INFO(NCCL_DUMP, "Dump profiler trace file to %s", dumpFilePath);
    printEventFn(comm->profilerContext, dumpFilePath);
  } else {
    INFO(NCCL_DUMP, "Profiler is not enabled or no print event function is loaded, skip dumping trace file to %s", dumpFilePath);
  }
  return ncclSuccess;
}

// ============================================================================
// CE Profiler Functions (Simple Wrappers)
// ============================================================================

/*
 * CE Collective start event - calls plugin startEvent callback
 */
ncclResult_t ncclProfilerStartCeCollEvent(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    // Check if CE Coll events are enabled (or child events CeSync/CeBatch which need CeColl)
    int ceCollMask = ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
    if (__atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ceCollMask) {
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileCeColl;
      eDescr.parentObj = args->collApiEventHandle;
      eDescr.rank = comm->rank;

//      eDescr.ceColl.seqNumber = comm->ceColl.ceSeqNum;
      eDescr.ceColl.func = ncclFuncToString(args->func);
      eDescr.ceColl.sendBuff = args->sendBuff;
      eDescr.ceColl.recvBuff = args->recvBuff;
      eDescr.ceColl.count = args->nElts;
      eDescr.ceColl.root = args->rootRank;
      eDescr.ceColl.datatype = ncclDatatypeToString(args->datatype);
      eDescr.ceColl.syncStrategy = "UC";
      eDescr.ceColl.intraBatchSync = false;
      eDescr.ceColl.batchSize = 0;
      eDescr.ceColl.numBatches = 0;
//      eDescr.ceColl.ceSeqNum = comm->ceColl.ceSeqNum;
      eDescr.ceColl.stream = (void*)stream;

      ncclProfiler->startEvent(comm->profilerContext, &args->ceCollProfHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

/*
 * CE Collective stop event - calls plugin stopEvent callback
 */
ncclResult_t ncclProfilerStopCeCollEvent(ncclComm_t comm, struct ncclCeCollArgs* args, hggcStream_t stream) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (args && args->ceCollProfHandle) {
      ncclProfiler->stopEvent(args->ceCollProfHandle);
    }
  }
  return ncclSuccess;
}

/*
 * CE Sync start event - calls plugin startEvent callback
 */
ncclResult_t ncclProfilerStartCeSyncEvent(ncclComm_t comm, struct ncclCeCollArgs* args,
                                          hggcStream_t stream, void** ceSyncHandle) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (args && args->ceCollProfHandle && (__atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ncclProfileCeSync)) {
      // CeSync only needs to check if it's enabled; parent CeColl is implicitly started via ceCollMask
      ncclProfilerEventDescr_t eDescr = { 0 };
      eDescr.type = ncclProfileCeSync;
      eDescr.parentObj = args->ceCollProfHandle;
      eDescr.rank = comm->rank;

//      eDescr.ceCollSync.isComplete = comm->ceColl.useCompletePtr;
      eDescr.ceCollSync.nRanks = comm->nRanks;

      ncclProfiler->startEvent(comm->profilerContext, ceSyncHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

/*
 * CE Sync stop event - calls plugin stopEvent callback
 */
ncclResult_t ncclProfilerStopCeSyncEvent(ncclComm_t comm, void* ceSyncHandle,
                                         hggcStream_t stream) {
  if (__builtin_expect(ncclProfiler != NULL, 0) && ceSyncHandle) {
    ncclProfiler->stopEvent(ceSyncHandle);
  }
  return ncclSuccess;
}

/*
 * CE Batch start event - calls plugin startEvent callback
 */
ncclResult_t ncclProfilerStartCeBatchEvent(ncclComm_t comm,
                                           struct ncclCeCollArgs* args,
                                           const struct ncclCeBatchOpsParams* params,
                                           hggcStream_t stream,
                                           void** ceBatchHandle) {
  if (__builtin_expect(ncclProfiler != NULL, 0)) {
    if (args && args->ceCollProfHandle) {
      // CeBatch only needs to check if it's enabled; parent CeColl is implicitly started via ceCollMask
      if (__atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ncclProfileCeBatch) {
        ncclProfilerEventDescr_t eDescr = { 0 };
        eDescr.type = ncclProfileCeBatch;
        eDescr.parentObj = args->ceCollProfHandle;
        eDescr.rank = comm->rank;

        eDescr.ceCollBatch.numOps = params->numOps;

        size_t totalBytes = 0;
        for (int i = 0; i < params->numOps; i++) {
          totalBytes += params->sizes[i];
        }
        eDescr.ceCollBatch.totalBytes = totalBytes;
        eDescr.ceCollBatch.useIntraSync = false;

        ncclProfiler->startEvent(comm->profilerContext, ceBatchHandle, &eDescr);
      }
    }
  }
  return ncclSuccess;
}

/*
 * CE Batch stop event - calls plugin stopEvent callback
 */
ncclResult_t ncclProfilerStopCeBatchEvent(ncclComm_t comm, void* ceBatchHandle, hggcStream_t stream) {
  if (__builtin_expect(ncclProfiler != NULL, 0) && ceBatchHandle) {
    ncclProfiler->stopEvent(ceBatchHandle);
  }
  return ncclSuccess;
}

