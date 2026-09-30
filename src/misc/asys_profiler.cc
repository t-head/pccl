/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdio.h>
#include "comm.h"
#include "profiler.h"
#include "asys_profiler.h"

#define PROFILER_MAX_STAGES 8

static const char* profilingStateSendStr[] = { "BufferWait", "GPUWait", "SendWait", "", "End" };
static const char* profilingStateRecvStr[] = { "BufferWait", "RecvWait", "FlushWait", "GPUWait", "End" };

// profiling with built-in profiler
#include "timer.h"
#include "alloc.h"

struct ncclProxyProfileEvent {
    double timestamp[PROFILER_MAX_STAGES];
    uint64_t opCount;
    int peer;
    int step;
    uint16_t channel;
    uint8_t type; // send / recv
    uint8_t opIndex;
    struct ncclProxyArgs* args;
};

struct ncclProxyProfileEvent* profilingEventsArr[NCCL_MAX_LOCAL_RANKS] = {NULL};
int profilingIndex[NCCL_MAX_LOCAL_RANKS] = {0};
double profilingStart[NCCL_MAX_LOCAL_RANKS] = {0};

// profiling with asys
static pcclAsysProfilerSubscriberHandle profilerHandle = 0;
static pcclProxyProfilerCallbackFunc profilerCallback = NULL;
static uint32_t callbackFlags[ncclProxyProfileStateNum] = {0};

static ncclResult_t pcclProfilingRecordAsys(struct ncclProxyArgs* args, int sub, int step, int state) {

  // make sure callback is valid and ready
  if (callbackFlags[state] == 0) return ncclSuccess;

  int rank = args->perfComm->rank;
  int peer = args->subs[sub].peer;
  int sendrecv = args->subs[sub].sendrecv;
  int channelId = args->subs[sub].channelId;
  int hggcDev = args->perfComm->hggcDev;
  const char** stateStr = sendrecv == proxySend ? profilingStateSendStr : profilingStateRecvStr;
  int callbackId = 0;

  // make sure callback is valid and ready
  if (state == ncclProxyProfileBegin) {
    if(args->subs[sub].profilingEvents[step%NCCL_STEPS]) return ncclSuccess;
    args->subs[sub].profilingEvents[step%NCCL_STEPS] = args;

    pcclProxyProfilerData changeData;
    pcclProxyProfilerBeginData beginData;

    changeData.stateName = stateStr[state];
    changeData.type = sendrecv;
    changeData.channelId = channelId;
    changeData.step = step;
    changeData.pipeIdx = step % NCCL_STEPS;

    beginData.stateData = &changeData;
    beginData.devId = hggcDev;
    beginData.rank = rank;
    beginData.peer = peer;
    beginData.sliceSteps = args->sliceSteps;
    beginData.chunkSteps = args->chunkSteps;
    beginData.chunkSize = args->chunkSize;
    beginData.done = args->done;
    beginData.nsteps = args->subs[sub].nsteps;
    callbackId = PCCL_PROXY_PROFILER_STATE_BEGIN;
    profilerCallback(callbackId, &beginData);
  } else if (state == ncclProxyProfileEnd) {
    pcclProxyProfilerFinishData finishData;
    finishData.type = sendrecv;
    finishData.channelId = channelId;
    finishData.pipeIdx = step % NCCL_STEPS;
    callbackId = PCCL_PROXY_PROFILER_STATE_END;
    profilerCallback(callbackId, &finishData);
    args->subs[sub].profilingEvents[step%NCCL_STEPS] = nullptr;
  } else {
    // state change
    pcclProxyProfilerData changeData;
    changeData.stateName = stateStr[state];
    changeData.type = sendrecv;
    changeData.channelId = channelId;
    changeData.step = step;
    changeData.pipeIdx = step % NCCL_STEPS;
    callbackId = PCCL_PROXY_PROFILER_STATE_CHANGE;
    profilerCallback(callbackId, &changeData);
  }

  return ncclSuccess;
}

// stub implementation
static ncclResult_t ncclProfilingRecordStub(struct ncclProxyArgs* args, int sub, int step, int state) { return ncclSuccess; }
static void ncclProfilingDumpStub(ncclComm_t comm) {}

void profilerInit(int type) {
  pcclAsysProxyProfilingRecord = (type == PROFILER_ASYS) ? pcclProfilingRecordAsys : ncclProfilingRecordStub;
}

int32_t pcclProfilerEnableCallback(uint32_t callback_id) {
  if (callback_id == PCCL_PROXY_PROFILER_STATE_BEGIN) {
    callbackFlags[ncclProxyProfileBegin] = 1;
  } else if (callback_id == PCCL_PROXY_PROFILER_STATE_END) {
    callbackFlags[ncclProxyProfileEnd] = 1;
  } else {
    callbackFlags[ncclProxyProfileRecvWait] = 1;
    callbackFlags[ncclProxyProfileRecvFlushWait] = 1;
    callbackFlags[ncclProxyProfileRecvGPUWait] = 1;
  }
  return 0;
}

int32_t pcclProfilerDisableCallback(uint32_t callback_id) {
  if (callback_id == PCCL_PROXY_PROFILER_STATE_BEGIN) {
    callbackFlags[ncclProxyProfileBegin] = 0;
  } else if (callback_id == PCCL_PROXY_PROFILER_STATE_END) {
    callbackFlags[ncclProxyProfileEnd] = 0;
  } else {
    callbackFlags[ncclProxyProfileRecvWait] = 0;
    callbackFlags[ncclProxyProfileRecvFlushWait] = 0;
    callbackFlags[ncclProxyProfileRecvGPUWait] = 0;
  }
  return 0;
}

int32_t pcclToolSubscribe(pcclAsysProfilerSubscriberHandle *handle, pcclProxyProfilerCallbackFunc callback) {
  profilerHandle = *handle;
  profilerCallback = callback;
  profilerInit(PROFILER_ASYS);
  return 0;
}

ncclResult_t (*pcclAsysProxyProfilingRecord)(struct ncclProxyArgs* args, int sub, int step, int state) = ncclProfilingRecordStub;
void (*ncclProfilingDump)(ncclComm_t comm) = ncclProfilingDumpStub;

int32_t pcclToolUnsubscribe(pcclAsysProfilerSubscriberHandle handle) {
  int32_t ret = -1;
  if (handle == profilerHandle) {
    profilerCallback = NULL;
    profilerInit(PROFILER_NONE);
    ret = 0;
  }
  return ret;
}
