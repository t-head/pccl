/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef ASYS_PROFILER_H
#define ASYS_PROFILER_H

enum pcclProfilerCallbackId {
  PCCL_PROXY_PROFILER_STATE_BEGIN = 0,
  PCCL_PROXY_PROFILER_STATE_CHANGE = 1,
  PCCL_PROXY_PROFILER_STATE_END = 2,
  PCCL_PROXY_PROFILER_STATE_NUM
};

struct pcclProxyProfilerData {
  const char *stateName;
  int type; // 1 means send and 0 means recv
  int channelId;
  int step; // current step
  int pipeIdx; // pipeline index for GUI
};

struct pcclProxyProfilerBeginData {
  struct pcclProxyProfilerData *stateData;

  int devId;
  int rank;
  int peer;

  int sliceSteps;
  int chunkSteps;
  int chunkSize;
  int done; // already finished steps
  int nsteps; // totoal steps
};

struct pcclProxyProfilerFinishData {
  int type; // 1 means send and 0 means recv
  int channelId;
  int pipeIdx; // pipeline index for GUI
};

typedef int32_t pcclAsysProfilerSubscriberHandle;
typedef void (*pcclProxyProfilerCallbackFunc)(uint32_t callback_id, const void *parameter);

#ifdef __cplusplus
extern "C" {
#endif

// enable one callback id, return 0 means success, others failed
int32_t pcclProfilerEnableCallback(uint32_t callback_id);

// disable one callback id, return 0 means success, others failed
int32_t pcclProfilerDisableCallback(uint32_t callback_id);

// register pccl profiler callback function, return 0 means success, others failed
int32_t pcclToolSubscribe(pcclAsysProfilerSubscriberHandle *handle, pcclProxyProfilerCallbackFunc callback);

// deregister pccl profiler callback function, return 0 means success, others failed
int32_t pcclToolUnsubscribe(pcclAsysProfilerSubscriberHandle handle);

#ifdef __cplusplus
}
#endif
#endif // End include guard

