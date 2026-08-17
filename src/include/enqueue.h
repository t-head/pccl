/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENQUEUE_H_
#define NCCL_ENQUEUE_H_

#include "comm.h"
#include "group.h"
#include "collectives.h"

ncclResult_t ncclInitKernelsForDevice(size_t* maxStackSize);
ncclResult_t ncclEnqueueCheck(struct ncclInfo* info);
ncclResult_t ncclLaunchPrepare(ncclComm_t comm);
ncclResult_t ncclLaunchKernelBefore_NoUncapturedHggc(ncclComm_t comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernel(ncclComm_t comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchKernelAfter_NoHggc(ncclComm_t comm, struct ncclKernelPlan* plan);
ncclResult_t ncclLaunchFinish(ncclComm_t comm);
ncclResult_t ncclLaunchReset(ncclComm_t comm);
ncclResult_t ncclPrepareTasks(ncclComm_t comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo = NULL);
ncclResult_t ncclTasksRegAndEnqueue(ncclComm_t comm);

static inline size_t ncclFuncSendCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncReduceScatter ? nRanks*count : count;
}
static inline size_t ncclFuncRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather ? nRanks*count : count;
}
static inline size_t ncclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count) {
  return func == ncclFuncAllGather || func == ncclFuncReduceScatter ? nRanks*count : count;
}

static inline bool ncclCeSupported(ncclComm_t comm) {
  return (comm->symmetricSupport && comm->nNodes == 1 && comm->config.CTAPolicy == PCCL_CTA_POLICY_ZERO);
}

ncclResult_t ncclAddProxyOpIfNeeded(ncclComm_t comm, struct ncclProxyOp* op);

#endif // End include guard
