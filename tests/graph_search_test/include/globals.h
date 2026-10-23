/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_GLOBALS_H
#define _NCCL_GLOBALS_H
#include "alloc.h"
#include "param.h"
#include "devcomm.h"
#include "gdrwrap.h"

#ifdef ENABLE_TRACE
std::chrono::high_resolution_clock::time_point ncclEpoch;
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+1] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv"};
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNet", "FC", "PAT"};
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple", "Signal", "SimpleBulk", "SignalBulk" };
const bool ncclSendRecvProtoSupport[NCCL_NUM_PROTOCOLS] = {false, false, true, true, true, true};

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);

PCCL_PARAM(DisableAbort, "DISABLE_ABORT", 0);

extern ncclNet_t* ncclNet;
extern ncclCollNet_t* ncclCollNet;

NCCL_PARAM(CollNetEnable, "COLLNET_ENABLE", 0);
PCCL_PARAM(NoMemset, "NO_MEMSET", 0);
PCCL_PARAM(SingleHostCrossNodeTest, "SINGLE_HOST_CROSS_NODE_TEST", 0);
PCCL_PARAM(PerfmodelSpeedup, "PERFMODEL_SPEEDUP", 0);
PCCL_PARAM(ChannelShuffle, "CHANNEL_SHUFFLE", 0);

int gModelType = HW_PPU;
int ncclPerfmodelSpeedup = 0;
const size_t ncclPerfmodelTotalDeviceMemSize = 1*1024*1024*1024;
const size_t ncclPerfmodelDevMemAlignSize = 2*1024*1024;
void* ncclPerfmodelHostPtr[8];
void* ncclPerfmodelDevicePtr[8];
size_t ncclPerfmodelMemOffset[8];
int ncclPerfmodelNeedCopy[8];

int ncclParamNoMemsetFlag;
int ncclParamSingleHostCrossNodeTestFlag;
int ncclParamChannelShuffleFlag;
int useMemcpy;

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;

#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_BUFFSIZE_ARM (1 << 20) /* 1MiB */
#define DEFAULT_SIGNAL_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_SIGNAL_BUFFSIZE_ARM (1 << 20) /* 1MiB */
#define DEFAULT_SIMPLE_BULK_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_SIMPLE_BULK_BUFFSIZE_ARM (1 << 20) /* 1MiB */
#define DEFAULT_SIGNAL_BULK_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_SIGNAL_BULK_BUFFSIZE_ARM (1 << 20) /* 1MiB */

NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

extern int64_t ncclParamCrossNic();
NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);

// GDRCOPY support
gdr_t ncclGdrCopy = NULL;

#endif  // _NCCL_GLOBALS_H
