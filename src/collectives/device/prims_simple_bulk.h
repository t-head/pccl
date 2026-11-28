/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PRIMS_SIMPLE_BULK_H_
#define NCCL_PRIMS_SIMPLE_BULK_H_
#include "prims_simple_base.h"

// Implementation of primitive types
template <typename T, typename Fan, typename RedOp, int SlicePerChunk, int StepPerSlice,
         int Unroll, int Direct, ncclConn_t SENDCONNTYPE, ncclConn_t RECVCONNTYPE, int P2p, bool isNetOffload>
class Primitives<T, RedOp, Fan, Direct, ProtoSimpleBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>, P2p, isNetOffload>:
    public PrimitivesSimpleBase<T, Fan, RedOp, SlicePerChunk, StepPerSlice, Unroll, Direct,
      ProtoSimpleBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>,
      Primitives<T, RedOp, Fan, Direct, ProtoSimpleBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>, P2p, isNetOffload>, P2p> {
 public:
  static_assert(!isNetOffload, "isNetOffload is not implemented on PPU");

  __device__ __forceinline__ Primitives(
      const int tid, const int nthreads, int* recvPeers, int* sendPeers,
      T* directBuff, struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs,
      int group, uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork=nullptr, int stepSize_=0, const uint64_t ncclSteps=NCCL_STEPS, bool noRed=false
    ): 
    PrimitivesSimpleBase<T, Fan, RedOp, SlicePerChunk, StepPerSlice, Unroll, Direct,
    ProtoSimpleBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>,
    Primitives<T, RedOp, Fan, Direct, ProtoSimpleBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>, P2p, isNetOffload>, P2p>
    (tid, nthreads, recvPeers, sendPeers, directBuff, comm, redOpArg, ptrs, group, connIndexRecv, connIndexSend, collWork, p2pWork, stepSize_, ncclSteps, noRed) {}
};

#endif
