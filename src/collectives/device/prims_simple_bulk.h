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
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND,
         int DIRECT, class FUNC, ncclConn_t SENDCONNTYPE, ncclConn_t RECVCONNTYPE, int P2P>
class Primitives<T, FUNC, ProtoSimpleBulk<UNROLL, SLICESPERCHUNK, SLICESTEPS, SENDCONNTYPE, RECVCONNTYPE>, NRECV, NSEND, DIRECT, P2P>:
    public PrimitivesSimpleBase<T, NRECV, NSEND, FUNC, UNROLL, SLICESPERCHUNK, SLICESTEPS, DIRECT,
      ProtoSimpleBulk<UNROLL, SLICESPERCHUNK, SLICESTEPS, SENDCONNTYPE, RECVCONNTYPE>,
      Primitives<T, FUNC, ProtoSimpleBulk<UNROLL, SLICESPERCHUNK, SLICESTEPS, SENDCONNTYPE, RECVCONNTYPE>, NRECV, NSEND, DIRECT, P2P>, P2P> {
 public:
  __device__ __forceinline__ Primitives(
      const int tid, const int nworkers, int* recvPeers, int* sendPeers,
      T* directBuff, struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs,
      int group, uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork=nullptr, const uint64_t ncclSteps=NCCL_STEPS, bool noRed=false
    ): 
    PrimitivesSimpleBase<T, NRECV, NSEND, FUNC, UNROLL, SLICESPERCHUNK, SLICESTEPS, DIRECT,
    ProtoSimpleBulk<UNROLL, SLICESPERCHUNK, SLICESTEPS, SENDCONNTYPE, RECVCONNTYPE>,
    Primitives<T, FUNC, ProtoSimpleBulk<UNROLL, SLICESPERCHUNK, SLICESTEPS, SENDCONNTYPE, RECVCONNTYPE>, NRECV, NSEND, DIRECT, P2P>, P2P>
    (tid, nworkers, recvPeers, sendPeers, directBuff, comm, redOpArg, ptrs, group, connIndexRecv, connIndexSend, collWork, p2pWork, ncclSteps, noRed) {}
};

#endif
