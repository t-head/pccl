/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PRIMS_SIMPLE_H_
#define NCCL_PRIMS_SIMPLE_H_
#include "prims_simple_base.h"
// Implementation of primitive types
template <typename T, int NRECV, int NSEND, class FUNC, int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, int DIRECT, int P2P>
class Primitives<T, FUNC, ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P>:
    public PrimitivesSimpleBase<T, NRECV, NSEND, FUNC, UNROLL, SLICESPERCHUNK, SLICESTEPS, DIRECT,
     ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>,
     Primitives<T, FUNC, ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P>, P2P> {

 public:
  __device__ __forceinline__ Primitives(
      const int tid, const int nworkers, int* recvPeers, int* sendPeers,
      T* directBuff, struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs,
      int group, uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork=nullptr, const uint64_t ncclSteps=NCCL_STEPS, bool noRed=false
    ): 
    PrimitivesSimpleBase<T, NRECV, NSEND, FUNC, UNROLL, SLICESPERCHUNK, SLICESTEPS, DIRECT,
    ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>,
    Primitives<T, FUNC, ProtoSimple<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P>, P2P>
    (tid, nworkers, recvPeers, sendPeers, directBuff, comm, redOpArg, ptrs, group, connIndexRecv, connIndexSend, collWork, p2pWork, ncclSteps, noRed) {}
};

#endif
