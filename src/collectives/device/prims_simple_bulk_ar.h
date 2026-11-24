/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PRIMS_SIMPLE_BULK_AR_H_
#define NCCL_PRIMS_SIMPLE_BULK_AR_H_

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, typename Fan,
         int DIRECT, class FUNC, ncclConn_t SENDCONNTYPE, ncclConn_t RECVCONNTYPE, int P2P>
class Primitives<T, FUNC, Fan, DIRECT, ProtoSimpleBulkAR<SLICESPERCHUNK, SLICESTEPS, UNROLL, SENDCONNTYPE, RECVCONNTYPE>, P2P> :
    public PrimitivesSimpleBase<T, Fan, FUNC, SLICESPERCHUNK, SLICESTEPS, UNROLL, DIRECT, ProtoSimpleBulkAR<SLICESPERCHUNK, SLICESTEPS, UNROLL, SENDCONNTYPE, RECVCONNTYPE>, Primitives<T, FUNC, Fan, DIRECT, ProtoSimpleBulkAR<SLICESPERCHUNK, SLICESTEPS, UNROLL, SENDCONNTYPE, RECVCONNTYPE>, P2P>, P2P> {
  static constexpr int NRECV = Fan::MaxRecv, NSEND = Fan::MaxSend;

 public:
  __device__ __forceinline__ Primitives(
      const int tid, const int nworkers, int* recvPeers, int* sendPeers,
      T* directBuff, struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs,
      int group, uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr, struct ncclDevWorkP2p* p2pWork=nullptr,
      const uint64_t ncclSteps=NCCL_STEPS, bool noRed=false
    ): PrimitivesSimpleBase<T, Fan, FUNC, SLICESPERCHUNK, SLICESTEPS, UNROLL, DIRECT, ProtoSimpleBulkAR<SLICESPERCHUNK, SLICESTEPS, UNROLL, SENDCONNTYPE, RECVCONNTYPE>, Primitives<T, FUNC, Fan, DIRECT, ProtoSimpleBulkAR<SLICESPERCHUNK, SLICESTEPS, UNROLL, SENDCONNTYPE, RECVCONNTYPE>, P2P>, P2P>
    (tid, nworkers, recvPeers, sendPeers, directBuff, comm, redOpArg, ptrs, group, connIndexRecv, connIndexSend, collWork, p2pWork, ncclSteps, noRed) {}
};

#endif
