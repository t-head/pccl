/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define SPINS_BEFORE_CHECK_ABORT 250000

// Unroll unconditionally the first send/recv since nsend/nrecv should be at
// least 1 if SEND/RECV is set.
#define FOR_SEND(func, ...) do { \
  if (SEND) { \
    /* Send to far first, then close */ \
    for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__); \
    func(0, ##__VA_ARGS__); \
  } \
} while (0)

#define FOR_RECV(func, ...) do { \
  if (RECV) { \
    /* Recv from close first, then far */ \
    func(0, ##__VA_ARGS__); \
    for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__); \
  } \
} while (0)

#define ROLE_SRC       0x01
#define ROLE_DST       0x02
#define ROLE_WAIT_RECV 0x04
#define ROLE_WAIT_SEND 0x08
#define ROLE_POST_SEND 0x10
#define ROLE_POST_RECV 0x20

struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;
  static constexpr int RTMD = RTMD_PA;
};

struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;
  static constexpr int RTMD = RTMD_PA;
};

template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr ncclConn_t sendConnType = CONN_NORMAL;
  static constexpr ncclConn_t recvConnType = CONN_NORMAL;
  static constexpr int RTMD = RTMD_PA;
};

template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS>
struct ProtoSignal {
  static constexpr int Id = NCCL_PROTO_SIGNAL;
  static constexpr ncclConn_t sendConnType = CONN_NORMAL;
  static constexpr ncclConn_t recvConnType = CONN_NORMAL;
  static constexpr int RTMD = RTMD_PA;
};

template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, ncclConn_t SENDCONNTYPE = CONN_ICN, ncclConn_t RECVCONNTYPE = CONN_ICN>
struct ProtoSimpleBulk {
  static constexpr int Id = NCCL_PROTO_SIMPLE_BULK;
  static constexpr ncclConn_t sendConnType = SENDCONNTYPE;
  static constexpr ncclConn_t recvConnType = RECVCONNTYPE;
  static constexpr int RTMD = RTMD_PA;
};

template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, ncclConn_t SENDCONNTYPE = CONN_ICN, ncclConn_t RECVCONNTYPE = CONN_ICN>
struct ProtoSignalBulk {
  static constexpr int Id = NCCL_PROTO_SIGNAL_BULK;
  static constexpr ncclConn_t sendConnType = SENDCONNTYPE;
  static constexpr ncclConn_t recvConnType = RECVCONNTYPE;
  static constexpr int RTMD = RTMD_PA;
};

template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, ncclConn_t SENDCONNTYPE = CONN_ICN, ncclConn_t RECVCONNTYPE = CONN_ICN>
struct ProtoSimpleBulkAR {
  static constexpr int Id = NCCL_PROTO_SIMPLE_BULK_AR;
  static constexpr ncclConn_t sendConnType = SENDCONNTYPE;
  static constexpr ncclConn_t recvConnType = RECVCONNTYPE;
  static constexpr int RTMD = RTMD_AR;
};

// The primitives class. Specialized per protocol in the other headers.
template<typename T, typename FUNC, typename Proto, int NRECV, int NSEND, int DIRECT, int PrimitivesWithoutDirect, bool NET_POLL = false>
class Primitives;

template <typename T, int NRECV, int NSEND, class FUNC, int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, int DIRECT, typename Proto, class RealPrimitives, int P2P, bool NET_POLL = false>
class PrimitivesSimpleBase;

// Used by LL & LL128 to implement direct members in the naive way.
template<typename T, typename RealPrimitives>
struct PrimitivesWithoutDirect {
  __device__ void directSend(const T* inpIx, ssize_t directOffset, int eltN) {
    static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
  }

  // __device__ void directSendFromOutput(T* outIx, T* remoteOutIx, int eltN) {
  //   static_cast<RealPrimitives*>(this)->sendFromOutput(outIx, eltN);
  // }

  __device__ void directRecv(T* outIx, ssize_t directOffset, int eltN) {
    static_cast<RealPrimitives*>(this)->recv(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directCopySend(const T* inpIx, T* outIx, ssize_t directOffset, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->copySend(inpIx, outIx, eltN, postOp);
  }
  __device__ void directRecvCopySend(T* outIx, ssize_t directOffset, int eltN) {
    static_cast<RealPrimitives*>(this)->recvCopySend(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directRecvReduceCopySend(const T* inpIx, T* outIx, ssize_t directOffset, int eltN, bool postOp=false, int reqChunkSize=0) {
    // Direct is only for the send part
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }

  __device__ void directGather(T* inpIx, int totalN, int eltN, int size, int chunkOffset) {
    static_cast<RealPrimitives*>(this)->gather(inpIx, totalN, eltN, size);
  }
};

inline __device__ int checkAbort(struct ncclKernelComm* comm, uint32_t& spins, uint32_t& abort, uint32_t& hangSpins) {
  spins++;
  if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
    abort = *(comm->abortFlag);
    spins = 0;
    if (++hangSpins == comm->channels[getShmemPtr(comm)->channelId].nSpinsBeforeReportHang) {
      hangSpins = 0;
      volatileStoreU64(comm->channels[getShmemPtr(comm)->channelId].hostDevIndexLevel1, getShmemPtr(comm)->args.kernelIdx);
      comm->channels[getShmemPtr(comm)->channelId].nSpinsBeforeReportHang += min(comm->channels[getShmemPtr(comm)->channelId].nSpinsBeforeReportHang, UINT32_MAX-comm->channels[getShmemPtr(comm)->channelId].nSpinsBeforeReportHang);
    }
  }
  return abort;
}

#include "prims_ll.h"
#include "prims_ll128.h"
#include "prims_simple.h"
#include "prims_simple_bulk.h"
#include "prims_simple_bulk_ar.h"
#include "prims_signal.h"
#include "prims_signal_bulk.h"

#endif
