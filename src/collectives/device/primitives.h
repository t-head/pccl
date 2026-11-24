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


#define ROLE_SRC       0x01
#define ROLE_DST       0x02
#define ROLE_WAIT_RECV 0x04
#define ROLE_WAIT_SEND 0x08
#define ROLE_POST_SEND 0x10
#define ROLE_POST_RECV 0x20

struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;
  static constexpr int RTMD = RTMD_PA;
  // Half of an LL line is flags, so only half the step carries data.
  __device__ static int calcBytePerStep(struct ncclKernelComm* comm) {
    return comm->buffSizes[NCCL_PROTO_LL] / NCCL_STEPS / 2;
  }
  // One 16-byte line has 8 bytes of data.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t);
  }
  // How many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};

struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;
  static constexpr int RTMD = RTMD_PA;
  __device__ static int calcBytePerStep(struct ncclKernelComm* comm) {
    return (comm->buffSizes[NCCL_PROTO_LL128] / NCCL_STEPS) * NCCL_LL128_DATAELEMS / NCCL_LL128_LINEELEMS;
  }
  __device__ static int calcBytePerGrain() {
    return NCCL_LL128_SHMEM_ELEMS_PER_THREAD * NCCL_LL128_DATAELEMS * sizeof(uint64_t) / NCCL_LL128_LINEELEMS;
  }
  static constexpr int MaxGroupWidth = 1;
};

/* Protocol classes: ProtoSimple, ProtoLL, ProtoLL128
 * We use these as template args to the Primitives class instead of integral
 * enums (e.g. NCCL_PROTO_LL) because for SIMPLE we need to carry a few extra
 * numbers. Also these types hold methods which let us compute numbers important
 * to how that protocol operates with a consistent interface so that our
 * algorithm code can operate protocol parametrically.
 */
template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL, int MultimemSrcs_1 = 0, int MultimemDsts_1 = 0>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerSlice_1;
  static constexpr int Unroll = Unroll_1;
  static constexpr int MultimemSrcs = MultimemSrcs_1;
  static constexpr int MultimemDsts = MultimemDsts_1;

  static constexpr ncclConn_t sendConnType = CONN_NORMAL;
  static constexpr ncclConn_t recvConnType = CONN_NORMAL;
  static constexpr int RTMD = RTMD_PA;

  static_assert(MultimemSrcs_1 == 0 && MultimemDsts_1 == 0,
                "PPU has no multicast address space.");

  __device__ static int calcBytePerStep(struct ncclKernelComm* comm) {
    return comm->buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 2;
};

#define PCCL_DEFINE_SIMPLE_FAMILY_PROTO_METRICS()                                  \
  static constexpr int Unroll = Unroll_1;                                          \
  static constexpr int SlicePerChunk = SlicePerChunk_1;                            \
  static constexpr int StepPerSlice = StepPerSlice_1;                              \
  static constexpr int MultimemSrcs = 0;                                           \
  static constexpr int MultimemDsts = 0;                                           \
  __device__ static int calcBytePerStep(struct ncclKernelComm* comm) {              \
    return comm->buffSizes[Id] / NCCL_STEPS;                                       \
  }                                                                                \
  __device__ static int calcBytePerGrain() {                                       \
    return sizeof(uint64_t);                                                       \
  }                                                                                \
  static constexpr int MaxGroupWidth = 2

template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL>
struct ProtoSignal {
  static constexpr int Id = NCCL_PROTO_SIGNAL;
  static constexpr ncclConn_t sendConnType = CONN_NORMAL;
  static constexpr ncclConn_t recvConnType = CONN_NORMAL;
  static constexpr int RTMD = RTMD_PA;
  PCCL_DEFINE_SIMPLE_FAMILY_PROTO_METRICS();
};

template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL, ncclConn_t SENDCONNTYPE = CONN_ICN, ncclConn_t RECVCONNTYPE = CONN_ICN>
struct ProtoSimpleBulk {
  static constexpr int Id = NCCL_PROTO_SIMPLE_BULK;
  static constexpr ncclConn_t sendConnType = SENDCONNTYPE;
  static constexpr ncclConn_t recvConnType = RECVCONNTYPE;
  static constexpr int RTMD = RTMD_PA;
  PCCL_DEFINE_SIMPLE_FAMILY_PROTO_METRICS();
};

template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL, ncclConn_t SENDCONNTYPE = CONN_ICN, ncclConn_t RECVCONNTYPE = CONN_ICN>
struct ProtoSignalBulk {
  static constexpr int Id = NCCL_PROTO_SIGNAL_BULK;
  static constexpr ncclConn_t sendConnType = SENDCONNTYPE;
  static constexpr ncclConn_t recvConnType = RECVCONNTYPE;
  static constexpr int RTMD = RTMD_PA;
  PCCL_DEFINE_SIMPLE_FAMILY_PROTO_METRICS();
};

template<int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL, ncclConn_t SENDCONNTYPE = CONN_ICN, ncclConn_t RECVCONNTYPE = CONN_ICN>
struct ProtoSimpleBulkAR {
  static constexpr int Id = NCCL_PROTO_SIMPLE_BULK_AR;
  static constexpr ncclConn_t sendConnType = SENDCONNTYPE;
  static constexpr ncclConn_t recvConnType = RECVCONNTYPE;
  static constexpr int RTMD = RTMD_AR;
  PCCL_DEFINE_SIMPLE_FAMILY_PROTO_METRICS();
};

#define PCCL_ASSERT_SIMPLE_FAMILY_ORDER(Proto)                             \
  static_assert(Proto<2, 1, 4>::SlicePerChunk == 2,                        \
                #Proto ": first positional arg must be SlicePerChunk");   \
  static_assert(Proto<2, 1, 4>::StepPerSlice == 1,                         \
                #Proto ": second positional arg must be StepPerSlice");   \
  static_assert(Proto<2, 1, 4>::Unroll == 4,                               \
                #Proto ": third positional arg must be Unroll")

PCCL_ASSERT_SIMPLE_FAMILY_ORDER(ProtoSimple);
PCCL_ASSERT_SIMPLE_FAMILY_ORDER(ProtoSignal);
PCCL_ASSERT_SIMPLE_FAMILY_ORDER(ProtoSimpleBulk);
PCCL_ASSERT_SIMPLE_FAMILY_ORDER(ProtoSignalBulk);
PCCL_ASSERT_SIMPLE_FAMILY_ORDER(ProtoSimpleBulkAR);
#undef PCCL_ASSERT_SIMPLE_FAMILY_ORDER

static_assert(ProtoSimple<2, 1>::Unroll == COLL_UNROLL,
              "ProtoSimple<2,1> must default the third arg to COLL_UNROLL");
static_assert(ProtoSimple<2, 1, 4>::MultimemSrcs == 0, "MultimemSrcs must default to 0");
static_assert(ProtoSimple<2, 1, 4, 0, 0>::MultimemDsts == 0, "explicit multimem form must bind");

/* Fan (as in fan-in & fan-out) classes hold recv and send counts. The template
 * arguments are static bounds on the maximum values. Asymmetric counts are
 * independent. Symmetric is a static guarantee that nrecv==nsend, so it only
 * stores one value at runtime. This optimization save 32-bit register, but more
 * importantly uses fewer predicate registers when unrolling loops.
 */
template<int MaxRecv_, int MaxSend_>
struct FanAsymmetric {
  static constexpr int MaxRecv = MaxRecv_, MaxSend = MaxSend_;
  int nr, ns;
  FanAsymmetric() = default;
  __device__ FanAsymmetric(int nrecv, int nsend): nr(nrecv), ns(nsend) {
    // assert(nrecv <= MaxRecv && nsend <= MaxSend);
  }
  __device__ int nrecv() const { return MaxRecv ? nr : 0; }
  __device__ int nsend() const { return MaxSend ? ns : 0; }
};

template<int MaxArity>
struct FanSymmetric {
  static constexpr int MaxRecv = MaxArity, MaxSend = MaxArity;
  int n;
  FanSymmetric() = default;
  __device__ FanSymmetric(int nrecv, int nsend): n(nrecv) {
    // assert(nrecv == nsend && nrecv <= MaxArity);
  }
  __device__ int nrecv() const { return n; }
  __device__ int nsend() const { return n; }
};

static_assert(FanAsymmetric<2, 0>::MaxRecv == 2, "FanAsymmetric MaxRecv");
static_assert(FanAsymmetric<2, 0>::MaxSend == 0, "FanAsymmetric MaxSend");
static_assert(FanSymmetric<3>::MaxRecv == 3 && FanSymmetric<3>::MaxSend == 3,
              "FanSymmetric arity bounds both directions");
static_assert(sizeof(FanSymmetric<1>) < sizeof(FanAsymmetric<1, 1>),
              "FanSymmetric must store one count, not two");

// The primitives class. Specialized per protocol in the other headers.
template<typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p, bool isNetOffload = false>
class Primitives;

template<typename T, typename Fan, typename RedOp, int SlicePerChunk, int StepPerSlice, int Unroll, int Direct, typename Proto, class RealPrimitives, int P2p>
class PrimitivesSimpleBase;

template <typename Proto, int MaxSend>
__device__ __forceinline__ int ncclPrimsThreadsFromWorkers(int nworkers) {
#ifdef NO_EXTRA_WARP
  return nworkers;
#else
  constexpr bool postWarp =
    MaxSend > 0 && Proto::Id != NCCL_PROTO_LL && Proto::Id != NCCL_PROTO_LL128;
  return nworkers + (postWarp && nworkers >= NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE - WARP_SIZE
                     ? WARP_SIZE : 0);
#endif
}

// Used by LL & LL128 to implement direct members in the naive way.
template<typename RealPrimitives>
struct PrimitivesWithoutDirect {
  __device__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
  }
  __device__ void directSendFromOutput(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->sendFromOutput(outIx, eltN);
  }
  __device__ void directRecv(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->recv(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->copySend(inpIx, outIx, eltN, postOp);
  }
  __device__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->recvCopySend(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    return;
  }
  __device__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false, int reqChunkSize=0) {
    // Direct is only for the send part
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false, int reqChunkSize=0) {
    static_cast<RealPrimitives*>(this)->recvReduceSend(inpIx, eltN);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false, int reqChunkSize=0) {
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }

  __device__ void directRecvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    static_cast<RealPrimitives*>(this)->recvReduceCopy(inpIx, outIx, eltN, postOp);
  }
  __device__ void directGather(intptr_t outIx, int totalN, int eltN, int size) {
    static_cast<RealPrimitives*>(this)->gather(outIx, totalN, eltN, size);
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

static_assert(std::is_base_of<
                  PrimitivesSimpleBase<float, FanAsymmetric<1, 1>, FuncSum<float>, 2, 1, 4, 0,
                                       ProtoSimple<2, 1, 4>,
                                       Primitives<float, FuncSum<float>, FanAsymmetric<1, 1>, 0,
                                                  ProtoSimple<2, 1, 4>, 0>,
                                       0>,
                  Primitives<float, FuncSum<float>, FanAsymmetric<1, 1>, 0, ProtoSimple<2, 1, 4>, 0>>::value,
              "PrimitivesSimpleBase must take <SlicePerChunk, StepPerSlice, Unroll> like the Protos");

// workersFromThreads() inverts the ctor's carve: it takes the TOTAL thread count and
// removes the post warp above NCCL's threshold. Reading it as a worker count comes
// out WARP_SIZE low, so the cases that define the carve are pinned here.
using PrimsCarveCheck =
    Primitives<float, FuncSum<float>, FanAsymmetric<1, 1>, 1, ProtoSimple<2, 1, 4>, 0>;
using PrimsNoSendCarveCheck =
    Primitives<float, FuncSum<float>, FanAsymmetric<1, 0>, 1, ProtoSimple<2, 1, 4>, 0>;
#ifdef NO_EXTRA_WARP
static_assert(PrimsCarveCheck::workersFromThreads(128) == 128,
              "NO_EXTRA_WARP builds have no post warp to carve");
#else
static_assert(PrimsCarveCheck::workersFromThreads(128) == 96,
              "SIMPLE primitives that can send must carve one post warp out of the total");
static_assert(PrimsCarveCheck::workersFromThreads(3*WARP_SIZE) == 2*WARP_SIZE,
              "the carve must start exactly at NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE");
static_assert(PrimsCarveCheck::workersFromThreads(2*WARP_SIZE) == 2*WARP_SIZE,
              "no post warp below the threshold - the block cannot spare it");
static_assert(PrimsNoSendCarveCheck::workersFromThreads(128) == 128,
              "a recv-only Fan must not carve a post warp");
#endif

#endif
