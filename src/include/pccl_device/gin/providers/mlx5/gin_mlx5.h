/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_GDAKI_MLX_H_
#define _PCCL_DEVICE_GIN_GDAKI_MLX_H_

#include <cstdint>
#ifndef DOCA_VERBS_USE_HGGC_WRAPPER
#define DOCA_VERBS_USE_HGGC_WRAPPER
#endif

#ifndef DOCA_VERBS_USE_NET_WRAPPER
#define DOCA_VERBS_USE_NET_WRAPPER
#endif

#ifdef PCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
#define DOCA_GPUNETIO_VERBS_ENABLE_DEBUG 1
#endif

#include "../../gin_gdaki_device_common.h"
#include "gin_mlx5_device_host_common.h"

// DOCA-GPUNetIO is third-party (NVIDIA): its packed-typedef structs draw
// Silence -Wignored-attributes for the doca_gpunetio include (the structs
// have explicit static_assert sizeof checks so __packed__ is a no-op here).
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#include "doca_gpunetio/doca_gpunetio_device.h"
#pragma GCC diagnostic pop

struct pcclGinGdakiMlxRequest {
  int peer;
  doca_gpu_dev_verbs_ticket_t docaTicket;
};
static_assert(sizeof(pcclGinGdakiMlxRequest) <= sizeof(pcclGinRequest_t),
              "pcclGinGdakiMlxRequest must fit in pcclGinRequest_t");

#ifdef PCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
#include <stdio.h>
#endif

namespace pccl {
namespace gin {
namespace gdaki_mlx {

PCCL_DEVICE_INLINE uint32_t docaOptFlagsFromGinOptFlags(uint32_t ginOptFlags) {
  return DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT
    | (!!(ginOptFlags & pcclGinOptFlagsMaySkipCreditCheck) * DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK)
    | (!!(ginOptFlags & pcclGinOptFlagsAggregateRequests) * DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING);
}

// GDAKI-only interface for a thread-issued group of one-WQE puts. `peer` is a
// GIN-relative peer (the public pcclGin layer must translate a team-relative
// rank before calling this internal interface).
struct PutGroupThread {
  doca_gpu_dev_verbs_put_group_thread_t docaGroup;
  int peer;
};

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
PCCL_DEVICE_INLINE static void putGroupThreadBeginMode(
    pcclGinCtx ctx, int peer, uint32_t count, PutGroupThread* group) {
  using pccl::utility::loadConst;
  pcclGinGdakiMlxGPUContext* gdaki =
    &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
  doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
  doca_gpu_dev_verbs_put_group_thread_begin<resource_sharing_mode>(
    qp, count, &group->docaGroup);
  group->peer = peer;
}

PCCL_DEVICE_INLINE static void putGroupThreadBegin(
    pcclGinCtx ctx, int peer, uint32_t count, PutGroupThread* group) {
  switch ((pcclGinResourceSharingMode)ctx.resourceSharingMode) {
    case PCCL_GIN_RESOURCE_SHARING_CTA:
      putGroupThreadBeginMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(
        ctx, peer, count, group);
      break;
    default:
      putGroupThreadBeginMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        ctx, peer, count, group);
      break;
  }
}

PCCL_DEVICE_INLINE static void putGroupThreadPrepare(
    pcclGinCtx /*ctx*/, PutGroupThread* group, uint32_t ordinal,
    pcclGinWindow_t dstWin, size_t dstOff,
    pcclGinWindow_t srcWin, size_t srcOff, size_t bytes) {
  using pccl::utility::loadConst;
  pcclGinGdakiMlxMemHandle* dstMh = (pcclGinGdakiMlxMemHandle*)dstWin;
  pcclGinGdakiMlxMemHandle* srcMh = (pcclGinGdakiMlxMemHandle*)srcWin;
  doca_gpu_dev_verbs_addr raddr, laddr;
  raddr.addr = dstOff;
  raddr.key = loadConst(loadConst(&dstMh->rkeys) + group->peer);
  laddr.addr = srcOff;
  laddr.key = loadConst(&srcMh->lkey);
  doca_gpu_dev_verbs_put_group_thread_prepare(
    &group->docaGroup, ordinal, raddr, laddr, bytes);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
PCCL_DEVICE_INLINE static void putGroupThreadCommitMode(
    PutGroupThread* group, doca_gpu_dev_verbs_ticket_t* outTicket) {
  doca_gpu_dev_verbs_put_group_thread_commit<resource_sharing_mode>(
    &group->docaGroup, outTicket);
}

PCCL_DEVICE_INLINE static void putGroupThreadCommit(
    pcclGinCtx ctx, PutGroupThread* group,
    hggc::thread_scope required, hggc::thread_scope given,
    doca_gpu_dev_verbs_ticket_t* outTicket) {
  // The grouped-put submission path releases prepared WQEs at GPU scope.
  // The caller must finish payload production and synchronize its producer
  // threads before entering commit; the WQE release cannot do that for them.
  // Add a system release when the requested scope is stronger than the one
  // already supplied by the caller.
  // hggc::thread_scope_system has the lowest value.
  if ((required == hggc::thread_scope_system) && (given > required))
    doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();

  switch ((pcclGinResourceSharingMode)ctx.resourceSharingMode) {
    case PCCL_GIN_RESOURCE_SHARING_CTA:
      putGroupThreadCommitMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(
        group, outTicket);
      break;
    default:
      putGroupThreadCommitMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        group, outTicket);
      break;
  }
#ifdef PCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
  doca_gpu_dev_verbs_wait(group->docaGroup.qp);
#endif
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop>
PCCL_DEVICE_INLINE static void putImplMode(pcclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                              pcclGinWindow_t dstWin, size_t dstOff, pcclGinWindow_t srcWin,
                                              size_t srcOff, size_t bytes, bool hasSignal,
                                              size_t signalOffset, __be32 signalKey, pcclGinSignalOp_t signalOp,
                                              uint64_t signalOpArg, bool hasCounter,
                                              pcclGinCounter_t counterId, bool hasDescriptor,
                                              pcclGinDescriptorSmem* descriptor,
                                              hggc::thread_scope required, hggc::thread_scope given,
                                              uint32_t optFlags) {
  using pccl::utility::loadConst;
  coop.sync();
  if (coop.thread_rank() == 0) {
    pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
    doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
    doca_gpu_dev_verbs_qp* companion_qp;
    pcclGinGdakiMlxMemHandle* dstMh = (pcclGinGdakiMlxMemHandle*)dstWin;
    pcclGinGdakiMlxMemHandle* srcMh = (pcclGinGdakiMlxMemHandle*)srcWin;
    uint32_t codeOpt = pccl::gin::gdaki_mlx::docaOptFlagsFromGinOptFlags(optFlags);

    doca_gpu_dev_verbs_addr raddr, laddr;
    if (hasWins) {
      raddr.addr = dstOff;
      raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);
      laddr.addr = srcOff, laddr.key = loadConst(&srcMh->lkey);
    }

    doca_gpu_dev_verbs_addr sig_raddr, sig_laddr;
    if (hasSignal) {
      if (signalOp == pcclGinSignalInc) signalOpArg = 1;
      sig_raddr.addr = signalOffset;
      sig_raddr.key = signalKey;
      sig_laddr.addr = 0;
      sig_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
    }

    doca_gpu_dev_verbs_addr counter_raddr, counter_laddr;
    if (hasCounter) {
      companion_qp = loadConst(&gdaki->companion_gdqp) + peer;
      counter_raddr.addr = sizeof(uint64_t) * (counterId + loadConst(&gdaki->counters_table.offset));
      counter_raddr.key = loadConst(loadConst(&gdaki->counters_table.rkeys) + ctx.rank);
      counter_laddr.addr = 0;
      counter_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
    }

    // hggc::thread_scope_system has the lowest value
    // DOCA guarantees SCOPE_GPU. Only add another release if SCOPE_SYSTEM is required.
    if ((required == hggc::thread_scope_system) && (given > required)) {
      doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    }

    if (hasWins) {
      if (hasSignal && hasCounter) {
        doca_gpu_dev_verbs_put_signal_counter<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
          qp, raddr, laddr, bytes, sig_raddr, sig_laddr, signalOpArg, companion_qp, counter_raddr,
          counter_laddr, 1, codeOpt);
      } else if (hasSignal) {
        doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
          qp, raddr, laddr, bytes, sig_raddr, sig_laddr, signalOpArg, codeOpt);
      } else if (hasCounter) {
        doca_gpu_dev_verbs_put_counter<resource_sharing_mode>(
          qp, raddr, laddr, bytes, companion_qp, counter_raddr, counter_laddr, 1, codeOpt);
      } else {
        doca_gpu_dev_verbs_put<resource_sharing_mode>(qp, raddr, laddr, bytes, codeOpt);
      }
    } else {
      if (hasCounter) {
        doca_gpu_dev_verbs_signal_counter<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
          qp, sig_raddr, sig_laddr, signalOpArg, companion_qp, counter_raddr, counter_laddr, 1, codeOpt);
      } else {
        doca_gpu_dev_verbs_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
          qp, sig_raddr, sig_laddr, signalOpArg, codeOpt);
      }
    }

#ifdef PCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
    doca_gpu_dev_verbs_wait(qp);
    if (hasCounter) doca_gpu_dev_verbs_wait(companion_qp);
#endif
  }
  coop.sync();
}

template <typename Coop>
PCCL_DEVICE_INLINE static void putImpl(pcclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                              pcclGinWindow_t dstWin, size_t dstOff, pcclGinWindow_t srcWin,
                                              size_t srcOff, size_t bytes, bool hasSignal,
                                              size_t signalOffset, __be32 signalKey, pcclGinSignalOp_t signalOp,
                                              uint64_t signalOpArg, bool hasCounter,
                                              pcclGinCounter_t counterId, bool hasDescriptor,
                                              pcclGinDescriptorSmem* descriptor,
                                              hggc::thread_scope required, hggc::thread_scope given,
                                              uint32_t optFlags) {
  switch ((pcclGinResourceSharingMode)ctx.resourceSharingMode) {
    case PCCL_GIN_RESOURCE_SHARING_CTA:
      putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(
        ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
        hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
        hasCounter, counterId, hasDescriptor, descriptor,
        required, given, optFlags);
      break;
    default:
      putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
        hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
        hasCounter, counterId, hasDescriptor, descriptor,
        required, given, optFlags);
      break;
  }
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop, typename T>
PCCL_DEVICE_INLINE static void putValueImplMode(pcclGinCtx ctx, Coop coop, int peer, pcclGinWindow_t dstWin,
                                      size_t dstOff, T srcData, bool hasSignal,
                                      size_t signalOffset, __be32 signalKey, pcclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags) {
  using pccl::utility::loadConst;

  coop.sync();
  if (coop.thread_rank() == 0) {
    pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
    doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
    pcclGinGdakiMlxMemHandle* dstMh = (pcclGinGdakiMlxMemHandle*)dstWin;
    uint32_t codeOpt = pccl::gin::gdaki_mlx::docaOptFlagsFromGinOptFlags(optFlags);

    doca_gpu_dev_verbs_addr raddr;
    raddr.addr = dstOff;
    raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);

    doca_gpu_dev_verbs_addr sig_raddr, sig_laddr;
    if (hasSignal) {
      if (signalOp == pcclGinSignalInc) signalOpArg = 1;
      sig_raddr.addr = signalOffset;
      sig_raddr.key = signalKey;
      sig_laddr.addr = 0;
      sig_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
    }

    // hggc::thread_scope_system has the lowest value
    // DOCA guarantees SCOPE_GPU. Only add another release if SCOPE_SYSTEM is required.
    if ((required == hggc::thread_scope_system) && (given > required)) {
      doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    }

    if (hasSignal) {
      doca_gpu_dev_verbs_p_signal<T, DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
        qp, raddr, srcData, sig_raddr, sig_laddr, signalOpArg, codeOpt);
    } else {
      doca_gpu_dev_verbs_p<T, resource_sharing_mode>(qp, raddr, srcData, codeOpt);
    }

#ifdef PCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
    doca_gpu_dev_verbs_wait(qp);
#endif
  }
  coop.sync();
}

template <typename Coop, typename T>
PCCL_DEVICE_INLINE static void putValueImpl(pcclGinCtx ctx, Coop coop, int peer, pcclGinWindow_t dstWin,
                                      size_t dstOff, T srcData, bool hasSignal,
                                      size_t signalOffset, __be32 signalKey, pcclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags) {
  switch ((pcclGinResourceSharingMode)ctx.resourceSharingMode) {
    case PCCL_GIN_RESOURCE_SHARING_CTA:
      putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(
        ctx, coop, peer, dstWin, dstOff, srcData,
        hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
        hasDescriptor, descriptor, required, given, optFlags);
      break;
    default:
      putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        ctx, coop, peer, dstWin, dstOff, srcData,
        hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
        hasDescriptor, descriptor, required, given, optFlags);
      break;
  }
}

} // namespace gdaki_mlx
} // namespace gin
} // namespace pccl

template <>
struct pcclGinGdakiApi_Get<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, pcclGinWindow_t remoteWin, size_t remoteOff,
                                      pcclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, pcclGinDescriptorSmem* descriptor,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    using pccl::utility::loadConst;
    coop.sync();
    if (coop.thread_rank() == 0) {
      pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
      doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
      pcclGinGdakiMlxMemHandle* remoteMh = (pcclGinGdakiMlxMemHandle*)remoteWin;
      pcclGinGdakiMlxMemHandle* localMh = (pcclGinGdakiMlxMemHandle*)localWin;
      doca_gpu_dev_verbs_addr raddr, laddr;
      raddr.addr = remoteOff;
      raddr.key = loadConst(loadConst(&remoteMh->rkeys) + peer);
      laddr.addr = localOff;
      laddr.key = loadConst(&localMh->lkey);
      doca_gpu_dev_verbs_addr uninitialized_daddr{};
      doca_gpu_dev_verbs_ticket_t unused_out_ticket;
      uint32_t codeOpt = pccl::gin::gdaki_mlx::docaOptFlagsFromGinOptFlags(optFlags);
      doca_gpu_dev_verbs_get(
          qp, raddr, laddr, bytes, uninitialized_daddr, &unused_out_ticket, codeOpt);
    }
    coop.sync();
  }
};

template <>
struct pcclGinGdakiApi_FlushAsync<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, int peer, pcclGinRequest_t* outRequest, uint32_t optFlags) {
    using pccl::utility::loadConst;
    pcclGinGdakiMlxRequest* req = reinterpret_cast<pcclGinGdakiMlxRequest*>(outRequest);
    req->peer = peer;
    pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
    doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
    doca_gpu_dev_verbs_addr daddr;
    daddr.addr = 0;
    daddr.key = loadConst(&gdaki->sink_buffer_lkey);
    doca_gpu_dev_verbs_mcst<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
        DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(
        qp, daddr, &req->docaTicket);
  }
};

template <>
struct pcclGinGdakiApi_Wait<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinRequest_t& request, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor, hggc::memory_order ord, uint32_t* abortFlag) {
    (void)ord; // Ignore. DOCA already guarantees memory_order_acquire
    using pccl::utility::loadConst;
    using pccl::utility::testAbort;
    pcclGinGdakiMlxRequest& req = reinterpret_cast<pcclGinGdakiMlxRequest&>(request);
    pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
    doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + req.peer;
    if (abortFlag) {
      uint32_t steps = 0;
      int status = EBUSY;
      while (status != 0 && !testAbort(abortFlag, steps)) {
        status = doca_gpu_dev_verbs_poll_one_cq_at(&qp->cq_sq, req.docaTicket);
      }
    } else {
      doca_gpu_dev_verbs_wait(qp, req.docaTicket);
    }
  }
};

template <>
struct pcclGinGdakiApi_Put<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      pcclGinWindow_t dstWin, size_t dstOff, pcclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      pcclGinSignalDescriptor signal, pcclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      pcclGinCounter_t counterId, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    using pccl::utility::loadConst;
    size_t signalOffset = 0;
    __be32 signalKey = 0;
    bool hasSignal = signal.type != PCCL_GIN_SIGNAL_TYPE_NONE;
    if (signal.type == PCCL_GIN_SIGNAL_TYPE_INDEXED) {
      pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
      signalOffset = sizeof(uint64_t) * (signal.indexedSignal.signalId + loadConst(&gdaki->signals_table.offset));
      signalKey = loadConst(loadConst(&gdaki->signals_table.rkeys) + peer);
    } else if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
      pcclGinGdakiMlxMemHandle* signalMh = (pcclGinGdakiMlxMemHandle*)signal.vaSignal.signalWindow;
      signalKey = loadConst(loadConst(&signalMh->rkeys) + peer);
      signalOffset = signal.vaSignal.signalOffset;
    }
    pccl::gin::gdaki_mlx::putImpl(
      ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
      hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
      hasCounter, counterId, hasDescriptor, descriptor,
      required, given, optFlags
    );
  }
};

template <>
struct pcclGinGdakiApi_PutValue<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  template <typename Coop, typename T>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, int peer, pcclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal,
                                      pcclGinSignalDescriptor signal, pcclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      pcclGinDescriptorSmem* descriptor,
                                      hggc::thread_scope required, hggc::thread_scope given,
                                      uint32_t optFlags = pcclGinOptFlagsDefault) {
    using pccl::utility::loadConst;
    size_t signalOffset = 0;
    __be32 signalKey = 0;
    bool hasSignal = signal.type != PCCL_GIN_SIGNAL_TYPE_NONE;
    if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
      pcclGinGdakiMlxMemHandle* signalMh = (pcclGinGdakiMlxMemHandle*)signal.vaSignal.signalWindow;
      signalKey = loadConst(loadConst(&signalMh->rkeys) + peer);
      signalOffset = signal.vaSignal.signalOffset;
    } else if (signal.type == PCCL_GIN_SIGNAL_TYPE_INDEXED) {
      pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
      signalOffset = sizeof(uint64_t) * (signal.indexedSignal.signalId + loadConst(&gdaki->signals_table.offset));
      signalKey = loadConst(loadConst(&gdaki->signals_table.rkeys) + peer);
    }
    pccl::gin::gdaki_mlx::putValueImpl(
      ctx, coop, peer, dstWin, dstOff, srcVal,
      hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
      hasDescriptor, descriptor, required, given, optFlags
    );
  }
};

template <>
struct pcclGinGdakiApi_ResetCounter<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinCounter_t counterId) {
    using pccl::utility::loadConst;
    pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
    loadConst(&gdaki->counters_table.buffer)[counterId] = 0;
  }
};

template <>
struct pcclGinGdakiApi_ResetSignal<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, pcclGinSignalDescriptor signal) {
    using pccl::utility::loadConst;
    if (signal.type == PCCL_GIN_SIGNAL_TYPE_VA) {
      uint64_t* signalPtr = (uint64_t*)pcclGetLocalPointer(signal.vaSignal.pcclWindow, signal.vaSignal.signalOffset);
      *signalPtr = 0;
    } else {
      pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
      loadConst(&gdaki->signals_table.buffer)[signal.indexedSignal.signalId] = 0;
    }
  }
};

template <>
struct pcclGinGdakiApi_GetCounterPtr<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx ctx, pcclGinCounter_t counterId) {
    using pccl::utility::loadConst;
    pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
    return loadConst(&gdaki->counters_table.buffer) + counterId;
  }
};

template <>
struct pcclGinGdakiApi_GetSignalPtr<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  PCCL_DEVICE_INLINE static uint64_t* call(pcclGinCtx ctx, pcclGinSignal_t signalId) {
    using pccl::utility::loadConst;
    pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
    return loadConst(&gdaki->signals_table.buffer) + signalId;
  }
};

template <>
struct pcclGinGdakiApi_Flush<PCCL_GIN_GDAKI_PROVIDER_MLX5> {
  template <typename Coop>
  PCCL_DEVICE_INLINE static void call(pcclGinCtx ctx, Coop coop, hggc::memory_order ord, uint32_t* abortFlag) {
    (void)ord; // Ignore. DOCA already guarantees memory_order_acquire
    using pccl::utility::loadConst;
    using pccl::utility::testAbort;

    pcclGinGdakiMlxGPUContext* gdaki = &((struct pcclGinGdakiMlxGPUContext*)ctx.handle)[ctx.contextId];
    doca_gpu_dev_verbs_qp* qps = loadConst(&gdaki->gdqp);

    if (abortFlag) {
      uint32_t steps = 0;
      #pragma unroll 1
      for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
        int status = EBUSY;
        uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qps[peer].sq_rsvd_index);
        if (ticket == 0)
          return;
        --ticket;
        while (status != 0 && !testAbort(abortFlag, steps)) {
          status = doca_gpu_dev_verbs_poll_one_cq_at(&qps[peer].cq_sq, ticket);
        }
      }
    } else {
      #pragma unroll 1
      for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
        doca_gpu_dev_verbs_wait(qps + peer);
      }
    }
  }
};

#endif /* _PCCL_DEVICE_GIN_GDAKI_MLX_H_ */
