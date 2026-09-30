/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_LL_A2A_H_
#define _PCCL_DEVICE_LL_A2A_H_
#include "impl/core__types.h"

struct pcclLLA2AHandle;

PCCL_EXTERN_C __host__ int pcclLLA2ACalcSlots(int maxElts, int maxEltSize);

PCCL_EXTERN_C __host__ pcclResult_t pcclLLA2ACreateRequirement(int nBlocks, int nSlots, pcclLLA2AHandle_t* outHandle, pcclDevResourceRequirements_t* outReq);

#if PCCL_CHECK_HGGCCC
template<typename Coop>
struct pcclLLA2ASession_internal;

template<typename Coop>
struct pcclLLA2ASession: pcclLLA2ASession_internal<Coop> {
  PCCL_DEVICE_INLINE pcclLLA2ASession(Coop, pcclDevComm const&, pcclTeam, pcclLLA2AHandle, uint32_t block, int maxElts, bool multimem=false, pcclMultimemHandle mmHandle={});

  PCCL_DEVICE_INLINE ~pcclLLA2ASession();

  pcclLLA2ASession(pcclLLA2ASession const&) = delete; // Sessions are not copyable

  template<typename T>
  PCCL_DEVICE_INLINE void send(int peer, int slot, T data);

  template<typename T>
  PCCL_DEVICE_INLINE void bcast(int slot, T data);

  template<typename T>
  PCCL_DEVICE_INLINE T recv(int slot);

  template<int MinEltCount, int MaxEltCount, typename T>
  PCCL_DEVICE_INLINE void recvUnrolled(int eltStart, int eltCount, int eltStride, T(&vals)[MaxEltCount]);

  template<int Unroll, typename Elt, typename EltToAcc, typename Reduce>
  PCCL_DEVICE_INLINE auto recvReduce(int eltStart, int eltCount, int eltStride, EltToAcc eltToAcc, Reduce red)
    -> decltype(eltToAcc(pccl::utility::declval<Elt>())) ;

  // End an alltoall region. For every peer in team you must have done both of the
  // following each of which can be accomplished using any thread in coop:
  //  1. Targeted that peer with at least one send().
  //  2. Received from a slot targeted by that peer.
  PCCL_DEVICE_INLINE void endEpoch(Coop);
};
#endif

#endif // _PCCL_DEVICE_LL_A2A_H_
