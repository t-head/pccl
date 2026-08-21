/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_BARRIER__FUNCS_H_
#define _PCCL_DEVICE_BARRIER__FUNCS_H_
#include "barrier__types.h"
#include "lsa_barrier__funcs.h"
#if defined(PCCL_OS_LINUX)
#include "gin_barrier__funcs.h"
#endif
#include "../utility.h"

#if PCCL_CHECK_HGGCCC
template<typename Coop>
PCCL_DEVICE_INLINE pcclBarrierSession<Coop>::pcclBarrierSession(
    Coop coop, pcclTeam innerTeam, pcclTeam outerTeam, pcclGin gin,
    pcclLsaBarrierHandle innerHandle, pcclGinBarrierHandle outerHandle,
    uint32_t index, bool multimem, pcclMultimemHandle innerMmHandle
  ):
  pcclBarrierSession_internal<Coop>(coop,
    pccl::utility::present(gin),
    pccl::utility::present(coop, gin.comm, innerTeam, innerHandle, index, multimem, innerMmHandle),
    pccl::utility::present(coop, gin, outerTeam, outerHandle, index)
  ) {
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclBarrierSession<Coop>::pcclBarrierSession(
    Coop coop, pcclTeamTagWorld, pcclGin gin, uint32_t index, bool multimem
  ):
  pcclBarrierSession<Coop>(
    coop, pcclTeamLsa(gin.comm), pcclTeamRail(gin.comm), gin,
    gin.comm.hybridLsaBarrier, gin.comm.hybridRailGinBarrier,
    index, multimem, gin.comm.lsaMultimem
  ) {
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclBarrierSession<Coop>::pcclBarrierSession(
    Coop coop, pcclTeamTagLsa, pcclDevComm const& comm, uint32_t index, bool multimem
  ):
  pcclBarrierSession_internal<Coop>(coop,
    pccl::utility::Absent(),
    pccl::utility::present(coop, comm, pcclTeamLsa(comm), comm.hybridLsaBarrier, index, multimem, comm.lsaMultimem),
    pccl::utility::Absent()
  ) {
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclBarrierSession<Coop>::pcclBarrierSession(
    Coop coop, pcclTeamTagRail, pcclGin gin, uint32_t index
  ):
  pcclBarrierSession_internal<Coop>(coop,
    pccl::utility::present(gin),
    pccl::utility::Absent(),
    pccl::utility::present(coop, gin, pcclTeamRail(gin.comm), gin.comm.hybridRailGinBarrier, index)
  ) {
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclLsaBarrierSession<Coop>& pcclBarrierSession<Coop>::lsaBarrier() {
  return this->innerLsaBar.thing;
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclGinBarrierSession<Coop>& pcclBarrierSession<Coop>::ginBarrier() {
  return this->outerGinBar.thing;
}

template<typename Coop>
PCCL_DEVICE_INLINE void pcclBarrierSession<Coop>::sync(Coop, hggc::memory_order ord, pcclGinFenceLevel fence) {
  if (this->innerLsaBar.present) {
    this->innerLsaBar.thing.sync(this->coop, this->outerGinBar.present ? pccl::utility::releaseOrderOf(ord) : ord);
  }
  if (this->outerGinBar.present) {
    this->outerGinBar.thing.sync(this->coop, this->innerLsaBar.present ? pccl::utility::acquireOrderOf(ord) : ord, fence);
  }
}

template<typename Coop>
PCCL_DEVICE_INLINE pcclResult_t pcclBarrierSession<Coop>::sync(
    Coop, hggc::memory_order ord, pcclGinFenceLevel fence, uint64_t timeoutCycles) {
  pcclResult_t lsaResult = pcclSuccess, railResult = pcclSuccess;

  // Inner LSA barrier (if present) - detects remote CTA/rank issues
  if (this->innerLsaBar.present) {
    uint64_t startCycle = clock64();
    lsaResult = this->innerLsaBar.thing.sync(
      this->coop,
      this->outerGinBar.present ? pccl::utility::releaseOrderOf(ord) : ord,
      timeoutCycles
    );
    uint64_t elapsed = clock64() - startCycle;
    timeoutCycles -= min(elapsed, timeoutCycles);
    // Because threads within a coop don't synchronize about the timeout condition,
    // we need to invoke the second barrier even if the first one times out,
    // to ensure that all the threads arrive at the coop sync.
  }

  // Outer GIN barrier (if present) - detects remote GPU/network issues
  if (this->outerGinBar.present) {
    railResult = this->outerGinBar.thing.sync(
      this->coop,
      this->innerLsaBar.present ? pccl::utility::acquireOrderOf(ord) : ord,
      fence,
      timeoutCycles
    );
  }
  return lsaResult != pcclSuccess ? lsaResult : railResult;
}
#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_BARRIER__FUNCS_H_
