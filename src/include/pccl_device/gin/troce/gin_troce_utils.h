/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCLX_DEVICE_GIN_TROCE_UTILS_H_
#define _PCCLX_DEVICE_GIN_TROCE_UTILS_H_

#include "../../utility.h"

namespace pcclx {
namespace gin {
namespace troce {

template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE void fenceAcquire() {
  if (scope == hggc::thread_scope_block)
    asm volatile("ppu.fence.acquire.blk;" ::: "memory");
  else if (scope == hggc::thread_scope_device)
    asm volatile("ppu.fence.acquire.gpu;" ::: "memory");
  else if (scope == hggc::thread_scope_system)
    asm volatile("ppu.fence.acquire.sys;" ::: "memory");
}

template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE void fenceRelease() {
  if (scope == hggc::thread_scope_block)
    asm volatile("ppu.fence.release.blk;" ::: "memory");
  else if (scope == hggc::thread_scope_device)
    asm volatile("ppu.fence.release.gpu;" ::: "memory");
  else if (scope == hggc::thread_scope_system)
    asm volatile("ppu.fence.release.sys;" ::: "memory");
}

PCCL_DEVICE_INLINE void storeRelaxed(uint8_t *ptr, uint8_t val) {
  asm volatile("ppu.st.weak.cs.u8 [%0], %1;" : : "l"(ptr), "r"(uint32_t(val)));
}

PCCL_DEVICE_INLINE void storeRelaxed(uint16_t *ptr, uint16_t val) {
  asm volatile("ppu.st.weak.cs.u16 [%0], %1;" : : "l"(ptr), "r"(uint32_t(val)));
}

PCCL_DEVICE_INLINE void storeRelaxed(uint32_t *ptr, uint32_t val) {
  asm volatile("ppu.st.weak.cs.u32 [%0], %1;" : : "l"(ptr), "r"(val));
}

// Publish preceding writes before updating the system-visible doorbell.
PCCL_DEVICE_INLINE void storeRelease(uint64_t *ptr, uint64_t val) {
  hggc::atomic_ref<uint64_t, hggc::thread_scope_system> ref(*ptr);
  fenceRelease();
  ref.store(val, hggc::memory_order_relaxed);
}

template <typename T>
PCCL_DEVICE_INLINE T atomicRead(T *ptr) {
  return *(volatile T *)ptr;
}

template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE uint64_t loadRelaxed(uint64_t *ptr) {
  if (scope == hggc::thread_scope_block) {
    uint64_t value;
    asm volatile("ppu.ld.relaxed.blk.b64 %0, [%1];" : "=l"(value) : "l"(ptr));
    return value;
  } else {
    uint64_t value;
    asm volatile("ppu.ld.relaxed.gpu.b64 %0, [%1];" : "=l"(value) : "l"(ptr));
    return value;
  }
}

PCCL_DEVICE_INLINE uint64_t loadRelaxedSysGlobal(uint64_t *ptr) {
  uint64_t value;
  asm volatile("ppu.ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(value) : "l"(ptr));
  return value;
}

}  // namespace troce
}  // namespace gin
}  // namespace pcclx

#endif  // _PCCLX_DEVICE_GIN_TROCE_UTILS_H_
