/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_GIN_TROCE_UTILS_H_
#define _PCCL_DEVICE_GIN_TROCE_UTILS_H_

#include "../../utility.h"

namespace pccl {
namespace gin {
namespace troce {

// Scoped fences and relaxed loads adapted from NCCL v2.32.3-1 GPI helpers.
template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE void fenceAcquire() {
  uint32_t dummy;
  uint32_t value;
  if (scope == hggc::thread_scope_block)
    asm volatile("ld.acquire.cta.b32 %0, [%1];" : "=r"(value) : "l"(&dummy));
  else if (scope == hggc::thread_scope_device)
    asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(value) : "l"(&dummy));
  else if (scope == hggc::thread_scope_system)
    asm volatile("ld.acquire.sys.b32 %0, [%1];" : "=r"(value) : "l"(&dummy));
}

template <hggc::thread_scope scope = hggc::thread_scope_device>
PCCL_DEVICE_INLINE void fenceRelease() {
  uint32_t dummy;
  const uint32_t value = 0;
  if (scope == hggc::thread_scope_block)
    asm volatile("st.release.cta.u32 [%0], %1;" : : "l"(&dummy), "r"(value));
  else if (scope == hggc::thread_scope_device)
    asm volatile("st.release.gpu.u32 [%0], %1;" : : "l"(&dummy), "r"(value));
  else if (scope == hggc::thread_scope_system)
    asm volatile("st.release.sys.u32 [%0], %1;" : : "l"(&dummy), "r"(value));
}

PCCL_DEVICE_INLINE void storeRelaxed(uint8_t *ptr, uint8_t val) {
  asm volatile("st.weak.cs.u8 [%0], %1;" : : "l"(ptr), "r"(uint32_t(val)));
}

PCCL_DEVICE_INLINE void storeRelaxed(uint16_t *ptr, uint16_t val) {
  asm volatile("st.weak.cs.u16 [%0], %1;" : : "l"(ptr), "r"(uint32_t(val)));
}

PCCL_DEVICE_INLINE void storeRelaxed(uint32_t *ptr, uint32_t val) {
  asm volatile("st.weak.cs.u32 [%0], %1;" : : "l"(ptr), "r"(val));
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
    asm volatile("ld.relaxed.cta.b64 %0, [%1];" : "=l"(value) : "l"(ptr));
    return value;
  } else {
    uint64_t value;
    asm volatile("ld.relaxed.gpu.b64 %0, [%1];" : "=l"(value) : "l"(ptr));
    return value;
  }
}

PCCL_DEVICE_INLINE uint64_t loadRelaxedSysGlobal(uint64_t *ptr) {
  uint64_t value;
  asm volatile("ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(value) : "l"(ptr));
  return value;
}

}  // namespace troce
}  // namespace gin
}  // namespace pccl

#endif  // _PCCL_DEVICE_GIN_TROCE_UTILS_H_
