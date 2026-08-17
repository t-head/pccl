/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_PTR_H_
#define _NCCL_DEVICE_PTR_H_
#include "core.h"
#include <stdint.h>

#if __cplusplus
template<typename T>
struct ncclSymPtr {
  using ElementType = T;
  pcclWindow_t window;
  size_t offset;

  NCCL_HOST_DEVICE_INLINE constexpr ncclSymPtr(pcclWindow_t window=nullptr, size_t offset=0)
    : window(window), offset(offset) {}

  template<typename U>
  NCCL_HOST_DEVICE_INLINE operator ncclSymPtr<U>() const;

  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(unsigned int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(unsigned long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(long long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator+=(unsigned long long d);

  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(unsigned int d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(unsigned long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(long long d);
  NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& operator-=(unsigned long long d);

  NCCL_DEVICE_INLINE T* localPtr() const;
  NCCL_DEVICE_INLINE T* lsaPtr(int peer) const;
  NCCL_DEVICE_INLINE T* peerPtr(int peer) const;
  NCCL_DEVICE_INLINE T* peerPtr(ncclTeam team, int peer) const;
  NCCL_DEVICE_INLINE T* multimemPtr(ncclMultimemHandle mmHandle) const;
  NCCL_DEVICE_INLINE T* lsaMultimemPtr(ncclDevComm const&) const;
};

template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator+(ncclSymPtr<T> p, Int d);
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator-(ncclSymPtr<T> p, Int d);
template<typename T>
NCCL_HOST_DEVICE_INLINE ptrdiff_t operator-(ncclSymPtr<T> a, ncclSymPtr<T> b);

template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator==(ncclSymPtr<T> a, ncclSymPtr<T> b);
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator!=(ncclSymPtr<T> a, ncclSymPtr<T> b);
#endif

#endif
