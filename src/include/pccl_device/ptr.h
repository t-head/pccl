/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_PTR_H_
#define _PCCL_DEVICE_PTR_H_
#include "core.h"
#include <stdint.h>

#if __cplusplus
template<typename T>
struct pcclSymPtr {
  using ElementType = T;
  pcclWindow_t window;
  size_t offset;

  PCCL_HOST_DEVICE_INLINE constexpr pcclSymPtr(pcclWindow_t window=nullptr, size_t offset=0)
    : window(window), offset(offset) {}

  template<typename U>
  PCCL_HOST_DEVICE_INLINE operator pcclSymPtr<U>() const;

  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator+=(int d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator+=(unsigned int d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator+=(long d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator+=(unsigned long d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator+=(long long d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator+=(unsigned long long d);

  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator-=(int d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator-=(unsigned int d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator-=(long d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator-=(unsigned long d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator-=(long long d);
  PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& operator-=(unsigned long long d);

  PCCL_DEVICE_INLINE T* localPtr() const;
  PCCL_DEVICE_INLINE T* lsaPtr(int peer) const;
  PCCL_DEVICE_INLINE T* peerPtr(int peer) const;
  PCCL_DEVICE_INLINE T* peerPtr(pcclTeam team, int peer) const;
  PCCL_DEVICE_INLINE T* multimemPtr(pcclMultimemHandle mmHandle) const;
  PCCL_DEVICE_INLINE T* lsaMultimemPtr(pcclDevComm const&) const;
};

template<typename T, typename Int>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T> operator+(pcclSymPtr<T> p, Int d);
template<typename T, typename Int>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T> operator-(pcclSymPtr<T> p, Int d);
template<typename T>
PCCL_HOST_DEVICE_INLINE ptrdiff_t operator-(pcclSymPtr<T> a, pcclSymPtr<T> b);

template<typename T, typename Int>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T> operator==(pcclSymPtr<T> a, pcclSymPtr<T> b);
template<typename T, typename Int>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T> operator!=(pcclSymPtr<T> a, pcclSymPtr<T> b);
#endif

#endif
