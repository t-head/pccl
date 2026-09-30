/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_PTR__FUNCS_H_
#define _PCCL_DEVICE_PTR__FUNCS_H_
#include "ptr__types.h"
#include "core__funcs.h"
#include "comm__types.h"

#if __cplusplus

template<typename T>
template<typename U>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>::operator pcclSymPtr<U>() const {
  return {window, offset};
}

template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator+=(int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}
template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator+=(unsigned int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}

template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator+=(long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}
template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator+=(unsigned long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}

template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator+=(long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}
template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator+=(unsigned long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}

template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator-=(int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}
template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator-=(unsigned int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}

template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator-=(long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}
template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator-=(unsigned long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}

template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator-=(long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}
template<typename T>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T>& pcclSymPtr<T>::operator-=(unsigned long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}

#if PCCL_CHECK_HGGCCC
template<typename T>
PCCL_DEVICE_INLINE T* pcclSymPtr<T>::localPtr() const {
  if (window) {
    return (T*)pcclGetLocalPointer(window, offset);
  } else {
    return (T*)offset;
  }
}

template<typename T>
PCCL_DEVICE_INLINE T* pcclSymPtr<T>::lsaPtr(int peer) const {
  return (T*)pcclGetLsaPointer(window, offset, peer);
}

template<typename T>
PCCL_DEVICE_INLINE T* pcclSymPtr<T>::peerPtr(int peer) const {
  return (T*)pcclGetPeerPointer(window, offset, peer);
}

template<typename T>
PCCL_DEVICE_INLINE T* pcclSymPtr<T>::peerPtr(pcclTeam team, int peer) const {
  return (T*)pcclGetPeerPointer(window, offset, team, peer);
}

template<typename T>
PCCL_DEVICE_INLINE T* pcclSymPtr<T>::multimemPtr(pcclMultimemHandle mmHandle) const {
  assert(false && "multimem is not supported on PPU");
  return (T*)pcclGetMultimemPointer(window, offset, mmHandle);
}

template<typename T>
PCCL_DEVICE_INLINE T* pcclSymPtr<T>::lsaMultimemPtr(pcclDevComm const& comm) const {
  assert(false && "multimem is not supported on PPU");
  return (T*)pcclGetLsaMultimemPointer(window, offset, comm);
}
#endif // PCCL_CHECK_HGGCCC

template<typename T, typename Int>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T> operator+(pcclSymPtr<T> p, Int d) {
  return p += d;
}
template<typename T, typename Int>
PCCL_HOST_DEVICE_INLINE pcclSymPtr<T> operator-(pcclSymPtr<T> p, Int d) {
  return p -= d;
}
template<typename T>
PCCL_HOST_DEVICE_INLINE ptrdiff_t operator-(pcclSymPtr<T> a, pcclSymPtr<T> b) {
  return reinterpret_cast<T*>(a.offset) - reinterpret_cast<T*>(b.offset);
}

template<typename T>
PCCL_HOST_DEVICE_INLINE bool operator==(pcclSymPtr<T> a, pcclSymPtr<T> b) {
  return a.window == b.window && a.offset == b.offset;
}
template<typename T>
PCCL_HOST_DEVICE_INLINE bool operator!=(pcclSymPtr<T> a, pcclSymPtr<T> b) {
  return a.window != b.window || a.offset != b.offset;
}

#endif // __cplusplus
#endif // _PCCL_DEVICE_PTR__FUNCS_H_
