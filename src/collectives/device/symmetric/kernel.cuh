/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_SYMMETRIC_KERNEL_H_
#define NCCL_DEVICE_SYMMETRIC_KERNEL_H_

#include "symmetric.h"

template<template<typename> class Red, typename T>
__device__ __forceinline__ void pccl_symRun_AllReduce_AGxLL_R(struct ncclSymDevArgs const* args);

template<template<typename> class Red, typename T>
__device__ __forceinline__ void pccl_symRun_AllReduce_RSxLD_AGxST(struct ncclSymDevArgs const* args);

__device__ __forceinline__ void pccl_symRun_AllGather_LL(struct ncclSymDevArgs const* args);
__device__ __forceinline__ void pccl_symRun_AllGather_ST(struct ncclSymDevArgs const* args);

template<template<typename> class Red, typename T>
__device__ __forceinline__ void pccl_symRun_ReduceScatter_LL(struct ncclSymDevArgs const* args);
template<template<typename> class Red, typename T>
__device__ __forceinline__ void pccl_symRun_ReduceScatter_LD(struct ncclSymDevArgs const* args);
template<template<typename> class Red, typename T>
#endif
