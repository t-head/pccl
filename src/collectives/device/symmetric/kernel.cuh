/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_SYMMETRIC_KERNEL_H_
#define NCCL_DEVICE_SYMMETRIC_KERNEL_H_

#include "sym_kernels.h"

template<template<typename> class Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R(ncclSymkDevWorkArgs const* args);

template<template<typename> class Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST(ncclSymkDevWorkArgs const* args);

__device__ __forceinline__ void ncclSymkRun_AllGather_LL(ncclSymkDevWorkArgs const* args);
__device__ __forceinline__ void ncclSymkRun_AllGather_ST(ncclSymkDevWorkArgs const* args);

template<template<typename> class Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL(ncclSymkDevWorkArgs const* args);
template<template<typename> class Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LD(ncclSymkDevWorkArgs const* args);
#endif
