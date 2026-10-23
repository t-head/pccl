/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_HOSTLIB_ONLY
#include "pccl_device/coop.h"
#include "pccl_device/impl/barrier__funcs.h"
#include "pccl_device/impl/comm__funcs.h"
#include "pccl_device/impl/core__funcs.h"
#include "pccl_device/impl/ll_a2a__funcs.h"
#include "pccl_device/impl/lsa_barrier__funcs.h"
#include "pccl_device/impl/lsa_bulk__funcs.h"
#if !defined(PCCL_OS_WINDOWS)
#include "pccl_device/impl/gin__funcs.h"
#include "pccl_device/impl/gin_barrier__funcs.h"
#endif
#include "pccl_device/impl/ptr__funcs.h"
#include "pccl_device/impl/reduce_copy__funcs.h"
#include "pccl_device/kernel_dispatch.h"
#else
// Include the types and declaration if PCCL_HOSTLIB_ONLY is defined
#include "pccl_device/coop.h"
#include "pccl_device/core.h"
#include "pccl_device/ll_a2a.h"
#include "pccl_device/lsa_bulk.h"
#include "pccl_device/barrier.h"
#include "pccl_device/ptr.h"
#include "pccl_device/reduce_copy.h"
#include "pccl_device/impl/comm__types.h"
#include "pccl_device/impl/core__types.h"
#include "pccl_device/impl/ll_a2a__types.h"
#include "pccl_device/impl/barrier__types.h"
#include "pccl_device/impl/ptr__types.h"
#include "pccl_device/impl/reduce_copy__types.h"
#include "pccl_device/kernel_dispatch.h"
#endif
