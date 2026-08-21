/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_NET_DEVICE_H_
#define PCCL_NET_DEVICE_H_

#define PCCL_NET_DEVICE_INVALID_VERSION      0x0
#define PCCL_NET_MTU_SIZE                    4096

// Arbitrary version number - A given PCCL build will only be compatible with a single device networking plugin
// version. PCCL will check the supplied version number from net->getProperties() and compare to its internal version.
#define PCCL_NET_DEVICE_UNPACK_VERSION 0x7

typedef enum {
  PCCL_NET_DEVICE_HOST=0,
  PCCL_NET_DEVICE_UNPACK=1,
  PCCL_NET_DEVICE_GIN_PROXY=2,
  PCCL_NET_DEVICE_GIN_GDAKI=3,
} pcclNetDeviceType;

typedef struct {
  pcclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  void* handle;
  size_t size;
  int needsProxyProgress;
} pcclNetDeviceHandle_v7_t;

typedef pcclNetDeviceHandle_v7_t pcclNetDeviceHandle_v8_t;
typedef pcclNetDeviceHandle_v8_t pcclNetDeviceHandle_v9_t;
typedef pcclNetDeviceHandle_v9_t pcclNetDeviceHandle_v10_t;
typedef pcclNetDeviceHandle_v10_t pcclNetDeviceHandle_v11_t;
typedef pcclNetDeviceHandle_v11_t pcclNetDeviceHandle_v12_t;
typedef pcclNetDeviceHandle_v12_t pcclNetDeviceHandle_t;

#endif
