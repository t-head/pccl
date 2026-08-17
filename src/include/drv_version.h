/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef DRV_VERSION_H_
#define DRV_VERSION_H_

#define PCCL_UNDEF_DRIVER_VERSION 0x55aaaa55

#define PCCL_SYSTEM_DRIVER_VERSION(major, minor, patch) \
    (((major) << 16) | ((minor) << 8) | (patch))

#define PCCL_HGGC_DRIVER_VERSION(major, minor, patch) \
    (((major) * 10000) + ((minor) * 100) + (patch))

#endif

