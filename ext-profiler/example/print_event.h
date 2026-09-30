/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PRINT_EVENT_H_
#define PRINT_EVENT_H_

#include "pccl_internal.h"

extern ncclDebugLogger_t logFn;

void debugEvent(void* eHandle, const char* tag);
void printEvent(FILE* fh, void* handle);

#endif
