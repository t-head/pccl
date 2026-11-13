/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RAS_DIAGNOSTICS_CHECKS_H_
#define NCCL_RAS_DIAGNOSTICS_CHECKS_H_

#include "diagnostics.h"

ncclResult_t rasDiagnosticsNcclEnvCollectLocal(const struct rasDiagnosticsContext* ctx,
                                            struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsNcclEnvSummarize(const struct rasDiagnosticsContext* ctx,
                                         const struct rasDiagnosticsReporter* reporter, const char* data, int nData);
ncclResult_t rasDiagnosticsRuntimeStateCollectLocal(const struct rasDiagnosticsContext* ctx,
                                                   struct rasDiagnosticsLocalData* data);
ncclResult_t rasDiagnosticsRuntimeStateSummarize(const struct rasDiagnosticsContext* ctx,
                                                const struct rasDiagnosticsReporter* reporter,
                                                const char* data, int nData);

#endif // NCCL_RAS_DIAGNOSTICS_CHECKS_H_
