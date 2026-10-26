/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_TUNER_H_
#define NCCL_INT_TUNER_H_

#include "pccl_tuner.h"

// Tuning plugin to override NCCL's default algorithm/protocol tuning.

// Attempts to load NCCL tuner from environmental variable.
// Returns ncclSuccess if the correct tuner symbol has been found and
// successfully loaded.  Otherwise returns an error and also logs the error.
ncclResult_t ncclLoadTunerPlugin(ncclTuner_t** tuner);

// Cleans up NCCL tuner plugin.
ncclResult_t ncclCloseTunerPlugin(ncclTuner_t** tuner);
#endif
