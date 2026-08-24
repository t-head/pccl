/**************************************************************************
* SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
*
* SPDX-License-Identifier: Apache-2.0
*
* See LICENSE.txt for more license information
**************************************************************************/

#ifndef PCCL_ANALYZER_H_
#define PCCL_ANALYZER_H_

#include <cstdio>

#include "ras_wire.h"

int analyzerFormatReport(FILE* fp, const struct rasCollComms* blob, int nLegTimeouts,
                     int singleCategory);
int analyzerFormatSummary(FILE* fp, const struct rasCollComms* blob);

int analyzerDumpData(const char* dir, const void* blob, int nBlobBytes, int nLegTimeouts);
struct rasCollComms* analyzerLoadData(FILE* fp, int* outBlobBytes, int* outLegTimeouts);
int analyzerDumpReport(const char* dir, const struct rasCollComms* blob, int nLegTimeouts);

#endif
