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

#include "diagnostics_runtime.h"

// The input remains owned by the caller. Views into records only live for this call.
// Analyze available rank snapshots; reports identify incomplete coverage.
int analyzerFormatRuntimeReport(FILE* fp, const struct rasDiagnosticsRuntimeRecord* records, int nRecords,
                                int nLegTimeouts, int singleCategory, const char* title = nullptr);
int analyzerFormatRuntimeSummary(FILE* fp, const struct rasDiagnosticsRuntimeRecord* records, int nRecords);
int analyzerDumpRuntimeData(const char* dir, const struct rasDiagnosticsRuntimeRecord* records,
                            int nRecords, int nLegTimeouts);
// Reads native v2 records. The returned array is malloc-owned; free it after use.
struct rasDiagnosticsRuntimeRecord* analyzerLoadRuntimeData(FILE* fp, int* outNRecords, int* outLegTimeouts);
int analyzerDumpRuntimeReport(const char* dir, const struct rasDiagnosticsRuntimeRecord* records,
                              int nRecords, int nLegTimeouts);

int analyzerConnTypeFromString(const char* s, int peer);

// True when S is a valid -T/--time argument ("YYYYMMDD-HHMMSS" or a shorter
// prefix)
bool analyzerTimeArgValid(const char* s);

int analyzerFormatReportFromCommDumps(FILE* fp, const char* dir, int round = 1,
                                      const char* timeStr = nullptr,
                                      int* outFilesParsed = nullptr);

#endif
