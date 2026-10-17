/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include <stdio.h>
#include <chrono>

#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include "pccl_internal.h"

#include <stdlib.h>
#include <stdarg.h>

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

/* we will redefine below log api to contain module & level */
#define NCCL_LOG_N_LEVEL 30
inline int get_hggc_device() {
  int hggcDev;
  hggcGetDevice(&hggcDev);
  return hggcDev;
}

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;
void ncclDebugInit();
extern pthread_mutex_t ncclDebugOutputLock;
extern FILE *ncclDebugFile;
extern ncclResult_t getHostName(char* hostname, int maxlen, const char delim);
void ncclSetThreadName(pthread_t thread, const char *fmt, ...);
#ifdef __cplusplus
#include <thread>
void ncclSetThreadName(std::thread& thread, const char *fmt, ...);
#endif

void ncclPerCommHashDebugInit(uint64_t commHash);
void ncclPerCommHashDebugClose();
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *file, const char *func, int line, bool record, const char *fmt, ...);
void ncclDebugLogForPlugin(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
extern char ncclLastError[];
#define LARGE_LOG_SIZE 2048

#define ERROR_IMPL(RECORD, ...) ncclDebugLog(PCCL_LOG_ERROR, NCCL_ALL, __FILENAME__, __func__, __LINE__, (RECORD), __VA_ARGS__)
#define WARN_IMPL(RECORD, ...) ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILENAME__, __func__, __LINE__, (RECORD), __VA_ARGS__)

#define ERROR(...) ERROR_IMPL(true, __VA_ARGS__)
#define WARN(...) WARN_IMPL(true, __VA_ARGS__)

#define NOWARN(EXPR, FLAGS) \
  do { \
    int oldNoWarn = ncclDebugNoWarn; \
    ncclDebugNoWarn = FLAGS; \
    (EXPR); \
    ncclDebugNoWarn = oldNoWarn; \
  } while(0)

#define INFO(FLAGS, ...) ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __FILENAME__, __func__, __LINE__, false, __VA_ARGS__)

// Log only the first time a given call site is reached.
#define LOG_ONCE(LOG) do { static bool logged = false; if (!logged) { logged = true; LOG; } } while(0)
#define ERROR_ONCE(...) LOG_ONCE(ERROR(__VA_ARGS__))
#define WARN_ONCE(...) LOG_ONCE(WARN(__VA_ARGS__))
#define INFO_ONCE(FLAGS, ...) LOG_ONCE(INFO((FLAGS), __VA_ARGS__))

#define VERBOSE(FLAGS, ...) ncclDebugLog(PCCL_LOG_VERBOSE, (FLAGS), __FILENAME__, __func__, __LINE__, false, __VA_ARGS__)

#define TRACE_CALL(...) ncclDebugLog(PCCL_LOG_VERBOSE, NCCL_CALL, __FILENAME__, __func__, __LINE__, false, __VA_ARGS__)
#define INFO_CALL(...) ncclDebugLog(NCCL_LOG_INFO, NCCL_CALL, __FILENAME__, __func__, __LINE__, false, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclDebugLog(NCCL_LOG_TRACE, (FLAGS), __FILENAME__, __func__, __LINE__, false,  __VA_ARGS__)
extern std::chrono::high_resolution_clock::time_point ncclEpoch;
#else
#define TRACE(...)
#endif

#endif
