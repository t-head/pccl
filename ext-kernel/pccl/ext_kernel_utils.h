/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*************************************************************************
 * Utility macros for ext-kernel plugin.
 * Replaces checks.h, debug.h, param.h, bitops.h from pccl internals.
 ************************************************************************/

#ifndef EXT_KERNEL_PCCL_UTILS_H_
#define EXT_KERNEL_PCCL_UTILS_H_

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <hggc.h>

#include "ext_kernel.h"

/*========================================================================
 * Debug logging — resolved at link time from libpccl.so
 *========================================================================*/
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ncclDebugLogger_t)(pcclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

extern ncclDebugLogger_t logFn;
#define ERROR(fmt, ...) logFn(PCCL_LOG_ERROR, PCCL_ALL,  __FILE__, __LINE__, "[%s] " fmt, __func__, ##__VA_ARGS__)
#define WARN(fmt, ...)  logFn(PCCL_LOG_WARN, PCCL_ALL, __FILE__, __LINE__, "[%s] " fmt, __func__, ##__VA_ARGS__)
#define INFO(FLAGS, fmt, ...) logFn(NCCL_LOG_INFO, (FLAGS), __FILE__, __LINE__, "[%s] " fmt, __func__, ##__VA_ARGS__)
#define VERBOSE(FLAGS, fmt, ...) logFn(PCCL_LOG_VERBOSE, (FLAGS), __FILE__, __LINE__, "[%s] " fmt, __func__, ##__VA_ARGS__)

/*========================================================================
 * Error checking macros
 *========================================================================*/
#define NCCLCHECK(call) do {                            \
  ncclResult_t _res = (call);                           \
  if (_res != ncclSuccess && _res != ncclInProgress) {  \
    ERROR("PCCLCHECK failed: call=%s res=%d(0x%x)", #call, (int)_res, (int)_res); \
    return _res;                                        \
  }                                                     \
} while(0)

#define NCCLCHECKGOTO(call, res, label) do {            \
  res = (call);                                         \
  if (res != ncclSuccess && res != ncclInProgress) {    \
    ERROR("PCCLCHECKGOTO failed: call=%s res=%d(0x%x)", #call, (int)res, (int)res); \
    goto label;                                         \
  }                                                     \
} while(0)

#define HGGCCHECK(cmd) do {                             \
  hggcError_t _err = (cmd);                             \
  if (_err != hggcSuccess) {                            \
    ERROR("Hggc failure: cmd=%s '%s'", #cmd, hggcGetErrorString(_err)); \
    return ncclUnhandledHggcError;                      \
  }                                                     \
} while(0)

#define HGGCCHECKGOTO(cmd, res, label) do {             \
  hggcError_t _err = (cmd);                             \
  if (_err != hggcSuccess) {                            \
    ERROR("Hggc failure: cmd=%s '%s'", #cmd, hggcGetErrorString(_err)); \
    res = ncclUnhandledHggcError;                       \
    goto label;                                         \
  }                                                     \
} while(0)

/*========================================================================
 * PCCL_PARAM — self-contained env var reading (no pccl dependency)
 * Pattern copied from ext-tuner/pccl/tuner.h
 *========================================================================*/
#define PCCL_PARAM(name, env, default_value)                      \
pthread_mutex_t pcclParamMutex##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t pcclParam##name() {                                       \
  static_assert(default_value != -1LL, "default value cannot be -1"); \
  static int64_t value = -1LL;                                    \
  pthread_mutex_lock(&pcclParamMutex##name);                      \
  if (value == -1LL) {                                            \
    value = default_value;                                        \
    char* str = getenv("PCCL_" env);                              \
    if (str && strlen(str) > 0) {                                 \
      errno = 0;                                                  \
      int64_t v = strtoll(str, NULL, 0);                          \
      if (errno) {                                                \
        INFO(PCCL_ALL, "Invalid value %s for %s, using default %lld.", \
             str, "PCCL_" env, (long long)value);                 \
      } else {                                                    \
        value = v;                                                \
        INFO(PCCL_ALL, "%s set by environment to %lld.",          \
             "PCCL_" env, (long long)value);                      \
      }                                                           \
    }                                                             \
  }                                                               \
  pthread_mutex_unlock(&pcclParamMutex##name);                    \
  return value;                                                   \
}

/*========================================================================
 * Memory helpers
 *========================================================================*/
template <typename T>
static inline ncclResult_t extFree(T*& ptr) {
  if (ptr != NULL) free(ptr);
  ptr = NULL;
  return ncclSuccess;
}

template <typename T>
static inline ncclResult_t extCalloc(T** ptr, size_t nelem) {
  if (nelem > 0) {
    void* p = malloc(nelem * sizeof(T));
    if (p == NULL) {
      ERROR("Failed to malloc %zu bytes", nelem * sizeof(T));
      return ncclSystemError;
    }
    memset(p, 0, nelem * sizeof(T));
    *ptr = (T*)p;
  } else {
    *ptr = NULL;
  }
  return ncclSuccess;
}

template <typename T>
static inline ncclResult_t extHggcHostCalloc(T** ptr, size_t nelem) {
  hggcError_t err = hggcMallocHost((void**)ptr, nelem * sizeof(T));
  if (err != hggcSuccess) {
    ERROR("Failed to hggcMallocHost %zu bytes", nelem * sizeof(T));
    *ptr = NULL;
    return ncclSystemError;
  }
  memset(*ptr, 0, nelem * sizeof(T));
  return ncclSuccess;
}

/*========================================================================
 * Math / alignment macros
 *========================================================================*/
#define DIVUP(x, y)    (((x)+(y)-1)/(y))
#define ROUNDUP(x, y)  (DIVUP((x),(y))*(y))

#ifdef __cplusplus
template<typename X, typename Y, typename Z = decltype(X()+Y())>
static __host__ __device__ constexpr Z divUp(X x, Y y) {
  return (x+y-1)/y;
}
#endif

/*========================================================================
 * Size literal operators
 *========================================================================*/
#ifdef __cplusplus
constexpr size_t operator""_KB(unsigned long long v) { return v * 1024; }
constexpr size_t operator""_MB(unsigned long long v) { return v * 1024 * 1024; }
#endif

#endif /* EXT_KERNEL_PCCL_UTILS_H_ */
