/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"
#include "drv_version.h"

// Check HGGC calls
#define HGGCCHECK(cmd) do {                                                            \
    hggcError_t err = cmd;                                                             \
    if( err != hggcSuccess ) {                                                         \
        ERROR("Hggc failure: cmd=%s '%s'", #cmd, hggcGetErrorString(err)); \
        (void)hggcGetLastError();                            \
        return ncclUnhandledHggcError;                                                 \
    }                                                                                  \
} while(false)

#define HGGCCHECKWITHWARN(cmd) do {                                                    \
    hggcError_t err = cmd;                                                             \
    if( err != hggcSuccess ) {                                                         \
        WARN("Hggc failure: cmd=%s '%s'", #cmd, hggcGetErrorString(err)); \
        (void)hggcGetLastError();                            \
        return ncclUnhandledHggcError;                                                 \
    }                                                                                  \
} while(false)

#define HGGCCHECKGOTO(cmd, res, label) do {                                            \
    hggcError_t err = cmd;                                                             \
    if( err != hggcSuccess ) {                                                         \
        ERROR("Hggc failure: cmd=%s '%s'", #cmd, hggcGetErrorString(err)); \
        (void)hggcGetLastError();                            \
        res = ncclUnhandledHggcError;                                                  \
        goto label;                                                                    \
    }                                                                                  \
} while(false)

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
#define LT_OKI_CHECK(call) do {                                  \
  ltOkiStatus_t status = call;                                    \
  if (status != LT_OKI_SUCCESS) {                                 \
    ERROR("lt Oki error: call=%s status=0x%x", #call, (int)status); \
    return ncclUnhandledLtOkiError;                               \
  }                                                               \
} while (false)

#define LT_OKI_CHECKGOTO(call, res, label) do {                   \
  ltOkiStatus_t status = call;                                    \
  if (status != LT_OKI_SUCCESS) {                                 \
    ERROR("lt Oki error: call=%s status=0x%x", #call, (int)status); \
    res = ncclUnhandledLtOkiError;                                \
    goto label;                                                   \
  }                                                              \
} while (false)
#endif

// Log-and-continue variant: errors are reported but do not abort the caller.
#define HGGCCHECKIGNORE(cmd) do {                            \
    hggcError_t err = cmd;                                   \
    if( err != hggcSuccess ) {                               \
        INFO(NCCL_ALL, "%s:%d Hggc failure '%s'",            \
             __FILE__, __LINE__, hggcGetErrorString(err));   \
        (void)hggcGetLastError();                            \
    }                                                        \
} while(false)

// Returns true when a hggcXxx call succeeded; usable inside if() conditions.
#define HGGCSUCCESS(cmd) ((cmd) == hggcSuccess)

#include <errno.h>
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    int err = errno; \
    ERROR("Call to %s failed: call=%s '%s'", name, #call, strerror(err)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKGOTO(call, name, res, label) do { \
  int retval; \
  SYSCHECKVALGOTO(call, name, retval, res, label); \
} while (false)

#define SYSCHECKVALGOTO(call, name, retval, res, label) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    int err = errno; \
    ERROR("Call to %s failed: call=%s '%s'", name, #call, strerror(err)); \
    res = ncclSystemError; \
    goto label; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    int err = errno; \
    INFO(NCCL_ALL, "Call to %s returned '%s', retrying: call=%s", name, strerror(err), #call); \
  } else { \
    break; \
  } \
} while(true)

// Pthread calls don't set errno and never return EINTR.
#define PTHREADCHECK(statement, name) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to %s failed: statement=%s '%s'", name, #statement, strerror(retval)); \
    return ncclSystemError; \
  } \
} while (0)

#define PTHREADCHECKGOTO(statement, name, RES, label) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to %s failed: statement=%s '%s'", name, #statement, strerror(retval)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)


// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess && res != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) WARN_IMPL(false, "PCCLCHECK failed: call=%s res=0x%x", #call, (int)res);    \
    return res; \
  } \
} while (0)

#define NCCLVERIFY(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess && res != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) WARN_IMPL(false, "PCCLVERIFY failed: call=%s res=0x%x", #call, (int)res);    \
  } \
} while (0)

#define NCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != ncclSuccess && res != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) WARN_IMPL(false, "PCCLCHECKGOTO failed: call=%s res=0x%x", #call, (int)res);    \
    goto label; \
  } \
} while (0)

// Log-and-keep-first-error variant: capture the failure code into RES (without
// overwriting an earlier failure) and keep going. Used in cleanup paths where
// we want to attempt every step even if one fails.
#define NCCLCHECKIGNORE(call, RES) do { \
  ncclResult_t TMPRES = call; \
  if (TMPRES != ncclSuccess && TMPRES != ncclInProgress) { \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL, "%s:%d -> %d", __FILE__, __LINE__, TMPRES); \
    if (RES == ncclSuccess) RES = TMPRES; \
  } \
} while (0)

#define NEQCHECK(statement, value) do {   \
  if ((statement) != value) {             \
    /* Print the back trace*/             \
    ERROR_IMPL(false, "NEQCHECK failed: statement=%s expected=%s res=0x%x", #statement, #value, (int)ncclSystemError);    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define NEQCHECKGOTO(statement, value, res, label) do { \
  if ((statement) != value) { \
    /* Print the back trace*/ \
    res = ncclSystemError;    \
    ERROR_IMPL(false, "NEQCHECKGOTO failed: statement=%s expected=%s res=0x%x", #statement, #value, (int)res);    \
    goto label; \
  } \
} while (0)


#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    ERROR_IMPL(false, "EQCHECK failed: statement=%s unexpected=%s res=0x%x", #statement, #value, (int)ncclSystemError);    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define EQCHECKGOTO(statement, value, res, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    res = ncclSystemError;    \
    ERROR_IMPL(false, "EQCHECKGOTO failed: statement=%s unexpected=%s res=0x%x", #statement, #value, (int)res);    \
    goto label; \
  } \
} while (0)

#endif
