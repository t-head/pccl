//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

/*
 * Wrapper assert.h for REL_NO_SYM builds.
 * Overrides glibc's assert() to use __func__ instead of __PRETTY_FUNCTION__,
 * preventing demangled C++ function signatures from leaking into .rodata.
 */
#ifndef PCCL_ASSERT_OVERRIDE_H
#define PCCL_ASSERT_OVERRIDE_H

/* Include the real system assert.h first */
#include_next <assert.h>

/* Explicitly declare __assert_fail since it's an internal glibc function */
#ifdef __cplusplus
extern "C" {
#endif
void __assert_fail(const char *__assertion, const char *__file,
                   unsigned int __line, const char *__function)
    __attribute__((__nothrow__)) __attribute__((__noreturn__));
#ifdef __cplusplus
}
#endif

#ifdef REL_NO_SYM
/* Undefine glibc's version and replace with one using __func__ */
#undef assert
#define assert(expr) \
  ((expr) ? (void)0 : __assert_fail("assertion failed", __FILE__, __LINE__, __func__))
#endif

#endif /* PCCL_ASSERT_OVERRIDE_H */
