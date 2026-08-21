/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#ifndef GIN_PROXY_DEFS_H
#define GIN_PROXY_DEFS_H

#include <stdint.h>
#include <stddef.h>

#define PCCL_GIN_PROXY_VERSION 100
#define PCCL_GIN_PROXY_GFD_VERSION 1

typedef enum {
  pcclGinProxyOpPut = 1 << 0,
  pcclGinProxyOpBaseMask = 1 << 0,
  pcclGinProxyOpWithInline = 1 << 1,
  pcclGinProxyOpWithCounter = 1 << 2,
  pcclGinProxyOpWithSignalInc = 1 << 3,
  pcclGinProxyOpWithSignalAdd = 1 << 4,
  pcclGinProxyOpVASignal = 1 << 5, // VA signals do not include put.
  pcclGinProxyOpGet = 1 << 6,
  pcclGinProxyOpFlush = 1 << 7,
} pcclGinProxyOp_t;

static_assert(sizeof(void *) == sizeof(uint64_t) && sizeof(size_t) == sizeof(uint64_t),
              "The proxy code is built on the assumption that the pointer size is 64 bits and at "
              "most 57 bits are used for the actual pointer.");

typedef union {
  uint64_t raw;
  struct {
    uint64_t v : 1;
    uint64_t resv : 63;
  } __attribute__((packed, aligned(1))) flag;
  struct {
    uint64_t flag : 1;
    uint64_t version : 4;
    uint64_t resv : 2;
    uint64_t size : 57;
  } __attribute__((packed, aligned(1))) header;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t srcOff : 63;
  } __attribute__((packed, aligned(1))) srcOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t srcHandle : 63;
  } __attribute__((packed, aligned(1))) srcHandle;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t vaSignalOff : 63;
  } __attribute__((packed, aligned(1))) vaSignalOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t vaSignalHandle : 63;
  } __attribute__((packed, aligned(1))) vaSignalHandle;
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint32_t inlineValLow;
    uint16_t inlineValLow2;
  } __attribute__((packed, aligned(1))) inlineLow;
  // inline supports a max of 96 bit / 12 byte values
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t inlineValHigh;
    uint8_t resv1;
    uint32_t resv2;
  } __attribute__((packed, aligned(1))) inlineHigh;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t dstOff : 63;
  } __attribute__((packed, aligned(1))) dstOff;
  struct {
    // the last bit is the flag, so we support 63 bit VAs
    uint64_t flag : 1;
    uint64_t dstHandle : 63;
  } __attribute__((packed, aligned(1))) dstHandle;
  struct {
    uint8_t flag : 1;
    // We need to keep the size of counterId and signalId in sync with the
    // PCCL_GIN_COUNTER_POOL_SIZE / PCCL_GIN_SIGNAL_POOL_SIZE upper limits
    // in gin_host.cc.
    // must be non-zero if WITH_COUNTER is set
    uint32_t counterId : 23;
    // must be non-zero if WITH_SIGNAL_INC, WITH_SIGNAL_ADD, or WITH_SIGNAL_SET is set
    uint32_t signalId : 24;
    uint16_t signalValLow;
  } __attribute__((packed, aligned(1))) completion;
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t signalValLow2;
    uint32_t signalValHigh;
  } __attribute__((packed, aligned(1))) signalVal;
  struct {
    uint8_t flag : 1;
    uint8_t resv : 7;
    uint16_t op;
    uint8_t resv2;
    uint32_t resv3;
  } __attribute__((packed, aligned(1))) headerExt;
} pcclGinProxyQword_t;
static_assert(sizeof(pcclGinProxyQword_t) == sizeof(uint64_t),
              "sizeof(pcclGinProxyQword_t) != sizeof(uint64_t)");
static_assert(PCCL_GIN_PROXY_GFD_VERSION < (1 << 4),
              "PCCL_GIN_PROXY_GFD_VERSION must be less than 2^4");

typedef enum {
  pcclGinProxyGfdHeader = 0,
  pcclGinProxyGfdInlineLow = 1,
  pcclGinProxyGfdInlineHigh = 2,
  pcclGinProxyGfdSrcOff = 1, // re-uses the inline word
  pcclGinProxyGfdSrcHandle = 2, // re-uses the inline word
  pcclGinProxyGfdVASignalOff = 1, // re-uses the inline word, VA signals with PUT must be split into two GFDs
  pcclGinProxyGfdVASignalHandle = 2, // re-uses the inline word, VA signals with PUT must be split into two GFDs
  pcclGinProxyGfdDstOff = 3,
  pcclGinProxyGfdDstHandle = 4,
  pcclGinProxyGfdCompletion = 5,
  pcclGinProxyGfdSignalVal = 6,
  pcclGinProxyGfdHeaderExt = 7,
  pcclGinProxyGfdQwords = 16,
} pcclGinProxyGfdQwordIdx_t;

// aligned(16): gin_proxy.h stores this as uint4. packed: the bitfield layouts need no padding.
typedef struct __attribute__((packed, aligned(16))) {
  pcclGinProxyQword_t qword[pcclGinProxyGfdQwords];
} pcclGinProxyGfd_t;
static_assert(sizeof(pcclGinProxyGfd_t) == 128,
              "sizeof(pcclGinProxyGfd_t) != 128 - Backwards compat requires pcclGinProxyGfd to be 128 bytes!");
static_assert(alignof(pcclGinProxyGfd_t) >= 16, "pcclGinProxyGfd_t must be at least 16-byte aligned");

typedef struct {
  int nranks;
  uint32_t queueSize;
  pcclGinProxyGfd_t *queues;
  uint32_t *pis;
  // The consumer indices will reside in CPU or GPU memory depending on the availability of GDR
  uint32_t *cis;

  uint64_t *counters;
  uint64_t *signals;
} pcclGinProxyGpuCtx_t;

#endif
