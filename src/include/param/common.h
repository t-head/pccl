/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Common types shared between param.h (C++) and param_c.h (C API).
//
// This header is intended to be includable from both C and C++.

#ifndef PARAM_COMMON_H_INCLUDED
#define PARAM_COMMON_H_INCLUDED

#include <stdint.h>
#include "pccl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  PCCL_PARAM_FLAG_NONE       = 0,
  PCCL_PARAM_FLAG_PUBLISHED  = 1ULL << 0, // public parameters in PCCL doc
  PCCL_PARAM_FLAG_DEPRECATED = 1ULL << 1,
  PCCL_PARAM_FLAG_CACHED     = 1ULL << 2, // value cached, subsequent change has no effect
  PCCL_PARAM_FLAG_UNUSED     = 1ULL << 3 // parameter has no effect
} pcclParamFlag_t;

// Type IDs for param info. non-integers, non-boolean and non-const-char* is mapped to RAW type.
typedef enum {
  PCCL_PARAM_TYPE_I8 = 1,
  PCCL_PARAM_TYPE_I16,
  PCCL_PARAM_TYPE_I32,
  PCCL_PARAM_TYPE_I64,
  PCCL_PARAM_TYPE_U8,
  PCCL_PARAM_TYPE_U16,
  PCCL_PARAM_TYPE_U32,
  PCCL_PARAM_TYPE_U64,
  PCCL_PARAM_TYPE_BOOL,
  PCCL_PARAM_TYPE_CSTR,
  PCCL_PARAM_TYPE_RAW
} pcclParamTypeId_t;

// Parameter metadata. All const char* fields must point to string literals.
typedef struct {
  pcclParamTypeId_t typeId;
  uint64_t flags;
  const char* typeStr;
  const char* key;
  const char* desc;
} pcclParamInfo_t;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>

struct pcclParamInterface {
  virtual ~pcclParamInterface() = default;
  virtual pcclResult_t getRawData(void* out, int maxLen, int* len) = 0;
  virtual std::string toString() = 0;
  virtual std::string dump() = 0;
};
#endif

#endif /* PARAM_COMMON_H_INCLUDED */
