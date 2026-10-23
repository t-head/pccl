/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// C API implementation for pcclParam access.
//
// This file implements the C-facing API declared in param/c_api.h.

#include "param/param.h"
#include "core.h"
#include "debug.h"

#include <cstdio>
#include <string>
#include <mutex>
#include <unordered_set>

USE_PCCL_PARAM(pcclParamDumpAllFlag, bool);

// Helper to unwrap opaque handle to pcclParamRegistry::mapEntry*
static inline pcclParamRegistry::mapEntry* unwrap(pcclParamHandle_t* h) {
  return reinterpret_cast<pcclParamRegistry::mapEntry*>(h);
}

static inline void pcclParamCheckFlag(uint64_t flags, const char* key) {
  static std::unordered_set<std::string> warningSet;

  // skip if this key was checked before, avoid repetitive warning messages
  if (warningSet.count(key)) {
    return;
  }

  warningSet.emplace(key);
  if (flags & PCCL_PARAM_FLAG_UNUSED) {
    INFO(NCCL_ENV, "Warning: ENV %s is unused. Setting it has no effect.", key);
  } else if (flags & PCCL_PARAM_FLAG_DEPRECATED) {
    INFO(NCCL_ENV, "Warning: ENV %s is deprecated. "
         "It may be removed or become unused in future versions.", key);
  }

  if (!(flags & PCCL_PARAM_FLAG_PUBLISHED)) {
    INFO(NCCL_ENV, "ENV: %s is private. Its function may change without notice across versions.", key);
  }
}

// ============================================================================
// Handle-based Access Interface
// ============================================================================

NCCL_API(pcclResult_t, pcclParamBind, pcclParamHandle_t** out, const char* key);
pcclResult_t pcclParamBind(pcclParamHandle_t** out, const char* key) {
  if (!out || !key) return pcclInvalidArgument;
  char keyBuf[PCCL_PARAM_KEY_MAX];
  const char* canonKey = pcclParamNormalizeKey(key, keyBuf, sizeof(keyBuf));
  auto* entry = pcclParamRegistry::find(canonKey);
  if (!entry) {
    INFO(NCCL_ENV, "PARAM: key \"%s\" not found in registry", key);
    return pcclInvalidArgument;
  }
  pcclParamCheckFlag(entry->info.flags, canonKey);
  *out = reinterpret_cast<pcclParamHandle_t*>(entry);
  return pcclSuccess;
}

// ============================================================================
// Typed Getters — dispatch via typeId + getRawData()
// ============================================================================

#define PCCL_PARAM_DEFINE_TYPED_GETTER(suffix, ctype, tid)                    \
  NCCL_API(pcclResult_t, pcclParamGet##suffix,                               \
           pcclParamHandle_t* h, ctype* out);                                 \
  pcclResult_t pcclParamGet##suffix(pcclParamHandle_t* h, ctype* out) {      \
    if (!h || !out) return pcclInvalidArgument;                               \
    auto* e = unwrap(h);                                                      \
    if (e->info.typeId != (tid)) {                                            \
      INFO(NCCL_ENV, "PARAM: type mismatch for key \"%s\"", e->info.key);    \
      return pcclInvalidArgument;                                             \
    }                                                                         \
    int len = 0;                                                              \
    return e->param->getRawData(out, sizeof(*out), &len);                     \
  }

PCCL_PARAM_DEFINE_TYPED_GETTER(I8,  int8_t,   PCCL_PARAM_TYPE_I8)
PCCL_PARAM_DEFINE_TYPED_GETTER(I16, int16_t,  PCCL_PARAM_TYPE_I16)
PCCL_PARAM_DEFINE_TYPED_GETTER(I32, int32_t,  PCCL_PARAM_TYPE_I32)
PCCL_PARAM_DEFINE_TYPED_GETTER(I64, int64_t,  PCCL_PARAM_TYPE_I64)
PCCL_PARAM_DEFINE_TYPED_GETTER(U8,  uint8_t,  PCCL_PARAM_TYPE_U8)
PCCL_PARAM_DEFINE_TYPED_GETTER(U16, uint16_t, PCCL_PARAM_TYPE_U16)
PCCL_PARAM_DEFINE_TYPED_GETTER(U32, uint32_t, PCCL_PARAM_TYPE_U32)
PCCL_PARAM_DEFINE_TYPED_GETTER(U64, uint64_t, PCCL_PARAM_TYPE_U64)

#undef PCCL_PARAM_DEFINE_TYPED_GETTER

// ============================================================================
// String Accessors
// ============================================================================

NCCL_API(pcclResult_t, pcclParamGetStr, pcclParamHandle_t* h, const char** out);
pcclResult_t pcclParamGetStr(pcclParamHandle_t* h, const char** out) {
  if (!h || !out) return pcclInvalidArgument;
  auto* e = unwrap(h);
  if (e->info.typeId != PCCL_PARAM_TYPE_CSTR) {
    INFO(NCCL_ENV, "PARAM: type mismatch for key \"%s\"", e->info.key);
    return pcclInvalidArgument;
  }
  // Thread-local buffer to keep the returned pointer valid for the caller
  static thread_local std::string buffer;
  buffer = unwrap(h)->param->toString();
  *out = buffer.c_str();
  return pcclSuccess;
}

NCCL_API(pcclResult_t, pcclParamGet, pcclParamHandle_t* h, void* out, int maxLen, int* len);
pcclResult_t pcclParamGet(pcclParamHandle_t* h, void* out, int maxLen, int* len) {
  if (!h || !out) return pcclInvalidArgument;
  return unwrap(h)->param->getRawData(out, maxLen, len);
}

// ============================================================================
// Typeless Access Interface
// ============================================================================
NCCL_API(pcclResult_t, pcclParamGetAllParameterKeys, const char*** table, int* tableLen);
pcclResult_t pcclParamGetAllParameterKeys(const char*** table, int* tableLen) {
  if (!table || !tableLen) return pcclInvalidArgument;

  // Thread-local storage: keyTable owns string copies so ptrTable pointers remain
  // valid even if the registry map changes between calls.
  static thread_local std::vector<std::string> keyTable;
  static thread_local std::vector<const char*> ptrTable;

  bool showAll = pcclParamDumpAllFlag();

  std::lock_guard<std::mutex> regLock(pcclParamRegistry::mutex());
  auto& map = pcclParamRegistry::instance();

  keyTable.clear();
  keyTable.reserve(map.size());
  for (const auto& kv : map) {
    if (!showAll && !(kv.second.info.flags & PCCL_PARAM_FLAG_PUBLISHED)) continue;
    keyTable.emplace_back(kv.first.data(), kv.first.size());
  }

  ptrTable.clear();
  ptrTable.reserve(keyTable.size());
  for (const auto& s : keyTable) {
    ptrTable.push_back(s.c_str());
  }

  *table = ptrTable.data();
  *tableLen = static_cast<int>(ptrTable.size());
  return pcclSuccess;
}

NCCL_API(pcclResult_t, pcclParamGetParameter, const char* key, const char** value, int* valueLen);
pcclResult_t pcclParamGetParameter(const char* key, const char** value, int* valueLen) {
  if (!key || !value || !valueLen) return pcclInvalidArgument;

  // Thread-local buffer to keep the returned pointer valid for the caller
  static thread_local std::string resultHolder;

  char keyBuf[PCCL_PARAM_KEY_MAX];
  auto* entry = pcclParamRegistry::find(pcclParamNormalizeKey(key, keyBuf, sizeof(keyBuf)));

  if (entry != nullptr) {
    pcclParamCheckFlag(entry->info.flags, entry->info.key);
    resultHolder = entry->param->toString();
    *value = resultHolder.c_str();
    *valueLen = static_cast<int>(resultHolder.length());
    return pcclSuccess;
  } else {
    *value = nullptr;
    *valueLen = 0;
    INFO(NCCL_ENV, "PARAM: key \"%s\" not found in registry", key);
    return pcclInvalidArgument;
  }
}

NCCL_API(void, pcclParamDumpAll);
void pcclParamDumpAll(){
  bool showAll = pcclParamDumpAllFlag();
  printf("=== pcclParam Registry Dump ===\n");
  std::lock_guard<std::mutex> regLock(pcclParamRegistry::mutex());
  for (auto& kv : pcclParamRegistry::instance()) {
    if (!showAll && !(kv.second.info.flags & PCCL_PARAM_FLAG_PUBLISHED)) continue;
    printf("%s\n", kv.second.param->dump().c_str());
  }
}
