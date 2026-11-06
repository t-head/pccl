/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Parameter definitions for the pcclParam system itself.

#include "param/param.h"
#include "param/parsers.h"
#include "param.h"
#include "debug.h"

#include <unordered_set>

DEFINE_PCCL_PARAM(pcclParamDumpAllFlag, bool, PCCL_PARAM_DUMP_ALL, false,
                  PCCL_PARAM_FLAG_NONE, PCCL_PARAM_DEFAULT,
                  "Print all parameters including private ones");

using pcclStringSet = std::unordered_set<std::string>;
DEFINE_PCCL_PARAM(pcclParamNoCacheStr, const char *, PCCL_NO_CACHE, nullptr,
                  PCCL_PARAM_FLAG_CACHED, PCCL_PARAM_DEFAULT,
                  "Comma-separated list of param keys to disable caching (or ALL)");

extern "C" PCCL_PARAM_COMPILER_EXPORT_SYMBOL bool pcclParamIsCacheDisabled(const char* key) {
  // Short-circuit for PCCL_NO_CACHE itself to prevent circular dependency
  if (std::strcmp(key, "PCCL_NO_CACHE") == 0) return false;

  static std::once_flag initFlag;
  static pcclStringSet set;
  static bool noCacheAll = false;

  std::call_once(initFlag, []() {
    auto parser = pcclParamListOf<pcclStringSet>(',');
    pcclStringSet listed;
    if (parser.resolve(pcclParamNoCacheStr(), listed) == pcclSuccess) {
      char buf[PCCL_PARAM_KEY_MAX];
      for (const auto& entry : listed) {
        set.emplace(pcclParamNormalizeKey(entry.c_str(), buf, sizeof(buf)));
      }
      noCacheAll = set.count("ALL") > 0;
    }
  });

  bool ret = noCacheAll || set.count(key) > 0;
  if (ret) INFO(NCCL_ENV, "PARAM: Disabling caching for environment variable %s.", key);
  return ret;
}

extern "C" PCCL_PARAM_COMPILER_EXPORT_SYMBOL const char* pcclParamEnvGet(const char* key,
                                                                        const char** src) {
  if (key == nullptr) return nullptr;
  initEnv();

  const char* value = pcclGetEnvWithDualPrefix(key);
  if (value && src) {
    *src = strncmp(key, "PCCL_", 5) == 0 ? pccl::param::utils::srcEnvPccl()
                                        : pccl::param::utils::srcEnvRaw();
  }
  return value;
}

extern "C" PCCL_PARAM_COMPILER_EXPORT_SYMBOL const char* pcclParamNormalizeKey(const char* key,
                                                                              char* buf, int bufLen) {
  (void)buf;
  (void)bufLen;
  return key;
}

