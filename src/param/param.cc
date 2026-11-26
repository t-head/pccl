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
      // The dual prefix lets a user name a parameter with either spelling, so
      // canonicalize every entry to the PCCL_* key that ensureLoaded() passes in.
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

// PCCL has no EnvPlugin. Resolve against the environment with the PCCL_/NCCL_
// dual prefix instead, mirroring pcclGetEnvWithDualPrefix(), and report which
// spelling supplied the value through *src.
extern "C" PCCL_PARAM_COMPILER_EXPORT_SYMBOL const char* pcclParamEnvGet(const char* key,
                                                                        const char** src) {
  if (key == nullptr) return nullptr;
  initEnv();

  char pcclName[PCCL_PARAM_KEY_MAX];
  char ncclName[PCCL_PARAM_KEY_MAX];
  pcclName[0] = '\0';
  ncclName[0] = '\0';

  if ((key[0] == 'N' || key[0] == 'P') && strncmp(key + 1, "CCL_", 4) == 0) {
    if (snprintf(pcclName, sizeof(pcclName), "PCCL%s", key + 4) >= (int)sizeof(pcclName))
      pcclName[0] = '\0';
    if (snprintf(ncclName, sizeof(ncclName), "PCCL%s", key + 4) >= (int)sizeof(ncclName))
      ncclName[0] = '\0';
    else
      ncclName[0] = 'N';
  }

  const char* v = nullptr;
  if (pcclName[0] != '\0' && (v = getenv(pcclName)) != nullptr) {
    if (src) *src = pccl::param::utils::srcEnvPccl();
    return v;
  }
  if (ncclName[0] != '\0' && (v = getenv(ncclName)) != nullptr) {
    if (src) *src = pccl::param::utils::srcEnvNccl();
    return v;
  }
  if ((v = getenv(key)) != nullptr) {
    if (src) *src = pccl::param::utils::srcEnvRaw();
    return v;
  }
  return nullptr;
}

// Rewrite a NCCL_* key to its canonical PCCL_* spelling so a caller may bind
// either spelling and reach the same parameter.
extern "C" PCCL_PARAM_COMPILER_EXPORT_SYMBOL const char* pcclParamNormalizeKey(const char* key,
                                                                              char* buf, int bufLen) {
  if (key == nullptr || buf == nullptr || bufLen <= 0) return key;
  if (key[0] == 'N' && strncmp(key + 1, "CCL_", 4) == 0) {
    if (snprintf(buf, bufLen, "PCCL%s", key + 4) < bufLen) return buf;
  }
  return key;
}

