/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "param/param_registry.h"
#include "debug.h"

extern "C" PCCL_PARAM_COMPILER_EXPORT_SYMBOL void* pcclParamRegistryInstance() {
  static pcclParamRegistry::registryState state;
  return &state;
}

pcclResult_t pcclParamRegistry::add(std::string key, pcclParamInfo_t info,
                                    pcclParamInterface* param) {
  auto& reg = state();
  std::lock_guard<std::mutex> lock(reg.mtx);
  if (reg.map.find(key) != reg.map.end()) {
    WARN("PARAM: Duplicate registration for key \"%s\", ignoring.", key.c_str());
    return pcclInternalError;
  }
  reg.map[key] = {info, param};
  return pcclSuccess;
}

pcclParamRegistry::mapEntry* pcclParamRegistry::find(std::string key) {
  auto& reg = state();
  std::lock_guard<std::mutex> lock(reg.mtx);
  auto it = reg.map.find(key);
  return (it != reg.map.end()) ? &it->second : nullptr;
}

pcclResult_t pcclParamRegistry::remove(std::string key) {
  auto& reg = state();
  std::lock_guard<std::mutex> lock(reg.mtx);
  reg.map.erase(key);
  return pcclSuccess;
}
