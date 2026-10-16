/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_REGISTRY_H_INCLUDED
#define PARAM_REGISTRY_H_INCLUDED

#include "pccl.h"
#include "param/common.h"
#include "param/utils.h"

#include <string>
#include <unordered_map>
#include <mutex>

// C-linkage singleton accessor — exported as "pcclParamRegistryInstance"
// Returns a process-wide RegistryState so map and mutex share identity across DSOs.
extern "C" PCCL_PARAM_COMPILER_EXPORT_SYMBOL void* pcclParamRegistryInstance();

// pcclParamRegistry is a global singleton list of all parameters. Parameters
// defined through the DEFINE_PCCL_PARAM macro are automatically registered here
// at program init (before main()). This also works for DEFINE_PCCL_PARAM in
// external .so files, where the parameter is registered when the .so is loaded
// and initialized, or at dlopen().
//
// Each entry is a map of (key -> { pcclParamInfo_t info, pcclParamInterface* param }),
// which are all the information for public APIs to check and query parameters.
//
// The underlying state (RegistryState) is held behind a C-linkage accessor
// (pcclParamRegistryInstance) so that all DSOs in the process share a single
// map and mutex, even when PCCL is statically linked into multiple libraries.
//
// Thread safety: all public methods (add, find, remove) acquire the internal
// mutex. Registration during static initialization is safe because each
// pcclParam constructor calls add() independently with the lock held.
//
// The class is non-instantiable (deleted constructor); all access is through
// static methods: add() to register, find() to look up by key, and remove()
// to unregister. The C API (c_api.cc) uses find() to resolve handles and then
// calls virtual methods on the pcclParamInterface* pointer (toString, dump,
// getRawData) to service queries.
class pcclParamRegistry {
public:
  struct mapEntry {
    pcclParamInfo_t info;
    pcclParamInterface* param;
  };

  using mapType = std::unordered_map<std::string, mapEntry>;
  struct registryState {
    mapType map;
    std::mutex mtx;
  };

  static registryState& state() {
    return *static_cast<registryState*>(pcclParamRegistryInstance());
  }

  static mapType& instance() {
    return state().map;
  }

  static std::mutex& mutex() {
    return state().mtx;
  }

  // Register a parameter; returns pcclInternalError on duplicate key.
  static pcclResult_t add(std::string key, pcclParamInfo_t info, pcclParamInterface* param);

  // Find a parameter by key; returns nullptr if not found.
  static mapEntry* find(std::string key);

  // Unregister a parameter by key.
  static pcclResult_t remove(std::string key);

  // Prevent instantiation
  pcclParamRegistry() = delete;
};

#endif /* PARAM_REGISTRY_H_INCLUDED */
