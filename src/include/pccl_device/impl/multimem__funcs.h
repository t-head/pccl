/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_MULTIMEM__FUNCS_H_
#define _PCCL_DEVICE_MULTIMEM__FUNCS_H_

#include "../utility.h"
#include "vector__types.h"
#include <cassert>

#if PCCL_CHECK_HGGCCC

namespace pccl {
namespace utility {

// PPU has no hardware multimem (multicast) support on any current model or
// architecture, so the multimem load-reduce / store paths are never selected.
// These helpers keep only the LSA (direct memory) path and hard-fail at runtime
// if a multimem path is ever requested. A runtime assert (not static_assert) is
// used so the templates still instantiate under C++14, where reduceCopy compiles
// both the multimem and non-multimem branches.

template<typename Pack, bool UseMultimem, typename RedOp, int Count = Pack::Count>
struct LoadImpl {
  PCCL_DEVICE_INLINE static Pack run(const Pack* addr) {
    if (UseMultimem) {
      assert(false && "multimem is not supported on PPU");
      return Pack{};
    }
    return *addr;
  }
};

// Empty packs (0 elements) - just return an empty pack.
template<typename Pack, bool UseMultimem, typename RedOp>
struct LoadImpl<Pack, UseMultimem, RedOp, 0> {
  PCCL_DEVICE_INLINE static Pack run(const Pack*) {
    return Pack{};
  }
};

template<typename Pack, bool UseMultimem, typename RedOp>
PCCL_DEVICE_INLINE Pack load(const Pack* addr) {
  return LoadImpl<Pack, UseMultimem, RedOp>::run(addr);
}

template<typename Pack, bool UseMultimem>
PCCL_DEVICE_INLINE void store(Pack* addr, const Pack& val) {
  if (UseMultimem) {
    assert(false && "multimem is not supported on PPU");
    return;
  }
  *addr = val;
}

} // namespace utility
} // namespace pccl

#endif // PCCL_CHECK_HGGCCC

#endif // _PCCL_DEVICE_MULTIMEM__FUNCS_H_
