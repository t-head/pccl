/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "alloc.h"
#include "devcomm.h"
#include "collectives.h"
#include "common_kernel.h"
#include "common.h"
#include "bitops.h"
#include <algorithm>

namespace {
  template<typename T>
  __global__ void __launch_bounds__(512, 1)
  oneRankReduce(void* dst, void* src, size_t nElts, uint64_t redOpArg, bool redOpArgIsPtr) {
    int tid = threadIdx.x;
    int tn = blockDim.x;
    int bid = blockIdx.x;
    int bn = gridDim.x;

    // each block/channel gets a roughly equal segment of 16 byte packs
    constexpr int EltPerPack = 16/sizeof(T);
    ssize_t chunk = (ssize_t)alignUp(divUp((ssize_t)nElts, bn), EltPerPack);
    ssize_t i0 = (ssize_t)(bid+0)*chunk;
    ssize_t i1 = (ssize_t)(bid+1)*chunk;
    i0 = i0 < (ssize_t)nElts ? i0 : (ssize_t)nElts;
    i1 = i1 < (ssize_t)nElts ? i1 : (ssize_t)nElts;
    const T* s = (const T*)src + i0;
    T* d = (T*)dst + i0;

    if (redOpArgIsPtr) {
      redOpArg = RedOpArg<FuncPreMulSum<T>>::loadArg(reinterpret_cast<void*>(redOpArg));
    }
    FuncPreMulSum<T> redOp(redOpArg);
    ReduceOrCopyMulti<ncclCollUnroll(), FuncPreMulSum<T>, T, 1, 1, 1, 1>(
        tid, tn, redOp, /*preOpSrc0=*/true, /*postOp=*/true,
        1, &s, 1, &d, (int)(i1 - i0));
  }
}

static size_t oneRankEltSize(ncclDataType_t t) {
  switch (t) {
  case ncclInt8: case ncclUint8: return 1;
  case ncclFloat16: case ncclBfloat16: return 2;
  case ncclInt32: case ncclUint32: case ncclFloat32: return 4;
  case ncclInt64: case ncclUint64: case ncclFloat64: return 8;
#if defined(__HGGC_FP8_TYPES_EXIST__)
  case ncclFloat8e4m3: case ncclFloat8e5m2: return 1;
#endif
  default: return 0;
  }
}

ncclResult_t ncclLaunchOneRank(void* dst, void const* src, size_t nElts, struct ncclDevRedOpFull redOp, ncclDataType_t eltType, hggcStream_t stream) {
  size_t eltSize = oneRankEltSize(eltType);
  if (redOp.op != ncclDevPreMulSum) {
    if (dst != src) {
      HGGCCHECK(hggcMemcpyAsync(dst, (void*)src, nElts*eltSize, hggcMemcpyDeviceToDevice, stream));
    }
    return ncclSuccess;
  }

  const void* kernel;
  switch (eltType) {
  case ncclInt8:       kernel = (const void*)&oneRankReduce<int8_t>;   break;
  case ncclUint8:      kernel = (const void*)&oneRankReduce<uint8_t>;  break;
  case ncclInt32:      kernel = (const void*)&oneRankReduce<int32_t>;  break;
  case ncclUint32:     kernel = (const void*)&oneRankReduce<uint32_t>; break;
  case ncclInt64:      kernel = (const void*)&oneRankReduce<int64_t>;  break;
  case ncclUint64:     kernel = (const void*)&oneRankReduce<uint64_t>; break;
  case ncclFloat16:    kernel = (const void*)&oneRankReduce<half>;     break;
  case ncclFloat32:    kernel = (const void*)&oneRankReduce<float>;    break;
  case ncclFloat64:    kernel = (const void*)&oneRankReduce<double>;   break;
  case ncclBfloat16:   kernel = (const void*)&oneRankReduce<__ppu_bfloat16>; break;
#if defined(__HGGC_FP8_TYPES_EXIST__)
  case ncclFloat8e4m3: kernel = (const void*)&oneRankReduce<__hg_fp8_e4m3>; break;
  case ncclFloat8e5m2: kernel = (const void*)&oneRankReduce<__hg_fp8_e5m2>; break;
#endif
  default: return ncclInvalidArgument;
  }

  int gridx = (int)std::min((size_t)32, divUp(nElts*eltSize, (size_t)(16<<10)));
  if (gridx < 1) gridx = 1;
  void* args[5] = {(void*)&dst, (void*)&src, (void*)&nElts, (void*)&redOp.scalarArg, (void*)&redOp.scalarArgIsPtr};
  HGGCCHECK(hggcLaunchKernel(kernel, gridx, 512, args, 0, stream));
  return ncclSuccess;
}
