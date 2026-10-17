/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PCCL_GIN_GDAKI_H_
#define PCCL_GIN_GDAKI_H_

#include "gin_gdaki_device_common.h"
#include "providers/mlx5/gin_mlx5.h"
#include "providers/troce/gin_troce.h"

#if PCCL_CHECK_HGGCCC
template <template <pcclGinGdakiProvider> class ApiFn, typename... Arg>
PCCL_DEVICE_INLINE decltype(auto) pcclGinGdakiCall(pcclGinCtx ctx, Arg&&... arg) {
  switch (ctx.gdaProvider) {
    case PCCL_GIN_GDAKI_PROVIDER_MLX5:
      return ApiFn<PCCL_GIN_GDAKI_PROVIDER_MLX5>::call(ctx, static_cast<Arg&&>(arg)...);
    case PCCL_GIN_GDAKI_PROVIDER_TROCE:
      return ApiFn<PCCL_GIN_GDAKI_PROVIDER_TROCE>::call(ctx, static_cast<Arg&&>(arg)...);
    default:
      __builtin_unreachable(); // Host validates provider and context ABI before publication.
  }
}

// The public GDAKI implementation delegates to provider-specific, inline
// templates. The provider and raw context pointer are cached at GIN init.
#define PCCL_GIN_GDAKI_FORWARD(Name) \
template <> struct pcclGinApi_##Name<PCCL_NET_DEVICE_GIN_GDAKI> { \
  template <typename... Arg> \
  PCCL_DEVICE_INLINE static decltype(auto) call(pcclGinCtx ctx, Arg&&... arg) { \
    return pcclGinGdakiCall<pcclGinGdakiApi_##Name>(ctx, static_cast<Arg&&>(arg)...); \
  } \
};

PCCL_GIN_GDAKI_FORWARD(Wait)
PCCL_GIN_GDAKI_FORWARD(FlushAsync)
PCCL_GIN_GDAKI_FORWARD(Get)
PCCL_GIN_GDAKI_FORWARD(Put)
PCCL_GIN_GDAKI_FORWARD(PutValue)
PCCL_GIN_GDAKI_FORWARD(GetSignalPtr)
PCCL_GIN_GDAKI_FORWARD(GetCounterPtr)
PCCL_GIN_GDAKI_FORWARD(ReadCounter)
PCCL_GIN_GDAKI_FORWARD(WaitCounter)
PCCL_GIN_GDAKI_FORWARD(ResetSignal)
PCCL_GIN_GDAKI_FORWARD(ResetCounter)
PCCL_GIN_GDAKI_FORWARD(Flush)

#undef PCCL_GIN_GDAKI_FORWARD
#endif
#endif
