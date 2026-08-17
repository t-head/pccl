/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef HGTX_PAYLOAD_SCHEMAS_H_
#define HGTX_PAYLOAD_SCHEMAS_H_

#include "pccl.h"
#include "hgtx3/hgToolsExtPayload.h"
#include "hgtx3/hgToolsExtPayloadHelper.h"

#define PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(struct_id, prefix, entries) \
  _HGTX_PAYLOAD_TYPEDEF_STRUCT(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries) \
  prefix _HGTX_PAYLOAD_SCHEMA_INIT_ENTRIES(struct_id, _HGTX_PAYLOAD_PASS_THROUGH entries)

// C strings used as HGTX payload entry names.
static constexpr char const* pccl_hgtxCommStr = "PCCL communicator ID";
static constexpr char const* pccl_hgtxHggcDevStr = "HGGC device";
static constexpr char const* pccl_hgtxRankStr = "Rank";
static constexpr char const* pccl_hgtxNranksStr = "No. of ranks";
static constexpr char const* pccl_hgtxMsgSizeStr = "Message size [bytes]";
static constexpr char const* pccl_hgtxReductionOpStrpStr = "Reduction operation";

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsCommInitAll, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, commhash, TYPE_UINT64, pccl_hgtxCommStr),
    (int, ndev, TYPE_INT, "No. of devices")
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsCommInitRank, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, newcomm, TYPE_UINT64, pccl_hgtxCommStr),
    (int, nranks, TYPE_INT, pccl_hgtxNranksStr),
    (int, myrank, TYPE_INT, pccl_hgtxRankStr),
    (int, hggcDev, TYPE_INT, pccl_hgtxHggcDevStr)
  )
)
// The typedef and payload schema for ncclCommInitRank is also used for,
// ncclCommInitRankConfig, ncclCommInitRankScalable, ncclCommDestroy, and ncclCommAbort.
typedef PcclHgtxParamsCommInitRank PcclHgtxParamsCommInitRankConfig;
typedef PcclHgtxParamsCommInitRank PcclHgtxParamsCommInitRankScalable;
typedef PcclHgtxParamsCommInitRank PcclHgtxParamsCommAbort;
typedef PcclHgtxParamsCommInitRank PcclHgtxParamsCommDestroy;

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsCommSplit, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, newcomm, TYPE_UINT64, pccl_hgtxCommStr),
    (uint64_t, parentcomm, TYPE_UINT64, "Parent PCCL communicator ID"),
    (int, nranks, TYPE_INT, pccl_hgtxNranksStr),
    (int, myrank, TYPE_INT, pccl_hgtxRankStr),
    (int, hggcDev, TYPE_INT, pccl_hgtxHggcDevStr),
    (int, color, TYPE_INT, "Color"),
    (int, key, TYPE_INT, "Key")
  )
)


PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsCommShrink, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, newcomm, TYPE_UINT64, pccl_hgtxCommStr),
    (int, nranks, TYPE_INT, pccl_hgtxNranksStr),
    (int, myrank, TYPE_INT, pccl_hgtxRankStr),
    (int, hggcDev, TYPE_INT, pccl_hgtxHggcDevStr),
    (int, num_exclude, TYPE_INT, "num_exclude")
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsCommFinalize, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr)
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsAllGather, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr),
    (size_t, bytes, TYPE_SIZE, pccl_hgtxMsgSizeStr)
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsAlltoAll, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr),
    (size_t, bytes, TYPE_SIZE, pccl_hgtxMsgSizeStr)
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsAllReduce, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr),
    (size_t, bytes, TYPE_SIZE, pccl_hgtxMsgSizeStr),
    (int, op, TYPE_INT, pccl_hgtxReductionOpStrpStr)
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsBroadcast, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr),
    (size_t, bytes, TYPE_SIZE, pccl_hgtxMsgSizeStr),
    (int, root, TYPE_INT, "Root")
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsGather, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr),
    (size_t, bytes, TYPE_SIZE, pccl_hgtxMsgSizeStr),
    (int, root, TYPE_INT, "Root")
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsReduce, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr),
    (size_t, bytes, TYPE_SIZE, pccl_hgtxMsgSizeStr),
    (int, root, TYPE_INT, "Root"),
    (int, op, TYPE_INT, pccl_hgtxReductionOpStrpStr)
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsReduceScatter, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr),
    (size_t, bytes, TYPE_SIZE, pccl_hgtxMsgSizeStr),
    (int, op, TYPE_INT, pccl_hgtxReductionOpStrpStr)
  )
)

PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsScatter, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr),
    (size_t, bytes, TYPE_SIZE, pccl_hgtxMsgSizeStr),
    (int, root, TYPE_INT, "Root")
  )
)

// Used in NCCL APIs `ncclSend` and `ncclRecv`.
PCCL_HGTX_DEFINE_STRUCT_WITH_SCHEMA_ENTRIES(PcclHgtxParamsSendRecv, static constexpr,
  HGTX_PAYLOAD_ENTRIES(
    (uint64_t, comm, TYPE_UINT64, pccl_hgtxCommStr),
    (size_t, bytes, TYPE_SIZE, pccl_hgtxMsgSizeStr),
    (int, peer, TYPE_INT, "Peer rank")
  )
)

#endif