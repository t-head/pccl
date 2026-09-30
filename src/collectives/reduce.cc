/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "hgtx_payload_schemas.h"
#include "hgtx.h"

NCCL_API(pcclResult_t, pcclReduce, const void* sendbuff, void* recvbuff, size_t count,
    pcclDataType_t datatype, pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclReduce(const void* sendbuff, void* recvbuff, size_t count,
    pcclDataType_t datatype, pcclRedOp_t op, int root, pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(Reduce, PcclHgtxParamsReduce,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root, op));
  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
