/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "enqueue.h"
#include "hgtx_payload_schemas.h"
#include "hgtx.h"

NCCL_API(pcclResult_t, pcclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    pcclDataType_t datatype, pcclRedOp_t op, pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    pcclDataType_t datatype, pcclRedOp_t op, pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(AllReduce, PcclHgtxParamsAllReduce,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), op));
  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
