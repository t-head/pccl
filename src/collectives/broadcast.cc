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

NCCL_API(pcclResult_t, pcclBroadcast, const void* sendbuff, void* recvbuff, size_t count, pcclDataType_t datatype, int root,
    pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclBroadcast(const void* sendbuff, void* recvbuff, size_t count, pcclDataType_t datatype, int root,
    pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(Broadcast, PcclHgtxParamsBroadcast,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), root));
  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(pcclResult_t, pcclBcast, void* buff, size_t count, pcclDataType_t datatype, int root,
    pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclBcast(void* buff, size_t count, pcclDataType_t datatype, int root,
    pcclComm_t comm, hggcStream_t stream) {
  return pcclBroadcast(buff, buff, count, datatype, root, comm, stream);
}
