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

NCCL_API(pcclResult_t, pcclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    pcclDataType_t datatype, pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    pcclDataType_t datatype, pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(AllGather, PcclHgtxParamsAllGather,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, sendcount * ncclTypeSize(datatype)));
  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
