/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "hgtx_payload_schemas.h"
#include "hgtx.h"

NCCL_API(pcclResult_t, pcclAlltoAll, const void* sendbuff, void* recvbuff, size_t count,
    pcclDataType_t datatype, pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    pcclDataType_t datatype, pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(AlltoAll, PcclHgtxParamsAlltoAll,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype)));
  struct ncclInfo info = { ncclFuncAlltoAll, "AlltoAll",
    sendbuff, recvbuff, count, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}
