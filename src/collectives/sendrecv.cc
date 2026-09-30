/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "argcheck.h" // Need some checks here since we access comm
#include "hgtx_payload_schemas.h"
#include "hgtx.h"

NCCL_API(pcclResult_t, pcclSend, const void* sendbuff, size_t count, pcclDataType_t datatype, int peer,
    pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclSend(const void* sendbuff, size_t count, pcclDataType_t datatype, int peer,
    pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(Send, PcclHgtxParamsSendRecv,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));
  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(pcclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(pcclGroupEnd());
  return ret;
}

NCCL_API(pcclResult_t, pcclRecv, void* recvbuff, size_t count, pcclDataType_t datatype, int peer,
    pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclRecv(void* recvbuff, size_t count, pcclDataType_t datatype, int peer,
    pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(Recv, PcclHgtxParamsSendRecv,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer));
  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(pcclGroupStart());
  ret = ncclEnqueueCheck(&info);
  NCCLCHECK(pcclGroupEnd());
  return ret;
}
