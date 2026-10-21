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

NCCL_API(pcclResult_t, pcclPutSignal, const void* localbuff, size_t count, pcclDataType_t datatype,
    int peer, pcclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags,
    pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclPutSignal(const void* localbuff, size_t count, pcclDataType_t datatype,
    int peer, pcclWindow_t peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags,
    pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(PutSignal, PcclHgtxParamsPutSignal,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, count * ncclTypeSize(datatype), peer, ctx));
  struct ncclInfo info = { ncclFuncPutSignal, "PutSignal",
    localbuff, NULL, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1, /* chunkSteps, sliceSteps */
    peerWinOffset, peerWin, sigIdx, ctx, flags,
    0, NULL /* nDesc, signalDescs */ };
  return ncclEnqueueCheck(&info);
}

NCCL_API(pcclResult_t, pcclSignal, int peer, int sigIdx, int ctx, unsigned int flags,
    pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclSignal(int peer, int sigIdx, int ctx, unsigned int flags,
    pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(Signal, PcclHgtxParamsSignal,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, peer, ctx));
  struct ncclInfo info = { ncclFuncSignal, "Signal",
    NULL, NULL, 0, ncclInt8, ncclSum, peer, comm, stream, /* Args */
    1, 1, /* chunkSteps, sliceSteps */
    0, NULL, sigIdx, ctx, flags,
    0, NULL /* nDesc, signalDescs */ };
  return ncclEnqueueCheck(&info);
}

NCCL_API(pcclResult_t, pcclWaitSignal, int nDesc, pcclWaitSignalDesc_t* signalDescs,
    pcclComm_t comm, hggcStream_t stream);
pcclResult_t pcclWaitSignal(int nDesc, pcclWaitSignalDesc_t* signalDescs,
    pcclComm_t comm, hggcStream_t stream) {
  HGTX3_FUNC_WITH_PARAMS(WaitSignal, PcclHgtxParamsWaitSignal,
    HGTX3_PAYLOAD(comm ? comm->commHash : 0, nDesc, 0));
  struct ncclInfo info = { ncclFuncWaitSignal, "WaitSignal",
    NULL, NULL, 0, ncclInt32, ncclSum, 0, comm, stream, /* Args */
    1, 1, /* chunkSteps, sliceSteps */
    0, NULL, 0, 0, 0,
    nDesc, signalDescs /* nDesc, signalDescs */ };
  return ncclEnqueueCheck(&info);
}
