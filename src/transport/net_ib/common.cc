/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

#include <mutex>

char ncclIbIfName[MAX_IF_NAME_SIZE+1];
union ncclSocketAddress ncclIbIfAddr;

int ncclNIbDevs = -1;
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];

const char* ncclIbReqTypeStr[] = {
  "Unused", "Send", "Recv", "Flush", "GIN_iput", "GIN_iget",
};

NCCL_PARAM(IbSplitDataOnQps, "IB_SPLIT_DATA_ON_QPS", 1);
NCCL_PARAM(IbAsyncEvents,"IB_RETURN_ASYNC_EVENTS",1);

ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  if (ncclParamIbAsyncEvents() && __atomic_load_n(&stat->fatalErrorCount, std::memory_order_relaxed)) {
    ERROR("communicator encountered a fatal error (detected in %s)", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex) {
  if (base->isSend) {
    struct ncclIbSendComm* sComm = (struct ncclIbSendComm*)base;
    return &sComm->dev.base;
  } else {
    struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)base;
    return &rComm->dev.base;
  }
}

ncclResult_t ncclIbBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend) {
  baseComm->nqps = -1;
  baseComm->splitDataOnQps = ncclParamIbSplitDataOnQps();
  baseComm->isSend = isSend;
  baseComm->ready = 0;
  baseComm->recvMatchingScheme = BY_INDEX;

  return ncclSuccess;
}

ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm* recvComm) {
  NCCLCHECK(ncclIbBaseCommInit(&recvComm->base, false));
  return ncclSuccess;
}

ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm* sendComm) {
  NCCLCHECK(ncclIbBaseCommInit(&sendComm->base, true));
  return ncclSuccess;
}

pthread_t ncclIbAsyncThread;
void* ncclIbAsyncThreadMain(void* args) {
  struct ncclIbDev* dev = (struct ncclIbDev*)args;
  while (1) {
    struct ibv_async_event event;
    if (ncclSuccess != wrap_ibv_get_async_event(dev->context, &event)) { break; }
    char *str;
    struct ibv_cq* cq = event.element.cq;    // only valid if CQ error
    struct ibv_qp* qp = event.element.qp;    // only valid if QP error
    struct ibv_srq* srq = event.element.srq; // only valid if SRQ error
    if (ncclSuccess != wrap_ibv_event_type_str(&str, event.event_type)) { break; }
    switch (event.event_type) {
      case IBV_EVENT_DEVICE_FATAL:
        // the above is device fatal error
        WARN("NET/IB : %s:%d async fatal event: %s", dev->devName, dev->port, str);
        ncclIbDevFatalError(dev);
        break;
      case IBV_EVENT_CQ_ERR:
        // the above is a CQ fatal error
        WARN("NET/IB : %s:%d async fatal event on CQ (%p): %s", dev->devName, dev->port, cq, str);
        ncclIbCqFatalError(cq);
        break;
      case IBV_EVENT_QP_FATAL:
      case IBV_EVENT_QP_REQ_ERR:
      case IBV_EVENT_QP_ACCESS_ERR:
        // the above are QP fatal errors
        WARN("NET/IB : %s:%d async fatal event on QP (%p): %s", dev->devName, dev->port, qp, str);
        ncclIbQpFatalError(qp);
        break;
      case IBV_EVENT_SRQ_ERR:
        // SRQ are not used in NCCL
        WARN("NET/IB : %s:%d async fatal event on SRQ, unused for now (%p): %s", dev->devName, dev->port, srq, str);
        break;
      case IBV_EVENT_GID_CHANGE:
        WARN("NET/IB : %s:%d GID table changed", dev->devName, dev->port);
        break;
      case IBV_EVENT_PATH_MIG_ERR:
      case IBV_EVENT_PORT_ERR:
      case IBV_EVENT_PATH_MIG:
      case IBV_EVENT_PORT_ACTIVE:
      case IBV_EVENT_SQ_DRAINED:
      case IBV_EVENT_LID_CHANGE:
      case IBV_EVENT_PKEY_CHANGE:
      case IBV_EVENT_SM_CHANGE:
      case IBV_EVENT_QP_LAST_WQE_REACHED:
      case IBV_EVENT_CLIENT_REREGISTER:
      case IBV_EVENT_SRQ_LIMIT_REACHED:
        // the above are non-fatal
        WARN("NET/IB : %s:%d Got non-fatal async event: %s(%d)", dev->devName, dev->port, str, event.event_type);
        break;
      case IBV_EVENT_COMM_EST:
        break;
      default:
        WARN("NET/IB : %s:%d unknown event type (%d)", dev->devName, dev->port, event.event_type);
        break;
    }
    // acknowledgment needs to happen last to avoid user-after-free
    if (ncclSuccess != wrap_ibv_ack_async_event(&event)) { break; }
  }
  return NULL;
}

ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, void* /*config*/) {
  return ncclIbInit(logFunction);
}

ncclResult_t ncclIbFinalizeDevices() {
  return ncclSuccess;
}

ncclResult_t ncclIbSetTrafficClass(void* ctx, int trafficClass) {
  if (ctx == nullptr) return ncclSuccess;
  struct ncclNetCommConfig_t* netCommConfig = (struct ncclNetCommConfig_t*)ctx;
  netCommConfig->trafficClass = trafficClass;
  return ncclSuccess;
}

ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props) {
  return ncclIbGetProperties(dev, props);
}

ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t /*offset*/,
                                       int fd, uint64_t mrFlags, void** mhandle) {
  if (fd != -1) {
    WARN("NET/IB: DMA-BUF MR registration is not supported");
    return ncclSystemError;
  }
  // GIN symmetric windows can exceed 2GiB, so keep the size in size_t.
  return ncclIbRegMrInternal(comm, data, size, type, mrFlags, mhandle);
}

ncclNet_t ncclNetIb = {
        "IB",
        ncclIbInit,
        ncclIbDevices,
        ncclIbGetProperties,
        ncclIbListen,
        ncclIbConnect,
        ncclIbAccept,
        ncclIbRegMr,
        NULL,
        ncclIbDeregMr,
        ncclIbIsend,
        ncclIbIrecv,
        ncclIbIflush,
        ncclIbTest,
        ncclIbCloseSend,
        ncclIbCloseRecv,
        ncclIbCloseListen
};
