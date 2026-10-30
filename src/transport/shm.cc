/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "shm.h"
#include "transport.h"

#define SHM_PATH_MAX 128

struct shmBuffInfo {
  void *hptr;
  void *dptr;
};

struct shmConnectInfo {
  int rank;
  ncclShmIpcDesc_t desc;
  struct shmBuffInfo buf;
};
static_assert(sizeof(shmConnectInfo) <= CONNECT_SIZE, "SHM Connect info is too large");

struct shmSendResources {
  struct ncclRecvMem* remHostMem;
  struct ncclRecvMem* devRemHostMem;
  ncclShmIpcDesc_t remDesc;
  struct ncclSendMem* hostMem;
  struct ncclSendMem* devHostMem;
};

struct shmRecvResources {
  struct ncclSendMem* remHostMem;
  struct ncclSendMem* devRemHostMem;
  ncclShmIpcDesc_t remDesc;
  struct ncclRecvMem* hostMem;
  struct ncclRecvMem* devHostMem;
};

struct shmProxyInfo {
  struct ncclRecvMem* ceRecvMem;
  char* devFifo;
  char* shmFifo;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  uint64_t step;
  hggcStream_t stream;
  hggcEvent_t events[NCCL_STEPS];

  ncclShmIpcDesc_t desc;
};

struct shmRequest {
  size_t size;
  bool legacy;
};

#define SHM_SEND_SIDE 1
#define SHM_RECV_SIDE 2
NCCL_PARAM(ShmDisable, "SHM_DISABLE", 0);
NCCL_PARAM(ShmUseHggcMemcpy, "SHM_USE_HGGC_MEMCPY", 0);
NCCL_PARAM(ShmMemcpyMode, "SHM_MEMCPY_MODE", SHM_SEND_SIDE);
static int useMemcpySend = 0;
static int useMemcpyRecv = 0;
NCCL_PARAM(ShmLocality, "SHM_LOCALITY", SHM_RECV_SIDE);
static int shmLocality = 0;
static void initCeOperation();

/* Determine two peers can communicate with SHM */
static ncclResult_t shmCanConnect(int* ret, ncclComm_t comm, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 0;
  initCeOperation();

  if (ncclParamShmDisable() == 1) return ncclSuccess;

  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
  if (useNet) return ncclSuccess;

  // Same host?
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 hostHash %lx peer2 hostHash %lx", info1->hostHash, info2->hostHash);
  if (info1->hostHash != info2->hostHash) return ncclSuccess;

  // Common /dev/shm (between containers) ?
  TRACE(NCCL_INIT|NCCL_SHM, "peer1 shmDev %lx peer2 shmDev %lx", info1->shmDev, info2->shmDev);
  if (info1->shmDev != info2->shmDev) return ncclSuccess;

  *ret = 1;

  return ncclSuccess;
}

#define MAX_SHM_NAME_LEN 1024

/* Create and return connect structures for this peer to connect to me */
static ncclResult_t shmSendSetup(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct shmSendResources* resources;
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  size_t shmSize = sizeof(struct ncclSendMem);
  struct shmRequest req;

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");

  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  if (shmLocality == SHM_SEND_SIDE) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (p == NCCL_PROTO_SIMPLE_BULK || p == NCCL_PROTO_SIGNAL || p == NCCL_PROTO_SIGNAL_BULK || p == NCCL_PROTO_SIMPLE_BULK_AR) {
        continue;
      }
      shmSize += comm->buffSizes[p];
    }
  }
  req.size = shmSize;
  if (myInfo->hostHash == peerInfo->hostHash && myInfo->pidHash == peerInfo->pidHash)
    req.legacy = true;
  else
    req.legacy = false;

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_SHM, 1, myInfo->rank, &send->proxyConn));
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, (void*)&req, sizeof(struct shmRequest), (void*)info, sizeof(struct shmConnectInfo)));

  info->rank = comm->rank;
  resources->hostMem = (struct ncclSendMem*)info->buf.hptr;
  resources->devHostMem = (struct ncclSendMem*)info->buf.dptr;

  INFO(NCCL_INIT|NCCL_SHM,"Channel %02d : %d[%lx] -> %d[%lx] via SHM/%s/%s", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, useMemcpySend?"CE":"direct", useMemcpyRecv?"CE":"direct");
  return ncclSuccess;
}

static ncclResult_t shmRecvSetup(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct shmRecvResources* resources;
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  size_t shmSize = sizeof(struct ncclRecvMem);
  struct shmRequest req;

  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");

  if (shmLocality == SHM_RECV_SIDE) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (p == NCCL_PROTO_SIMPLE_BULK || p == NCCL_PROTO_SIGNAL || p == NCCL_PROTO_SIGNAL_BULK || p == NCCL_PROTO_SIMPLE_BULK_AR) {
        continue;
      }
      shmSize += comm->buffSizes[p];
    }
  }
  req.size = shmSize;
  if (myInfo->hostHash == peerInfo->hostHash && myInfo->pidHash == peerInfo->pidHash)
    req.legacy = true;
  else
    req.legacy = false;

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_SHM, 0, myInfo->rank, &recv->proxyConn));
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, (void*)&req, sizeof(struct shmRequest), (void*)info, sizeof(struct shmConnectInfo)));

  info->rank = comm->rank;
  resources->hostMem = (struct ncclRecvMem*)info->buf.hptr;
  resources->devHostMem = (struct ncclRecvMem*)info->buf.dptr;

  return ncclSuccess;
}

/* Connect to this peer */
static ncclResult_t shmSendConnect(ncclComm_t comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  struct shmSendResources* resources = (struct shmSendResources*)send->transportResources;
  char* buff;

  NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, &resources->remDesc));

  buff = shmLocality == SHM_SEND_SIDE ? (char*)(resources->devHostMem + 1) : (char*)(resources->devRemHostMem + 1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    send->conn.buffs[p] = buff;
    buff += comm->buffSizes[p];
  }
  send->conn.tail = &resources->devRemHostMem->tail;
  send->conn.head = &resources->devHostMem->head;

  if (useMemcpyRecv) {
    send->conn.sizesFifo = resources->devRemHostMem->sizesFifo;
  }
  if (useMemcpySend) {
    struct shmProxyInfo proxyInfo = { NULL, NULL, send->conn.buffs[NCCL_PROTO_SIMPLE], resources->hostMem, resources->remHostMem };
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &proxyInfo, sizeof(struct shmProxyInfo), &proxyInfo, sizeof(struct shmProxyInfo)));
    send->conn.buffs[NCCL_PROTO_SIMPLE] = proxyInfo.devFifo;
    send->conn.tail = &proxyInfo.ceRecvMem->tail;
    send->conn.sizesFifo = proxyInfo.ceRecvMem->sizesFifo;
  }

  return ncclSuccess;
}

static ncclResult_t shmRecvConnect(ncclComm_t comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  char* buff;

  NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, &resources->remDesc));

  buff = shmLocality == SHM_RECV_SIDE ? (char*)(resources->devHostMem + 1) : (char*)(resources->devRemHostMem + 1);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = buff;
    buff += comm->buffSizes[p];
  }
  recv->conn.head = &resources->devRemHostMem->head;
  recv->conn.tail = &resources->devHostMem->tail;

  if (useMemcpyRecv) {
    struct shmProxyInfo proxyInfo = { NULL, NULL, recv->conn.buffs[NCCL_PROTO_SIMPLE], resources->remHostMem, resources->hostMem };
    NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgConnect, &proxyInfo, sizeof(struct shmProxyInfo), &proxyInfo, sizeof(struct shmProxyInfo)));
    recv->conn.buffs[NCCL_PROTO_SIMPLE] = proxyInfo.devFifo;
    recv->conn.tail = &proxyInfo.ceRecvMem->tail;
  }

  return ncclSuccess;
}

static ncclResult_t shmSendFree(struct ncclConnector* send) {
  struct shmRecvResources* resources = (struct shmRecvResources*)send->transportResources;
  if (resources) {
    NCCLCHECK(ncclShmIpcClose(&resources->remDesc));
    free(resources);
    send->transportResources = NULL;
  }
  return ncclSuccess;
}

static ncclResult_t shmRecvFree(struct ncclConnector* recv) {
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  if (resources) {
    NCCLCHECK(ncclShmIpcClose(&resources->remDesc));
    free(resources);
    recv->transportResources = NULL;
  }
  return ncclSuccess;
}

static ncclResult_t shmSendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t ret = ncclSuccess;
  if (reqSize != sizeof(struct shmProxyInfo) || respSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  struct shmProxyInfo* proxyInfo;
  struct shmProxyInfo* reqInfo = (struct shmProxyInfo*)reqBuff;

  proxyInfo = (struct shmProxyInfo*)connection->transportResources;
  proxyInfo->shmFifo = reqInfo->shmFifo;
  proxyInfo->sendMem = reqInfo->sendMem;
  proxyInfo->recvMem = reqInfo->recvMem;
  NCCLCHECKGOTO(ncclHggcCalloc(&proxyInfo->devFifo, proxyState->buffSizes[NCCL_PROTO_SIMPLE], proxyState->pDevMemUsed, proxyState->memManager), ret, fail);
  NCCLCHECKGOTO(ncclHggcHostCalloc(&proxyInfo->ceRecvMem, 1, proxyState->pPinnedMemUsed), ret, fail);
  HGGCCHECKGOTO(hggcEitInterface::getInstance().wrapHggcStreamCreateForCopyEngine(&proxyInfo->stream), ret, fail);
  for (int i=0; i<NCCL_STEPS; i++) {
    HGGCCHECKGOTO(hggcEventCreate(proxyInfo->events+i), ret, fail);
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->transportResources = proxyInfo;
  if (respSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  memcpy(respBuff, proxyInfo, respSize);
  *done = 1;
exit:
  return ret;
fail:
  if (proxyInfo->ceRecvMem) ncclHggcHostFree(proxyInfo->ceRecvMem);
  if (proxyInfo->devFifo) (void)ncclHggcFree(proxyInfo->devFifo, MEM_NORMAL, proxyState->memManager);
  free(proxyInfo);
  goto exit;
}

static ncclResult_t shmRecvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t ret = ncclSuccess;
  if (reqSize != sizeof(struct shmProxyInfo) || respSize != sizeof(struct shmProxyInfo)) return ncclInternalError;
  struct shmProxyInfo* proxyInfo;
  struct shmProxyInfo* reqInfo = (struct shmProxyInfo*)reqBuff;

  proxyInfo = (struct shmProxyInfo*)connection->transportResources;
  proxyInfo->shmFifo = reqInfo->shmFifo;
  proxyInfo->sendMem = reqInfo->sendMem;
  proxyInfo->recvMem = reqInfo->recvMem;
  NCCLCHECKGOTO(ncclHggcCalloc(&proxyInfo->devFifo, proxyState->buffSizes[NCCL_PROTO_SIMPLE], proxyState->pDevMemUsed, proxyState->memManager), ret, fail);
  NCCLCHECKGOTO(ncclHggcHostCalloc(&proxyInfo->ceRecvMem, 1, proxyState->pPinnedMemUsed), ret, fail);
  HGGCCHECKGOTO(hggcEitInterface::getInstance().wrapHggcStreamCreateForCopyEngine(&proxyInfo->stream), ret, fail);
  for (int i=0; i<NCCL_STEPS; i++) {
    HGGCCHECKGOTO(hggcEventCreate(proxyInfo->events+i), ret, fail);
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  memcpy(respBuff, proxyInfo, respSize);
  *done = 1;
exit:
  return ret;
fail:
  if (proxyInfo->ceRecvMem) ncclHggcHostFree(proxyInfo->ceRecvMem);
  if (proxyInfo->devFifo) (void)ncclHggcFree(proxyInfo->devFifo, MEM_NORMAL, proxyState->memManager);
  free(proxyInfo);
  goto exit;
}

static ncclResult_t shmSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct shmProxyInfo* resources = (struct shmProxyInfo*)connection->transportResources;

  if (resources) {
    if (useMemcpySend) {
      HGGCCHECK(hggcStreamDestroy(resources->stream));
      NCCLCHECK(ncclHggcFree(resources->devFifo, MEM_NORMAL, proxyState->memManager));
      NCCLCHECK(ncclHggcHostFree(resources->ceRecvMem));
      for (int i=0; i<NCCL_STEPS; i++) {
        HGGCCHECK(hggcEventDestroy(resources->events[i]));
      }
    }
    NCCLCHECK(ncclShmIpcClose(&resources->desc));
    free(connection->transportResources);
    connection->transportResources = NULL;
  }
  return ncclSuccess;
}

static ncclResult_t shmRecvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct shmProxyInfo* resources = (struct shmProxyInfo*)connection->transportResources;

  if (resources) {
    if (useMemcpyRecv) {
      HGGCCHECK(hggcStreamDestroy(resources->stream));
      NCCLCHECK(ncclHggcFree(resources->devFifo, MEM_NORMAL, proxyState->memManager));
      NCCLCHECK(ncclHggcHostFree(resources->ceRecvMem));
      for (int i=0; i<NCCL_STEPS; i++) {
        HGGCCHECK(hggcEventDestroy(resources->events[i]));
      }
    }
    NCCLCHECK(ncclShmIpcClose(&resources->desc));
    free(connection->transportResources);
    connection->transportResources = NULL;
  }
  return ncclSuccess;
}

static ncclResult_t shmSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = proxyState->buffSizes[p] / NCCL_STEPS;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      // PCCL's Simple-class protocols share the SIMPLE buffer.
      if (!PROTO_IS_SIMPLE_CLASS(p)) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
          continue;
      }
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile int* sizesFifo = resources->ceRecvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->ceRecvMem->tail;
        if ((*recvTail > sub->base+sub->transmitted)) {
          int size = sizesFifo[buffSlot];
          HGGCCHECK(hggcMemcpyAsync(resources->shmFifo+buffSlot*stepSize, resources->devFifo+buffSlot*stepSize, size, hggcMemcpyDeviceToHost, resources->stream));
          HGGCCHECK(hggcEventRecord(resources->events[buffSlot], resources->stream));
          resources->recvMem->sizesFifo[buffSlot] = size;
          __sync_synchronize();
          sub->transmitted += args->sliceSteps;
        }
      }
      if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        hggcError_t res = hggcEventQuery(resources->events[buffSlot]);
        if (res != hggcErrorNotReady) HGGCCHECK(res);
        if (res == hggcSuccess) {
          sub->done += args->sliceSteps;
          resources->recvMem->tail = sub->base + sub->done;
        }
        if (sub->done == sub->nsteps) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

static ncclResult_t shmRecvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int stepSize = proxyState->buffSizes[p] / NCCL_STEPS;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmProxyInfo* resources = (struct shmProxyInfo*) (sub->connection->transportResources);
      // PCCL's Simple-class protocols share the SIMPLE buffer.
      if (!PROTO_IS_SIMPLE_CLASS(p)) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
          continue;
      }
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        if ((*recvTail > sub->base+sub->transmitted)) {
          int size = sizesFifo[buffSlot];
          HGGCCHECK(hggcMemcpyAsync(resources->devFifo+buffSlot*stepSize, resources->shmFifo+buffSlot*stepSize, size, hggcMemcpyHostToDevice, resources->stream));
          HGGCCHECK(hggcEventRecord(resources->events[buffSlot], resources->stream));
          sub->transmitted += args->sliceSteps;
        }
      }
      if (sub->done < sub->transmitted) {
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        hggcError_t res = hggcEventQuery(resources->events[buffSlot]);
        if (res != hggcErrorNotReady) HGGCCHECK(res);
        if (res == hggcSuccess) {
          sub->done += args->sliceSteps;
          resources->ceRecvMem->tail = sub->base + sub->done;
        }
        if (sub->done == sub->nsteps) {
          resources->step = sub->base + sub->nsteps;
          args->done++;
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

static ncclResult_t shmSendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t result = ncclSuccess;
  struct shmRequest* req = (struct shmRequest*)reqBuff;
  if (reqSize != sizeof(struct shmRequest)) return ncclInternalError;
  if (respSize != sizeof(struct shmConnectInfo)) return ncclInternalError;

  struct shmConnectInfo* info = (struct shmConnectInfo*)respBuff;
  struct shmProxyInfo* proxyInfo;

  NCCLCHECK(ncclCalloc(&proxyInfo, 1));
  NCCLCHECKGOTO(ncclShmAllocateShareableBuffer(req->size, req->legacy, &proxyInfo->desc, &info->buf.hptr, &info->buf.dptr), result, fail);
  memcpy(&info->desc, &proxyInfo->desc, sizeof(ncclShmIpcDesc_t));
  connection->transportResources = proxyInfo;
exit:
  return result;
fail:
  free(proxyInfo);
  goto exit;
}

static ncclResult_t shmRecvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  ncclResult_t result = ncclSuccess;
  struct shmRequest* req = (struct shmRequest*)reqBuff;
  if (reqSize != sizeof(struct shmRequest)) return ncclInternalError;
  if (respSize != sizeof(struct shmConnectInfo)) return ncclInternalError;

  struct shmConnectInfo* info = (struct shmConnectInfo*)respBuff;
  struct shmProxyInfo* proxyInfo;

  NCCLCHECK(ncclCalloc(&proxyInfo, 1));
  NCCLCHECKGOTO(ncclShmAllocateShareableBuffer(req->size, req->legacy, &proxyInfo->desc, &info->buf.hptr, &info->buf.dptr), result, fail);
  memcpy(&info->desc, &proxyInfo->desc, sizeof(ncclShmIpcDesc_t));
  connection->transportResources = proxyInfo;
exit:
  return result;
fail:
  free(proxyInfo);
  goto exit;
}

static void initCeOperation() {
  static int init = 0;
  if (!init) {
    useMemcpySend = ncclParamShmUseHggcMemcpy() && (ncclParamShmMemcpyMode() & 1);
    useMemcpyRecv = ncclParamShmUseHggcMemcpy() && (ncclParamShmMemcpyMode() & 2);
    if (useMemcpySend) {
      shmTransport.send.proxyConnect = shmSendProxyConnect;
      shmTransport.send.proxyProgress = shmSendProxyProgress;
    }
    if (useMemcpyRecv) {
      shmTransport.recv.proxyConnect = shmRecvProxyConnect;
      shmTransport.recv.proxyProgress = shmRecvProxyProgress;
    }
    shmLocality = ncclParamShmLocality();
    if (shmLocality != SHM_SEND_SIDE && shmLocality != SHM_RECV_SIDE) {
      WARN("Ignoring SHM locality, must be 1 (sender side) or 2 (receiver side, default)");
      shmLocality = SHM_RECV_SIDE;
    }
    init = 1;
  }
}

ncclResult_t ncclShmAllocateShareableBuffer(size_t size, bool legacy, ncclShmIpcDesc_t *desc, void **hptr, void **dptr) {
  if (desc == NULL || hptr == NULL) {
    WARN("Invalid argument desc %p, hptr %p", desc, hptr);
    return ncclInvalidArgument;
  }
  char shmPath[SHM_PATH_MAX] = { '\0' };
  desc->shmli.shmSize = size;
  NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), size, hptr, dptr, 1, &desc->shmli.handle));
  memcpy(desc->shmli.shmSuffix, shmPath + sizeof("/dev/shm/pccl-") - 1, sizeof(desc->shmli.shmSuffix));
  desc->legacy = true;
  INFO(NCCL_SHM, "MMAP allocated shareable host buffer %s size %zi ptr %p", shmPath, size, *hptr);
  return ncclSuccess;
}

ncclResult_t ncclShmImportShareableBuffer(ncclComm_t comm, int proxyRank, ncclShmIpcDesc_t *desc, void **hptr, void **dptr, ncclShmIpcDesc_t *descOut) {
  if (comm == NULL || desc == NULL || hptr == NULL || descOut == NULL) {
    WARN("Invalid argument comm %p, desc %p, hptr %p, descOut %p", comm, desc, hptr, descOut);
    return ncclInvalidArgument;
  }
  char shmPath[SHM_PATH_MAX];
  snprintf(shmPath, sizeof(shmPath), "/dev/shm/pccl-%s", desc->shmli.shmSuffix);
  NCCLCHECK(ncclShmOpen(shmPath, sizeof(shmPath), desc->shmli.shmSize, hptr, dptr, -1, &descOut->shmli.handle));
  descOut->legacy = true;
  INFO(NCCL_SHM, "MMAP imported shareable host buffer %s size %zi ptr %p", shmPath, desc->shmli.shmSize, *hptr);
  return ncclSuccess;
}

ncclResult_t ncclShmIpcClose(ncclShmIpcDesc_t *desc) {
  if (desc) {
    NCCLCHECK(ncclShmClose(desc->shmli.handle));
  }

  return ncclSuccess;
}

struct ncclTransport shmTransport = {
  "SHM",
  shmCanConnect,
  { shmSendSetup, shmSendConnect, shmSendFree, NULL, shmSendProxySetup, NULL, shmSendProxyFree, NULL, NULL, NULL },
  { shmRecvSetup, shmRecvConnect, shmRecvFree, NULL, shmRecvProxySetup, NULL, shmRecvProxyFree, NULL, NULL, NULL }
};
