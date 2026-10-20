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

struct shmConnectInfo {
  char shmName[7];
  int shmSize;
};
static_assert(sizeof(shmConnectInfo) <= CONNECT_SIZE, "SHM Connect info is too large");

struct shmSendResources {
  int remShmSize;
  struct ncclRecvMem* remHostMem;
  struct ncclRecvMem* devRemHostMem;
  int shmSize;
  struct ncclSendMem* hostMem;
  struct ncclSendMem* devHostMem;
};

struct shmRecvResources {
  int remShmSize;
  struct ncclSendMem* remHostMem;
  struct ncclSendMem* devRemHostMem;
  int shmSize;
  struct ncclRecvMem* hostMem;
  struct ncclRecvMem* devHostMem;
};

NCCL_PARAM(ShmDisable, "SHM_DISABLE", 0);

/* Determine two peers can communicate with SHM */
ncclResult_t shmCanConnect(int* ret, ncclComm_t comm, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 0;

  if (ncclParamShmDisable() == 1) return ncclSuccess;

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
ncclResult_t shmSendSetup(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct shmSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;

  char shmPath[PATH_MAX];
  shmPath[0] = '\0';
  info->shmSize = resources->shmSize = sizeof(struct ncclSendMem);
  NCCLCHECK(ncclShmOpen(shmPath, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));
  TRACE(NCCL_SHM,"Opened shmName %s shmSize %d", shmPath, info->shmSize);
  memcpy(info->shmName, shmPath+sizeof("/dev/shm/pccl-")-1, sizeof(info->shmName));

  INFO(NCCL_INIT|NCCL_SHM,"Channel %02d : %d[%lx] -> %d[%lx] via direct shared memory", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  return ncclSuccess;
}

ncclResult_t shmRecvSetup(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct shmRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  static_assert(sizeof(struct shmConnectInfo) <= sizeof(struct ncclConnect), "shm Connect Info is too big");
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;

  char shmPath[PATH_MAX];
  shmPath[0] = '\0';
  int shmSize = sizeof(struct ncclRecvMem);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) shmSize += comm->buffSizes[p];
  info->shmSize = resources->shmSize = shmSize;
  NCCLCHECK(ncclShmOpen(shmPath, resources->shmSize, (void**)&resources->hostMem, (void**)&resources->devHostMem, 1));
  TRACE(NCCL_SHM,"Opened shmName %s shmSize %d", shmPath, info->shmSize);
  memcpy(info->shmName, shmPath+sizeof("/dev/shm/pccl-")-1, sizeof(info->shmName));

  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t shmSendConnect(ncclComm_t comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;
  struct shmSendResources* resources = (struct shmSendResources*)send->transportResources;

  char shmPath[PATH_MAX];
  sprintf(shmPath, "/dev/shm/pccl-%s", info->shmName);
  resources->remShmSize = info->shmSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmPath, info->shmSize);
  NCCLCHECK(ncclShmOpen(shmPath, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));
  // Remove the file to ensure proper clean-up
  NCCLCHECK(ncclShmUnlink(shmPath));

  send->transportResources = resources;
  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    send->conn.buffs[p] = (char*)(resources->devRemHostMem+1) + offset;
    offset += comm->buffSizes[p];
  }
  send->conn.tail = &resources->devRemHostMem->tail;

  send->conn.head = &resources->devHostMem->head;
  return ncclSuccess;
}

ncclResult_t shmRecvConnect(ncclComm_t comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  struct shmConnectInfo* info = (struct shmConnectInfo*)connectInfo;

  char shmPath[PATH_MAX];
  sprintf(shmPath, "/dev/shm/pccl-%s", info->shmName);
  resources->remShmSize = info->shmSize;
  TRACE(NCCL_SHM,"Open shmName %s shmSize %d", shmPath, info->shmSize);
  NCCLCHECK(ncclShmOpen(shmPath, resources->remShmSize, (void**)&resources->remHostMem, (void**)&resources->devRemHostMem, 0));
  NCCLCHECK(ncclShmUnlink(shmPath));
  recv->conn.head = &resources->devRemHostMem->head;

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = (char*)(resources->devHostMem+1) + offset;
    offset += comm->buffSizes[p];
  }
  recv->conn.tail = &resources->devHostMem->tail;
  return ncclSuccess;
}

ncclResult_t shmSendFree(ncclComm_t, struct ncclConnector* send) {
  struct shmRecvResources* resources = (struct shmRecvResources*)send->transportResources;
  NCCLCHECK(ncclShmClose(resources->hostMem, resources->devHostMem, resources->shmSize));
  NCCLCHECK(ncclShmClose(resources->remHostMem, resources->devRemHostMem, resources->remShmSize));
  NCCLCHECK(ncclFree(resources));
  return ncclSuccess;
}

ncclResult_t shmRecvFree(ncclComm_t, struct ncclConnector* recv) {
  struct shmRecvResources* resources = (struct shmRecvResources*)recv->transportResources;
  NCCLCHECK(ncclShmClose(resources->hostMem, resources->devHostMem, resources->shmSize));
  NCCLCHECK(ncclShmClose(resources->remHostMem, resources->devRemHostMem, resources->remShmSize));
  NCCLCHECK(ncclFree(resources));
  return ncclSuccess;
}

struct ncclTransport shmTransport = {
  "SHM",
  shmCanConnect,
  { shmSendSetup, shmSendConnect, shmSendFree, NULL, NULL, NULL, NULL, NULL, NULL, NULL},
  { shmRecvSetup, shmRecvConnect, shmRecvFree, NULL, NULL, NULL, NULL, NULL, NULL, NULL}
};
