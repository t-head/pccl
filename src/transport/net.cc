/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "transport.h"
#include "net.h"
#include "graph.h"
#include "proxy.h"
#include "collectives.h"
#include "gdrwrap.h"
#include "shm.h"
#include "p2p.h"
#include "profiler.h"

bool nccl_use_eic = false;
// EIC plugins resolve these variables with dlsym.
__attribute__((visibility("default"))) int nccl_use_u2mm_host = 0;
__attribute__((visibility("default"))) int nccl_use_u2mm_dev = 0;
__attribute__((visibility("default"))) int nccl_eic_gpu_offset = 0;

static_assert(sizeof(ncclNetHandle_t) <= CONNECT_SIZE, "NET Connect info is too large");

#define NCCL_NET_MAP_HOSTMEM 0
#define NCCL_NET_MAP_DEVMEM 1
#define NCCL_NET_MAP_SHARED_HOSTMEM 2
#define NCCL_NET_MAP_SHARED_DEVMEM 3
#define NCCL_NET_MAP_GDCMEM 4
#define NCCL_NET_MAP_MEMS 5

#define NCCL_NET_MAP_MASK_DEVMEM 0x40000000
#define NCCL_NET_MAP_MASK_SHARED 0x80000000
#define NCCL_NET_MAP_MASK_USED   0x20000000
#define NCCL_NET_MAP_MASK_OFFSET 0x1fffffff

#define NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName) \
  ((mapStruct)->offsets.offsetName >> 30)

#define NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName >> 29) == 0)

#define NCCL_NET_MAP_GET_POINTER(mapStruct, cpuOrGpu, offsetName) \
  (NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) ? NULL : \
   (mapStruct)->mems[NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName)].cpuOrGpu##Ptr + ((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_OFFSET))

#define NCCL_NET_MAP_DEV_MEM(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_DEVMEM) != 0)

#define NCCL_NET_MAP_ADD_POINTER(mapStruct, shared, dev, memSize, offsetName) do { \
    int bank = NCCL_NET_MAP_MASK_USED + (dev)*NCCL_NET_MAP_MASK_DEVMEM + (shared)*NCCL_NET_MAP_MASK_SHARED; \
    if ((shared) == 0) { \
      if (dev) { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size += memSize; \
      } else { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size += memSize; \
      } \
    } else { \
      (mapStruct)->offsets.offsetName = bank; \
    } \
} while (0);

struct connectMapMem{
  char* gpuPtr;
  char* cpuPtr;
  int size;
  ncclIpcDesc ipcDesc;
  union {
    char shmPath[PATH_MAX];
    char u2mmMkey[U2MM_MKEY_MAX_LEN];
  };
};

struct connectMap {
  struct ncclMemManager* memManager;
  // Memory modes belong to the selected backend, including imported mappings.
  int u2mmHost;
  int u2mmDev;
  int sameProcess;
  int shared;
  int hggcDev;
  // First 3 bits of offsets determine the mem bank. 001 is host mem, 011 is dev mem, 101 is shared host mem and 111 is shared dev mem.
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  struct {
    uint32_t sendMem;
    uint32_t recvMem;
    uint32_t buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};

struct sendResources {
  struct connectMap map;
  void* netSendComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int netDev;
  int useGdr;
  int maxRecvs;
  uint64_t* gdcSync;
  void* gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  char* u2mmMkeyPtr;
};

struct recvResources {
  struct connectMap map;
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int tpRemoteProxyRank;
  int netDev;
  int useGdr;
  int maxRecvs;
  uint64_t* gdcSync;
  uint64_t* gdcFlush;
  void* gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
};

NCCL_PARAM(NetDisableIntra, "NET_DISABLE_INTRA", -2);

/* Determine if two peers can communicate with NET */
static ncclResult_t canConnect(int* ret, ncclComm_t comm, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  // Same host?
  if (info1->hostHash == info2->hostHash) {
    // User disabled NET for intra-node?
    if (ncclParamNetDisableIntra() == 1) {
      *ret = 0;
      return ncclSuccess;
    }
  }
  *ret = 1;
  return ncclSuccess;
}

NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);
NCCL_PARAM(NetSharedComms, "NET_SHARED_COMMS", 1);
#ifdef NET_BYPASS
PCCL_PARAM(NetByPass, "NET_BYPASS", 0);
#endif
PCCL_PARAM(GdrCpuFlush, "GDR_CPU_FLUSH", 0);
PCCL_PARAM(SnicCpuFlush, "SNIC_CPU_FLUSH", 1);
PCCL_PARAM(CpuFlushAsync, "CPU_FLUSH_ASYNC", 0);
PCCL_PARAM(GdrUseDevMemForRxTail, "GDR_USE_DEV_MEM_FOR_RX_TAIL", 1);

// Sentinel stored in subGroup->requests[slot] to mark that this step issued an
// async CPU flush (so recv progress stage-B polls the ticket instead of ncclNetTest).
static char ncclAsyncFlushSentinelObj;
#define NCCL_ASYNC_FLUSH_PENDING ((void*)&ncclAsyncFlushSentinelObj)

struct setupReq {
  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int shared;
  int netDev;
  int useGdr;
  int channelId;
  int connIndex;
};

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
static ncclResult_t sendSetup(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req;
  int localRank;

  send->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;

  int proxyRank;
  int64_t netId;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &req.netDev, &proxyRank));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 1, &req.useGdr));
  send->conn.direct |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, proxyRank, &send->proxyConn));
  NCCLCHECK(ncclTopoGetLocalRank(comm, myInfo->rank, &localRank));
  req.tpLocalRank = comm->topParentLocalRanks[localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  if (proxyRank == myInfo->rank) {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d%s%s", channelId, connIndex, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), req.netDev,
        req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  } else {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%lx] -> %d[%lx] [send] via NET/%s/%d(%d)%s%s", channelId, connIndex, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, ncclNetName(), req.netDev,
        proxyRank, req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  }
  *((int*)connectInfo) = comm->topParentRanks[proxyRank];
  return ncclSuccess;
}

// GDRCOPY support: TAIL_ENABLE When enabled locates the RX proxy tail in CUDA memory
NCCL_PARAM(GdrCopySyncEnable, "GDRCOPY_SYNC_ENABLE", 0);
// GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E read to flush GDRDMA buffers
NCCL_PARAM(GdrCopyFlushEnable, "GDRCOPY_FLUSH_ENABLE", 0);

/* Setup recv connector */
static ncclResult_t recvSetup(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req;
  int localRank;

  recv->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;

  // Use myInfo->rank as the receiver uses its own NIC
  int64_t netId;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, myInfo->rank, &netId, &req.netDev, NULL));
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->rank, netId, 0, &req.useGdr));

  // We don't support PXN on receive yet
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 0, myInfo->rank, &recv->proxyConn));

  NCCLCHECK(ncclTopoGetLocalRank(comm, myInfo->rank, &localRank));
  req.tpLocalRank = comm->topParentLocalRanks[localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), connectInfo, sizeof(ncclNetHandle_t)));

  INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%lx] -> %d[%lx] [receive] via NET/%s/%d%s%s", channelId, connIndex, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId, ncclNetName(), req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  return ncclSuccess;
}

static ncclResult_t netMapU2mmShm(struct connectMapMem* mem) {
  NCCLCHECK(ncclU2mmShmOpen(mem->u2mmMkey, mem->size, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr));
  return ncclSuccess;
}

static ncclResult_t netMapShm(struct connectMapMem* mem) {
  NCCLCHECK(ncclShmOpen(mem->shmPath, mem->size, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr, 0));
  NCCLCHECK(ncclShmUnlink(mem->shmPath));
  return ncclSuccess;
}

static ncclResult_t netCreateU2mmShm(struct connectMapMem* mem, ncclNet_t* net, int netDev, char** mkeyPtr) {
  char* mkey = NULL;
  NCCLCHECK(ncclU2mmShmCreate(&mkey, mem->size, (void**)&mem->cpuPtr, net, netDev));
  memcpy((void*)mem->u2mmMkey, (void*)mkey, U2MM_MKEY_MAX_LEN);
  *mkeyPtr = mkey;
  return ncclSuccess;
}

static ncclResult_t netCreateShm(struct connectMapMem* mem) {
  mem->shmPath[0] = '\0'; // Let ncclShmOpen create a tmp file
  NCCLCHECK(ncclShmOpen(mem->shmPath, mem->size, (void**)&mem->cpuPtr, NULL, 1));
  return ncclSuccess;
}

static ncclResult_t netDumpMap(struct connectMap* map) {
  WARN("Dump map same process %d shared %d\n", map->sameProcess, map->shared);
  struct connectMapMem *mem = map->mems+NCCL_NET_MAP_HOSTMEM;
  WARN("Mem 0: Host mem %s (%x B) CPU %p GPU %p\n", mem->shmPath, mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_DEVMEM;
  WARN("Mem 1: Vid  mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_HOSTMEM;
  WARN("Mem 2: Shared Host mem %s (%x B) CPU %p GPU %p\n", mem->shmPath, mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_DEVMEM;
  WARN("Mem 3: Shared Vid mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  WARN("SendMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.sendMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, sendMem), map->offsets.sendMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem), NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem));
  WARN("RecvMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.recvMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, recvMem), map->offsets.recvMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem), NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem));
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    WARN("Proto %d -> Used %d Bank %d Offset %x, cpu %p, gpu %p\n", p,
        map->offsets.buffs[p] & NCCL_NET_MAP_MASK_USED ? 1 : 0,
        NCCL_NET_MAP_OFFSET_BANK(map, buffs[p]), map->offsets.buffs[p] & NCCL_NET_MAP_MASK_OFFSET,
        NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]), NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]));
  }
  WARN("End of dump\n");
  return ncclSuccess;
}

static ncclResult_t sendConnect(ncclComm_t comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct connectMap* map = (struct connectMap*)send->transportResources;
  void* opId;

  // map isn't allocated thus this op hasn't been submitted yet
  if (!map) {
    NCCLCHECK(ncclCalloc(&map, 1));
    send->transportResources = map;
    opId = send;
    INFO(NCCL_PROXY, "sendConnect pcclProxyCallAsync opId=%p", opId);
    NCCLCHECK(ncclProxyCallAsync(comm, &send->proxyConn, ncclProxyMsgConnect, connectInfo, sizeof(ncclNetHandle_t), sizeof(struct connectMap), opId));
  } else {
    opId = send;
  }

  ncclResult_t ret;
  ret = ncclPollProxyResponse(comm, &send->proxyConn, map, opId);
  if (ret != ncclSuccess) {
    if (ret != ncclInProgress) {
      NCCLCHECK(ncclFree(map));
      send->transportResources = NULL;
    }
    return ret;
  }
  INFO(NCCL_PROXY, "sendConnect pcclPollProxyResponse opId=%p", opId);
  map->memManager = comm->memManager;

  if (map->sameProcess && !ncclCuMemEnable()) {
    if (map->hggcDev != comm->hggcDev) {
      // Enable P2P access for Legacy IPC
      hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
      HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
      hggcError_t err = hggcDeviceEnablePeerAccess(map->hggcDev, 0);
      HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
      if (err == hggcErrorPeerAccessAlreadyEnabled) {
        hggcGetLastError();
      } else if (err != hggcSuccess) {
        ERROR("failed to peer with device %d: %d %s", map->hggcDev, err, hggcGetErrorString(err));
        return ncclInternalError;
      }
    }
  } else if (!(map->sameProcess && map->hggcDev == comm->hggcDev)) {
    if (!map->sameProcess) {
      if (map->u2mmHost) {
        NCCLCHECK(netMapU2mmShm(map->mems+NCCL_NET_MAP_HOSTMEM));
      } else {
        NCCLCHECK(netMapShm(map->mems+NCCL_NET_MAP_HOSTMEM));
      }
    }

    int memMode = (!map->sameProcess && map->u2mmDev) ? MEM_U2MM_DIRECT : map->u2mmDev;
    if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
      NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.rank,
                                             map->mems[NCCL_NET_MAP_DEVMEM].size,
                                             &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                             (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr,
                                             memMode));
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = NULL;
    }
    if (map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size) {
      void** sharedDevMemPtr = comm->proxyState->sharedDevMems+send->proxyConn.tpLocalRank;
      comm->proxyState->sharedDevMemModes[send->proxyConn.tpLocalRank] = memMode;
      if (*sharedDevMemPtr == NULL) {
        NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.rank,
                                               map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size,
                                               &map->mems[NCCL_NET_MAP_SHARED_DEVMEM].ipcDesc,
                                               sharedDevMemPtr, memMode));
      }
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = (char*)(*sharedDevMemPtr);
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].cpuPtr = NULL;
    }
  }
  // NCCLCHECK(netDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.sizesFifo = recvMem->sizesFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  send->conn.offsFifo = map->shared ? recvMem->offsFifo: NULL;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
  return ncclSuccess;
}

/* Connect to this peer */
static ncclResult_t recvConnect(ncclComm_t comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct connectMap* map = (struct connectMap*)recv->transportResources;
  void* opId;

  if (!map) {
    NCCLCHECK(ncclCalloc(&map, 1));
    recv->transportResources = map;
    // Use recv connector as unique identifier
    opId = recv;
    INFO(NCCL_PROXY, "recvConnect pcclProxyCallAsync opId=%p &recv->proxyConn=%p connectInfo=%p",
       opId, &recv->proxyConn, connectInfo);

    NCCLCHECK(ncclProxyCallAsync(comm, &recv->proxyConn, ncclProxyMsgConnect, connectInfo, sizeof(int), sizeof(struct connectMap), opId));
  } else {
    opId = recv;
  }

  ncclResult_t ret;
  ret = ncclPollProxyResponse(comm, &recv->proxyConn, map, opId);
  if (ret != ncclSuccess) {
    if (ret != ncclInProgress) {
      NCCLCHECK(ncclFree(map));
      recv->transportResources = NULL;
    }
    return ret;
  }
  INFO(NCCL_PROXY, "recvConnect pcclPollProxyResponse opId=%p", opId);
  //NCCLCHECK(netDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  recv->conn.head = &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;
  recv->conn.sizesFifo = recvMem->sizesFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  recv->conn.offsFifo = map->shared ? recvMem->offsFifo: NULL;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);
  }
  return ncclSuccess;
}

static ncclResult_t sendFree(struct ncclConnector* send) {
  struct connectMap* map = (struct connectMap*)(send->transportResources);
  if (map) {
    int hggcDev;
    HGGCCHECK(hggcGetDevice(&hggcDev));
    if (map->hggcDev != hggcDev && map->mems[NCCL_NET_MAP_DEVMEM].size) {
      int memMode = (!map->sameProcess && map->u2mmDev) ? MEM_U2MM_DIRECT : map->u2mmDev;
      if (ncclCuMemEnable() && (memMode == MEM_NORMAL)) {
        // hgMem API support
        NCCLCHECK(ncclP2pFreeShareableBuffer(&map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
        NCCLCHECK(ncclHgMemFree(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->memManager));
      } else if (!map->sameProcess) {
        // Legacy CUDA IPC support
        HGGCCHECK(hggcIpcCloseMemHandle(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      }
    }
    if (!map->sameProcess) {
      if (map->u2mmHost) {
        NCCLCHECK(ncclU2mmShmClose(map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr));
      } else {
        NCCLCHECK(ncclShmClose(map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
      }
    }
    NCCLCHECK(ncclFree(map));
  }
  return ncclSuccess;
}

static ncclResult_t recvFree(struct ncclConnector* recv) {
  NCCLCHECK(ncclFree(recv->transportResources));
  return ncclSuccess;
}

#define NCCL_SHARED_STEPS 16
static ncclResult_t sharedBuffersInit(struct ncclProxyState* proxyState, int hggc, int tpLocalRank, int type, int sameProcess,
    int nChannels, char** gpuPtr, char** cpuPtr, int* size, ncclIpcDesc* ipcDesc) {
  if (hggc == 0 && sameProcess == 0) {
      ERROR("PXN should not use host buffers for data");
      return ncclInternalError;
  }
  struct ncclProxyProgressState* progressState = &proxyState->progressState;
  if (progressState->localPeers == NULL) {
    NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
  }
  struct ncclProxyPeer** localPeers = progressState->localPeers;
  if (localPeers[tpLocalRank] == NULL) {
    NCCLCHECK(ncclCalloc(localPeers+tpLocalRank, 1));
  }
  struct ncclProxyPeer* peer = localPeers[tpLocalRank];
  struct ncclProxySharedP2p* state = type == 0 ? &peer->send : &peer->recv;
  state->refcount++;
  if (state->size == 0) {
    state->size = nChannels*(NCCL_SHARED_STEPS/NCCL_STEPS)*proxyState->buffSizes[NCCL_PROTO_SIMPLE]/SENDRECV_SLICEFACTOR;
  }

  if (size) *size = state->size;

  if (hggc && state->hggcBuff == NULL) {
    int memMode = (!sameProcess && proxyState->u2mmDev) ? MEM_U2MM_DIRECT : proxyState->u2mmDev;
    if ((sameProcess == 0 && state->size > 0) || ncclCuMemEnable()) {
      NCCLCHECK(ncclP2pAllocateShareableBuffer(state->size, 0, &state->ipcDesc, (void**)&state->hggcBuff,
        proxyState->pDevMemUsed, memMode, -1, proxyState->memManager, ncclMemPersist));
    } else {
      NCCLCHECK(ncclHggcCallocU2mm(&state->hggcBuff,
                                    state->size,
                                    proxyState->pDevMemUsed,
                                    memMode, proxyState->memManager));
    }
  }
  if (!hggc && state->hostBuff == NULL) {
    NCCLCHECK(ncclHggcHostCallocU2mm(&state->hostBuff,
                                      state->size,
                                      proxyState->pPinnedMemUsed,
                                      proxyState->u2mmHost,
                                      0, proxyState->ncclNet));
  }
  if (cpuPtr) *cpuPtr = hggc ? state->hggcBuff : state->hostBuff;
  if (gpuPtr) *gpuPtr = (cpuPtr && sameProcess) ? *cpuPtr : NULL;
  if (ipcDesc) memcpy(ipcDesc, &state->ipcDesc, sizeof(state->ipcDesc));
  return ncclSuccess;
}

static ncclResult_t sharedBuffersGet(struct ncclProxyState* proxyState, int channel, int slot, int* offset, int *size) {
  // Use different pools for different channels and also separate send/recv.
  int slotSize = proxyState->buffSizes[NCCL_PROTO_SIMPLE]/(NCCL_STEPS*SENDRECV_SLICEFACTOR);
  int globalSlot = (channel*NCCL_SHARED_STEPS)+slot;
  *offset = slotSize * globalSlot;
  if (size) *size = slotSize;
  return ncclSuccess;
}

static ncclResult_t sharedBuffersDestroy(struct ncclProxyState* proxyState, int tpLocalRank, int type, struct ncclProxyConnection* connection) {
  if (proxyState->progressState.localPeers == NULL) NCCLCHECK(ncclInternalError);
  struct ncclProxyPeer* peer = proxyState->progressState.localPeers[tpLocalRank];
  if (peer == NULL) NCCLCHECK(ncclInternalError);
  struct ncclProxySharedP2p* state = type == 0 ? &peer->send : &peer->recv;
  //if (state->size == 0) NCCLCHECK(ncclInternalError);
  if (ncclAtomicRefCountDecrement(&state->refcount) == 0) {
    if (state->hggcBuff) {
      if (!connection->sameProcess || ncclCuMemEnable()) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&state->ipcDesc));
      }
      NCCLCHECK(ncclHggcFree(state->hggcBuff, proxyState->u2mmDev, proxyState->memManager));
    }
    if (state->hostBuff) NCCLCHECK(ncclHggcHostFree(state->hostBuff, proxyState->u2mmHost));
  }
  if (peer->send.refcount || peer->recv.refcount) return ncclSuccess;
  NCCLCHECK(ncclFree(proxyState->progressState.localPeers[tpLocalRank]));
  for (int r=0; r<proxyState->tpLocalnRanks; r++) {
    if (proxyState->progressState.localPeers[r]) return ncclSuccess;
  }
  // All peers are freed, free array
  NCCLCHECK(ncclFree(proxyState->progressState.localPeers));
  return ncclSuccess;
}

static ncclResult_t proxySharedInit(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels) {
  NCCLCHECK(sharedBuffersInit(proxyState, 1, connection->tpLocalRank, 0, connection->sameProcess, nChannels, NULL, NULL, NULL, NULL));
  return ncclSuccess;
}

static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct sendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->netDev = req->netDev;
  resources->shared = connection->shared = req->shared;
  resources->useGdr = req->useGdr;
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  resources->u2mmMkeyPtr = NULL;
  ncclNetProperties_t props;
  NCCLCHECK(ncclNetGetProperties(req->netDev, &props));
  resources->maxRecvs = props.maxRecvs;

  // We don't return any data
  if (respSize != 0) return ncclInternalError;
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct recvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->netDev = req->netDev;
  resources->shared = connection->shared = req->shared;
  resources->useGdr = req->useGdr;
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  ncclNetProperties_t props;
  NCCLCHECK(ncclNetGetProperties(req->netDev, &props));
  resources->maxRecvs = props.maxRecvs;

  if (respSize != sizeof(ncclNetHandle_t)) return ncclInternalError;
  if (ncclEicNet(proxyState->ncclNet) && nccl_eic_gpu_offset && proxyState->ownerComm->hgmlDeviceCount > 8)
    *(int *)((uintptr_t)respBuff + nccl_eic_gpu_offset) = proxyState->ownerComm->hgmlDev + 1;
  NCCLCHECK(ncclNetListen(req->netDev, respBuff, &resources->netListenComm));
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);
  if (reqSize != sizeof(ncclNetHandle_t)) return ncclInternalError;

  if (ncclEicNet(proxyState->ncclNet) && nccl_eic_gpu_offset && proxyState->ownerComm->hgmlDeviceCount > 8)
    *(int *)((uintptr_t)reqBuff + nccl_eic_gpu_offset) = proxyState->ownerComm->hgmlDev + 1;
  if (resources->shared) {
    // Shared buffers
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->tpLocalRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers+resources->tpLocalRank, 1));
    }
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->send.proxyAppend+resources->channelId;

    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      if (progressState->netComms[resources->netDev] == NULL) {
        NCCLCHECK(ncclCalloc(progressState->netComms+resources->netDev, proxyState->tpnRanks));
      }
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev]+resources->tpRemoteRank;
      // let only one local rank connect to a tpRemoteRank to avoid duplicate connections
      if (comms->activeConnect[resources->channelId] == 0)
        comms->activeConnect[resources->channelId] = (resources->tpLocalRank + 1);
      if (comms->sendComm[resources->channelId] == NULL &&
          comms->activeConnect[resources->channelId] == (resources->tpLocalRank + 1)) {
        NCCLCHECK(ncclNetConnect(resources->netDev, reqBuff, comms->sendComm + resources->channelId));
      }
      resources->netSendComm = comms->sendComm[resources->channelId];
      if (comms->sendComm[resources->channelId]) comms->sendRefCount[resources->channelId]++;
    } else {
      NCCLCHECK(ncclNetConnect(resources->netDev, reqBuff, &resources->netSendComm));
    }
  } else {
    // Connect to remote peer
    NCCLCHECK(ncclNetConnect(resources->netDev, reqBuff, &resources->netSendComm));
    connection->proxyAppendPtr = &connection->proxyAppend;
  }
  if (resources->netSendComm == NULL) {
    *done = 0;
    return ncclInProgress;
  }
  *done = 1;

  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess = connection->sameProcess;
  map->u2mmHost = proxyState->u2mmHost;
  map->u2mmDev = proxyState->u2mmDev;
  map->shared = resources->shared;
  HGGCCHECK(hggcGetDevice(&map->hggcDev));

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      resources->buffSizes[p] = proxyState->buffSizes[p];
      // skip zero bytes regMr
      if (resources->buffSizes[p] == 0) continue;

      // for signal/bulk shared buffer
      if (p > NCCL_PROTO_SIMPLE) {
        map->offsets.buffs[p] = map->offsets.buffs[NCCL_PROTO_SIMPLE];
        continue;
      }

      NCCL_NET_MAP_ADD_POINTER(map, 0, p!= NCCL_PROTO_LL && resources->useGdr, resources->buffSizes[p], buffs[p]);
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(sharedBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 0, map->sameProcess, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, &mapMem->ipcDesc));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;

    if (proxyState->allocP2pNetLLBuffers) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, 0, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
      resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
    }

    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    int memMode = (!map->sameProcess && map->u2mmDev) ? MEM_U2MM_DIRECT : map->u2mmDev;
    if (resources->shared == 0) {
      if (!map->sameProcess || ncclCuMemEnable()) {
        ALIGN_SIZE(map->mems[NCCL_NET_MAP_DEVMEM].size, HGGC_IPC_MIN);
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0,
          &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc, (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr,
          proxyState->pDevMemUsed, memMode, -1, proxyState->memManager, ncclMemPersist));
      } else {
        NCCLCHECK(ncclHggcCallocU2mm(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr,
                                      map->mems[NCCL_NET_MAP_DEVMEM].size,
                                      proxyState->pDevMemUsed,
                                      memMode, proxyState->memManager));
      }
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  if (map->sameProcess) {
    NCCLCHECK(ncclHggcHostCallocU2mm(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr,
                                      map->mems[NCCL_NET_MAP_HOSTMEM].size,
                                      proxyState->pPinnedMemUsed,
                                      map->u2mmHost,
                                      resources->netDev, proxyState->ncclNet));
    map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  } else {
    if (map->u2mmHost) {
      NCCLCHECK(netCreateU2mmShm(map->mems+NCCL_NET_MAP_HOSTMEM, proxyState->ncclNet, resources->netDev, &resources->u2mmMkeyPtr));
    } else {
      NCCLCHECK(netCreateShm(map->mems+NCCL_NET_MAP_HOSTMEM));
    }
    void* sendMem = (void*)NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
    void* recvMem = (void*)NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
    memset(sendMem, 0, sizeof(struct ncclSendMem));
    memset(recvMem, 0, sizeof(struct ncclRecvMem));
  }
  if (ncclGdrCopy && map->sameProcess && ncclParamGdrCopySyncEnable()) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 1, &resources->gdrDesc, proxyState->pDevMemUsed, proxyState->memManager));

    resources->gdcSync = cpuPtr;
    struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
    gdcMem->cpuPtr = (char*)cpuPtr;
    gdcMem->gpuPtr = (char*)gpuPtr;
    gdcMem->size = sizeof(uint64_t); // sendMem->head
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);

  // Don't give credits yet in shared mode.
  resources->sendMem->head = map->shared ? -NCCL_STEPS : 0;
  for (int i=0; i<NCCL_STEPS; i++) resources->recvMem->sizesFifo[i] = -1;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (p > NCCL_PROTO_SIMPLE) {
      resources->mhandles[p] = resources->mhandles[NCCL_PROTO_SIMPLE];
      continue;
    }
    if (resources->buffers[p]) {
      NCCLCHECK(ncclNetRegMr(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_HGGC : NCCL_PTR_HOST, &resources->mhandles[p]));
    }
  }

  // NCCLCHECK(netDumpMap(map));
  if (respSize != sizeof(struct connectMap)) return ncclInternalError;
  memcpy(respBuff, map, sizeof(struct connectMap));

  return ncclSuccess;
}

// Comma-separated list of device name prefixes identifying a SNIC, can be
// overridden with PCCL_SNIC_NAME_PREFIX.
#define SNIC_NAME_PREFIX_DEFAULT "troce_,tsoe_,srdma_"
#define MAX_SNIC_NAME_PREFIXES 16

static struct netIf snicPrefixes[MAX_SNIC_NAME_PREFIXES];
static int nSnicPrefixes = 0;
static pthread_once_t snicPrefixOnce = PTHREAD_ONCE_INIT;

static void initSnicPrefixes() {
  const char* env = pcclGetEnvWithDualPrefix("PCCL_SNIC_NAME_PREFIX");
  if (env == NULL || strlen(env) == 0) {
    env = SNIC_NAME_PREFIX_DEFAULT;
  } else {
    INFO(NCCL_NET|NCCL_ENV, "PCCL_SNIC_NAME_PREFIX set by environment to %s", env);
  }
  nSnicPrefixes = parseStringList(env, snicPrefixes, MAX_SNIC_NAME_PREFIXES);
}

static bool isSnicDevice(int netDev) {
  ncclNetProperties_t props;
  if (ncclNetGetProperties(netDev, &props) != ncclSuccess) return false;
  pthread_once(&snicPrefixOnce, initSnicPrefixes);
  if (nSnicPrefixes == 0) return false;
  return matchIfList(props.name, -1, snicPrefixes, nSnicPrefixes, false);
}

static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(int)) return ncclInternalError;
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);
  resources->tpRemoteProxyRank = *(int*)reqBuff;

  // Finish connection establishment from remote peer
  if (resources->shared) {
    // Shared buffers
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->tpLocalRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers+resources->tpLocalRank, 1));
    }
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->recv.proxyAppend+resources->channelId;

    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      if (progressState->netComms[resources->netDev] == NULL) {
        NCCLCHECK(ncclCalloc(progressState->netComms+resources->netDev, proxyState->tpnRanks));
      }
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev]+resources->tpRemoteProxyRank;
      // reuse handle to for netdev/remote rank to avoid duplicate connections
      if (comms->activeAccept[resources->channelId] == 0)
        comms->activeAccept[resources->channelId] = (resources->tpLocalRank + 1);
      //try connecting while comm is null
      if (comms->recvComm[resources->channelId] == NULL &&
          comms->activeAccept[resources->channelId] == (resources->tpLocalRank + 1)) {
        NCCLCHECK(ncclNetAccept(resources->netListenComm, comms->recvComm + resources->channelId));
      }
      resources->netRecvComm = comms->recvComm[resources->channelId];
      if (comms->recvComm[resources->channelId]) comms->recvRefCount[resources->channelId]++;
    } else {
      NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
    }
  } else {
    // Connect to remote peer
    NCCLCHECK(ncclNetAccept(resources->netListenComm, &resources->netRecvComm));
    connection->proxyAppendPtr = &connection->proxyAppend;
  }
  if (resources->netRecvComm == NULL) {
    *done = 0;
    return ncclInProgress;
  }
  *done = 1;
  NCCLCHECK(ncclNetCloseListen(resources->netListenComm));

  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess = connection->sameProcess;
  map->u2mmHost = proxyState->u2mmHost;
  map->u2mmDev = proxyState->u2mmDev;
  if (map->sameProcess == 0) return ncclInternalError; // We don't support remote proxy for recv
  map->shared = resources->shared;

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      resources->buffSizes[p] = proxyState->buffSizes[p];
      // skip zero bytes regMr
      if (resources->buffSizes[p] == 0) continue;

      // for signal/bulk shared buffer
      if (p > NCCL_PROTO_SIMPLE) {
        map->offsets.buffs[p] = map->offsets.buffs[NCCL_PROTO_SIMPLE];
        continue;
      }

      NCCL_NET_MAP_ADD_POINTER(map, 0, resources->useGdr, resources->buffSizes[p], buffs[p]);
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(sharedBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 1, 1, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, NULL));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;
    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  if (proxyState->allocP2pNetLLBuffers) {
    NCCL_NET_MAP_ADD_POINTER(map, 0, 0, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
    resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
  }

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    if (resources->shared == 0) {
      if (ncclCuMemEnable()) {
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0,
          &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc, (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr,
          proxyState->pDevMemUsed, map->u2mmDev, -1, proxyState->memManager, ncclMemPersist));
      } else {
        NCCLCHECK(ncclHggcCallocU2mm(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr,
                                      map->mems[NCCL_NET_MAP_DEVMEM].size,
                                      proxyState->pDevMemUsed,
                                      map->u2mmDev, proxyState->memManager));
      }
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  NCCLCHECK(ncclHggcHostCallocU2mm(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr,
                                    map->mems[NCCL_NET_MAP_HOSTMEM].size,
                                    proxyState->pPinnedMemUsed,
                                    map->u2mmHost,
                                    resources->netDev, proxyState->ncclNet));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  if (map->sameProcess) {
    uint64_t *cpuPtr = NULL;
    uint64_t *gpuPtr = NULL;
    bool useCpuFlush = pcclParamGdrCpuFlush() || (pcclParamSnicCpuFlush() && isSnicDevice(resources->netDev));
    // go through nccl gdr copy path
    if (ncclGdrCopy) {
      NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 2, &resources->gdrDesc, proxyState->pDevMemUsed, proxyState->memManager));
    } else {
      // go through pccl mem lock path
      if (ncclParamGdrCopySyncEnable() || pcclParamGdrUseDevMemForRxTail() || ncclParamGdrCopyFlushEnable() || useCpuFlush) {
        // not used for regMr, do not need u2mm
        NCCLCHECK(ncclHggcCalloc(&gpuPtr, 2, proxyState->pDevMemUsed, proxyState->memManager));
        resources->gdrDesc = gpuPtr; // re-use this pointer for free
        hggcError_t res = hggcEitInterface::getInstance().wrapHggcMemLock((HGdeviceptr)gpuPtr, (void**)&cpuPtr);
        if (hggcSuccess != res) {
          cpuPtr = NULL;
          ERROR("wrapHggcMemLock fail %d", res);
        }
      }
    }
    // assign gdc sync and flush pointer
    if ((ncclParamGdrCopySyncEnable() || pcclParamGdrUseDevMemForRxTail()) && cpuPtr) {
      resources->gdcSync = cpuPtr;
      struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
      gdcMem->cpuPtr = (char*)cpuPtr;
      gdcMem->gpuPtr = (char*)gpuPtr;
      gdcMem->size = sizeof(uint64_t);
    }
    if ((ncclParamGdrCopyFlushEnable() || useCpuFlush) && cpuPtr) {
      resources->gdcFlush = cpuPtr + 1;
    }
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (p > NCCL_PROTO_SIMPLE) {
      resources->mhandles[p] = resources->mhandles[NCCL_PROTO_SIMPLE];
      continue;
    }
    if (resources->buffers[p]) {
      NCCLCHECK(ncclNetRegMr(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_HGGC : NCCL_PTR_HOST, &resources->mhandles[p]));
    }
  }

  // NCCLCHECK(netDumpMap(map));
  if (respSize != sizeof(struct connectMap)) return ncclInternalError;
  memcpy(respBuff, map, sizeof(struct connectMap));

  return ncclSuccess;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct sendResources* resources = (struct sendResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect or unused p2p conn for coll
    NCCLCHECK(sharedBuffersDestroy(proxyState, connection->tpLocalRank, 0, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (p > NCCL_PROTO_SIMPLE) break;
      if (resources->buffers[p]) {
        NCCLCHECK(ncclNetDeregMr(resources->netSendComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    if (resources->map.sameProcess) {
      NCCLCHECK(ncclHggcHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, resources->map.u2mmHost));
    } else {
      if (resources->map.u2mmHost) {
        NCCLCHECK(ncclU2mmShmDestroy(resources->u2mmMkeyPtr, mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
      } else {
        NCCLCHECK(ncclShmClose(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, NULL, mems[NCCL_NET_MAP_HOSTMEM].size));
      }
    }
    NCCLCHECK(ncclHggcFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr, resources->map.u2mmDev, proxyState->memManager));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // hgMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc, proxyState->memManager));
    if (resources->shared) {
      NCCLCHECK(sharedBuffersDestroy(proxyState, resources->tpLocalRank, 0, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev]+resources->tpRemoteRank;
        comms->sendRefCount[resources->channelId]--;
        if (comms->sendRefCount[resources->channelId] == 0) NCCLCHECK(ncclNetCloseSend(comms->sendComm[resources->channelId]));
      } else {
        NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
      }
    } else {
      NCCLCHECK(ncclNetCloseSend(resources->netSendComm));
    }
  }
  NCCLCHECK(ncclFree(resources));

  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct recvResources* resources = (struct recvResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect or unused p2p conn for coll
    NCCLCHECK(sharedBuffersDestroy(proxyState, connection->tpLocalRank, 1, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (p > NCCL_PROTO_SIMPLE) break;
      if (resources->buffers[p]) {
        NCCLCHECK(ncclNetDeregMr(resources->netRecvComm, resources->mhandles[p]));
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclHggcHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, resources->map.u2mmHost));
    NCCLCHECK(ncclHggcFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr, resources->map.u2mmDev, proxyState->memManager));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // hgMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (ncclGdrCopy) {
      if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc, proxyState->memManager));
    } else {
    if (resources->gdcFlush) hggcEitInterface::getInstance().wrapHggcMemUnlock((HGdeviceptr)resources->gdrDesc);
    if (resources->gdrDesc) {
      NCCLCHECK(ncclHggcFree(resources->gdrDesc, MEM_NORMAL, proxyState->memManager));
    }
    }
    if (resources->shared) {
      NCCLCHECK(sharedBuffersDestroy(proxyState, resources->tpLocalRank, 1, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev]+resources->tpRemoteProxyRank;
        comms->recvRefCount[resources->channelId]--;
        if (comms->recvRefCount[resources->channelId] == 0) NCCLCHECK(ncclNetCloseRecv(comms->recvComm[resources->channelId]));
      } else {
        NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
      }
    } else {
      NCCLCHECK(ncclNetCloseRecv(resources->netRecvComm));
    }
  }

  NCCLCHECK(ncclFree(resources));
  return ncclSuccess;
}

static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->transmitted = sub->done = 0;
      ncclProfilerRecordProxyOpEventState(s, args, ncclProfilerProxyOpInProgress_v4);
    }
#ifdef NET_BYPASS
    args->bypass = pcclParamNetByPass();
#endif
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS / args->nsubs);
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      int postedStepId = sub->posted;
      int transmittedStepId = sub->transmitted;
      int doneStepId = sub->done;
      if (sub->done == sub->nsteps) continue;
      struct sendResources* resources = (struct sendResources*) (sub->connection->transportResources);
      void* mhandle = resources->mhandles[p];
      int stepSize = resources->buffSizes[p] / NCCL_STEPS;
      char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
      // Post buffers to the GPU
      if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
        ncclProfilerStartSendProxyStepEvent(s, args, postedStepId);
        pcclAsysProxyProfilingRecord(args, s, postedStepId, ncclProxyProfileBegin);
        int buffSlot = (sub->base + sub->posted) % NCCL_STEPS;
        if (resources->shared) {
          int sharedBuffSlot = sub->posted % maxDepth;
          int offset;
          NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot * args->nsubs + s, &offset, NULL));
          resources->recvMem->offsFifo[buffSlot] = offset;
          __sync_synchronize();
          volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
          sub->posted += args->sliceSteps;
          *sendHead = sub->base + sub->posted - NCCL_STEPS;
          if (resources->gdcSync) wc_store_fence(); // Flush out WC write
        } else {
          sub->posted += args->sliceSteps;
        }
        ncclProfilerRecordProxyStepEventState(s, args, postedStepId, ncclProfilerProxyStepSendGPUWait);
        pcclAsysProxyProfilingRecord(args, s, sub->posted - args->sliceSteps, ncclProxyProfileSendGPUWait);
        args->idle = 0;
        continue;
      }
      // Check whether we received data from the GPU and send it to the network
      if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base + sub->transmitted) % NCCL_STEPS;
        volatile int* sizesFifo = resources->recvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        uint64_t tail = sub->base + sub->transmitted;
        if (sizesFifo[buffSlot] != -1 && (*recvTail > tail || p == NCCL_PROTO_LL)) {
          // We have something to receive, let's check if it's completely ready.
          int size = sizesFifo[buffSlot];
          bool shared = (p == NCCL_PROTO_SIMPLE || p == NCCL_PROTO_SIMPLE_BULK) && resources->shared;
          char* buff = shared ? localBuff + resources->recvMem->offsFifo[buffSlot] : localBuff + buffSlot * stepSize;
          int ready = 1;
#if !defined(STORE_BYPASS)
          if (p == NCCL_PROTO_LL128) {
            ready = resources->useGdr;
            if (!ready) {
              // When data is in sysmem, we need to wait until all flags are correct since the GPU only
              // called threadfence()
              uint64_t flag = sub->base + sub->transmitted + 1;
              int nFifoLines = DIVUP(sizesFifo[buffSlot], sizeof(uint64_t) * NCCL_LL128_LINEELEMS);
              volatile uint64_t* lines = (volatile uint64_t*)buff;
              ready = 1;
              for (int i = 0; i < nFifoLines; i++) {
                if (lines[i * NCCL_LL128_LINEELEMS + NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
              }
            }
          } else if (p == NCCL_PROTO_LL) {
            uint32_t flag = NCCL_LL_FLAG(sub->base + sub->transmitted + 1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
            for (int i = 0; i < nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
          }
#endif
          if (ready) {
            ncclProfilerRecordProxyStepEventState(s, args, transmittedStepId, ncclProfilerProxyStepSendPeerWait_v4);
            // Data is ready, try to send.
#ifdef NET_BYPASS
            if (args->bypass)
              sub->requests[buffSlot] = &sub;
            else
              NCCLCHECK(ncclNetIsend(resources->netSendComm, buff, size, resources->tpRank, mhandle, sub->requests + buffSlot));
#else
            NCCLCHECK(ncclNetIsend(resources->netSendComm, buff, size, resources->tpRank, mhandle, sub->requests + buffSlot));
#endif
            if (sub->requests[buffSlot] != NULL) {
              TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] Isend posted, req %p, buff %p, size %d, proto %d, myRank %d, channelId %d",
                sub->transmitted, buffSlot, sub->nsteps, sub->requests[buffSlot], buff, size, p, proxyState->tpRank, sub->channelId);
              sizesFifo[buffSlot] = -1;
              pcclAsysProxyProfilingRecord(args, s, sub->transmitted, ncclProxyProfileSendWait);
              __sync_synchronize();
              sub->transmitted += args->sliceSteps;
              ncclProfilerRecordProxyStepEventState(s, args, transmittedStepId, ncclProfilerProxyStepSendWait);
              args->idle = 0;
              continue;
            }
          }
        }
      }
      // Check whether the network has completed some send operations.
      if (sub->done < sub->transmitted) {
        int done;
        int buffSlot = (sub->base + sub->done) % NCCL_STEPS;
#ifdef NET_BYPASS
        if (args->bypass)
          done = 1;
        else
          NCCLCHECK(ncclNetTest(sub->requests[buffSlot], &done, NULL));
#else
        NCCLCHECK(ncclNetTest(sub->requests[buffSlot], &done, NULL));
#endif
        if (done) {
          TRACE(NCCL_NET, "sendProxy [%ld/%d/%d] request %p done", sub->done, buffSlot, sub->nsteps, sub->requests[buffSlot]);
          pcclAsysProxyProfilingRecord(args, s, sub->done, ncclProxyProfileEnd);
          sub->done += args->sliceSteps;
          ncclProfilerStopProxyStepEvent(s, args, doneStepId);
          if (resources->shared == 0) {
            volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
            *sendHead = sub->base + sub->done;
            if (resources->gdcSync) wc_store_fence(); // Flush out WC write
          }
          args->idle = 0;
          if (sub->done == sub->nsteps) {
            args->done++;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      for (int s = 0; s < args->nsubs; s++) {
          ncclProfilerStopProxyOpEvent(s, args);
      }
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    // Initialize subs and group them by same recvComm.
    void* recvComm;
    int groupSize = 0;
    int maxRecvs = 1;
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      if (groupSize == maxRecvs) {
        groupSize = 0;
      } else if (s > 0) { // Find next sub with the same recvComm
        int next;
        for (next = s; next < args->nsubs; next++) {
          struct recvResources* nextRes = (struct recvResources*) (args->subs[next].connection->transportResources);
          if (nextRes->netRecvComm == recvComm) break;
        }
        if (next == args->nsubs) { // Not found
          groupSize = 0;
        } else if (s != next) { // We found a sub later with the same recvComm ; swap subs
          struct ncclProxySubArgs temp;
          memcpy(&temp, sub, sizeof(struct ncclProxySubArgs));
          memcpy(sub, args->subs + next, sizeof(struct ncclProxySubArgs));
          memcpy(args->subs + next, &temp, sizeof(struct ncclProxySubArgs));
        }
      }
      groupSize++;
      struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
      maxRecvs = resources->maxRecvs;
      recvComm = resources->netRecvComm;
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
      for (int i = 0; i < groupSize; i++) sub[-i].groupSize = groupSize;
      ncclProfilerRecordProxyOpEventState(s, args, ncclProfilerProxyOpInProgress_v4);
    }
    args->state = ncclProxyOpProgress;
#ifdef NET_BYPASS
    args->bypass = pcclParamNetByPass();
#endif
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS / args->nsubs);
    for (int s = 0; s < args->nsubs; s += args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs + s;
      int subCount = 0;
      void* ptrs[NCCL_PROXY_MAX_SUBS];
      int sizes[NCCL_PROXY_MAX_SUBS];
      int tags[NCCL_PROXY_MAX_SUBS];
      void* mhandles[NCCL_PROXY_MAX_SUBS];

      for (int i = 0; i < subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        int postedStepId = sub->posted;
        if (sub->posted < sub->nsteps) {
          if (sub->posted >= sub->done + maxDepth) { subCount = 0; break; }
          ncclProfilerStartRecvProxyStepEvent(s + i, args, postedStepId);
          pcclAsysProxyProfilingRecord(args, s + i, postedStepId, ncclProxyProfileBegin);
          struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
          int stepSize = resources->buffSizes[p] / NCCL_STEPS;
          char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
          int buffSlot = (sub->base + sub->posted) % NCCL_STEPS;
          volatile int* offsFifo = (volatile int*)resources->recvMem->offsFifo;
          if ((p == NCCL_PROTO_SIMPLE || p == NCCL_PROTO_SIMPLE_BULK) && resources->shared) {
            int sharedBuffSlot = sub->posted % maxDepth;
            int offset;
            NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot * args->nsubs + s + i, &offset, sizes + subCount));
            offsFifo[buffSlot] = offset;
            ptrs[subCount] = localBuff + offset;
          } else {
            ptrs[subCount] = localBuff + buffSlot * stepSize;
            sizes[subCount] = stepSize * args->sliceSteps;
          }
          if (sub->nbytes < sizes[subCount]) sizes[subCount] = sub->nbytes;
          tags[subCount] = resources->tpRemoteRank;
          mhandles[subCount] = resources->mhandles[p];
          subCount++;
        }
      }
      if (subCount) {
        uint64_t step = subGroup->posted;
        struct recvResources* resources = (struct recvResources*) (subGroup->connection->transportResources);
        void** requestPtr = subGroup->requests + (step%NCCL_STEPS);

#ifdef NET_BYPASS
        if (args->bypass)
          requestPtr = (void**)&subGroup;
        else
          NCCLCHECK(ncclNetIrecv(resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, requestPtr));
#else
        NCCLCHECK(ncclNetIrecv(resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, requestPtr));
#endif
        if (*requestPtr) {
          for (int i = 0; i < subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;
            int postedStepId = sub->posted;
            TRACE(NCCL_NET, "recvProxy [%ld/%ld/%d] Irecv posted, buff %p, size %ld, myRank %d, channelId %d",
              sub->posted, (sub->base + sub->posted) % NCCL_STEPS, sub->nsteps, ptrs[i], sizes[i], proxyState->tpRank, sub->channelId);
            sub->posted += args->sliceSteps;
            ncclProfilerRecordProxyStepEventState(s + i, args, postedStepId, ncclProfilerProxyStepRecvWait);
            pcclAsysProxyProfilingRecord(args, s + i, postedStepId, ncclProxyProfileRecvWait);
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s = 0; s < args->nsubs; s += args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs + s;
      if (subGroup->posted > subGroup->received) {
        uint64_t step = subGroup->received;
        int done;
        void* ptrs[NCCL_PROXY_MAX_SUBS];
        int sizes[NCCL_PROXY_MAX_SUBS];
        void* mhandles[NCCL_PROXY_MAX_SUBS];
        for (int i = 0; i < NCCL_PROXY_MAX_SUBS; i++) sizes[i] = 0;
#ifdef NET_BYPASS
        if (args->bypass) {
          done = 1;
        } else {
          NCCLCHECK(ncclNetTest(subGroup->requests[step % NCCL_STEPS], &done, sizes));
        }
#else
        NCCLCHECK(ncclNetTest(subGroup->requests[step % NCCL_STEPS], &done, sizes));
#endif
        if (done) {
          int needFlush = 0;
          int totalSize = 0;
          for (int i = 0; i < NCCL_PROXY_MAX_SUBS; i++) totalSize += sizes[i];
          for (int i = 0; i < subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;
            int receivedStepId = sub->received;
            int buffSlot = (sub->base + sub->received) % NCCL_STEPS;
            struct recvResources* resources = (struct recvResources*) (subGroup->connection->transportResources);
            volatile int* sizesFifo = resources->recvMem->sizesFifo;
            sizesFifo[buffSlot] = -1;
            sub->received += args->sliceSteps;
            ncclProfilerRecordProxyStepEventState(s + i, args, receivedStepId, ncclProfilerProxyStepRecvFlushWait);
            pcclAsysProxyProfilingRecord(args, s + i, receivedStepId, ncclProxyProfileRecvFlushWait);
            if (step < sub->nsteps) {
              struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
              needFlush |= resources->useGdr;
            }
          }
          subGroup->requests[step % NCCL_STEPS] = NULL;
          if (totalSize > 0 && (p == NCCL_PROTO_SIMPLE || p == NCCL_PROTO_SIMPLE_BULK) && needFlush) {
            // GDRCOPY support
            struct recvResources* resources = (struct recvResources*) (subGroup->connection->transportResources);
            if (resources->gdcFlush) {
#if defined (__x86_64__)
              // Only take the async path once the flush thread's dedicated
              // scratch is ready; otherwise fall back to the synchronous read
              // so we never signal completion without a real PCI-E flush.
              if (pcclParamCpuFlushAsync()
                  && __atomic_load_n(&proxyState->progressState.flushScratch, __ATOMIC_ACQUIRE)) {
                // Async: take a ticket (bump flushReqSeq) to signal demand; the
                // per-proxyState flush thread performs the real PCI-E read and
                // coalesces across tickets. The read that advances flushDoneSeq
                // to >= this ticket observed flushReqSeq >= ticket, so it was
                // issued after this point (after the step's data arrived).
                uint32_t slot = step % NCCL_STEPS;
                subGroup->asyncFlushTicket[slot] = __atomic_add_fetch(&proxyState->progressState.flushReqSeq, 1, __ATOMIC_ACQ_REL);
                subGroup->requests[slot] = NCCL_ASYNC_FLUSH_PENDING;
              } else {
                // Force a PCI-E read from GPU memory
                asm volatile ("mov (%0), %%eax" :: "l"(resources->gdcFlush) : "%eax");
              }
#else
              ERROR("NET: GDR CPU Flush only supported on x86_64");
              return ncclInternalError;
#endif
            } else {
              int subCount = 0;
              for (int i = 0; i < subGroup->groupSize; i++) {
                struct ncclProxySubArgs* sub = subGroup + i;
                if (step < sub->nsteps) {
                  struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
                  int stepSize = resources->buffSizes[p] / NCCL_STEPS;
                  char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
                  int buffSlot = (sub->base + sub->posted - args->sliceSteps) % NCCL_STEPS;
                  if (resources->shared) {
                    ptrs[subCount] = localBuff + resources->recvMem->offsFifo[buffSlot];
                  } else {
                    ptrs[subCount] = localBuff + buffSlot * stepSize;
                  }
                  mhandles[subCount] = resources->mhandles[p];
                  subCount++;
                }
              }
              struct recvResources* resources = (struct recvResources*) (subGroup->connection->transportResources);
              NCCLCHECK(ncclNetIflush(resources->netRecvComm, subCount, ptrs, sizes, mhandles, subGroup->requests + (step % NCCL_STEPS)));
            }
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s = 0; s < args->nsubs; s += args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs + s;
      if (subGroup->received > subGroup->transmitted) {
        uint64_t step = subGroup->transmitted;
        int done = 1;
        void* request = subGroup->requests[step % NCCL_STEPS];
        if (request == NCCL_ASYNC_FLUSH_PENDING) {
          // Async CPU flush: done once the flush thread's completed sequence has
          // caught up to this ticket.
          done = (__atomic_load_n(&proxyState->progressState.flushDoneSeq, __ATOMIC_ACQUIRE)
                  >= subGroup->asyncFlushTicket[step % NCCL_STEPS]) ? 1 : 0;
        } else if (request) {
          NCCLCHECK(ncclNetTest(request, &done, NULL));
        }
#ifdef NET_BYPASS
        if (args->bypass) done = 1;
#endif
        if (done) {
          for (int i = 0; i < subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;
            int transmittedStepId = sub->transmitted;
            sub->transmitted += args->sliceSteps;
            ncclProfilerRecordProxyStepEventState(s + i, args, transmittedStepId, ncclProfilerProxyStepRecvGPUWait);
            pcclAsysProxyProfilingRecord(args, s + i, transmittedStepId, ncclProxyProfileRecvGPUWait);
            if (step < sub->nsteps) {
              __sync_synchronize();
              struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
              volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
              *recvTail = sub->base + sub->transmitted;
              if (resources->gdcSync) wc_store_fence(); // Flush out WC write
            }
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s = 0; s < args->nsubs; s += args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs + s;
      for (int i = 0; i < subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        if (sub->done == sub->nsteps) continue;
        if (sub->transmitted > sub->done) {
          struct recvResources* resources = (struct recvResources*) (sub->connection->transportResources);
          volatile uint64_t* sendHead = &resources->sendMem->head;
          uint64_t done = *sendHead;
          while (done > sub->base + sub->done &&
              // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
              sub->transmitted > sub->done) {
            int doneStepId = sub->done;
            sub->done += args->sliceSteps;
            ncclProfilerStopProxyStepEvent(s + i, args, doneStepId);
            pcclAsysProxyProfilingRecord(args, s + i, doneStepId, ncclProxyProfileEnd);
            args->idle = 0;
            if (sub->done == sub->nsteps) {
              args->done++;
              break;
            }
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
      for (int s = 0; s < args->nsubs; s++) {
        ncclProfilerStopProxyOpEvent(s, args);
      }
    }
  }
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  canConnect,
  { sendSetup, sendConnect, sendFree, proxySharedInit, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress, NULL, NULL},
  { recvSetup, recvConnect, recvFree, proxySharedInit, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress, NULL, NULL}
};
