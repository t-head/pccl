/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#include "register_inline.h"
#include "shm.h"
#include "transport.h"
#include "cudawrap.h"

enum p2pType { P2P_DIRECT, P2P_INTERMEDIATE, P2P_IPC, P2P_HGMEM };

struct ncclP2pBuff {
  void* directPtr;
  size_t size;
  ncclIpcDesc ipcDesc;
};

struct ncclP2pRequest {
  size_t size;
  int refcount;
  int peerRank;
};

struct p2pConnectInfo {
  int rank;
  int read;
  // CE and device-buffer descriptors are exclusive and must fit CONNECT_SIZE.
  union {
    struct ncclP2pBuff p2pBuff;
    ncclShmIpcDesc_t desc;
  };
  int minPathNum;
};
static_assert(sizeof(struct p2pConnectInfo) <= CONNECT_SIZE, "p2pConnectInfo is too large");

struct p2pIpcExpInfo {
  ncclIpcDesc ipcDesc;
  bool legacyIpcCap;
  int impFd;
  size_t phySize;
  size_t virSize;
  uintptr_t offset;
};

struct p2pShm {
  struct ncclSendMem sendMem;
  struct ncclRecvMem recvMem;
};
struct p2pShmProxyInfo {
  // Shared memory between proxy and receiving GPU
  struct p2pShm* shm;
  struct p2pShm* devShm;
  ncclShmIpcDesc_t desc;

  // Intermediate step for sender
  struct ncclRecvMem* ceRecvMem;
  char* ceDevBuff;

  // Receiver buffer
  char* recvFifo;

  // Used by progress only
  uint64_t step;
  hggcStream_t stream;
  hggcEvent_t events[NCCL_STEPS];
};

static_assert(sizeof(p2pConnectInfo) <= CONNECT_SIZE, "P2P Connect info is too large");

struct p2pResources {
  enum p2pType type;
  union {
    struct ncclSendMem* sendDevMem;
    struct ncclRecvMem* recvDevMem;
  };
  void* sendMemIpc;
  int sendMemSameProc;
  void* recvMemIpc;
  int recvMemSameProc;

  struct ncclMemManager* ipcMemManager;
  // CE memcpy support
  struct p2pShmProxyInfo proxyInfo;
  struct p2pShm* shm;
  struct p2pShm* devShm;
  ncclShmIpcDesc_t desc;
};

// hgMem API support
struct p2pHgMemProxyInfo {
  struct ncclP2pBuff p2pBuff;
};

#include <sys/types.h>

/* Convert a PCI busId string into a local hggcDev device index (cf. HGGC_VISIBLE_DEVICES) */
static int busIdToHggcDev(int64_t busId) {
  int ndev;
  if (hggcGetDeviceCount(&ndev) != hggcSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (hggcDeviceGetPCIBusId(devBusIdStr, HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != hggcSuccess)
      return -1;
    int64_t devBusId;
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible hggc devices
  return -1;
}

static void initCeOperation();

/* Determine if two peers can communicate through p2p */
ncclResult_t p2pCanConnect(int* ret, ncclComm_t comm, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  initCeOperation();

  // Check topology / p2p level.
  int intermediateRank;
  NCCLCHECK(ncclTopoCheckP2p(comm, topo, info1->rank, info2->rank, ret, NULL, &intermediateRank));
  if (*ret == 0) return ncclSuccess;

  // Check if NET would work better
  int useNet = 0;
  NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, &useNet));
  if (useNet) {
    *ret = 0;
    return ncclSuccess;
  }

  if (info1->hostHash != comm->peerInfo[comm->rank].hostHash ||
      info1->hostHash != info2->hostHash) {
    // If either peer is non-local then we are done.
    return ncclSuccess;
  }

  // Convert the peer's busId into a local hggcDev index (cf. hggc_VISIBLE_DEVICES)
  int hggcDev1 = busIdToHggcDev(info1->busId);
  int hggcDev2 = busIdToHggcDev(info2->busId);
  if (hggcDev1 == -1 || hggcDev2 == -1) {
    // hggc 10.1 and later (or PPU) can use P2P with invisible devices.
    return ncclSuccess;
  }

  // Check that hggc can do P2P
  int p2p;
  if (hggcDeviceCanAccessPeer(&p2p, hggcDev1, hggcDev2) != hggcSuccess) {
    INFO(NCCL_INIT|NCCL_P2P,"peer query failed between dev %d(=%lx) and dev %d(=%lx)",
         hggcDev1, info1->busId, hggcDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }

  // This will always fail when using NCCL_CUMEM_ENABLE=1
  if (p2p != 0 && !ncclCuMemEnable()) {
    // Cached result of the legacyIPC detection
    static int legacyIPC = -1;
    if (legacyIPC >= 0) {
      *ret = legacyIPC;
      return ncclSuccess;
    }
    // Check that legacy IPC support is available (WSL WAR)
    char *dummy;
    hggcIpcMemHandle_t ipc;
    NCCLCHECK(ncclHggcCalloc(&dummy, HGGC_IPC_MIN, NULL));
    if (hggcIpcGetMemHandle(&ipc, dummy) != hggcSuccess) {
      INFO(NCCL_INIT|NCCL_P2P,"Legacy IPC not supported");
      *ret = 0;
    }
    NCCLCHECK(ncclHggcFree(dummy));
    legacyIPC = *ret;
    return ncclSuccess;
  }

  if (p2p == 0) {
    INFO(NCCL_INIT|NCCL_P2P,"Could not enable P2P between dev %d(=%lx) and dev %d(=%lx)",
         hggcDev1, info1->busId, hggcDev2, info2->busId);
    *ret = 0;
    return ncclSuccess;
  }
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

// hgMem API support
ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int refcount, ncclIpcDesc *ipcDesc, void **ptr, uint64_t* pDevMemUsed, int memMode, int peerRank, struct ncclMemManager* manager, ncclMemType_t memType) {
  if (ncclCuMemEnable() && (memMode == MEM_NORMAL)) {
    // hgMem API support
    HGmemAllocationHandleType type = ncclHgMemHandleType;
    HGmemGenericAllocationHandle handle;

    struct ncclMemManager* trackManager = manager;
    NCCLCHECK(ncclHgMemAlloc(ptr, &handle, type, size, pDevMemUsed, nullptr, memMode, trackManager, memType));
    NCCLCHECK(ncclHggcMemset(*ptr, 0, size));
    if (type == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // Return the native hgMem handle for later Export/Import via UDS
      memcpy(&ipcDesc->hgDesc.data, &handle, sizeof(handle));
    } else {
      HGCHECK(hgMemExportToShareableHandle(&ipcDesc->hgDesc, handle, type, 0));
    }
    if (refcount) {
      memcpy(&ipcDesc->memHandle, &handle, sizeof(handle));
      for (int r = 0; r < refcount; ++r) HGCHECK(hgMemRetainAllocationHandle(&handle, *ptr));
    }
    if (trackManager != nullptr && peerRank >= 0 && memType != ncclMemPersist) {
      NCCLCHECK(ncclDynMemMarkExportToPeer(trackManager, *ptr, peerRank));
    }
  } else {
    // Allocate a HGGC buffer and generate an IPC handle for it
    NCCLCHECK(ncclHggcCallocU2mm((char **)ptr, size, pDevMemUsed, memMode, manager));
    hggcError_t res = hggcIpcGetMemHandle(&ipcDesc->devIpc, *ptr);
    if (res != hggcSuccess) {
      WARN("hggcIpcGetMemHandle failed : %s", hggcGetErrorString(res));
      ncclHggcFree(*ptr, memMode, manager);
      HGGCCHECK(res);
    }
  }
  INFO(NCCL_P2P|NCCL_ALLOC, "Allocated shareable buffer %p size %zi ipcDesc %p", *ptr, size, ipcDesc);

  return ncclSuccess;
}

ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc) {
  return ncclSuccess;
}

ncclResult_t ncclP2pImportShareableBuffer(ncclComm_t comm, int peer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr, int memMode, void* ownerPtr, ncclMemType_t memType) {
  if (ncclCuMemEnable() && (memMode == MEM_NORMAL)) {
    // hgMem API support
    HGdeviceptr dptr = 0;
    HGmemAllocationHandleType type = ncclHgMemHandleType;
    HGmemGenericAllocationHandle handle;
    ncclHgDesc *hgDesc = &ipcDesc->hgDesc;
    HGmemAllocationProp prop = {};
    size_t granularityVir = 0;
    size_t granularityPhy = 0;
    size_t sizeAlignedVir = 0;
    size_t sizeAlignedPhy = 0;

    prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    prop.requestedHandleTypes = type;
    prop.location.id = comm->hggcDev;
    HGCHECK(hgMemGetAllocationGranularity(&granularityVir, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));
    HGCHECK(hggcEitInterface::getInstance().wrapHgMemGetAllocationGranularityEx(&granularityPhy, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));
    ALIGN_SIZE_WITH_RTN(size, granularityPhy, sizeAlignedPhy);
    ALIGN_SIZE_WITH_RTN(size, granularityVir, sizeAlignedVir);

    // Import and map the remote memory descriptor to the local GPU
    if (type == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      // UDS fd support
      int fd = -1;
      // Send hgMem handle to remote for conversion to an fd
      NCCLCHECK(ncclProxyClientGetFdBlocking(comm, peer, &hgDesc->data, &fd));
      INFO(NCCL_P2P, "UDS converted handle 0x%lx to fd %d on remote peer %d", *(uint64_t*)&hgDesc->data, fd, peer);
      HGCHECK(hgMemImportFromShareableHandle(&handle, (void *)(uintptr_t)fd, type));
      SYSCHECK(close(fd), "close");
    } else {
      HGCHECK(hgMemImportFromShareableHandle(&handle, (void*)hgDesc, type));
    }
    HGCHECK(hgMemAddressReserve(&dptr, sizeAlignedVir, /* alignment */ 0, /* addr */ 0, /* flags */ 0));
    HGCHECK(hgMemMap(dptr, sizeAlignedPhy, /* offset */ 0, handle, /* flags */ 0));
    VERBOSE(NCCL_P2P, "Imported shareable buffer reqSize %zi sizeAlignedPhy %zi sizeAlignedVir %zi handle 0x%llx dptr %p", size, sizeAlignedPhy, sizeAlignedVir, handle, (void*)dptr);

    // Allow access by the local GPU
    HGmemAccessDesc accessDesc = {};
    accessDesc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = comm->hggcDev;
    accessDesc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
    HGCHECK(hgMemSetAccess(dptr, sizeAlignedPhy, &accessDesc, 1));
    VERBOSE(NCCL_P2P, "Set Access for %p size %zi sizeAlignedPhy %zi sizeAlignedVir %zi on dev %d", (void*)dptr, size, sizeAlignedPhy, sizeAlignedVir, accessDesc.location.id);

    *devMemPtr = (void *)dptr;

    // Track imported buffer
    if (comm->memManager != nullptr && (memType == ncclMemPersist || ownerPtr != nullptr)) {
      int ownerDev = (comm->peerInfo && comm->peerInfoValid && peer >= 0 && peer < comm->nRanks) ? comm->peerInfo[peer].hggcDev : -1;
      // import sizeAlignedPhy
      ncclResult_t tracked = ncclMemTrackImportFromPeer(comm->memManager, *devMemPtr, sizeAlignedPhy, handle, type, memType, peer, ownerDev, ownerPtr, sizeAlignedVir);
      if (tracked != ncclSuccess) {
        HGCHECKIGNORE(hgMemUnmap(dptr, sizeAlignedPhy));
        HGCHECKIGNORE(hgMemRelease(handle));
        HGCHECKIGNORE(hgMemAddressFree(dptr, sizeAlignedVir));
        *devMemPtr = nullptr;
        return tracked;
      }
    }

    INFO(NCCL_P2P, "Imported shareable buffer device %d size %zi sizeAlignedPhy %zi sizeAlignedVir %zi ptr %p via HGMEM", comm->hggcDev, size, sizeAlignedPhy, sizeAlignedVir, *devMemPtr);
  } else {
    // Legacy HGGC IPC
    HGGCCHECK(hggcIpcOpenMemHandle(devMemPtr, ipcDesc->devIpc, hggcIpcMemLazyEnablePeerAccess));

    INFO(NCCL_P2P, "Imported shareable buffer device %d size %zi ptr %p via HGGC IPC", comm->hggcDev, size, *devMemPtr);
  }

  return ncclSuccess;
}

// Setting this to non zero causes P2P to use Reads rather than Writes
NCCL_PARAM(P2pReadEnable, "P2P_READ_ENABLE", -2);
NCCL_PARAM(P2pDirectDisable, "P2P_DIRECT_DISABLE", 0);

static ncclResult_t p2pGetInfo(ncclComm_t comm, struct ncclTopoSystem* topo, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2, int* read, int* intermediateRank) {
  int p2p;
  // Queries the topology to see if the GPUs are Ampere and
  // connected via NVLink, if so we enable P2P Read by default
  NCCLCHECK(ncclTopoCheckP2p(comm, topo, info1->rank, info2->rank, &p2p, read, intermediateRank));

  int readEnable = ncclParamP2pReadEnable();
  if (readEnable != -2) *read = readEnable;
  return ncclSuccess;
}

static ncclResult_t p2pMap(ncclComm_t comm, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct p2pConnectInfo* p2pInfo, void** devMem, void** ipcPtr) {
  struct ncclTopoSystem* topo = comm->topo;
  struct ncclP2pBuff* p2pBuff = &p2pInfo->p2pBuff;
  if (PCCL_SAME_PID(myInfo, peerInfo)) {
    if (peerInfo->hggcDev != myInfo->hggcDev) {
      // Enable P2P access
      hggcStreamCaptureMode mode = hggcStreamCaptureModeRelaxed;
      HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
      hggcError_t err = hggcDeviceEnablePeerAccess(peerInfo->hggcDev, 0);
      HGGCCHECK(hggcThreadExchangeStreamCaptureMode(&mode));
      if (err == hggcErrorPeerAccessAlreadyEnabled) {
        hggcGetLastError();
      } else if (err != hggcSuccess) {
        WARN("failed to peer with device %d(=%lx): %d %s",
            peerInfo->hggcDev, peerInfo->busId, err, hggcGetErrorString(err));
        return ncclInternalError;
      }
      // hgMem API support
      if (ncclCuMemEnable()) {
        // for intra-process ranks, we should map memHandle of the peers to increase refcount.
        // Otherwise, if peers abort and free the buffer, the rank can suffer invalid access.
        size_t reservationSize = 0;
        NCCLCHECK(ncclHgMemAllocAddr(devMem, &p2pBuff->ipcDesc.memHandle, p2pBuff->size, usingMemcpy() ? &peerInfo->hggcDev : NULL, &reservationSize));
        if (comm->memManager) {
          // Transfer the exporter-retained reference to the tracked import.
          size_t mappedSize = 0;
          HGCHECK(hgMemGetAddressRange(NULL, &mappedSize, (HGdeviceptr)*devMem));
          ncclResult_t tracked = ncclMemTrackImportFromPeer(comm->memManager, *devMem,
            mappedSize, p2pBuff->ipcDesc.memHandle, ncclHgMemHandleType, ncclMemOffload,
            peerInfo->rank, peerInfo->hggcDev, p2pBuff->directPtr, reservationSize);
          if (tracked != ncclSuccess) {
            NCCLCHECK(ncclHgMemFreeAddr(*devMem));
            HGCHECK(hgMemRelease(p2pBuff->ipcDesc.memHandle));
            *devMem = nullptr;
            return tracked;
          }
        } else {
          HGCHECK(hgMemRelease(p2pBuff->ipcDesc.memHandle));
        }
        *ipcPtr = *devMem;
      } else {
        *devMem = p2pBuff->directPtr;
        *ipcPtr = NULL;
      }
    } else {
      *devMem = p2pBuff->directPtr;
      *ipcPtr = NULL;
    }
  } else {
    // Different PID
    NCCLCHECK(ncclP2pImportShareableBuffer(comm, peerInfo->rank, p2pBuff->size, &p2pBuff->ipcDesc, devMem, MEM_NORMAL, p2pBuff->directPtr, ncclMemOffload));
    *ipcPtr = *devMem;
  }
  p2pInfo->minPathNum = 0;
  // Get MinPath between 2 hggc devices
  if (peerInfo->rank != myInfo->rank) {
    NCCLCHECK(ncclTopoGetMinPaths(topo, myInfo->rank, peerInfo->rank, (int*)&p2pInfo->minPathNum));
    VERBOSE(NCCL_P2P, "Min ICN link path num %02u : Rank%d/PPU%d/%lx -> Rank%d/PPU%d/%lx;",
            p2pInfo->minPathNum, myInfo->rank, myInfo->ppuid, myInfo->busId, peerInfo->rank, peerInfo->ppuid, peerInfo->busId);
  }
  return ncclSuccess;
}

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct p2pResources* resources;
  struct ncclP2pRequest req;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm, comm->topo, myInfo, peerInfo, &useRead, &intermediateRank));
  if (usingMemcpy()) useRead = 0;

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  if (graph && connIndex == 1) info->read = 0;
  const char* useReadStr = info->read ? "/read" : "";

  int sendSize = sizeof(struct ncclSendMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  if (info->read) sendSize += comm->buffSizes[NCCL_PROTO_SIMPLE];
  ALIGN_SIZE(sendSize, HGGC_IPC_MIN);

  intermediateRank = -1; // force to -1 to go through ICN transfer
  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (useRead == 0 && ncclParamP2pDirectDisable() == 0) send->conn.direct |= NCCL_DIRECT_GPU;
    INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%lx] -> %d[%lx] via P2P/IPC pointer %s%s",
        channelId, connIndex, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, useReadStr,
        myInfo->hostHash == peerInfo->hostHash ? "intra-node" : "inter-node");
  } else {
    info->rank = intermediateRank;
    INFO(NCCL_INIT|NCCL_P2P, "Channel %02d/%01d : %d[%lx] -> %d[%lx] via P2P/indirect/%d[%lx]%s",
        channelId, connIndex, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId, intermediateRank,
        comm->peerInfo[intermediateRank].busId, useReadStr);
  }

  memset(&req, '\0', sizeof(req));
  req.size = sendSize;
  req.refcount = 0;
  // peerInfo->rank is the rank that will import this buffer in p2pRecvConnect.
  req.peerRank = peerInfo->rank;
  if (PCCL_SAME_PID((comm->peerInfo + info->rank), peerInfo) && (comm->peerInfo[info->rank].hggcDev != peerInfo->hggcDev)) req.refcount++;
  if (PCCL_SAME_PID((comm->peerInfo + info->rank), myInfo) && (comm->peerInfo[info->rank].hggcDev != myInfo->hggcDev)) req.refcount++;
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 1, info->rank, &send->proxyConn));
  if (usingMemcpy()) {
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, NULL, 0, &resources->proxyInfo, sizeof(struct p2pShmProxyInfo)));
    memcpy(&info->desc, &resources->proxyInfo.desc, sizeof(ncclShmIpcDesc_t));
  } else {
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest), &info->p2pBuff, sizeof(struct ncclP2pBuff)));
    NCCLCHECK(p2pMap(comm, myInfo, comm->peerInfo+info->rank, info, (void**)&resources->sendDevMem, &resources->sendMemIpc));
    resources->sendMemSameProc = PCCL_SAME_PID(myInfo, (comm->peerInfo + info->rank));
  }
  if (ncclCuMemEnable()) resources->ipcMemManager = comm->memManager;

  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int channelId, int connIndex) {
  struct p2pResources* resources;
  struct ncclP2pRequest req;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  int useRead, intermediateRank;
  NCCLCHECK(p2pGetInfo(comm, comm->topo, myInfo, peerInfo, &useRead, &intermediateRank));

  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  info->read = useRead;
  // For CollNet, use write for scatter-reduce (conn 1), read for broadcast-gather (conn 0)
  if (graph && connIndex == 1) info->read = 0;

  int recvSize = sizeof(struct ncclRecvMem);
  // For P2P Read the SIMPLE buffer is tagged on the end of the ncclSendMem structure
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (p == NCCL_PROTO_SIMPLE_BULK || p == NCCL_PROTO_SIGNAL || p == NCCL_PROTO_SIGNAL_BULK || p == NCCL_PROTO_SIMPLE_BULK_AR) {
      continue;
    }
    if (!(useRead && p == NCCL_PROTO_SIMPLE)) {
      recvSize += comm->buffSizes[p];
    }
  }
  ALIGN_SIZE(recvSize, HGGC_IPC_MIN);

  intermediateRank = -1; // force to -1 to go through ICN transfer
  if (intermediateRank == -1) {
    info->rank = myInfo->rank;
    if (useRead == 0 && ncclParamP2pDirectDisable() == 0) recv->conn.direct |= NCCL_DIRECT_GPU;
  } else {
    info->rank = intermediateRank;
  }

  memset(&req, '\0', sizeof(req));
  req.size = recvSize;
  req.refcount = 0;
  req.peerRank = peerInfo->rank;
  if (PCCL_SAME_PID((comm->peerInfo + info->rank), peerInfo) && (comm->peerInfo[info->rank].hggcDev != peerInfo->hggcDev)) req.refcount++;
  if (PCCL_SAME_PID((comm->peerInfo + info->rank), myInfo) && (comm->peerInfo[info->rank].hggcDev != myInfo->hggcDev)) req.refcount++;
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_P2P, 0, info->rank, &recv->proxyConn));
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(struct ncclP2pRequest), &info->p2pBuff, sizeof(struct ncclP2pBuff)));
  NCCLCHECK(p2pMap(comm, myInfo, comm->peerInfo+info->rank, info, (void**)&resources->recvDevMem, &resources->recvMemIpc));
  resources->recvMemSameProc = PCCL_SAME_PID(myInfo, (comm->peerInfo + info->rank));
  if (ncclCuMemEnable()) resources->ipcMemManager = comm->memManager;

  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t p2pSendConnect(ncclComm_t comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  struct ncclRecvMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  NCCLCHECK(p2pMap(comm, comm->peerInfo+rank, comm->peerInfo+info->rank, info, (void**)&remDevMem, &resources->recvMemIpc));
  resources->recvMemSameProc = PCCL_SAME_PID((comm->peerInfo + rank), (comm->peerInfo + info->rank));

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is local (ncclSendMem) */
      send->conn.buffs[p] = resources->sendDevMem->buff;
    } else {
      send->conn.buffs[p] = remDevMem->buff + offset;
      offset += comm->buffSizes[p];
    }
  }
  if (usingMemcpy()) {
    send->conn.tail = &resources->proxyInfo.ceRecvMem->tail;
    send->conn.sizesFifo = resources->proxyInfo.ceRecvMem->sizesFifo;
    send->conn.head = &resources->proxyInfo.devShm->sendMem.head;
    // Send SIMPLE buff to proxy, and replace it by local buffer
    NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgConnect, &send->conn.buffs[NCCL_PROTO_SIMPLE], sizeof(void*), NULL, 0));
    send->conn.buffs[NCCL_PROTO_SIMPLE] = resources->proxyInfo.ceDevBuff;
  } else {
    send->conn.tail = &remDevMem->tail;
    send->conn.head = &resources->sendDevMem->head;
    send->conn.ptrExchange = &resources->sendDevMem->ptrExchange;
  }
  send->conn.minPathNum = info->minPathNum;
  send->conn.signalMarker = remDevMem->signalMarker;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(ncclComm_t comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  struct ncclSendMem* remDevMem = NULL;

  if (usingMemcpy()) {
    NCCLCHECK(ncclShmImportShareableBuffer(comm, info->rank, &info->desc, (void**)&resources->shm, (void**)&resources->devShm, &resources->desc));

    recv->conn.tail = &resources->devShm->recvMem.tail;
    recv->conn.head = &resources->devShm->sendMem.head;
  } else {
    NCCLCHECK(p2pMap(comm, comm->peerInfo+rank, comm->peerInfo+info->rank, info, (void**)&remDevMem, &resources->sendMemIpc));
    resources->sendMemSameProc = PCCL_SAME_PID((comm->peerInfo + rank), (comm->peerInfo + info->rank));

    recv->conn.tail = &resources->recvDevMem->tail;
    recv->conn.head = &remDevMem->head;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
  }
  recv->conn.minPathNum = info->minPathNum;
  recv->conn.signalMarker = resources->recvDevMem->signalMarker;

  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    if (info->read && p == NCCL_PROTO_SIMPLE) {
      /* For P2P Read the SIMPLE buffer is remote (ncclSendMem) */
      recv->conn.buffs[p] = remDevMem->buff;
    } else {
      recv->conn.buffs[p] = resources->recvDevMem->buff + offset;
      offset += comm->buffSizes[p];
    }
  }
  return ncclSuccess;
}

ncclResult_t p2pSendFree(struct ncclConnector* send) {
  struct p2pResources* resources = (struct p2pResources*)send->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // hgMem API support
      if (resources->sendMemIpc) {
        if (resources->sendMemSameProc && !resources->ipcMemManager) {
          NCCLCHECK(ncclHgMemFreeAddr(resources->sendMemIpc));
        } else {
          NCCLCHECK(ncclHggcFree(resources->sendMemIpc, MEM_NORMAL, resources->ipcMemManager));
        }
      }

      if (resources->recvMemIpc) {
        if (resources->recvMemSameProc && !resources->ipcMemManager) {
          NCCLCHECK(ncclHgMemFreeAddr(resources->recvMemIpc));
        } else {
          NCCLCHECK(ncclHggcFree(resources->recvMemIpc, MEM_NORMAL, resources->ipcMemManager));
        }
      }
    }
    else {
      if (resources->sendMemIpc) HGGCCHECK(hggcIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) HGGCCHECK(hggcIpcCloseMemHandle(resources->recvMemIpc));
    }
    NCCLCHECK(ncclFree(resources));
  }
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(struct ncclConnector* recv) {
  struct p2pResources* resources = (struct p2pResources*)recv->transportResources;
  if (resources) {
    if (ncclCuMemEnable()) {
      // hgMem API support
      if (resources->sendMemIpc) {
        if (resources->sendMemSameProc && !resources->ipcMemManager) {
          NCCLCHECK(ncclHgMemFreeAddr(resources->sendMemIpc));
        } else {
          NCCLCHECK(ncclHggcFree(resources->sendMemIpc, MEM_NORMAL, resources->ipcMemManager));
        }
      }

      if (resources->recvMemIpc) {
        if (resources->recvMemSameProc && !resources->ipcMemManager) {
          NCCLCHECK(ncclHgMemFreeAddr(resources->recvMemIpc));
        } else {
          NCCLCHECK(ncclHggcFree(resources->recvMemIpc, MEM_NORMAL, resources->ipcMemManager));
        }
      }
    }
    else {
      if (resources->sendMemIpc) HGGCCHECK(hggcIpcCloseMemHandle(resources->sendMemIpc));
      if (resources->recvMemIpc) HGGCCHECK(hggcIpcCloseMemHandle(resources->recvMemIpc));
      if (usingMemcpy()) {
        NCCLCHECK(ncclShmIpcClose(&resources->desc));
      }
    }
    NCCLCHECK(ncclFree(resources));
  }
  return ncclSuccess;
}

static ncclResult_t p2pSendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (usingMemcpy()) {
    struct p2pShmProxyInfo* proxyInfo;
    size_t shmSize;

    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    connection->transportResources = proxyInfo;

    NCCLCHECK(ncclHggcCalloc(&proxyInfo->ceDevBuff, proxyState->buffSizes[NCCL_PROTO_SIMPLE], proxyState->pDevMemUsed, proxyState->memManager));

    shmSize = sizeof(struct ncclSendMem) + sizeof(struct ncclRecvMem);
    NCCLCHECK(ncclShmAllocateShareableBuffer(shmSize, false, &proxyInfo->desc, (void**)&proxyInfo->shm, (void**)&proxyInfo->devShm));

    NCCLCHECK(ncclHggcHostCalloc(&proxyInfo->ceRecvMem, 1, proxyState->pPinnedMemUsed));

    if (respSize != sizeof(struct p2pShmProxyInfo)) return ncclInternalError;
    memcpy(respBuff, proxyInfo, sizeof(struct p2pShmProxyInfo));
  } else {
    struct ncclP2pRequest* req = (struct ncclP2pRequest*)reqBuff;
    if (reqSize != sizeof(struct ncclP2pRequest)) return ncclInternalError;
    int size = req->size;
    if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
    struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
    struct ncclMemManager* manager = proxyState->memManager;
    NCCLCHECK(ncclP2pAllocateShareableBuffer(size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr, proxyState->pDevMemUsed, MEM_NORMAL, req->peerRank, manager, ncclMemOffload));
    p2pBuff->size = size;
    if (ncclCuMemEnable()) {
      // hgMem API support
      struct p2pHgMemProxyInfo* proxyInfo;
      NCCLCHECK(ncclCalloc(&proxyInfo, 1));
      memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
      connection->transportResources = proxyInfo;
    } else {
      connection->transportResources = p2pBuff->directPtr;
    }
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t p2pRecvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct ncclP2pRequest* req = (struct ncclP2pRequest*)reqBuff;
  if (reqSize != sizeof(struct ncclP2pRequest)) return ncclInternalError;
  int size = req->size;
  if (respSize != sizeof(struct ncclP2pBuff)) return ncclInternalError;
  struct ncclP2pBuff* p2pBuff = (struct ncclP2pBuff*)respBuff;
  struct ncclMemManager* manager = proxyState->memManager;
  NCCLCHECK(ncclP2pAllocateShareableBuffer(size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr, proxyState->pDevMemUsed, MEM_NORMAL, req->peerRank, manager, ncclMemOffload));
  p2pBuff->size = size;
  if (ncclCuMemEnable()) {
    // hgMem API support
    struct p2pHgMemProxyInfo* proxyInfo;
    NCCLCHECK(ncclCalloc(&proxyInfo, 1));
    memcpy(&proxyInfo->p2pBuff, p2pBuff, sizeof(*p2pBuff));
    connection->transportResources = proxyInfo;
  } else {
    connection->transportResources = p2pBuff->directPtr;
  }
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t p2pSendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;

  if (reqSize != sizeof(void*)) return ncclInternalError;
  proxyInfo->recvFifo = *((char**)reqBuff);

  HGGCCHECK(hggcEitInterface::getInstance().wrapHggcStreamCreateForCopyEngine(&proxyInfo->stream));
  for (int i=0; i<NCCL_STEPS; i++) {
    HGGCCHECK(hggcEventCreate(proxyInfo->events+i));
  }
  connection->proxyAppendPtr = &connection->proxyAppend;
  return ncclSuccess;
}

static ncclResult_t p2pSendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  if (usingMemcpy()) {
    struct p2pShmProxyInfo* proxyInfo = (struct p2pShmProxyInfo*)connection->transportResources;
    if (proxyInfo) {
      NCCLCHECK(ncclShmIpcClose(&proxyInfo->desc));
      NCCLCHECK(ncclHggcHostFree(proxyInfo->ceRecvMem));
      NCCLCHECK(ncclHggcFree(proxyInfo->ceDevBuff, MEM_NORMAL, proxyState->memManager));
      HGGCCHECK(hggcStreamDestroy(proxyInfo->stream));
      for (int i=0; i<NCCL_STEPS; i++) {
        HGGCCHECK(hggcEventDestroy(proxyInfo->events[i]));
      }
      NCCLCHECK(ncclFree(proxyInfo));
    }
  } else {
    if (ncclCuMemEnable()) {
      // hgMem API support
      struct p2pHgMemProxyInfo *proxyInfo = (struct p2pHgMemProxyInfo *) connection->transportResources;
      if (proxyInfo) {
        struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff;
        NCCLCHECK(ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc));
        NCCLCHECK(ncclHggcFree(p2pBuff->directPtr, MEM_NORMAL, proxyState->memManager));
        NCCLCHECK(ncclFree(proxyInfo));
      }
    } else {
      // Do not check return code as CUDA may have already shut down
      NCCLCHECK(ncclHggcFree(connection->transportResources, MEM_NORMAL, proxyState->memManager));
    }
  }
  return ncclSuccess;
}

static ncclResult_t p2pRecvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  if (ncclCuMemEnable()) {
    struct p2pHgMemProxyInfo *proxyInfo = (struct p2pHgMemProxyInfo *) connection->transportResources;
    if (proxyInfo) {
      struct ncclP2pBuff *p2pBuff = &proxyInfo->p2pBuff;
      NCCLCHECK(ncclP2pFreeShareableBuffer(&p2pBuff->ipcDesc));
      NCCLCHECK(ncclHggcFree(p2pBuff->directPtr, MEM_NORMAL, proxyState->memManager));
      NCCLCHECK(ncclFree(proxyInfo));
    }
  } else {
    // Do not check return code as CUDA may have already shut down
    NCCLCHECK(ncclHggcFree(connection->transportResources, MEM_NORMAL, proxyState->memManager));
  }
  return ncclSuccess;
}

static ncclResult_t p2pSendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
      // Round to next multiple of sliceSteps
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
      struct p2pShmProxyInfo* resources = (struct p2pShmProxyInfo*) (sub->connection->transportResources);
      if (p != NCCL_PROTO_SIMPLE &&
          p != NCCL_PROTO_SIMPLE_BULK &&
          p != NCCL_PROTO_SIMPLE_BULK_AR &&
          p != NCCL_PROTO_SIGNAL &&
          p != NCCL_PROTO_SIGNAL_BULK) { // Only Simple uses hggcMemcpy
          resources->step = sub->base + sub->nsteps;
          args->done++;
          continue;
      }
      if (sub->transmitted < sub->done + NCCL_STEPS && sub->transmitted < sub->nsteps) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile int* sizesFifo = resources->ceRecvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->ceRecvMem->tail;
        // Check GPU has sent everything
        if ((*recvTail > sub->base+sub->transmitted)) {
          int size = sizesFifo[buffSlot];
          HGGCCHECK(hggcMemcpyAsync(resources->recvFifo+buffSlot*stepSize, resources->ceDevBuff+buffSlot*stepSize, size, hggcMemcpyDeviceToDevice, resources->stream));
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
          // Notify SHM
          resources->shm->recvMem.tail = sub->base + sub->done;
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

static ncclResult_t ipcRegisterBuffer(ncclComm_t comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, struct ncclReg* regRecord, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, bool* isLegacyIpc) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIpcRegInfo* newInfo = NULL;
  uintptr_t* peerRmtAddrs = NULL;
  int legacyIpcCap = 0;
  size_t sizeAlignedPhy = 0;
  size_t sizeAlignedVir = 0;
  void* baseAddr = NULL;
  bool needUpdate = false;

  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (isLegacyIpc) *isLegacyIpc = false;
  if (regRecord) {
    // buffer was registered by by users, we need to start to register or reuse it
    int peerLocalRank = -1;
    for (int p = 0; p < nPeers; p++) {
      int peerRank = peerRanks[p];
      peerLocalRank = comm->rankToLocalRank[peerRank];
      if (regRecord->ipcInfos[peerLocalRank]) {
        // We already have IPC info for peerLocalRank, no need to register it, we can reuse it
        *regBufFlag = 1;
        if (isLegacyIpc) *isLegacyIpc = regRecord->ipcInfos[peerLocalRank]->impInfo.legacyIpcCap;
        INFO(NCCL_REG, "rank %d - IPC reuse buffer %p size %ld (baseAddr %p size %ld) to peer %d regAddr %p", comm->rank, userbuff, buffSize, (void*)regRecord->addr, regRecord->pages * comm->regCache.pageSize, peerRank, regRecord->ipcInfos[peerLocalRank]->impInfo.rmtRegAddr);
      } else {
        // Register buffer with peerLocalRank
        struct ncclProxyConnector* proxyConn = NULL;
        struct p2pIpcExpInfo ipcInfo;

        if (baseAddr == NULL) {
          HGCHECKGOTO(hgMemGetAddressRange((HGdeviceptr*)&baseAddr, &sizeAlignedPhy, (HGdeviceptr)userbuff), ret, fail);
          sizeAlignedVir = sizeAlignedPhy;
          HGCHECKGOTO(hgPointerGetAttribute((void*)&legacyIpcCap, HG_POINTER_ATTRIBUTE_IS_LEGACY_HGGC_IPC_CAPABLE, (HGdeviceptr)baseAddr), ret, fail);
        }
        if (comm->gproxyConn[peerRank].initialized == false)
          NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_P2P, 1, peerRank, &comm->gproxyConn[peerRank]), ret, fail);
        proxyConn = &comm->gproxyConn[peerRank];

        // Get the mem handle for that buffer. It may have been allocated through hggcMalloc in which case we'll
        // get the hggc legacy mem handle, or through hgMem*.
        if (ncclCuMemEnable()) {
          HGmemGenericAllocationHandle handle;
          if (hgMemRetainAllocationHandle(&handle, baseAddr) != HGGC_SUCCESS) {
            // if hgMem* export fails, retry legacy export
            if (comm->directMode || !ncclLegacyCudaRegister()) goto fail;
            HGGCCHECKGOTO(hggcIpcGetMemHandle(&ipcInfo.ipcDesc.devIpc, baseAddr), ret, fail);
            ipcInfo.legacyIpcCap = true;
            if (isLegacyIpc) *isLegacyIpc = true;
          } else {
            ipcInfo.legacyIpcCap = false;
            if (isLegacyIpc) *isLegacyIpc = false;
            // hgMem* export to file descriptor or fabric handle
            if (proxyConn->sameProcess) {
              memcpy(&ipcInfo.ipcDesc.memHandle, &handle, sizeof(HGmemGenericAllocationHandle));
            } else {
              if (ncclHgMemHandleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
                int expFd = -1;
                HGCHECKGOTO(hgMemExportToShareableHandle(&expFd, handle, ncclHgMemHandleType, 0), ret, fail);
                NCCLCHECKGOTO(ncclProxyClientQueryFdBlocking(comm, proxyConn, expFd, &ipcInfo.impFd), ret, fail);
                SYSCHECKGOTO(close(expFd), "close", ret, fail);
              } else {
                // Allow this to silently fail for cases where the user buff cannot be registered
                if ((hgMemExportToShareableHandle(&ipcInfo.ipcDesc.hgDesc.handle, handle, ncclHgMemHandleType, 0)) != HGGC_SUCCESS) {
                  HGCHECKGOTO(hgMemRelease(handle), ret, fail);
                  goto fail;
                }
              }
            }
            HGmemAllocationProp prop = {};
            size_t granularityVir = 0;
            HGCHECK(hgMemGetAllocationPropertiesFromHandle(&prop, handle));
            HGCHECK(hgMemGetAllocationGranularity(&granularityVir, &prop, HG_MEM_ALLOC_GRANULARITY_MINIMUM));
            ALIGN_SIZE_WITH_RTN(sizeAlignedPhy, granularityVir, sizeAlignedVir);
            HGCHECKGOTO(hgMemRelease(handle), ret, fail);
          }
        } else if (legacyIpcCap) {
          // legacy export
          if (comm->directMode || !ncclLegacyCudaRegister()) goto fail;
          HGGCCHECKGOTO(hggcIpcGetMemHandle(&ipcInfo.ipcDesc.devIpc, baseAddr), ret, fail);
          ipcInfo.legacyIpcCap = true;
          if (isLegacyIpc) *isLegacyIpc = true;
        } else {
          // nothing works, just return
          goto fail;
        }

        void* rmtRegAddr = NULL;
        ipcInfo.phySize = sizeAlignedPhy;
        ipcInfo.virSize = sizeAlignedVir;
        ipcInfo.offset = regRecord->addr - (uintptr_t)baseAddr;
        // Now ipcInfo contains all necessary registration info. Start to register buffer on proxy side
        // and get the remote register address back.
        if (proxyConn) {
          INFO(NCCL_REG, "rank %d - IPC registering buffer %p buffSize %ld (baseAddr %p phySize %ld) to peer %d",
            comm->rank, userbuff, buffSize, (void*)regRecord->addr, ipcInfo.phySize, peerRank);
          NCCLCHECKGOTO(ncclProxyCallBlocking(comm, proxyConn, ncclProxyMsgRegister, &ipcInfo, sizeof(p2pIpcExpInfo), &rmtRegAddr, sizeof(void*)), ret, fail);
        }
        if (rmtRegAddr) {
          NCCLCHECKGOTO(ncclCalloc(&newInfo, 1), ret, fail);
          if (regRecord->ipcInfos[peerLocalRank] != NULL) {
            ERROR("IPC registration already exists for peerRank %d peerLocalRank %d", peerRank, peerLocalRank);
            ret = ncclInternalError;
            goto fail;
          }
          regRecord->state |= IPC_REG_COMPLETE;
          newInfo->peerRank = peerRank;
          newInfo->baseAddr = baseAddr;
          newInfo->impInfo.rmtRegAddr = rmtRegAddr;
          newInfo->impInfo.offset = ipcInfo.offset;
          newInfo->impInfo.legacyIpcCap = ipcInfo.legacyIpcCap;
          newInfo->ipcProxyconn = proxyConn;
          regRecord->ipcInfos[peerLocalRank] = newInfo;
          if (regRecord->regIpcAddrs.hostPeerRmtAddrs == NULL) {
            NCCLCHECKGOTO(ncclCalloc(&regRecord->regIpcAddrs.hostPeerRmtAddrs, comm->localRanks), ret, fail);
          }
          regRecord->regIpcAddrs.hostPeerRmtAddrs[peerLocalRank] = (uintptr_t)rmtRegAddr;
          needUpdate = true;
          *regBufFlag = 1;
          INFO(NCCL_REG, "rank %d - IPC registered buffer %p phySize %ld virSize %ld (baseAddr %p size %ld) to peer %d regAddr %p offsetOut %ld",
            comm->rank, userbuff, buffSize, (void*)regRecord->addr, ipcInfo.phySize, ipcInfo.virSize, peerRank, rmtRegAddr, (uintptr_t)userbuff - regRecord->addr);
        }
      }
    }

    if (*regBufFlag) {
      if (type == NCCL_IPC_COLLECTIVE) {
        // for collective, store registered remote buffers into dev memory for future reference
        if (regRecord->regIpcAddrs.devPeerRmtAddrs == NULL || needUpdate) {
          NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->hostStream), ret, fail);
          NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(), &comm->sharedRes->deviceStream), ret, fail);
          if (regRecord->regIpcAddrs.devPeerRmtAddrs == NULL)
            NCCLCHECKGOTO(ncclHggcCallocAsync(&regRecord->regIpcAddrs.devPeerRmtAddrs, comm->localRanks, NULL, comm->sharedRes->hostStream.hggcStream, comm->memManager), ret, fail);
          if (needUpdate)
            NCCLCHECKGOTO(ncclHggcMemcpyAsync(regRecord->regIpcAddrs.devPeerRmtAddrs, regRecord->regIpcAddrs.hostPeerRmtAddrs, comm->localRanks, comm->sharedRes->hostStream.hggcStream), ret, fail);
          NCCLCHECKGOTO(ncclStrongStreamWaitStream(ncclCudaGraphNone(), &comm->sharedRes->deviceStream, &comm->sharedRes->hostStream), ret, fail);
          NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->hostStream), ret, fail);
          NCCLCHECKGOTO(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream), ret, fail);
        }
        peerRmtAddrs = regRecord->regIpcAddrs.devPeerRmtAddrs;
      } else {
        assert(nPeers == 1);
        // p2p always returns remote addr here since remote buffer addr is passed in ncclDevWorkP2p struct
        peerRmtAddrs = (uintptr_t*)regRecord->regIpcAddrs.hostPeerRmtAddrs[peerLocalRank];
      }
      *offsetOut = (uintptr_t)userbuff - regRecord->addr;
      *peerRmtAddrsOut = peerRmtAddrs;
    }
  }
exit:
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  ncclFree(newInfo);
  INFO(NCCL_REG, "rank %d failed to IPC register userbuff %p buffSize %ld nPeers %d isLegacyIpc %d type %s", comm->rank, userbuff, buffSize, nPeers, isLegacyIpc ? *isLegacyIpc : -1, ncclHgMemHandleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ? "POSIX_FD" : "FABRIC");
  goto exit;
}

ncclResult_t ncclIpcLocalRegisterBuffer(ncclComm_t comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut) {
  ncclResult_t ret = ncclSuccess;
  struct ncclReg *regRecord = NULL;
  bool isValid = false;

  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    NCCLCHECKGOTO(ncclRegFind(comm, userbuff, buffSize, &regRecord), ret, fail);
    NCCLCHECKGOTO(ncclRegLocalIsValid(regRecord, &isValid), ret, fail);
    if (isValid)
      NCCLCHECKGOTO(ipcRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regRecord, regBufFlag, offsetOut, peerRmtAddrsOut, NULL), ret, fail);
  }

exit:
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  goto exit;
}

struct ncclIpcCleanupCallback {
  struct ncclCommCallback base;
  ncclComm_t comm;
  struct ncclReg *reg;
};

static ncclResult_t cleanupIpc(ncclComm_t comm, struct ncclCommCallback* cb) {
  struct ncclIpcCleanupCallback* obj = (struct ncclIpcCleanupCallback*)cb;
  NCCLCHECK(ncclCommGraphDeregister(obj->comm, obj->reg));
  ncclFree(obj);
  return ncclSuccess;
}

ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm_t comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts) {
  ncclResult_t ret = ncclSuccess;
  void* baseAddr;
  size_t sizeAlignedPhy;
  struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue = reinterpret_cast<struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>*>(cleanupQueuePtr);
  bool isLegacyIpc = false;
  struct ncclReg *regRecord = NULL;

  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    NCCLCHECKGOTO(ncclHgMemGetAddressRange((HGdeviceptr)userbuff, buffSize, (HGdeviceptr*)&baseAddr, &sizeAlignedPhy, NULL), ret, fail);
    NCCLCHECKGOTO(ncclCommGraphRegister(comm, baseAddr, sizeAlignedPhy, (void**)&regRecord), ret, fail);
    NCCLCHECKGOTO(ipcRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regRecord, regBufFlag, offsetOut, peerRmtAddrsOut, &isLegacyIpc), ret, fail);
    if (*regBufFlag) {
      struct ncclIpcCleanupCallback* record;
      NCCLCHECKGOTO(ncclCalloc(&record, 1), ret, fail);
      record->base.fn = cleanupIpc;
      record->comm = comm;
      record->reg = regRecord;
      if (isLegacyIpc) {
        ncclIntruQueueEnqueue(&comm->legacyRegCleanupQueue, (struct ncclCommCallback*)record);
      } else {
        ncclIntruQueueEnqueue(cleanupQueue, (struct ncclCommCallback*)record);
        if (nCleanupQueueElts) *nCleanupQueueElts += 1;
      }
    } else {
      NCCLCHECKGOTO(ncclCommGraphDeregister(comm, regRecord), ret, fail);
    }
  }

exit:
  // coverity[leaked_storage:FALSE] => normally, addrsRecord is added to the cleanupQueue
  return ret;
fail:
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  goto exit;
}

ncclResult_t ncclIpcDeregBuffer(ncclComm_t comm, struct ncclIpcRegInfo* regInfo) {
  NCCLCHECK(ncclProxyCallBlocking(comm, regInfo->ipcProxyconn, ncclProxyMsgDeregister, &regInfo->impInfo, sizeof(struct ncclIpcImpInfo), NULL, 0));
  INFO(NCCL_REG, "rank %d - IPC deregistered buffer %p peer %d ipc remote buffer %p", comm->rank, regInfo->baseAddr, regInfo->peerRank, regInfo->impInfo.rmtRegAddr);
  return ncclSuccess;
}

static ncclResult_t p2pProxyRegister(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct p2pIpcExpInfo* ipcExpInfo = (struct p2pIpcExpInfo*)reqBuff;
  void* regAddr = NULL;
  ncclResult_t ret = ncclSuccess;
  bool mapped = false;
  bool imported = false;
  HGmemGenericAllocationHandle handle;

  if (reqSize != sizeof(struct p2pIpcExpInfo)) {
    ERROR("Invalid P2P IPC register request size %d, expected %zu", reqSize, sizeof(struct p2pIpcExpInfo));
    return ncclInternalError;
  }
  if (respSize != sizeof(void*)) {
    ERROR("Invalid P2P IPC register response size %d, expected %zu", respSize, sizeof(void*));
    return ncclInternalError;
  }

  INFO(NCCL_REG, "Proxy rank %d register reqBuff %p phySize %ld virSize %ld offset %ld legacyIpcCap %d sameProcess %d",
    proxyState->tpRank, reqBuff, ipcExpInfo->phySize, ipcExpInfo->virSize, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap, connection->sameProcess);

  // request peer passes all necessary buffer info to import. The proxy thread would register
  // the buffer locally and return register addr back
  if (ipcExpInfo->legacyIpcCap) {
    // legacy import
    HGGCCHECKGOTO(hggcIpcOpenMemHandle(&regAddr, ipcExpInfo->ipcDesc.devIpc, hggcIpcMemLazyEnablePeerAccess), ret, fail);
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo->offset);
  } else {
    // hgMem import
    if (connection->sameProcess) {
      // if proxy is same process as request peer, we just need to map the handle.
      memcpy(&handle, &ipcExpInfo->ipcDesc.memHandle, sizeof(HGmemGenericAllocationHandle));
    } else {
      if (ncclHgMemHandleType == HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
        HGCHECKGOTO(hgMemImportFromShareableHandle(&handle, (void*)(uintptr_t)ipcExpInfo->impFd, ncclHgMemHandleType), ret, fail);
        SYSCHECKGOTO(close(ipcExpInfo->impFd), "close", ret, fail);
      } else {
        HGCHECKGOTO(hgMemImportFromShareableHandle(&handle, (void*)&ipcExpInfo->ipcDesc.hgDesc, ncclHgMemHandleType), ret, fail);
      }
    }
    imported = true;
    HGCHECKGOTO(hgMemAddressReserve((HGdeviceptr*)&regAddr, ipcExpInfo->virSize, /* alignment */ 0, /* addr */ 0, /* flags */ 0), ret, fail);
    HGCHECKGOTO(hgMemMap((HGdeviceptr)regAddr, ipcExpInfo->phySize, /* offset */ 0, handle, /* flags */ 0), ret, fail);
    mapped = true;
    // Allow access by the local GPU
    HGmemAccessDesc accessDesc = {};
    accessDesc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = proxyState->hggcDev;
    accessDesc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
    HGCHECKGOTO(hgMemSetAccess((HGdeviceptr)regAddr, ipcExpInfo->phySize, &accessDesc, 1), ret, fail);
    regAddr = (void*)((uintptr_t)regAddr + ipcExpInfo->offset);
  }
  INFO(NCCL_REG, "Proxy rank %d register success regAddr %p phySize %ld virSize %ld offset %ld legacyIpcCap %d sameProcess %d", proxyState->tpRank, regAddr, ipcExpInfo->phySize, ipcExpInfo->virSize, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap, connection->sameProcess);

exit:
  memcpy(respBuff, (void*)&regAddr, sizeof(void*));
  *done = 1;
  return ret;
fail:
  if (!ipcExpInfo->legacyIpcCap) {
    if (mapped) HGCHECK(hgMemUnmap((HGdeviceptr)regAddr, ipcExpInfo->phySize));
    if (regAddr) HGCHECK(hgMemAddressFree((HGdeviceptr)regAddr, ipcExpInfo->virSize));
    if (imported) HGCHECK(hgMemRelease(handle));
  }
  regAddr = NULL;
  goto exit;
}

static ncclResult_t p2pProxyDeregister(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done) {
  ncclResult_t ret = ncclSuccess;
  struct ncclIpcImpInfo* ipcInfo = (struct ncclIpcImpInfo*)reqBuff;
  if (reqSize != sizeof(struct ncclIpcImpInfo)) {
    ERROR("Invalid P2P IPC deregister request size %d, expected %zu", reqSize, sizeof(struct ncclIpcImpInfo));
    return ncclInternalError;
  }

  if (ipcInfo->legacyIpcCap) {
    HGGCCHECKGOTO(hggcIpcCloseMemHandle((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)), ret, fail);
  } else {
    if (connection->sameProcess) {
      NCCLCHECKGOTO(ncclHgMemFreeAddr((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)), ret, fail);
    } else {
      NCCLCHECKGOTO(ncclHggcFree((void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset)), ret, fail);
    }
  }

exit:
  *done = 1;
  return ret;
fail:
  goto exit;
}


struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  {
    p2pSendSetup,
    p2pSendConnect,
    p2pSendFree,
    NULL,
    p2pSendProxySetup,
    NULL,
    p2pSendProxyFree,
    NULL,
    p2pProxyRegister,
    p2pProxyDeregister
  },
  {
    p2pRecvSetup,
    p2pRecvConnect,
    p2pRecvFree,
    NULL,
    p2pRecvProxySetup,
    NULL,
    p2pRecvProxyFree,
    NULL,
    p2pProxyRegister,
    p2pProxyDeregister
  }
};

static void initCeOperation() {
  static int init = 0;
  if (!init) {
    if (usingMemcpy()) {
      p2pTransport.send.proxyConnect = p2pSendProxyConnect;
      p2pTransport.send.proxyProgress = p2pSendProxyProgress;
    }
    init = 1;
  }
}
