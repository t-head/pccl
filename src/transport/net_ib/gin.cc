/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include "pccl_compat.h"
#include "alloc.h"
#include "connect.h"
#include "pccl_gin.h"
#include "gin/gin_host.h"
#include "gin.h"
#include "gdaki/gin_host_gdaki.h"
#include "param.h"
#include <mutex>

const int NCCL_GIN_IB_ALLGATHER_TAG = 0xa0;
const int NCCL_GIN_IB_ALLTOALL_TAG = 0xa1;

extern int64_t ncclParamDmaBufEnable();

// Check GDR support for GIN. This is run at init, so we don't know yet whether
// the GPU will support DMA-BUF.
static ncclResult_t ncclGinIbGdrSupport(bool* gdrSupport, bool gdaki) {
  *gdrSupport = true;
  bool peerMemSupport =
      gdaki ? ncclIbPeerMemSupport() == ncclSuccess
            : ncclIbGdrSupport() == ncclSuccess;
  if (ncclParamDmaBufEnable() && ncclIbDmaBufSupport(0) == ncclSuccess) {
    INFO_ONCE(NCCL_INIT|NCCL_NET, "GIN prefers DMA-BUF over alixpu_peermem to map GPU memory for RDMA.");
  } else if (peerMemSupport) {
    INFO_ONCE(NCCL_INIT|NCCL_NET, "GIN uses alixpu_peermem to map GPU memory for RDMA.");
  } else if (ncclParamDmaBufEnable() == 0) {
    *gdrSupport = false;
    WARN_ONCE("Unable to use GIN: alixpu_peermem is not loaded and DMA-BUF is disabled by PCCL_DMABUF_ENABLE=0.");
  } else {
    *gdrSupport = false;
    WARN_ONCE("Unable to use GIN: alixpu_peermem is not loaded and the NIC does not support DMA-BUF.");
  }
  return ncclSuccess;
}

// Check the current GPU supports GDR for GIN. This is run during connect().
static ncclResult_t ncclGinIbGdrGpuSupport(bool gdaki) {
  int hggcDev;
  HGGCCHECK(hggcGetDevice(&hggcDev));
  int dmaBufSupportOnDevice = 0;
  if (ncclParamDmaBufEnable())
    HGCHECK(hgDeviceGetAttribute(&dmaBufSupportOnDevice, HG_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, hggcDev));
  bool peerMemSupport =
      gdaki ? ncclIbPeerMemSupport() == ncclSuccess
            : ncclIbGdrSupport() == ncclSuccess;
  if (dmaBufSupportOnDevice == 1) {
    INFO_ONCE(NCCL_INIT|NCCL_NET, "GIN prefers DMA-BUF over alixpu_peermem to map GPU memory for RDMA.");
    return ncclSuccess;
  }
  if (peerMemSupport) {
    INFO_ONCE(NCCL_INIT|NCCL_NET, "GIN uses alixpu_peermem to map GPU memory for RDMA.");
    return ncclSuccess;
  }
  if (ncclParamDmaBufEnable() == 0) {
    WARN_ONCE("Unable to use GIN: alixpu_peermem is not loaded and DMA-BUF is disabled by PCCL_DMABUF_ENABLE=0.");
    return ncclInvalidUsage;
  }
  WARN_ONCE("Unable to use GIN: alixpu_peermem is not loaded and device %d does not support DMA-BUF.", hggcDev);
  return ncclInvalidUsage;
}

NCCL_PARAM(GinType, "GIN_TYPE", -1);

static std::mutex ncclGinIbGdakiLockMutex;
static int ncclGinIbGdakiNDevs = -1;

ncclResult_t ncclGinIbGdakiInit() {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (ncclGinIbGdakiNDevs == -1) {
    int ndevs = 0;
    for (int i = 0; i < ncclNIbDevs; i++) {
      if (ncclIbDevs[i].ibProvider == IB_PROVIDER_MLX5) {
        ncclGinIbGdakiDevIndexes[ndevs] = i;
        ++ndevs;
      }
    }
    ncclGinIbGdakiNDevs = ndevs;
  }
  return ncclSuccess;
}

extern ncclGin_t ncclGinIb;
extern ncclGin_t ncclGinIbGdaki;
extern ncclGin_t ncclGinIbProxy;

// Initialize GDAKI or PROXY backend. ginType can force a particular backend.
// If provided, overwrite ginIb with the backend (generic ginIb case).
ncclResult_t ncclGinIbInitType(void** ctx, uint64_t /*commId*/, ncclDebugLogger_t logFunction, int ginType, ncclGin_t* ginIb) {
  NCCLCHECK(ncclIbInitDevices(logFunction, nullptr));
  if (ncclNIbDevs == 0) return ncclInternalError;

  if (ginType == NCCL_GIN_TYPE_GDAKI) goto try_gdaki;
  if (ginType == NCCL_GIN_TYPE_PROXY) goto try_proxy;
  if (ginType != -1) {
    INFO(NCCL_INIT|NCCL_NET, "NET_IB: no support for GIN type %ld", ncclParamGinType());
    return ncclInternalError;
  }

  bool gdrSupport;

try_gdaki:
  NCCLCHECK(ncclGinIbGdakiInit());
  if (ncclGinIbGdakiNDevs == 0 && ginType == -1) goto try_proxy;
  NCCLCHECK(ncclGinIbGdrSupport(&gdrSupport, /*gdaki*/ true));
  if (!gdrSupport && ginType == -1) goto try_proxy;
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &ncclGinIbGdaki, sizeof(ncclGinIb));
  goto end;

try_proxy:
  NCCLCHECK(ncclGinIbGdrSupport(&gdrSupport, /*gdaki*/ false));
  if (!gdrSupport) return ncclInternalError;
  if (ginIb) memcpy(ginIb, &ncclGinIbProxy, sizeof(ncclGinIb));

end:
  ncclNetCommConfig_t* netCommConfig = nullptr;
  NCCLCHECK(ncclCalloc(&netCommConfig, 1));
  netCommConfig->trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
  *ctx = netCommConfig;
  return ncclSuccess;
}

ncclResult_t ncclGinIbInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, ncclParamGinType(), &ncclGinIb);
}

// GIN entry point, which will then morph into either the GDAKI or PROXY backend.
ncclGin_t ncclGinIb = {
  "GIN_IB",
  ncclGinIbInit,
  NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL
};

ncclResult_t ncclGinIbFinalize(void* ctx) {
  if (ctx) free(ctx);
  return ncclIbFinalizeDevices();
}

// Copy v6 IB device properties into the v12 GIN-facing property struct.
static void ncclGinIbFillV12Properties(const ncclNetProperties_v6_t& src, ncclNetProperties_v12_t* dst) {
  dst->name = src.name;
  dst->pciPath = src.pciPath;
  dst->guid = src.guid;
  dst->ptrSupport = src.ptrSupport;
  dst->regIsGlobal = 0;
  dst->forceFlush = 0;
  dst->speed = src.speed;
  dst->port = src.port;
  dst->latency = src.latency;
  dst->maxComms = src.maxComms;
  dst->maxRecvs = src.maxRecvs;
  dst->netDeviceType = NCCL_NET_DEVICE_HOST;
  dst->netDeviceVersion = 0;
  dst->vProps.ndevs = 1;
  dst->vProps.devs[0] = 0;
  dst->maxP2pBytes = MAX_GIN_SIZE;
  dst->maxCollBytes = MAX_GIN_SIZE;
  dst->maxMultiRequestSize = 0;
  dst->railId = NCCL_NET_ID_UNDEF;
  dst->planeId = NCCL_NET_ID_UNDEF;
}

static ncclResult_t ncclGinIbAllGather(struct ncclGinIbCollComm* cComm, void* srcBuf, void* recvBuf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void* rMhandle = NULL;
  void* sMhandle = NULL;
  void* srequest = NULL;
  void* rrequest = NULL;
  int speer;
  int rpeer;
  void* rbuf;
  int tag;
  int done;
  int recvLen = (int)len;

  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->recvComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &rMhandle),
                status, out);
  NCCLCHECKGOTO(ncclNetIb.regMr(cComm->sendComm, recvBuf,
                                cComm->nranks * len, NCCL_PTR_HOST,
                                &sMhandle),
                status, out);

  speer = cComm->rank;
  memcpy((void*)((uintptr_t)recvBuf + speer * len), srcBuf, len);
  for (int i = 0; i < cComm->nranks - 1; i++) {
    rpeer = (speer - 1 + cComm->nranks) % cComm->nranks;
    while (srequest == NULL || rrequest == NULL) {
      rbuf = (void*)((uintptr_t)recvBuf + rpeer * len);
      tag = NCCL_GIN_IB_ALLGATHER_TAG;
      if (srequest == NULL)
        NCCLCHECKGOTO(ncclNetIb.isend(cComm->sendComm,
                                      (void*)((uintptr_t)recvBuf + speer * len),
                                      len, tag, sMhandle, &srequest),
                      status, out);
      if (rrequest == NULL) {
        NCCLCHECKGOTO(ncclNetIb.irecv(cComm->recvComm, 1, &rbuf, &recvLen,
                                      &tag, &rMhandle, &rrequest),
                      status, out);
      }
    }
    while (srequest || rrequest) {
      if (rrequest)
        NCCLCHECKGOTO(ncclNetIb.test(rrequest, &done, NULL),
                      status, out);
      if (done) rrequest = NULL;
      if (srequest)
        NCCLCHECKGOTO(ncclNetIb.test(srequest, &done, NULL),
                      status, out);
      if (done) srequest = NULL;
    }
    speer = rpeer;
  }

out:
  if (rMhandle) ncclNetIb.deregMr(cComm->recvComm, rMhandle);
  if (sMhandle) ncclNetIb.deregMr(cComm->sendComm, sMhandle);
  return status;
}

static ncclResult_t ncclGinIbAllToAll(struct ncclGinIbCollComm* cComm, void* src_buf, void* recv_buf, size_t len) {
  ncclResult_t status = ncclSuccess;
  void* tmp_buf = nullptr;
  NCCLCHECK(ncclIbMalloc((void**)&tmp_buf, cComm->nranks * cComm->nranks * len));
  NCCLCHECKGOTO(cComm->allGather(cComm, src_buf, tmp_buf, cComm->nranks * len), status, out);

  for (int i = 0; i < cComm->nranks; i++) {
    memcpy((void*)((uintptr_t)recv_buf + i * len),
           (void*)((uintptr_t)tmp_buf + i * cComm->nranks * len + cComm->rank * len),
           len);
  }

out:
  if (tmp_buf) free(tmp_buf);
  return status;
}

ncclResult_t ncclGinIbP2PBarrier(struct ncclGinIbCollComm* cComm) {
  int* dummy;
  NCCLCHECK(ncclIbMalloc((void**)&dummy, cComm->nranks * sizeof(int)));
  NCCLCHECK(ncclGinIbAllGather(cComm, dummy + cComm->rank, dummy, sizeof(int)));
  free(dummy);
  return ncclSuccess;
}

static ncclResult_t ncclGinIbGetGidIndexAdapter(struct ibv_context* context, uint8_t portNum,
                                                struct ibv_port_attr* portAttr, int* gidIndex) {
  return ncclIbGetGidIndex(context, portNum, portAttr->gid_tbl_len, gidIndex);
}

ncclResult_t ncclGinIbConnect(void* ctx, void* handles[], int nranks, int rank,
                              void* listenComm, void** collComm) {
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclGinIbCollComm* cCommArray = nullptr;
  int next;

  *collComm = NULL;
  NCCLCHECK(ncclIbMalloc((void**)&cCommArray, sizeof(*cCommArray)));

  struct ncclGinIbCollComm* cComm = cCommArray;
  cComm->ctx = ctx;
  cComm->nranks = nranks;
  cComm->rank = rank;

  next = (cComm->rank + 1) % nranks;
  do {
    if (cComm->sendComm == NULL) {
      NCCLCHECK(ncclNetIb.connect(lComm->dev, handles[next], &cComm->sendComm));
    }
    if (cComm->recvComm == NULL) {
      NCCLCHECK(ncclNetIb.accept(lComm, &cComm->recvComm));
    }
  } while (cComm->sendComm == NULL || cComm->recvComm == NULL);

  cComm->getProperties = (ncclResult_t(*)(int, void*))ncclIbGetProperties;
  cComm->allGather = ncclGinIbAllGather;
  cComm->allToAll = ncclGinIbAllToAll;
  cComm->getGidIndex = ncclGinIbGetGidIndexAdapter;
  cComm->dev = lComm->dev;

  cComm->ib.context = ncclIbDevs[cComm->dev].context;
  cComm->ib.pd = ncclIbDevs[cComm->dev].pd;

  *collComm = cCommArray;
  return ncclSuccess;
}

ncclResult_t ncclGinIbCloseColl(void* collComm) {
  struct ncclGinIbCollComm* cCommArray = (struct ncclGinIbCollComm*)collComm;
  if (!cCommArray) return ncclSuccess;

  struct ncclGinIbCollComm* cComm = cCommArray;
  if (cComm->recvComm) {
    NCCLCHECK(ncclNetIb.closeRecv(cComm->recvComm));
    cComm->recvComm = NULL;
  }

  if (cComm->sendComm) {
    NCCLCHECK(ncclNetIb.closeSend(cComm->sendComm));
    cComm->sendComm = NULL;
  }

  memset(cComm, 0, sizeof(*cComm));
  free(cCommArray);
  return ncclSuccess;
}

// ---------------------- GDAKI backend ------------------------------------
// The GDAKI backend dispatches every device-facing entry point into
// gin_host_gdaki.cc, which drives the DOCA-GPUNetIO host library.

ncclResult_t ncclGinIbGdakiInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_GDAKI, NULL);
}

ncclResult_t ncclGinIbGdakiDevices(int* ndev) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  *ndev = ncclGinIbGdakiNDevs;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiGetProperties(int dev, ncclNetProperties_v12_t* props) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  if (dev >= ncclGinIbGdakiNDevs) {
    WARN("NET/IB : Requested properties for GIN GDAKI NIC %d, only %d GIN GDAKI NICs have been created",
         dev, ncclGinIbGdakiNDevs);
    return ncclInvalidUsage;
  }
  ncclNetProperties_v6_t physProps;
  NCCLCHECK(ncclIbGetPhysProperties(ncclGinIbGdakiDevIndexes[dev], &physProps));
  ncclGinIbFillV12Properties(physProps, props);
  props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiListen(void* /*ctx*/, int dev, void* opaqueHandle, void** listenComm) {
  std::lock_guard<std::mutex> lock(ncclGinIbGdakiLockMutex);
  return ncclNetIb.listen(ncclGinIbGdakiDevIndexes[dev], opaqueHandle, listenComm);
}

ncclResult_t ncclGinIbGdakiConnect(void* ctx, void* handles[], int nranks, int rank,
                                   void* listenComm, void** collComm) {
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ true));
  NCCLCHECK(ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));

  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)*collComm;
  cComm->getProperties = (ncclResult_t(*)(int, void*))ncclGinIbGdakiGetProperties;
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiCreateContext(void* collComm, ncclGinConfig_v13_t* config,
                                         void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  NCCLCHECK(ncclGinGdakiCreateContext(cComm, config->nSignals, config->nCounters,
                                      config->nContexts, config->queueDepth,
                                      config->trafficClass, ginCtx, devHandle));
  return ncclSuccess;
}

ncclResult_t ncclGinIbGdakiRegMrSym(void* collComm, void* data, size_t size, int type,
                                    uint64_t mr_flags, void** mhandle, void** ginHandle) {
  return ncclGinGdakiRegMrSym((struct ncclGinIbCollComm*)collComm, data, size, type,
                              mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbGdakiDeregMrSym(void* collComm, void* mhandle) {
  return ncclGinGdakiDeregMrSym((struct ncclGinIbCollComm*)collComm, mhandle);
}

ncclResult_t ncclGinIbGdakiDestroyContext(void* ginCtx) {
  return ncclGinGdakiDestroyContext(ginCtx);
}

ncclResult_t ncclGinIbGdakiProgress(void* collComm) {
  return ncclGinGdakiProgress(collComm);
}

ncclResult_t ncclGinIbGdakiQueryLastError(void* ginCtx, bool* hasError) {
  return ncclGinGdakiQueryLastError(ginCtx, hasError);
}

// Adapter so the v13 listen-close signature matches the v6 IB closeListen.
static ncclResult_t ncclGinIbCloseListen(void* listenComm) {
  return ncclNetIb.closeListen(listenComm);
}

ncclGin_t ncclGinIbGdaki = {
  "GIN_IB_GDAKI",
  ncclGinIbGdakiInit,
  ncclGinIbGdakiDevices,
  ncclGinIbGdakiGetProperties,
  ncclGinIbGdakiListen,
  ncclGinIbGdakiConnect,
  ncclGinIbGdakiCreateContext,
  ncclGinIbGdakiRegMrSym,
  NULL,
  ncclGinIbGdakiDeregMrSym,
  ncclGinIbGdakiDestroyContext,
  ncclGinIbCloseColl,
  ncclGinIbCloseListen,
  NULL,
  NULL,
  NULL,
  NULL,
  NULL,
  ncclGinIbGdakiProgress,
  ncclGinIbGdakiQueryLastError,
  ncclGinIbFinalize
};

// ---------------------- PROXY backend ------------------------------------

struct ncclIbGinProxyMrHandle {
  void*       mrHandle;
  uintptr_t*  base_vas;
  uint32_t*   rkeys;
  uint32_t    lkey;
  // Stash the local MR so we can recover lkey/rkey without referencing the
  // (opaque) v6 mhandle layout.
};

ncclResult_t ncclGinIbProxyInit(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  return ncclGinIbInitType(ctx, commId, logFunction, NCCL_GIN_TYPE_PROXY, NULL);
}

ncclResult_t ncclGinIbProxyDevices(int* ndev) {
  return ncclIbDevices(ndev);
}

ncclResult_t ncclGinIbProxyGetProperties(int dev, ncclNetProperties_v12_t* props) {
  ncclNetProperties_v6_t physProps;
  NCCLCHECK(ncclNetIb.getProperties(dev, &physProps));
  ncclGinIbFillV12Properties(physProps, props);
  props->vProps.devs[0] = dev;
  props->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyListen(void* /*ctx*/, int dev, void* opaqueHandle, void** listenComm) {
  return ncclNetIb.listen(dev, opaqueHandle, listenComm);
}

ncclResult_t ncclGinIbProxyConnect(void* ctx, void* handles[], int nranks, int rank,
                                   void* listenComm, void** collComm) {
  NCCLCHECK(ncclGinIbGdrGpuSupport(/*gdaki*/ false));
  NCCLCHECK(ncclGinIbConnect(ctx, handles, nranks, rank, listenComm, collComm));
  return ncclSuccess;
}

struct ncclGinIbProxyCtx {
  void**  fullRecvComm;
  void**  fullSendComm;
  int rank, nranks;
  int nContexts;
};

// Forward decl: ProxyDestroyContext is reused on ProxyCreateContext's error path.
ncclResult_t ncclGinIbProxyDestroyContext(void* ginCtx);

ncclResult_t ncclGinIbProxyCreateContext(void* collComm, ncclGinConfig_v13_t* config,
                                         void** ginCtx, ncclNetDeviceHandle_v11_t** /*devHandle*/) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  ncclIbSetTrafficClass(cComm->ctx, config->trafficClass);

  if (config->queueDepth != 0) {
    WARN("GIN_IB_PROXY does not support specifying qp depth");
    return ncclInvalidUsage;
  }

  int nranks;
  struct ncclGinIbProxyCtx* ginProxyCtx = NULL;
  *ginCtx = NULL;
  NCCLCHECK(ncclCalloc(&ginProxyCtx, config->nContexts));
  ginProxyCtx[0].nContexts = config->nContexts;
  ginProxyCtx[0].nranks = nranks = cComm->nranks;

  void* lComm = NULL;
  char* handle = NULL;
  char* handles = NULL;
  NCCLCHECKGOTO(ncclIbMalloc((void**)&handles, NCCL_NET_HANDLE_MAXSIZE * cComm->nranks), ret, end);
  handle = handles + NCCL_NET_HANDLE_MAXSIZE * cComm->rank;

  NCCLCHECKGOTO(ncclNetIb.listen(cComm->dev, handle, &lComm), ret, end);
  NCCLCHECKGOTO(cComm->allGather(cComm, handle, handles, NCCL_NET_HANDLE_MAXSIZE), ret, end);

  for (int c = 0; c < config->nContexts; c++) {
    struct ncclGinIbProxyCtx* gc = ginProxyCtx + c;
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullSendComm, sizeof(void*) * nranks), ret, end);
    NCCLCHECKGOTO(ncclIbMalloc((void**)&gc->fullRecvComm, sizeof(void*) * nranks), ret, end);
    gc->rank = cComm->rank;

    for (int i = 0; i < nranks; i++) {
      int connectPeer = (cComm->rank + i) % nranks;
      int acceptPeer = (cComm->rank - i + nranks) % nranks;
      do {
        if (gc->fullSendComm[connectPeer] == NULL)
          NCCLCHECKGOTO(ncclNetIb.connect(cComm->dev,
                                          handles + NCCL_NET_HANDLE_MAXSIZE * connectPeer,
                                          &gc->fullSendComm[connectPeer]),
                        ret, end);
        if (gc->fullRecvComm[acceptPeer] == NULL)
          NCCLCHECKGOTO(ncclNetIb.accept(lComm, &gc->fullRecvComm[acceptPeer]),
                        ret, end);
      } while ((gc->fullSendComm[connectPeer] == NULL) ||
               (gc->fullRecvComm[acceptPeer] == NULL));
      // Phase-2: complete QP transition (INIT → RTR → RTS) on the sender
      // side and synchronize with the receiver.  Without this the sender's
      // QP stays in INIT and ibv_post_send in IPut/IGet will fail.
      while (((struct ncclIbSendComm*)gc->fullSendComm[connectPeer])->base.ready == 0)
        NCCLCHECKGOTO(ncclSendCheck((struct ncclIbSendComm*)gc->fullSendComm[connectPeer]), ret, end);
      while (((struct ncclIbRecvComm*)gc->fullRecvComm[acceptPeer])->base.ready == 0)
        NCCLCHECKGOTO(ncclRecvCheck((struct ncclIbRecvComm*)gc->fullRecvComm[acceptPeer]), ret, end);
      NCCLCHECKGOTO(ncclGinIbP2PBarrier(cComm), ret, end);
    }
  }

end:
  free(handles);
  if (lComm) ncclNetIb.closeListen(lComm);
  if (ret != ncclSuccess) {
    if (ginProxyCtx != NULL) ncclGinIbProxyDestroyContext(ginProxyCtx);
  } else {
    *ginCtx = ginProxyCtx;
  }
  return ret;
}

ncclResult_t ncclGinIbProxyDestroyContext(void* ginCtx) {
  if (ginCtx == NULL) return ncclSuccess;
  struct ncclGinIbProxyCtx* gc = (struct ncclGinIbProxyCtx*)ginCtx;
  int nContexts = gc[0].nContexts;
  int nranks = gc[0].nranks;
  for (int c = 0; c < nContexts; c++) {
    if (gc[c].fullRecvComm) {
      for (int i = 0; i < nranks; i++) {
        if (gc[c].fullRecvComm[i] != NULL)
          NCCLCHECK(ncclNetIb.closeRecv(gc[c].fullRecvComm[i]));
      }
      free(gc[c].fullRecvComm);
      gc[c].fullRecvComm = NULL;
    }

    if (gc[c].fullSendComm) {
      for (int i = 0; i < nranks; i++) {
        if (gc[c].fullSendComm[i] != NULL)
          NCCLCHECK(ncclNetIb.closeSend(gc[c].fullSendComm[i]));
      }
      free(gc[c].fullSendComm);
      gc[c].fullSendComm = NULL;
    }
  }
  free(ginCtx);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSymDmaBuf(void* collComm, void* data, size_t size, int type,
                                          uint64_t offset, int fd, uint64_t mr_flags,
                                          void** mhandle, void** ginHandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  struct ncclIbGinProxyMrHandle* ginMrHandle;
  NCCLCHECK(ncclCalloc(&ginMrHandle, 1));

  NCCLCHECK(ncclIbRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags,
                                      &ginMrHandle->mrHandle));

  struct ibv_mr* mr = (struct ibv_mr*)ginMrHandle->mrHandle;
  ginMrHandle->lkey = mr->lkey;
  uint32_t rkey = mr->rkey;

  NCCLCHECK(ncclCalloc(&ginMrHandle->base_vas, cComm->nranks));
  NCCLCHECK(ncclCalloc(&ginMrHandle->rkeys, cComm->nranks));

  NCCLCHECK(cComm->allGather(cComm, &data, ginMrHandle->base_vas, sizeof(uintptr_t)));
  NCCLCHECK(cComm->allGather(cComm, &rkey, ginMrHandle->rkeys, sizeof(uint32_t)));

  *mhandle = ginMrHandle;
  *ginHandle = ginMrHandle;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyRegMrSym(void* collComm, void* data, size_t size, int type,
                                    uint64_t mr_flags, void** mhandle, void** ginHandle) {
  return ncclGinIbProxyRegMrSymDmaBuf(collComm, data, size, type, 0, -1, mr_flags, mhandle, ginHandle);
}

ncclResult_t ncclGinIbProxyDeregMrSym(void* collComm, void* mhandle) {
  struct ncclGinIbCollComm* cComm = (struct ncclGinIbCollComm*)collComm;
  struct ncclIbGinProxyMrHandle* ginMrHandle = (struct ncclIbGinProxyMrHandle*)mhandle;
  NCCLCHECK(ncclNetIb.deregMr(cComm->recvComm, ginMrHandle->mrHandle));
  free(ginMrHandle->base_vas);
  free(ginMrHandle->rkeys);
  free(ginMrHandle);
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPut(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                uint64_t dstOff, void* dstMhandle, uint32_t rank,
                                void** request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle* srcMrHandle = (struct ncclIbGinProxyMrHandle*)srcMhandle;
  struct ncclIbGinProxyMrHandle* dstMrHandle = (struct ncclIbGinProxyMrHandle*)dstMhandle;

  void* srcPtr = (void*)(srcMrHandle->base_vas[ginProxyCtx->rank] + srcOff);
  void* dstPtr = (void*)(dstMrHandle->base_vas[rank] + dstOff);
  uint32_t lkey = srcMrHandle->lkey;
  uint32_t rkey = dstMrHandle->rkeys[rank];

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ibv_qp* qp = comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  // pcclIbGetRequest presets events=1 for p2p, but GIN counts one event per signaled WQE via
  // ncclIbAddEvent below. Start from zero or the request never drains back to 0.
  req->events = 0;
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->ginPeerRank = rank;
  req->devBase = &comm->dev.base;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode              = IBV_WR_RDMA_WRITE;
  wr.send_flags          = IBV_SEND_SIGNALED;
  wr.wr_id               = req - comm->base.reqs;
  wr.next                = NULL;
  wr.wr.rdma.remote_addr = (uint64_t)dstPtr;
  wr.wr.rdma.rkey        = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)srcPtr;
  sge.length = size;
  sge.lkey = lkey;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp, &wr, &bad_wr));
  ncclIbAddEvent(req, 0);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIGet(void* ginCtx, int context, uint64_t remoteOffset, void* remoteMhandle,
                                size_t size, uint64_t localOffset, void* localMhandle, uint32_t rank,
                                void** request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle* remoteMrHandle = (struct ncclIbGinProxyMrHandle*)remoteMhandle;
  struct ncclIbGinProxyMrHandle* localMrHandle = (struct ncclIbGinProxyMrHandle*)localMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ibv_qp* qp = comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  // See ncclGinIbProxyIPut.
  req->events = 0;
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IGET;
  req->sock = &comm->base.sock;
  req->ginPeerRank = rank;
  req->devBase = &comm->dev.base;

  void* remotePtr = (void*)(remoteMrHandle->base_vas[rank] + remoteOffset);
  void* localPtr = (void*)(localMrHandle->base_vas[ginProxyCtx->rank] + localOffset);
  uint32_t rkey = remoteMrHandle->rkeys[rank];
  uint32_t lkey = localMrHandle->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));

  wr.opcode              = IBV_WR_RDMA_READ;
  wr.send_flags          = IBV_SEND_SIGNALED;
  wr.wr_id               = req - comm->base.reqs;
  wr.next                = NULL;
  wr.wr.rdma.remote_addr = (uint64_t)remotePtr;
  wr.wr.rdma.rkey        = rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  sge.addr = (uintptr_t)localPtr;
  sge.length = size;
  sge.lkey = lkey;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp, &wr, &bad_wr));
  ncclIbAddEvent(req, 0);

  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIPutSignal(void* ginCtx, int context, uint64_t srcOff, void* srcMhandle,
                                      size_t size, uint64_t dstOff, void* dstMhandle, uint32_t rank,
                                      uint64_t signalOff, void* signalMhandle, uint64_t signalValue,
                                      uint32_t signalOp, void** request) {
  if (signalOp != NCCL_NET_SIGNAL_OP_INC && signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("pcclGinIbProxyIPutSignal: Unsupported signalOp %u", signalOp);
    return ncclInvalidArgument;
  }

  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];

  struct ncclIbGinProxyMrHandle* srcMrHandle = (struct ncclIbGinProxyMrHandle*)srcMhandle;
  struct ncclIbGinProxyMrHandle* dstMrHandle = (struct ncclIbGinProxyMrHandle*)dstMhandle;
  struct ncclIbGinProxyMrHandle* signalMrHandle = (struct ncclIbGinProxyMrHandle*)signalMhandle;

  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
  struct ibv_qp* qp = comm->base.qps[0];

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  // See ncclGinIbProxyIPut. Only the atomic WQE is signaled, so one completion either way.
  req->events = 0;
  req->ginProxyCtx = ginProxyCtx;
  req->type = NCCL_NET_IB_REQ_GIN_IPUT;
  req->sock = &comm->base.sock;
  req->ginPeerRank = rank;
  req->devBase = &comm->dev.base;

  struct ibv_send_wr wr[2];
  memset(&wr, 0, sizeof(wr));
  struct ibv_sge sge[2];
  memset(&sge, 0, sizeof(sge));

  // If size is 0, we only need to send the signal. srcMrHandle must be non-NULL.
  if (size > 0 && dstMrHandle) {
    void* srcPtr = (void*)(srcMrHandle->base_vas[ginProxyCtx->rank] + srcOff);
    void* dstPtr = (void*)(dstMrHandle->base_vas[rank] + dstOff);
    uint32_t lkey = srcMrHandle->lkey;
    uint32_t rkey = dstMrHandle->rkeys[rank];

    wr[0].opcode              = IBV_WR_RDMA_WRITE;
    wr[0].send_flags          = 0;
    wr[0].wr_id               = req - comm->base.reqs;
    wr[0].next                = &wr[1];
    wr[0].wr.rdma.remote_addr = (uint64_t)dstPtr;
    wr[0].wr.rdma.rkey        = rkey;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;

    sge[0].addr = (uintptr_t)srcPtr;
    sge[0].length = size;
    sge[0].lkey = lkey;
  }

  void* signalPtr = (void*)(signalMrHandle->base_vas[rank] + signalOff);
  uint32_t signalRkey = signalMrHandle->rkeys[rank];

  wr[1].opcode                = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr[1].send_flags            = IBV_SEND_SIGNALED;
  wr[1].wr_id                 = req - comm->base.reqs;
  wr[1].next                  = NULL;
  wr[1].wr.atomic.remote_addr = (uint64_t)signalPtr;
  wr[1].wr.atomic.compare_add = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  wr[1].wr.atomic.rkey        = signalRkey;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;

  sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;
  sge[1].length = sizeof(comm->putSignalScratchpad);
  sge[1].lkey = comm->dev.putSignalScratchpadMr->lkey;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp, size > 0 ? &wr[0] : &wr[1], &bad_wr));
  ncclIbAddEvent(req, 0);
  *request = req;
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyTest(void* /*collComm*/, void* request, int* done) {
  struct ncclIbRequest* req = (struct ncclIbRequest*)request;
  struct ncclGinIbProxyCtx* ginProxyCtx = (struct ncclGinIbProxyCtx*)req->ginProxyCtx;
  uint32_t rank = req->ginPeerRank;
  *done = 0;

  if (req->events == 0) {
    *done = 1;
    NCCLCHECK(ncclIbFreeRequest(req));
    return ncclSuccess;
  }
  int wrDone = 0;
  struct ibv_wc wc[4];

  ncclIbNetCommBase* commBase;
  ncclIbNetCommDevBase* devBase;
  if (req->type == NCCL_NET_IB_REQ_FLUSH) {
    struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
    commBase = &comm->base;
    devBase = &comm->dev.base;
  } else {
    struct ncclIbSendComm* comm = (struct ncclIbSendComm*)ginProxyCtx->fullSendComm[rank];
    commBase = &comm->base;
    devBase = &comm->dev.base;
  }
  NCCLCHECK(wrap_ibv_poll_cq(devBase->cq, 4, wc, &wrDone));
  for (int i = 0; i < wrDone; i++) {
    if (wc[i].status != IBV_WC_SUCCESS) {
      union ncclSocketAddress addr;
      ncclSocketGetAddr(req->sock, &addr);
      char line[SOCKET_NAME_MAXLEN + 1];
      char* hcaName = devBase->pd->context->device->name;
      WARN("NET/IB/GIN: Got completion from peer %s with status=%d opcode=%d len=%u vendor err %u (%s) hca %s",
           ncclSocketToString(&addr, line), wc[i].status, wc[i].opcode, wc[i].byte_len,
           wc[i].vendor_err, ncclIbReqTypeStr[req->type], hcaName);
      return ncclRemoteError;
    }

    struct ncclIbRequest* wcReq = commBase->reqs + wc[i].wr_id;

    wcReq->events--;
    if (wcReq == req && wcReq->events == 0) {
      *done = 1;
      NCCLCHECK(ncclIbFreeRequest(wcReq));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclGinIbProxyIFlush(void* ginCtx, int context, void* mhandle, uint32_t rank, void** request) {
  struct ncclGinIbProxyCtx* ginProxyCtx = &((struct ncclGinIbProxyCtx*)ginCtx)[context];
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)ginProxyCtx->fullRecvComm[rank];
  struct ncclIbGinProxyMrHandle* ginMrHandle = (struct ncclIbGinProxyMrHandle*)mhandle;
  struct ibv_qp* qp = comm->gpuFlush.qp;

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  // See ncclGinIbProxyIPut.
  req->events = 0;
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  req->ginPeerRank = rank;
  req->ginProxyCtx = ginProxyCtx;
  req->devBase = &comm->dev.base;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;

  void* flushPtr = (void*)(ginMrHandle->base_vas[rank]);
  wr.wr.rdma.remote_addr = (uint64_t)flushPtr;
  wr.wr.rdma.rkey = ginMrHandle->rkeys[rank];
  wr.sg_list = &comm->gpuFlush.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(qp, &wr, &bad_wr));

  ncclIbAddEvent(req, 0);
  *request = req;
  return ncclSuccess;
}

ncclGin_t ncclGinIbProxy = {
  "GIN_IB_PROXY",
  ncclGinIbProxyInit,
  ncclGinIbProxyDevices,
  ncclGinIbProxyGetProperties,
  ncclGinIbProxyListen,
  ncclGinIbProxyConnect,
  ncclGinIbProxyCreateContext,
  ncclGinIbProxyRegMrSym,
  ncclGinIbProxyRegMrSymDmaBuf,
  ncclGinIbProxyDeregMrSym,
  ncclGinIbProxyDestroyContext,
  ncclGinIbCloseColl,
  ncclGinIbCloseListen,
  ncclGinIbProxyIPut,
  ncclGinIbProxyIPutSignal,
  ncclGinIbProxyIGet,
  ncclGinIbProxyIFlush,
  ncclGinIbProxyTest,
  NULL,
  NULL,
  ncclGinIbFinalize
};
