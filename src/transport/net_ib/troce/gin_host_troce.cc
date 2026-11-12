/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "alloc.h"
#include <assert.h>
#include "pccl_compat.h"
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <map>
#include <mutex>

#include "ibvwrap.h"
#include "gin/gin_host.h"
#include "gin_host_troce.h"
#include "trocedvwrap.h"
#include "param.h"

#include "pccl_device/gin/troce/gin_troce_device_host_common.h"
#include "../common.h"
#include "../connect.h"
#include "../gin.h"

#define MAX_PCI_ADDRESS_LEN 32U

extern int64_t ncclParamGinGdakiQpDepth();
extern int64_t ncclParamGinGdakiMaxDestRdAtomic();
extern int64_t ncclParamGinGdakiMaxQpRdAtomic();
extern int64_t ncclParamGinErrorQuerySec();

extern int64_t ncclParamIbPkey();
extern int64_t ncclParamIbTc();
extern int64_t ncclParamIbSl();
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamDmaBufEnable();
extern int64_t ncclParamIbPciRelaxedOrdering();

// ---------------------------------------------------------------------------
// Memory helpers
//
// MRs use IOVA zero, so WQE addresses are byte offsets within the MR.
// Registered buffers are page-aligned to satisfy the provider's IOVA constraint.
// ---------------------------------------------------------------------------

static constexpr size_t TROCE_PAGE_SIZE = 4096;

// ---------------------------------------------------------------------------
// NIC MMIO -> GPU mapping
//
// Map provider-returned SQ/CQ MMIO registers into the device address space.
// Reference counts allow queues to share the same MMIO register.
// ---------------------------------------------------------------------------

struct troceMmioMapping {
  void *devPtr;
  int refcount;
};

static std::mutex troceMmioMutex;
static std::map<void *, struct troceMmioMapping> troceMmioMap;

static ncclResult_t troceMapNicRegToGpu(void *reg, size_t size, void **outDevPtr) {
  std::lock_guard<std::mutex> lock(troceMmioMutex);
  std::map<void *, struct troceMmioMapping>::iterator it = troceMmioMap.find(reg);
  if (it != troceMmioMap.end()) {
    it->second.refcount++;
    *outDevPtr = it->second.devPtr;
    return ncclSuccess;
  }

  HGGCCHECK(hggcHostRegister(reg, size,
                             hggcHostRegisterPortable | hggcHostRegisterMapped |
                             hggcHostRegisterIoMemory));
  ncclResult_t ret = ncclSuccess;
  void *devPtr = nullptr;
  HGGCCHECKGOTO(hggcHostGetDevicePointer(&devPtr, reg, 0), ret, fail);

  troceMmioMap[reg] = {devPtr, 1};
  *outDevPtr = devPtr;
  return ncclSuccess;

fail:
  HGGCCHECKIGNORE(hggcHostUnregister(reg));
  return ret;
}

static ncclResult_t troceUnmapNicReg(void *reg) {
  if (reg == nullptr) {
    return ncclSuccess;
  }

  std::lock_guard<std::mutex> lock(troceMmioMutex);
  std::map<void *, struct troceMmioMapping>::iterator it = troceMmioMap.find(reg);
  if (it == troceMmioMap.end()) {
    return ncclSuccess;
  }
  if (it->second.refcount > 1) {
    it->second.refcount--;
    return ncclSuccess;
  }
  HGGCCHECK(hggcHostUnregister(reg));
  troceMmioMap.erase(it);
  return ncclSuccess;
}

static ncclResult_t troceAlignedHggcMalloc(void **ptr, size_t size) {
  size_t aligned = (size + TROCE_PAGE_SIZE - 1) & ~(TROCE_PAGE_SIZE - 1);
  if (aligned == 0) {
    aligned = TROCE_PAGE_SIZE;
  }
  HGGCCHECK(hggcMalloc(ptr, aligned));
  return ncclSuccess;
}

// Queue storage uses 64 KiB alignment and SYNC_MEMOPS for NIC access.
// Retain the original allocation pointer for deallocation.
static constexpr size_t TROCE_GPAGE_SIZE = 1ULL << 16;

static ncclResult_t troceCtrlAlloc(void **base, void **aligned, size_t size, int fillByte = 0) {
  size_t bufSize = size + TROCE_GPAGE_SIZE - 1;
  HGGCCHECK(hggcMalloc(base, bufSize));

  int attrVal = 1;
  HGCHECK(hgPointerSetAttribute(&attrVal, HG_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                (HGdeviceptr)*base));
  // CQ callers use 0xff to initialize the completion index to an invalid value.
  // Zero initialization could be interpreted as completion of the first WQE.
  HGGCCHECK(hggcMemset(*base, fillByte, bufSize));

  *aligned = (void *)(((uintptr_t)*base + TROCE_GPAGE_SIZE - 1) & ~(TROCE_GPAGE_SIZE - 1));
  return ncclSuccess;
}

static ncclResult_t troceGetDmaBufFd(void *addr, size_t length, int *fd) {
  if (ncclParamDmaBufEnable() == 0) {
    return ncclInvalidUsage;
  }

  static size_t hostPageSize = (size_t)sysconf(_SC_PAGESIZE);
  size_t alignedSize = ((length + hostPageSize - 1) / hostPageSize) * hostPageSize;

  if (ncclParamIbDataDirect()) {
    HGresult st = hgMemGetHandleForAddressRange(
      (void *)fd, (HGdeviceptr)addr, alignedSize, HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      HG_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (st == HGGC_SUCCESS) {
      return ncclSuccess;
    }
  }

  HGresult st = hgMemGetHandleForAddressRange(
    (void *)fd, (HGdeviceptr)addr, alignedSize, HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (st == HGGC_SUCCESS) {
    return ncclSuccess;
  }

  return ncclInvalidUsage;
}

static ncclResult_t troceRegMr(struct ibv_mr **mr, struct ibv_pd *pd, void *addr, size_t length,
                               int access) {
  int dmabufFd = -1;
  if (troceGetDmaBufFd(addr, length, &dmabufFd) == ncclSuccess) {
    ncclResult_t ret = wrap_ibv_reg_dmabuf_mr(mr, pd, 0ULL /*offset*/, length, 0ULL /*iova*/,
                                              dmabufFd, access);
    (void)close(dmabufFd);
    if (ret == ncclSuccess) {
      INFO(NCCL_NET, "troceRegMr: DMA-BUF addr=%p length=%zu lkey=0x%x rkey=0x%x", addr, length,
           (*mr)->lkey, (*mr)->rkey);
      return ncclSuccess;
    }
  }

  NCCLCHECK(wrap_ibv_reg_mr_iova2(mr, pd, addr, length, 0, (unsigned int)access));
  INFO(NCCL_NET, "troceRegMr: iova=0 addr=%p length=%zu lkey=0x%x rkey=0x%x", addr, length,
       (*mr)->lkey, (*mr)->rkey);
  return ncclSuccess;
}

// Host staging buffer with a device mirror for descriptor initialization.
template <typename T>
class TroceHostGpuMem {
 private:
  unsigned int numElements;

 public:
  T *hostBuf;
  T *gpuBuf;

  TroceHostGpuMem() : numElements(0), hostBuf(nullptr), gpuBuf(nullptr) {}

  ncclResult_t allocate(unsigned int n) {
    // Control collectives may register these host mirrors with verbs.
    NCCLCHECK(ncclIbMalloc((void**)&this->hostBuf, (size_t)n * sizeof(T)));
    HGGCCHECK(hggcMalloc((void **)&this->gpuBuf, (size_t)n * sizeof(T)));
    HGGCCHECK(hggcMemset(this->gpuBuf, 0, (size_t)n * sizeof(T)));
    this->numElements = n;
    return ncclSuccess;
  }

  ncclResult_t copyHtoD() {
    HGGCCHECK(hggcMemcpy(this->gpuBuf, this->hostBuf, (size_t)this->numElements * sizeof(T),
                         hggcMemcpyHostToDevice));
    return ncclSuccess;
  }

  ncclResult_t deallocate() {
    if (this->hostBuf) {
      NCCLCHECK(ncclFree(this->hostBuf));
      this->hostBuf = nullptr;
    }
    if (this->gpuBuf) {
      HGGCCHECK(hggcFree(this->gpuBuf));
      this->gpuBuf = nullptr;
    }
    return ncclSuccess;
  }
};

// Communicator-wide signal storage. Register the buffer, exchange
// remote keys, and assign each context a separate slice.
template <typename T>
class TroceGlobalTable {
 private:
  unsigned int numElements;
  unsigned int nextUnusedIdx;
  TroceHostGpuMem<__be32> rkeys;

 public:
  T *gpuPtr;
  struct ibv_mr *mr;

  TroceGlobalTable() : numElements(0), nextUnusedIdx(0), gpuPtr(nullptr), mr(nullptr) {}

  ncclResult_t allocate(unsigned int n, unsigned int nranks) {
    this->numElements = n;
    this->nextUnusedIdx = 0;
    if (n == 0) {
      return ncclSuccess;
    }
    NCCLCHECK(troceAlignedHggcMalloc((void **)&this->gpuPtr, (size_t)n * sizeof(T)));
    HGGCCHECK(hggcMemset(this->gpuPtr, 0, (size_t)n * sizeof(T)));
    NCCLCHECK(this->rkeys.allocate(nranks));
    return ncclSuccess;
  }

  ncclResult_t registerMr(struct ibv_pd *pd) {
    if (this->numElements == 0) {
      return ncclSuccess;
    }
    NCCLCHECK(troceRegMr(&this->mr, pd, this->gpuPtr, (size_t)this->numElements * sizeof(T),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                         IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC));
    return ncclSuccess;
  }

  ncclResult_t exchangeInfo(struct ncclGinIbCollComm *cComm) {
    if (this->numElements == 0) {
      return ncclSuccess;
    }
    __be32 rkey = htobe32(this->mr->rkey);
    NCCLCHECK(cComm->allGather(cComm, &rkey, this->rkeys.hostBuf, sizeof(__be32)));
    NCCLCHECK(this->rkeys.copyHtoD());
    return ncclSuccess;
  }

  ncclResult_t allocateElements(unsigned int n, unsigned int *outStartIdx) {
    if (this->nextUnusedIdx + n > this->numElements) {
      WARN("GIN/TROCE: not enough space in global table (%u requested, %u free)", n,
           this->numElements - this->nextUnusedIdx);
      return ncclInvalidUsage;
    }
    *outStartIdx = this->nextUnusedIdx;
    this->nextUnusedIdx += n;
    return ncclSuccess;
  }

  __be32 *rkeysDev() {
    return this->rkeys.gpuBuf;
  }

  ncclResult_t destroy() {
    if (this->mr) {
      NCCLCHECK(wrap_ibv_dereg_mr(this->mr));
      this->mr = nullptr;
    }
    if (this->gpuPtr) {
      HGGCCHECK(hggcFree(this->gpuPtr));
      this->gpuPtr = nullptr;
    }
    NCCLCHECK(this->rkeys.deallocate());
    return ncclSuccess;
  }
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct troceQpHost {
  struct ibv_qp *qp;
  struct ibv_cq *cq;      // send CQ, the only one the device polls
  struct ibv_cq *recvCq;  // Separate receive CQ required by the provider; not polled by GIN.

  // Provider queue descriptors retained for resource destruction.
  struct troce_qp_context qpCtx;
  struct troce_cq_context cqCtx;
  struct troce_cq_context recvCqCtx;

  // Ring buffers, all in device memory.
  void *sqBuf;
  void *rqBuf;
  void *cqBuf;
  void *recvCqBuf;
  void *sqBase;
  void *rqBase;
  void *cqBase;
  void *recvCqBase;
  size_t sqLen;
  size_t rqLen;
  size_t cqLen;

  // Tail registers mapped into the GPU address space.
  void *sqDbGpu;
  uint8_t trafficClass;
};

struct troceExchInfo {
  uint32_t qpn;
  uint32_t lid;
  uint8_t ibPort;
  uint64_t spn;
  uint64_t iid;
  enum ibv_mtu mtu;
  uint32_t sinkRkey;
};

struct troceContext {
  struct ibv_device_attr ibDevAttr;
  int hggcDev;
  uint8_t portNum;
  int gidIndex;
  enum ibv_mtu portMtu;  // local port active_mtu, negotiated to min() with the peer at RTR
  int nContexts;
  int nRanks;
  uint32_t sqDepth;

  struct ncclGinIbCollComm *collComm;
  ncclNetDeviceHandle_t *devHandle;
  struct ncclTroceFunctionTable *troce;

  // nContexts * nRanks queue pairs, indexed by context and then peer rank.
  struct troceQpHost *qps;

  TroceGlobalTable<uint64_t> *signalsTable;

  // Device-visible per-context descriptors and the per-context QP arrays they
  // point at.
  TroceHostGpuMem<struct pcclGinTroceGPUContext> *gpuCtx;
  TroceHostGpuMem<struct pcclGinTroceQp> *gpuQps;

  struct {
    void *addr;
    struct ibv_mr *mr;
  } sinkBuffer;

  uint64_t lastErrorQueryTime;
};

struct troceMemHandle {
  int type;
  struct ibv_mr *mr;
  TroceHostGpuMem<struct pcclGinTroceMemHandle> *mhandleHd;
  TroceHostGpuMem<__be32> *rkeysHd;
};

// Create an RC QP with device-resident queue storage through the provider's
// TROCE_ALLOC_GPU interface. Register its doorbells for device access.
static ncclResult_t troceCreateQp(struct troceContext *ctx, struct ibv_pd *pd,
                                  struct ibv_context *ibCtx, struct troceQpHost *out) {
  struct ibv_qp_init_attr qpInitAttr;
  // Apply the provider's minimum work-request capacity before sizing queue storage.
  const uint32_t minWqebb = (uint32_t)(TROCE_SQ_MIN_WQEBB / TROCE_SEND_WQE_BB_FACTOR);
  uint32_t numWqebb = ctx->sqDepth > minWqebb ? ctx->sqDepth : minWqebb;
  // SQ indices use bit masks, so their capacities must be
  // powers of two. Round the requested work-request capacity up accordingly.
  if (numWqebb & (numWqebb - 1)) {
    uint32_t pow2 = 1u;
    while (pow2 < numWqebb) {
      pow2 <<= 1;
    }
    INFO(NCCL_NET, "GIN/TROCE: rounding queue depth %u up to %u (must be a power of two)",
         numWqebb, pow2);
    numWqebb = pow2;
  }
  // Single-slot completion mode overwrites CQE zero with the latest completion.
  // Allocate a page-aligned buffer for this entry and enable IGNORE_OVERRUN.
  const int numCqe = 1;

  out->sqLen = ((size_t)numWqebb * TROCE_SEND_WQE_BB * TROCE_SEND_WQE_BB_FACTOR +
                TROCE_SQ_ALIGN - 1) & ~(TROCE_SQ_ALIGN - 1);
  out->rqLen = ((size_t)numWqebb * TROCE_RECV_WQE_BB + TROCE_RQ_ALIGN - 1) &
               ~(TROCE_RQ_ALIGN - 1);
  out->cqLen = TROCE_PAGE_SIZE;

  NCCLCHECK(troceCtrlAlloc(&out->sqBase, &out->sqBuf, out->sqLen));
  NCCLCHECK(troceCtrlAlloc(&out->rqBase, &out->rqBuf, out->rqLen));
  NCCLCHECK(troceCtrlAlloc(&out->cqBase, &out->cqBuf, out->cqLen, /*fillByte=*/0xff));

  // The provider requires separate send and receive CQs. Device progress polls
  // only the send CQ; the receive CQ remains unused by GIN.
  memset(&out->cqCtx, 0, sizeof(out->cqCtx));
  out->cqCtx.alloc_type = TROCE_ALLOC_GPU;
  out->cqCtx.addr = out->cqBuf;
  out->cqCtx.len = out->cqLen;
  out->cqCtx.flags = TROCE_CQ_ATTR_IGNORE_OVERRUN;
  out->cq = ctx->troce->troce_create_cq_with_context(ibCtx, numCqe, nullptr, 0, &out->cqCtx);
  if (out->cq == nullptr) {
    WARN("GIN/TROCE: troce_create_cq_with_context(send) failed (numCqe=%d len=%zu): %s", numCqe,
         out->cqLen, strerror(errno));
    return ncclSystemError;
  }

  NCCLCHECK(troceCtrlAlloc(&out->recvCqBase, &out->recvCqBuf, out->cqLen, /*fillByte=*/0xff));
  memset(&out->recvCqCtx, 0, sizeof(out->recvCqCtx));
  out->recvCqCtx.alloc_type = TROCE_ALLOC_GPU;
  out->recvCqCtx.addr = out->recvCqBuf;
  out->recvCqCtx.len = out->cqLen;
  out->recvCqCtx.flags = TROCE_CQ_ATTR_IGNORE_OVERRUN;
  out->recvCq =
    ctx->troce->troce_create_cq_with_context(ibCtx, numCqe, nullptr, 0, &out->recvCqCtx);
  if (out->recvCq == nullptr) {
    WARN("GIN/TROCE: troce_create_cq_with_context(recv) failed (numCqe=%d len=%zu): %s", numCqe,
         out->cqLen, strerror(errno));
    return ncclSystemError;
  }

  memset(&qpInitAttr, 0, sizeof(qpInitAttr));
  qpInitAttr.send_cq = out->cq;
  qpInitAttr.recv_cq = out->recvCq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = numWqebb;
  qpInitAttr.cap.max_recv_wr = numWqebb;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;

  memset(&out->qpCtx, 0, sizeof(out->qpCtx));
  out->qpCtx.alloc_type = TROCE_ALLOC_GPU;
  out->qpCtx.sq_addr = out->sqBuf;
  out->qpCtx.sq_len = out->sqLen;
  out->qpCtx.rq_addr = out->rqBuf;
  out->qpCtx.rq_len = out->rqLen;
  out->qp = ctx->troce->troce_create_qp_with_context(pd, &qpInitAttr, &out->qpCtx);
  if (out->qp == nullptr) {
    // The provider does not guarantee errno on failure. Enable TROCE_DEBUG_FILE
    // for provider diagnostics instead of interpreting a potentially stale errno.
    WARN("GIN/TROCE: troce_create_qp_with_context returned NULL (errno not meaningful). "
         "sqAddr=%p sqLen=%zu rqAddr=%p rqLen=%zu wqebb=%u maxSendWr=%u maxRecvWr=%u. "
      "Re-run with TROCE_DEBUG_FILE=/dev/stderr for the driver-side reason.",
      out->sqBuf, out->sqLen, out->rqBuf, out->rqLen, numWqebb,
      qpInitAttr.cap.max_send_wr, qpInitAttr.cap.max_recv_wr);
    return ncclSystemError;
  }

  // Map the provider's SQ and CQ doorbells for device access.
  NCCLCHECK(troceMapNicRegToGpu(out->qpCtx.sq_tail_reg, TROCE_SQ_UAR_SIZE, &out->sqDbGpu));

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = ctx->portNum;
  qpAttr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  NCCLCHECK(wrap_ibv_modify_qp(
    out->qp, &qpAttr,
    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return ncclSuccess;
}

// Query the QP traffic class after RTR and pass it to the device context.
// Reject initialization if the provider cannot supply the requested value.
static ncclResult_t troceQueryTrafficClass(struct troceContext *ctx, struct troceQpHost *qp) {
  struct trocedv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.attr_mask = TROCEDV_QP_ATTR_TC;
  if (ctx->troce->trocedv_query_qp_attr(qp->qp, &attr) != 0) {
    WARN("GIN/TROCE: trocedv_query_qp_attr failed for qpn=%#x", qp->qp->qp_num);
    return ncclSystemError;
  }
  if (!(attr.actual_mask & TROCEDV_QP_ATTR_TC)) {
    WARN("GIN/TROCE: trocedv_query_qp_attr did not fill TC for qpn=%#x (actual_mask=%#x)",
         qp->qp->qp_num, attr.actual_mask);
    return ncclSystemError;
  }
  qp->trafficClass = attr.tc;
  return ncclSuccess;
}

static ncclResult_t troceDestroyQp(struct troceContext *ctx, struct troceQpHost *qp) {
  if (qp->qp) {
    NEQCHECK(ctx->troce->troce_destroy_qp_with_context(qp->qp, &qp->qpCtx), 0);
    qp->qp = nullptr;
  }
  if (qp->cq) {
    NEQCHECK(ctx->troce->troce_destroy_cq_with_context(qp->cq, &qp->cqCtx), 0);
    qp->cq = nullptr;
  }
  if (qp->recvCq) {
    NEQCHECK(ctx->troce->troce_destroy_cq_with_context(qp->recvCq, &qp->recvCqCtx), 0);
    qp->recvCq = nullptr;
  }
  NCCLCHECK(troceUnmapNicReg(qp->qpCtx.sq_tail_reg));
  qp->qpCtx.sq_tail_reg = nullptr;
  if (qp->sqBase) {
    HGGCCHECK(hggcFree(qp->sqBase));
    qp->sqBase = nullptr;
    qp->sqBuf = nullptr;
  }
  if (qp->rqBase) {
    HGGCCHECK(hggcFree(qp->rqBase));
    qp->rqBase = nullptr;
    qp->rqBuf = nullptr;
  }
  if (qp->cqBase) {
    HGGCCHECK(hggcFree(qp->cqBase));
    qp->cqBase = nullptr;
    qp->cqBuf = nullptr;
  }
  if (qp->recvCqBase) {
    HGGCCHECK(hggcFree(qp->recvCqBase));
    qp->recvCqBase = nullptr;
    qp->recvCqBuf = nullptr;
  }
  return ncclSuccess;
}

static ncclResult_t troceConnectQp(struct troceContext *ctx, struct troceQpHost *qp,
                                   struct troceExchInfo *remote, int trafficClass) {
  struct ncclIbQpInfo info;
  memset(&info, 0, sizeof(info));
  info.lid = remote->lid;
  info.ib_port = ctx->portNum;
  info.spn = remote->spn;
  info.iid = remote->iid;
  // Use the smaller local and remote active MTU for a consistent RC path MTU.
  info.mtu = (remote->mtu < ctx->portMtu) ? remote->mtu : ctx->portMtu;
  int64_t requestedMaxDestRdAtomic = ncclParamGinGdakiMaxDestRdAtomic();
  int64_t requestedMaxRdAtomic = ncclParamGinGdakiMaxQpRdAtomic();
  // Nonpositive values select the device limit, as in the GDAKI configuration.
  // Keep the requester and responder bounded by their respective capabilities.
  int64_t maxDestRdAtomic = ctx->ibDevAttr.max_qp_rd_atom;
  int64_t maxRdAtomic = ctx->ibDevAttr.max_qp_init_rd_atom;
  if (requestedMaxDestRdAtomic > 0) maxDestRdAtomic = std::min(requestedMaxDestRdAtomic, maxDestRdAtomic);
  if (requestedMaxRdAtomic > 0) maxRdAtomic = std::min(requestedMaxRdAtomic, maxRdAtomic);
  if (maxDestRdAtomic < 1 || maxDestRdAtomic > UINT8_MAX || maxRdAtomic < 1 || maxRdAtomic > UINT8_MAX) {
    WARN("GIN/TROCE: invalid Read/atomic concurrency (responder=%lld requester=%lld)",
         (long long)maxDestRdAtomic, (long long)maxRdAtomic);
    return ncclInvalidArgument;
  }
  const int sl = ncclParamIbSl() != -1 ? ncclParamIbSl() :
                   trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF ? trafficClass : 0;
  const int tc = ncclParamIbTc() != -1 ? ncclParamIbTc() :
                   trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF ? trafficClass : 0;
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = info.mtu;
  attr.dest_qp_num = remote->qpn;
  attr.max_dest_rd_atomic = (uint8_t)maxDestRdAtomic;
  attr.min_rnr_timer = 12;
  attr.ah_attr.sl = sl;
  attr.ah_attr.port_num = info.ib_port;
  if (info.lid == 0) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid.global.subnet_prefix = info.spn;
    attr.ah_attr.grh.dgid.global.interface_id = info.iid;
    attr.ah_attr.grh.sgid_index = ctx->gidIndex;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.grh.traffic_class = tc;
  } else {
    attr.ah_attr.dlid = info.lid;
  }
  NCCLCHECK(wrap_ibv_modify_qp(qp->qp, &attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                               IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
                               IBV_QP_MIN_RNR_TIMER));
  // Query the traffic class established by RTR for the device doorbell encoding.
  NCCLCHECK(troceQueryTrafficClass(ctx, qp));
  NCCLCHECK(ncclIbRtsQp(qp->qp, (uint8_t)maxRdAtomic));
  INFO(NCCL_NET, "GIN/TROCE QP %#x Read/atomic concurrency: requester=%d responder=%d",
       qp->qp->qp_num, (int)maxRdAtomic, (int)maxDestRdAtomic);
  return ncclSuccess;
}

static void troceFillExchInfo(struct troceExchInfo *info, struct troceContext *ctx,
                              struct ibv_port_attr *portAttr, union ibv_gid *gid,
                              struct ibv_qp *qp) {
  info->qpn = qp->qp_num;
  info->lid = portAttr->lid;
  info->ibPort = ctx->portNum;
  info->spn = gid->global.subnet_prefix;
  info->iid = gid->global.interface_id;
  info->mtu = portAttr->active_mtu;
}

ncclResult_t ncclGinTroceCreateContext(void *collComm, int nSignals, int nCounters, int nContexts,
                                       int queueDepth, int trafficClass, void **outGinCtx,
                                       ncclNetDeviceHandle_t **outDevHandle) {
  if (nCounters != 0) {
    WARN("GIN/TROCE does not support counter resources");
    return ncclInvalidUsage;
  }
  ncclResult_t status = ncclSuccess;
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  const int rank = cComm->rank;
  const int nranks = cComm->nranks;
  const int nqps = nContexts * nranks;
  int64_t sqDepth = std::max<int64_t>(1024, queueDepth > 0 ? queueDepth : ncclParamGinGdakiQpDepth());
  const uint32_t maxSqDepth = PCCL_GIN_TROCE_MAX_SQ_WQES / TROCE_SEND_WQE_BB_FACTOR;
  if (sqDepth > maxSqDepth) {
    WARN("GIN/TROCE: requested SQ depth %lld exceeds maximum %u", (long long)sqDepth, maxSqDepth);
    return ncclInvalidArgument;
  }

  struct troceContext *ctx = nullptr;
  ncclNetDeviceHandle_t *devHandle = nullptr;
  struct troceExchInfo *localExch = nullptr;
  struct troceExchInfo *remoteExch = nullptr;
  struct ibv_port_attr portAttr;
  union ibv_gid gid;
  uint64_t *sinkBuffer = nullptr;
  struct ibv_mr *sinkMr = nullptr;

  TroceHostGpuMem<struct pcclGinTroceGPUContext> *gpuCtx =
    new TroceHostGpuMem<struct pcclGinTroceGPUContext>();
  TroceHostGpuMem<struct pcclGinTroceQp> *gpuQps = new TroceHostGpuMem<struct pcclGinTroceQp>();
  TroceGlobalTable<uint64_t> *signalsTable = new TroceGlobalTable<uint64_t>();

  NCCLCHECKGOTO(ncclCalloc(&ctx, 1), status, fail);

  ctx->collComm = cComm;
  ctx->nContexts = nContexts;
  ctx->nRanks = nranks;
  ctx->portNum = ncclIbDevs[cComm->dev].port;
  ctx->sqDepth = (uint32_t)sqDepth;
  ctx->signalsTable = signalsTable;
  ctx->gpuCtx = gpuCtx;
  ctx->gpuQps = gpuQps;

  HGGCCHECKGOTO(hggcGetDevice(&ctx->hggcDev), status, fail);
  NCCLCHECKGOTO(wrap_ibv_query_device(cComm->ib.context, &ctx->ibDevAttr), status, fail);

  // Resolve the optional provider queue interface before allocating resources.
  NCCLCHECKGOTO(ncclTroceFtableInit(&ctx->troce), status, fail);

  NCCLCHECKGOTO(gpuCtx->allocate(nContexts), status, fail);
  NCCLCHECKGOTO(gpuQps->allocate(nqps), status, fail);
  NCCLCHECKGOTO(signalsTable->allocate(nSignals * nContexts, nranks), status, fail);

  NCCLCHECKGOTO(signalsTable->registerMr(cComm->ib.pd), status, fail);
  NCCLCHECKGOTO(signalsTable->exchangeInfo(cComm), status, fail);

  // Atomic return values are not consumed; all QPs share the sink.
  NCCLCHECKGOTO(troceAlignedHggcMalloc((void **)&sinkBuffer, sizeof(uint64_t)), status, fail);
  HGGCCHECKGOTO(hggcMemset(sinkBuffer, 0, sizeof(uint64_t)), status, fail);
  NCCLCHECKGOTO(troceRegMr(&sinkMr, cComm->ib.pd, sinkBuffer, sizeof(uint64_t),
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC),
                status, fail);
  ctx->sinkBuffer.addr = sinkBuffer;
  ctx->sinkBuffer.mr = sinkMr;

  NCCLCHECKGOTO(wrap_ibv_query_port(cComm->ib.context, ctx->portNum, &portAttr), status, fail);
  ctx->portMtu = portAttr.active_mtu;
  NCCLCHECKGOTO(cComm->getGidIndex(cComm->ib.context, ctx->portNum, &portAttr, &ctx->gidIndex),
                status, fail);
  NCCLCHECKGOTO(wrap_ibv_query_gid(cComm->ib.context, ctx->portNum, ctx->gidIndex, &gid), status,
                fail);

  NCCLCHECKGOTO(ncclCalloc(&ctx->qps, nqps), status, fail);
  NCCLCHECKGOTO(ncclCalloc(&localExch, nranks), status, fail);
  NCCLCHECKGOTO(ncclCalloc(&remoteExch, nranks), status, fail);

  for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
    NCCLCHECKGOTO(troceCreateQp(ctx, cComm->ib.pd, cComm->ib.context, &ctx->qps[qpIdx]), status,
                  fail);
  }

  // Full mesh: for every context, exchange one QP per peer and connect it.
  for (int ctxIdx = 0; ctxIdx < nContexts; ctxIdx++) {
    for (int peer = 0; peer < nranks; peer++) {
      troceFillExchInfo(&localExch[peer], ctx, &portAttr, &gid,
                        ctx->qps[ctxIdx * nranks + peer].qp);
      localExch[peer].sinkRkey = sinkMr->rkey;
    }
    NCCLCHECKGOTO(cComm->allToAll(cComm, localExch, remoteExch, sizeof(struct troceExchInfo)),
                  status, fail);
    for (int peer = 0; peer < nranks; peer++) {
      NCCLCHECKGOTO(troceConnectQp(ctx, &ctx->qps[ctxIdx * nranks + peer], &remoteExch[peer], trafficClass),
                    status, fail);
      struct pcclGinTroceQp *devQp = &gpuQps->hostBuf[ctxIdx * nranks + peer];
      devQp->sink_rkey = htobe32(remoteExch[peer].sinkRkey);
      INFO(NCCL_NET, "[%d] GIN/TROCE connected ctx=%d peer=%d qpn=%#x -> remote qpn=%#x tc=%u", rank,
           ctxIdx, peer, ctx->qps[ctxIdx * nranks + peer].qp->qp_num, remoteExch[peer].qpn,
           ctx->qps[ctxIdx * nranks + peer].trafficClass);
    }
  }

  // Build the device-visible view.
  for (int ctxIdx = 0; ctxIdx < nContexts; ctxIdx++) {
    struct pcclGinTroceGPUContext *devCtx = &gpuCtx->hostBuf[ctxIdx];
    unsigned int bufferStart = 0;

    devCtx->qps = gpuQps->gpuBuf + (size_t)ctxIdx * nranks;
    devCtx->sink_buffer_lkey = htobe32(sinkMr->lkey);

    if (nSignals) {
      NCCLCHECKGOTO(signalsTable->allocateElements(nSignals, &bufferStart), status, fail);
      devCtx->signals_table.buffer = signalsTable->gpuPtr + bufferStart;
      devCtx->signals_table.rkeys = signalsTable->rkeysDev();
      devCtx->signals_table.lkey = htobe32(signalsTable->mr->lkey);
      devCtx->signals_table.offset = bufferStart;
    }

    for (int peer = 0; peer < nranks; peer++) {
      int qpIdx = ctxIdx * nranks + peer;
      struct pcclGinTroceQp *devQp = &gpuQps->hostBuf[qpIdx];
      devQp->qpn = ctx->qps[qpIdx].qp->qp_num;
      // Convert SQ storage bytes to WQEBB capacity for device-side index masks.
      // Queue sizing guarantees a power-of-two capacity.
      devQp->nwqes = (uint32_t)(ctx->qps[qpIdx].sqLen / TROCE_SEND_WQE_BB);
      devQp->cqn = ctx->qps[qpIdx].cqCtx.cqn;
      devQp->traffic_class = ctx->qps[qpIdx].trafficClass;
      // Device-resident queues and mapped doorbells support direct submission.
      devQp->sq_wqe = ctx->qps[qpIdx].sqBuf;
      devQp->sq_db = ctx->qps[qpIdx].sqDbGpu;
      devQp->cq_cqe = ctx->qps[qpIdx].cqBuf;
      devQp->resv_head = 0;
      devQp->ready_head = 0;
      devQp->prod_idx = 0;
      devQp->cons_idx = 0;
      devQp->progress_lock = 0;
      devQp->db_lock = 0;
    }
  }

  NCCLCHECKGOTO(gpuQps->copyHtoD(), status, fail);
  NCCLCHECKGOTO(gpuCtx->copyHtoD(), status, fail);

  NCCLCHECKGOTO(ncclCalloc(&devHandle, 1), status, fail);
  devHandle->netDeviceType = PCCL_NET_DEVICE_GIN_TROCE;
  devHandle->netDeviceVersion = PCCL_GIN_TROCE_VERSION;
  devHandle->handle = gpuCtx->gpuBuf;
  devHandle->size = sizeof(pcclGinTroceGPUContext) * nContexts;
  devHandle->needsProxyProgress = 0;
  ctx->devHandle = devHandle;

  INFO(NCCL_INIT | NCCL_NET,
       "GIN/TROCE context ready: rank=%d nranks=%d nContexts=%d nSignals=%d nCounters=%d "
       "sqDepth=%u gidIndex=%d sqBuf=%p sqDb=%p cqBuf=%p",
       rank, nranks, nContexts, nSignals, nCounters, ctx->sqDepth, ctx->gidIndex,
       ctx->qps[0].sqBuf, ctx->qps[0].sqDbGpu, ctx->qps[0].cqBuf);

  NCCLCHECK(ncclFree(localExch));
  NCCLCHECK(ncclFree(remoteExch));
  *outGinCtx = ctx;
  *outDevHandle = devHandle;
  return ncclSuccess;

fail:
  if (localExch) {
    NCCLCHECKIGNORE(ncclFree(localExch), status);
  }
  if (remoteExch) {
    NCCLCHECKIGNORE(ncclFree(remoteExch), status);
  }
  if (devHandle) {
    NCCLCHECKIGNORE(ncclFree(devHandle), status);
  }
  if (ctx) {
    if (ctx->qps) {
      ncclResult_t cleanupStatus = ncclSuccess;
      for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
        if (ctx->troce) {
          NCCLCHECKIGNORE(troceDestroyQp(ctx, &ctx->qps[qpIdx]), cleanupStatus);
        }
      }
      if (cleanupStatus != ncclSuccess) {
        NCCLCHECKIGNORE(cleanupStatus, status);
        WARN("GIN/TROCE: retaining context after queue cleanup failure");
        return status;
      }
      NCCLCHECKIGNORE(ncclFree(ctx->qps), status);
    }
  }
  if (sinkMr) {
    ncclResult_t cleanupStatus = wrap_ibv_dereg_mr(sinkMr);
    NCCLCHECKIGNORE(cleanupStatus, status);
    if (cleanupStatus != ncclSuccess) {
      WARN("GIN/TROCE: retaining context and sink storage after MR deregistration failure");
      return status;
    }
  }
  if (sinkBuffer) {
    HGGCCHECKIGNORE(hggcFree(sinkBuffer));
  }
  NCCLCHECKIGNORE(signalsTable->destroy(), status);
  NCCLCHECKIGNORE(gpuCtx->deallocate(), status);
  NCCLCHECKIGNORE(gpuQps->deallocate(), status);
  delete signalsTable;
  delete gpuCtx;
  delete gpuQps;
  NCCLCHECKIGNORE(ncclFree(ctx), status);
  return status;
}

ncclResult_t ncclGinTroceDestroyContext(void *ginCtx) {
  if (!ginCtx) {
    return ncclInvalidArgument;
  }
  struct troceContext *ctx = (struct troceContext *)ginCtx;
  const int nqps = ctx->nContexts * ctx->nRanks;

  if (ctx->qps) {
    for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
      NCCLCHECK(troceDestroyQp(ctx, &ctx->qps[qpIdx]));
    }
    NCCLCHECK(ncclFree(ctx->qps));
    ctx->qps = nullptr;
  }

  if (ctx->signalsTable) {
    NCCLCHECK(ctx->signalsTable->destroy());
    delete ctx->signalsTable;
  }
  if (ctx->gpuQps) {
    NCCLCHECK(ctx->gpuQps->deallocate());
    delete ctx->gpuQps;
  }
  if (ctx->gpuCtx) {
    NCCLCHECK(ctx->gpuCtx->deallocate());
    delete ctx->gpuCtx;
  }

  if (ctx->sinkBuffer.mr) {
    NCCLCHECK(wrap_ibv_dereg_mr(ctx->sinkBuffer.mr));
  }
  if (ctx->sinkBuffer.addr) {
    HGGCCHECK(hggcFree(ctx->sinkBuffer.addr));
  }
  if (ctx->devHandle) {
    NCCLCHECK(ncclFree(ctx->devHandle));
  }

  memset(ctx, 0, sizeof(*ctx));
  NCCLCHECK(ncclFree(ctx));
  return ncclSuccess;
}

ncclResult_t ncclGinTroceRegMrSym(void *collComm, void *data, size_t size, int type,
                                  uint64_t /*mr_flags*/, void **mhandle, void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  ncclResult_t status = ncclSuccess;
  struct ibv_mr *mr = nullptr;
  __be32 rkey;

  TroceHostGpuMem<struct pcclGinTroceMemHandle> *mhandleHd =
    new TroceHostGpuMem<struct pcclGinTroceMemHandle>();
  TroceHostGpuMem<__be32> *rkeysHd = new TroceHostGpuMem<__be32>();

  struct troceMemHandle *handle = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&handle, 1), status, fail);

  NCCLCHECKGOTO(troceRegMr(&mr, cComm->ib.pd, data, size,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC),
                status, fail);

  rkey = htobe32(mr->rkey);
  NCCLCHECKGOTO(rkeysHd->allocate(cComm->nranks), status, fail);
  NCCLCHECKGOTO(cComm->allGather(cComm, &rkey, rkeysHd->hostBuf, sizeof(__be32)), status, fail);
  NCCLCHECKGOTO(rkeysHd->copyHtoD(), status, fail);

  NCCLCHECKGOTO(mhandleHd->allocate(1), status, fail);
  mhandleHd->hostBuf->rkeys = rkeysHd->gpuBuf;
  mhandleHd->hostBuf->lkey = htobe32(mr->lkey);
  NCCLCHECKGOTO(mhandleHd->copyHtoD(), status, fail);

  handle->type = type;
  handle->mr = mr;
  handle->mhandleHd = mhandleHd;
  handle->rkeysHd = rkeysHd;

  *mhandle = (void *)handle;
  *ginHandle = (void *)mhandleHd->gpuBuf;
  return ncclSuccess;

fail:
  NCCLCHECKIGNORE(mhandleHd->deallocate(), status);
  NCCLCHECKIGNORE(rkeysHd->deallocate(), status);
  delete mhandleHd;
  delete rkeysHd;
  if (mr) {
    NCCLCHECKIGNORE(wrap_ibv_dereg_mr(mr), status);
  }
  if (handle) {
    NCCLCHECKIGNORE(ncclFree(handle), status);
  }
  return status;
}

ncclResult_t ncclGinTroceDeregMrSym(void * /*collComm*/, void *mhandle) {
  struct troceMemHandle *handle = (struct troceMemHandle *)mhandle;
  if (!handle) {
    return ncclInvalidArgument;
  }

  NCCLCHECK(wrap_ibv_dereg_mr(handle->mr));
  NCCLCHECK(handle->mhandleHd->deallocate());
  delete handle->mhandleHd;
  NCCLCHECK(handle->rkeysHd->deallocate());
  delete handle->rkeysHd;

  memset(handle, 0, sizeof(*handle));
  NCCLCHECK(ncclFree(handle));
  return ncclSuccess;
}

ncclResult_t ncclGinTroceProgress(void * /*ginCtx*/) {
  // Device code performs submission and CQ progress; no host progress is required.
  return ncclSuccess;
}

ncclResult_t ncclGinTroceQueryLastError(void *ginCtx, bool *hasError) {
  struct troceContext *ctx = (struct troceContext *)ginCtx;
  bool err = false;
  const int nqps = ctx->nContexts * ctx->nRanks;

  *hasError = false;

  // Limit full QP-state scans to the configured error-query interval.
  uint64_t now = clockNano();
  if ((now - ctx->lastErrorQueryTime) / 1e9 < ncclParamGinErrorQuerySec()) {
    return ncclSuccess;
  }
  ctx->lastErrorQueryTime = now;

  for (int qpIdx = 0; qpIdx < nqps && !err; qpIdx++) {
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr initAttr;
    memset(&attr, 0, sizeof(attr));
    memset(&initAttr, 0, sizeof(initAttr));
    if (wrap_ibv_query_qp(ctx->qps[qpIdx].qp, &attr, IBV_QP_STATE, &initAttr) != ncclSuccess) {
      err = true;
      break;
    }
    if (attr.qp_state == IBV_QPS_ERR) {
      WARN("GIN/TROCE: QP %#x is in ERROR state", ctx->qps[qpIdx].qp->qp_num);
      err = true;
    }
  }

  *hasError = err;
  return ncclSuccess;
}
