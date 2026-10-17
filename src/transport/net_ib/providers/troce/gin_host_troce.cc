/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include "pccl_compat.h"
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <map>
#include <mutex>

#include "ibvwrap.h"
#include "gin/gin_host.h"
#include "gin_host_troce.h"
#include "trocedvwrap.h"
#include "param.h"

#include "pccl_device/gin/providers/troce/gin_troce_device_host_common.h"
#include "../../common.h"
#include "../../connect.h"
#include "../../gin.h"

#define MAX_PCI_ADDRESS_LEN 32U

// SQ depth in WQEs. 1024 matches the depth sailSHMEM's GDAKI_TROCE transport uses on
// troce, which is a parameter set the driver is known to accept.
NCCL_PARAM(GinGdakiTroceQpDepth, "GIN_GDAKI_TROCE_QP_DEPTH", 1024);
NCCL_PARAM(GinGdakiTroceAtomicConcurrency, "GIN_GDAKI_TROCE_ATOMIC_CONCURRENCY", 1);
NCCL_PARAM(GinGdakiTroceErrorQuerySec, "GIN_GDAKI_TROCE_ERROR_QUERY_SEC", 10);
// Debug-only: issue one intentionally-invalid troce QP creation to verify that
// the driver's troce_err() diagnostics actually reach this process.
NCCL_PARAM(GinGdakiTroceProbeDiag, "GIN_GDAKI_TROCE_PROBE_DIAG", 0);

extern int64_t ncclParamIbPkey();
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamDmaBufEnable();
extern int64_t ncclParamIbPciRelaxedOrdering();

// ---------------------------------------------------------------------------
// Memory helpers
//
// As in GDAKI, MRs are registered with iova=0 so a WQE remote address is a byte
// offset inside the MR. The NIC requires (iova ^ addr) & (PAGE_SIZE-1) == 0, so
// registered buffers must be page aligned.
// ---------------------------------------------------------------------------

static constexpr size_t GDAKI_TROCE_PAGE_SIZE = 4096;

// ---------------------------------------------------------------------------
// NIC MMIO -> GPU mapping
//
// The SQ/CQ tail registers come back as host MMIO addresses; the GPU can only
// ring them once they are pinned as I/O memory and mapped into the device
// address space. Queues share register pages, so mappings are refcounted.
// ---------------------------------------------------------------------------

struct gdakiTroceMmioMapping {
  void *devPtr;
  int refcount;
};

static std::mutex gdakiTroceMmioMutex;
static std::map<void *, struct gdakiTroceMmioMapping> gdakiTroceMmioMap;

static ncclResult_t gdakiTroceMapNicRegToGpu(void *reg, void **outDevPtr) {
  size_t pageSize = (size_t)sysconf(_SC_PAGESIZE);
  void *pageAddr = (void *)((uintptr_t)reg & ~(pageSize - 1));
  size_t offsetInPage = (uintptr_t)reg - (uintptr_t)pageAddr;

  std::lock_guard<std::mutex> lock(gdakiTroceMmioMutex);
  auto it = gdakiTroceMmioMap.find(pageAddr);
  if (it != gdakiTroceMmioMap.end()) {
    it->second.refcount++;
    *outDevPtr = (void *)((uintptr_t)it->second.devPtr + offsetInPage);
    return ncclSuccess;
  }

  HGGCCHECK(hggcHostRegister(pageAddr, pageSize,
                             hggcHostRegisterPortable | hggcHostRegisterMapped |
                               hggcHostRegisterIoMemory));
  void *pageDevPtr = nullptr;
  HGGCCHECK(hggcHostGetDevicePointer(&pageDevPtr, pageAddr, 0));

  gdakiTroceMmioMap[pageAddr] = {pageDevPtr, 1};
  *outDevPtr = (void *)((uintptr_t)pageDevPtr + offsetInPage);
  return ncclSuccess;
}

static void gdakiTroceUnmapNicReg(void *reg) {
  if (reg == nullptr) return;
  size_t pageSize = (size_t)sysconf(_SC_PAGESIZE);
  void *pageAddr = (void *)((uintptr_t)reg & ~(pageSize - 1));

  std::lock_guard<std::mutex> lock(gdakiTroceMmioMutex);
  auto it = gdakiTroceMmioMap.find(pageAddr);
  if (it == gdakiTroceMmioMap.end()) return;
  if (--it->second.refcount > 0) return;
  hggcHostUnregister(pageAddr);
  gdakiTroceMmioMap.erase(it);
}

static ncclResult_t gdakiTroceAlignedHggcMalloc(void **ptr, size_t size) {
  size_t aligned = (size + GDAKI_TROCE_PAGE_SIZE - 1) & ~(GDAKI_TROCE_PAGE_SIZE - 1);
  if (aligned == 0) aligned = GDAKI_TROCE_PAGE_SIZE;
  HGGCCHECK(hggcMalloc(ptr, aligned));
  return ncclSuccess;
}

// NIC control memory (SQ/RQ/CQ rings) has stricter requirements than a plain
// device buffer: troce wants it aligned to a 64KB GPU page, and the pages must
// have SYNC_MEMOPS set so NIC DMA and GPU accesses stay coherent. Over-allocate
// and align by hand, keeping the base pointer around for the free path (this
// mirrors sailSHMEM's ibgda_gpu_mem_alloc).
static constexpr size_t GDAKI_TROCE_GPAGE_SIZE = 1ULL << 16;

static ncclResult_t gdakiTroceCtrlAlloc(void **base, void **aligned, size_t size, int fillByte = 0) {
  size_t bufSize = size + GDAKI_TROCE_GPAGE_SIZE - 1;
  HGGCCHECK(hggcMalloc(base, bufSize));

  int attrVal = 1;
  HGCHECK(hgPointerSetAttribute(&attrVal, HG_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                (HGdeviceptr)*base));
  // CQ buffers must be pre-filled with 0xff, not zeroes: the device-side poll
  // derives CQE validity from the owner bit, and an all-zero CQE is
  // indistinguishable from a completion the NIC has already written. sailSHMEM
  // initialises its CQ memory the same way.
  HGGCCHECK(hggcMemset(*base, fillByte, bufSize));

  *aligned = (void *)(((uintptr_t)*base + GDAKI_TROCE_GPAGE_SIZE - 1) & ~(GDAKI_TROCE_GPAGE_SIZE - 1));
  return ncclSuccess;
}

static ncclResult_t gdakiTroceGetDmaBufFd(void *addr, size_t length, int *fd) {
  if (ncclParamDmaBufEnable() == 0) return ncclInvalidUsage;

  static size_t hostPageSize = (size_t)sysconf(_SC_PAGESIZE);
  size_t alignedSize = ((length + hostPageSize - 1) / hostPageSize) * hostPageSize;

  if (ncclParamIbDataDirect()) {
    HGresult st = hgMemGetHandleForAddressRange(
      (void *)fd, (HGdeviceptr)addr, alignedSize, HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      HG_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (st == HGGC_SUCCESS) return ncclSuccess;
  }

  HGresult st = hgMemGetHandleForAddressRange(
    (void *)fd, (HGdeviceptr)addr, alignedSize, HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (st == HGGC_SUCCESS) return ncclSuccess;

  return ncclInvalidUsage;
}

static ncclResult_t gdakiTroceRegMr(struct ibv_mr **mr, struct ibv_pd *pd, void *addr, size_t length,
                               int access) {
  int dmabufFd = -1;
  if (gdakiTroceGetDmaBufFd(addr, length, &dmabufFd) == ncclSuccess) {
    ncclResult_t ret = wrap_ibv_reg_dmabuf_mr(mr, pd, 0ULL /*offset*/, length, 0ULL /*iova*/,
                                              dmabufFd, access);
    (void)close(dmabufFd);
    if (ret == ncclSuccess) {
      INFO(NCCL_NET, "gdakiTroceRegMr: DMA-BUF addr=%p length=%zu lkey=0x%x rkey=0x%x", addr, length,
           (*mr)->lkey, (*mr)->rkey);
      return ncclSuccess;
    }
  }

  NCCLCHECK(wrap_ibv_reg_mr_iova2(mr, pd, addr, length, 0, (unsigned int)access));
  INFO(NCCL_NET, "gdakiTroceRegMr: iova=0 addr=%p length=%zu lkey=0x%x rkey=0x%x", addr, length,
       (*mr)->lkey, (*mr)->rkey);
  return ncclSuccess;
}

// Host buffer with a device mirror. Same role as GdakiMlxHostGPUMemHandle: build
// the structure on the host, then copy it once into device memory.
template <typename T>
class GdakiTroceHostGpuMem {
 private:
  unsigned int numElements;

 public:
  T *hostBuf;
  T *gpuBuf;

  GdakiTroceHostGpuMem() : numElements(0), hostBuf(nullptr), gpuBuf(nullptr) {}

  ncclResult_t allocate(unsigned int n) {
    this->hostBuf = (T *)calloc(n, sizeof(T));
    EQCHECK(this->hostBuf, nullptr);
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
      free(this->hostBuf);
      this->hostBuf = nullptr;
    }
    if (this->gpuBuf) {
      HGGCCHECK(hggcFree(this->gpuBuf));
      this->gpuBuf = nullptr;
    }
    return ncclSuccess;
  }
};

// Global signals/counters table, shared by all contexts of this comm. Registered
// once, rkeys allGather'd once, then sliced per context.
template <typename T>
class GdakiTroceGlobalTable {
 private:
  unsigned int numElements;
  unsigned int nextUnusedIdx;
  GdakiTroceHostGpuMem<__be32> rkeys;

 public:
  T *gpuPtr;
  struct ibv_mr *mr;

  GdakiTroceGlobalTable() : numElements(0), nextUnusedIdx(0), gpuPtr(nullptr), mr(nullptr) {}

  ncclResult_t allocate(unsigned int n, unsigned int nranks) {
    this->numElements = n;
    this->nextUnusedIdx = 0;
    if (n == 0) return ncclSuccess;
    NCCLCHECK(gdakiTroceAlignedHggcMalloc((void **)&this->gpuPtr, (size_t)n * sizeof(T)));
    HGGCCHECK(hggcMemset(this->gpuPtr, 0, (size_t)n * sizeof(T)));
    NCCLCHECK(this->rkeys.allocate(nranks));
    return ncclSuccess;
  }

  ncclResult_t registerMr(struct ibv_pd *pd) {
    if (this->numElements == 0) return ncclSuccess;
    NCCLCHECK(gdakiTroceRegMr(&this->mr, pd, this->gpuPtr, (size_t)this->numElements * sizeof(T),
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC));
    return ncclSuccess;
  }

  ncclResult_t exchangeInfo(struct ncclGinIbCollComm *cComm) {
    if (this->numElements == 0) return ncclSuccess;
    __be32 rkey = htobe32(this->mr->rkey);
    NCCLCHECK(cComm->allGather(cComm, &rkey, this->rkeys.hostBuf, sizeof(__be32)));
    NCCLCHECK(this->rkeys.copyHtoD());
    return ncclSuccess;
  }

  ncclResult_t allocateElements(unsigned int n, unsigned int *outStartIdx) {
    if (this->nextUnusedIdx + n > this->numElements) {
      WARN("GIN/GDAKI_TROCE: not enough space in global table (%u requested, %u free)", n,
           this->numElements - this->nextUnusedIdx);
      return ncclInvalidUsage;
    }
    *outStartIdx = this->nextUnusedIdx;
    this->nextUnusedIdx += n;
    return ncclSuccess;
  }

  __be32 *rkeysDev() { return this->rkeys.gpuBuf; }

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

struct gdakiTroceQpHost {
  struct ibv_qp *qp;
  struct ibv_cq *cq;      // send CQ, the only one the device polls
  struct ibv_cq *recvCq;  // unused by GIN, but troce wants a distinct CQ

  // troce GPU-mode queue descriptors. Kept for the destroy path.
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
  void *cqDbGpu;
  uint8_t trafficClass;
};

struct gdakiTroceExchInfo {
  uint32_t qpn;
  uint32_t lid;
  uint8_t ibPort;
  uint64_t spn;
  uint64_t iid;
  enum ibv_mtu mtu;
};

struct gdakiTroceContext {
  int hggcDev;
  uint8_t portNum;
  int gidIndex;
  enum ibv_mtu portMtu;  // local port active_mtu, negotiated to min() with the peer at RTR
  int nContexts;
  int nRanks;
  uint32_t sqDepth;
  uint32_t cqDepth;

  struct ncclGinIbCollComm *collComm;
  ncclNetDeviceHandle_t *devHandle;
  struct ncclTroceFunctionTable *troce;

  // nContexts * nRanks queue pairs, laid out ctx-major just like GDAKI's gqps.
  struct gdakiTroceQpHost *qps;

  GdakiTroceGlobalTable<uint64_t> *countersTable;
  GdakiTroceGlobalTable<uint64_t> *signalsTable;

  // Device-visible per-context descriptors and the per-context QP arrays they
  // point at.
  GdakiTroceHostGpuMem<struct pcclGinGdakiTroceGPUContext> *gpuCtx;
  GdakiTroceHostGpuMem<struct pcclGinGdakiTroceQp> *gpuQps;

  // Mutable SQ/CQ indices, one set per QP, kept in device memory.
  uint64_t *idxPool;  // 5 * nContexts * nRanks entries
  uint32_t *pendingCounterPool;  // nqps * nwqes entries, per-WQE pending counter ids

  struct {
    void *addr;
    struct ibv_mr *mr;
  } sinkBuffer;

  uint64_t lastErrorQueryTime;
};

struct gdakiTroceMemHandle {
  int type;
  struct ibv_mr *mr;
  GdakiTroceHostGpuMem<struct pcclGinGdakiTroceMemHandle> *mhandleHd;
  GdakiTroceHostGpuMem<__be32> *rkeysHd;
};

// Create one RC QP whose SQ/RQ/CQ rings live in device memory, so the GPU can
// build WQEs and ring the doorbell without the CPU. Uses the troce private path
// (TROCE_ALLOC_GPU) via the dlopen'd table; sizing and alignment rules come from
// the troce ABI (see trocedvwrap.h).
static ncclResult_t gdakiTroceCreateQp(struct gdakiTroceContext *ctx, struct ibv_pd *pd,
                                  struct ibv_context *ibCtx, struct gdakiTroceQpHost *out) {
  struct ibv_qp_init_attr qpInitAttr;
  // Sizing copied verbatim from sailSHMEM's troce GDAKI_TROCE path so we present the
  // driver with a parameter set it already accepts:
  //   num_wqebb   = max(depth, TROCE_SQ_MIN_WQEBB / TROCE_SEND_WQE_BB_FACTOR)
  //   sq_buf_size = ROUNDUP(num_wqebb * TROCE_SEND_WQE_BB * TROCE_SEND_WQE_BB_FACTOR, TROCE_SQ_ALIGN)
  //   rq_buf_size = ROUNDUP(num_wqebb * TROCE_RECV_WQE_BB, TROCE_RQ_ALIGN)
  //   num_cqe     = num_wqebb * TROCE_SEND_WQE_BB_FACTOR
  const uint32_t minWqebb = (uint32_t)(TROCE_SQ_MIN_WQEBB / TROCE_SEND_WQE_BB_FACTOR);
  uint32_t numWqebb = ctx->sqDepth > minWqebb ? ctx->sqDepth : minWqebb;
  // The device code indexes the SQ, the CQ and the pending-counter table with
  // `idx & (n - 1)`, which is only equivalent to a modulo when the counts are
  // powers of two. queueDepth is configurable, so round up here rather than
  // trusting the caller.
  if (numWqebb & (numWqebb - 1)) {
    uint32_t pow2 = 1u;
    while (pow2 < numWqebb) pow2 <<= 1;
    INFO(NCCL_NET, "GIN/GDAKI_TROCE: rounding queue depth %u up to %u (must be a power of two)",
         numWqebb, pow2);
    numWqebb = pow2;
  }
  // Collapsed single-slot CQ: the NIC overwrites CQE #0 on every completion
  // (IGNORE_OVERRUN) and the device polls only that slot, so the CQ needs just
  // one entry regardless of SQ depth. The provider still requires a page-aligned
  // buffer, so allocate one page even though a single CQE is used.
  const int numCqe = 1;

  out->sqLen = ((size_t)numWqebb * TROCE_SEND_WQE_BB * TROCE_SEND_WQE_BB_FACTOR +
                TROCE_SQ_ALIGN - 1) & ~(TROCE_SQ_ALIGN - 1);
  out->rqLen = ((size_t)numWqebb * TROCE_RECV_WQE_BB + TROCE_RQ_ALIGN - 1) &
               ~(TROCE_RQ_ALIGN - 1);
  out->cqLen = GDAKI_TROCE_PAGE_SIZE;

  NCCLCHECK(gdakiTroceCtrlAlloc(&out->sqBase, &out->sqBuf, out->sqLen));
  NCCLCHECK(gdakiTroceCtrlAlloc(&out->rqBase, &out->rqBuf, out->rqLen));
  NCCLCHECK(gdakiTroceCtrlAlloc(&out->cqBase, &out->cqBuf, out->cqLen, /*fillByte=*/0xff));

  // troce GPU mode wants a distinct CQ object per queue direction, so allocate
  // the send CQ (the only one the device polls) and a recv CQ that stays idle.
  memset(&out->cqCtx, 0, sizeof(out->cqCtx));
  out->cqCtx.alloc_type = TROCE_ALLOC_GPU;
  out->cqCtx.addr = out->cqBuf;
  out->cqCtx.len = out->cqLen;
  out->cqCtx.flags = TROCE_CQ_ATTR_IGNORE_OVERRUN;
  out->cq = ctx->troce->troce_create_cq_with_context(ibCtx, numCqe, nullptr, 0, &out->cqCtx);
  if (out->cq == nullptr) {
    WARN("GIN/GDAKI_TROCE: troce_create_cq_with_context(send) failed (numCqe=%d len=%zu): %s", numCqe,
         out->cqLen, strerror(errno));
    return ncclSystemError;
  }

  NCCLCHECK(gdakiTroceCtrlAlloc(&out->recvCqBase, &out->recvCqBuf, out->cqLen, /*fillByte=*/0xff));
  memset(&out->recvCqCtx, 0, sizeof(out->recvCqCtx));
  out->recvCqCtx.alloc_type = TROCE_ALLOC_GPU;
  out->recvCqCtx.addr = out->recvCqBuf;
  out->recvCqCtx.len = out->cqLen;
  out->recvCqCtx.flags = TROCE_CQ_ATTR_IGNORE_OVERRUN;
  out->recvCq =
    ctx->troce->troce_create_cq_with_context(ibCtx, numCqe, nullptr, 0, &out->recvCqCtx);
  if (out->recvCq == nullptr) {
    WARN("GIN/GDAKI_TROCE: troce_create_cq_with_context(recv) failed (numCqe=%d len=%zu): %s", numCqe,
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

  // Diagnostic probe: deliberately pass an out-of-range alloc_type once, which
  // troce_create_qp_with_context must reject via troce_err(). If we see nothing
  // in the log, our process has no visibility into the driver's diagnostics and
  // any later NULL return cannot be attributed from the driver side.
  // Enable with PCCL_GIN_GDAKI_TROCE_PROBE_DIAG=1.
  if (ncclParamGinGdakiTroceProbeDiag()) {
    static bool probed = false;
    if (!probed) {
      probed = true;
      struct troce_qp_context badCtx;
      struct ibv_qp_init_attr badAttr;
      memset(&badCtx, 0, sizeof(badCtx));
      memset(&badAttr, 0, sizeof(badAttr));
      badCtx.alloc_type = 0x7fffffff;  // out of range on purpose
      badAttr.send_cq = out->cq;
      badAttr.recv_cq = out->recvCq;
      badAttr.qp_type = IBV_QPT_RC;
      badAttr.cap.max_send_wr = 1;
      badAttr.cap.max_recv_wr = 1;
      badAttr.cap.max_send_sge = 1;
      badAttr.cap.max_recv_sge = 1;
      struct ibv_qp *badQp = ctx->troce->troce_create_qp_with_context(pd, &badAttr, &badCtx);
      INFO(NCCL_NET,
           "GIN/GDAKI_TROCE diag probe: bad alloc_type returned %p (expect NULL plus a troce_err line "
           "above if TROCE_DEBUG_FILE is set)",
           (void *)badQp);
      if (badQp) ctx->troce->troce_destroy_qp_with_context(badQp, &badCtx);
    }
  }

  memset(&out->qpCtx, 0, sizeof(out->qpCtx));
  out->qpCtx.alloc_type = TROCE_ALLOC_GPU;
  out->qpCtx.sq_addr = out->sqBuf;
  out->qpCtx.sq_len = out->sqLen;
  out->qpCtx.rq_addr = out->rqBuf;
  out->qpCtx.rq_len = out->rqLen;
  out->qp = ctx->troce->troce_create_qp_with_context(pd, &qpInitAttr, &out->qpCtx);
  if (out->qp == nullptr) {
    // troce_create_qp_with_context returns NULL without promising to set errno,
    // so errno is NOT reported here: reading it produced a misleading "Invalid
    // argument" that came from an unrelated earlier call. The authoritative
    // reason is printed by the driver itself when TROCE_DEBUG_FILE is set.
    WARN("GIN/GDAKI_TROCE: troce_create_qp_with_context returned NULL (errno not meaningful). "
         "sqAddr=%p sqLen=%zu rqAddr=%p rqLen=%zu wqebb=%u maxSendWr=%u maxRecvWr=%u. "
         "Re-run with TROCE_DEBUG_FILE=/dev/stderr for the driver-side reason.",
         out->sqBuf, out->sqLen, out->rqBuf, out->rqLen, numWqebb,
         qpInitAttr.cap.max_send_wr, qpInitAttr.cap.max_recv_wr);
    return ncclSystemError;
  }

  // Expose the tail registers to the GPU: these become the doorbells the device
  // code writes to.
  NCCLCHECK(gdakiTroceMapNicRegToGpu(out->qpCtx.sq_tail_reg, &out->sqDbGpu));
  NCCLCHECK(gdakiTroceMapNicRegToGpu(out->cqCtx.cq_tail_reg, &out->cqDbGpu));

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

// The doorbell carries the QP's traffic class, so it is read back after RTR and
// handed to the device code. A missing TC is fatal rather than defaulted: ringing
// with the wrong TC makes the NIC drop the doorbell and no completion arrives.
static ncclResult_t gdakiTroceQueryTrafficClass(struct gdakiTroceContext *ctx, struct gdakiTroceQpHost *qp) {
  struct trocedv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.attr_mask = TROCEDV_QP_ATTR_TC;
  if (ctx->troce->trocedv_query_qp_attr(qp->qp, &attr) != 0) {
    WARN("GIN/GDAKI_TROCE: trocedv_query_qp_attr failed for qpn=%#x", qp->qp->qp_num);
    return ncclSystemError;
  }
  if (!(attr.actual_mask & TROCEDV_QP_ATTR_TC)) {
    WARN("GIN/GDAKI_TROCE: trocedv_query_qp_attr did not fill TC for qpn=%#x (actual_mask=%#x)",
         qp->qp->qp_num, attr.actual_mask);
    return ncclSystemError;
  }
  qp->trafficClass = attr.tc;
  return ncclSuccess;
}

static void gdakiTroceDestroyQp(struct gdakiTroceContext *ctx, struct gdakiTroceQpHost *qp) {
  if (qp->qp) {
    ctx->troce->troce_destroy_qp_with_context(qp->qp, &qp->qpCtx);
    qp->qp = nullptr;
  }
  if (qp->cq) {
    ctx->troce->troce_destroy_cq_with_context(qp->cq, &qp->cqCtx);
    qp->cq = nullptr;
  }
  if (qp->recvCq) {
    ctx->troce->troce_destroy_cq_with_context(qp->recvCq, &qp->recvCqCtx);
    qp->recvCq = nullptr;
  }
  gdakiTroceUnmapNicReg(qp->qpCtx.sq_tail_reg);
  gdakiTroceUnmapNicReg(qp->cqCtx.cq_tail_reg);
  if (qp->sqBase) hggcFree(qp->sqBase);
  if (qp->rqBase) hggcFree(qp->rqBase);
  if (qp->cqBase) hggcFree(qp->cqBase);
  if (qp->recvCqBase) hggcFree(qp->recvCqBase);
  qp->sqBase = qp->rqBase = qp->cqBase = qp->recvCqBase = nullptr;
  qp->sqBuf = qp->rqBuf = qp->cqBuf = qp->recvCqBuf = nullptr;
}

static ncclResult_t gdakiTroceConnectQp(struct gdakiTroceContext *ctx, struct gdakiTroceQpHost *qp,
                                   struct gdakiTroceExchInfo *remote) {
  struct ncclIbQpInfo info;
  memset(&info, 0, sizeof(info));
  info.lid = remote->lid;
  info.ib_port = remote->ibPort;
  info.spn = remote->spn;
  info.iid = remote->iid;
  // RC requires both ends to use the same path MTU; negotiate to the smaller of
  // the local and remote active MTUs (the standard IB path does the same). The
  // exchange carries each side's local active_mtu, so min() here is symmetric.
  info.mtu = (remote->mtu < ctx->portMtu) ? remote->mtu : ctx->portMtu;
  int atomicConcurrency = (int)ncclParamGinGdakiTroceAtomicConcurrency();
  if (atomicConcurrency < 1 || atomicConcurrency > 128) {
    WARN("GIN/GDAKI_TROCE: invalid atomic concurrency %d, expected 1..128", atomicConcurrency);
    return ncclInvalidArgument;
  }
  NCCLCHECK(ncclIbRtrQpWithAtomic(qp->qp, remote->qpn, &info, ctx->gidIndex,
                                  (uint8_t)atomicConcurrency));
  // Read the TC back while the QP is in RTR, matching sailSHMEM: the value is
  // established by the RTR transition and is what the doorbell has to carry.
  NCCLCHECK(gdakiTroceQueryTrafficClass(ctx, qp));
  NCCLCHECK(ncclIbRtsQpWithAtomic(qp->qp, (uint8_t)atomicConcurrency));
  return ncclSuccess;
}

static void gdakiTroceFillExchInfo(struct gdakiTroceExchInfo *info, struct gdakiTroceContext *ctx,
                              struct ibv_port_attr *portAttr, union ibv_gid *gid,
                              struct ibv_qp *qp) {
  info->qpn = qp->qp_num;
  info->lid = portAttr->lid;
  info->ibPort = ctx->portNum;
  info->spn = gid->global.subnet_prefix;
  info->iid = gid->global.interface_id;
  info->mtu = portAttr->active_mtu;
}

ncclResult_t ncclGinGdakiTroceCreateContext(void *collComm, int nSignals, int nCounters, int nContexts,
                                       int queueDepth, int /*trafficClass*/, void **outGinCtx,
                                       ncclNetDeviceHandle_t **outDevHandle) {
  ncclResult_t status = ncclSuccess;
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  const int rank = cComm->rank;
  const int nranks = cComm->nranks;
  const int nqps = nContexts * nranks;

  struct gdakiTroceContext *ctx = nullptr;
  ncclNetDeviceHandle_t *devHandle = nullptr;
  struct gdakiTroceExchInfo *localExch = nullptr;
  struct gdakiTroceExchInfo *remoteExch = nullptr;
  struct ibv_port_attr portAttr;
  union ibv_gid gid;
  uint64_t *sinkBuffer = nullptr;
  struct ibv_mr *sinkMr = nullptr;

  GdakiTroceHostGpuMem<struct pcclGinGdakiTroceGPUContext> *gpuCtx =
    new GdakiTroceHostGpuMem<struct pcclGinGdakiTroceGPUContext>();
  GdakiTroceHostGpuMem<struct pcclGinGdakiTroceQp> *gpuQps = new GdakiTroceHostGpuMem<struct pcclGinGdakiTroceQp>();
  GdakiTroceGlobalTable<uint64_t> *countersTable = new GdakiTroceGlobalTable<uint64_t>();
  GdakiTroceGlobalTable<uint64_t> *signalsTable = new GdakiTroceGlobalTable<uint64_t>();

  ctx = (struct gdakiTroceContext *)calloc(1, sizeof(*ctx));
  EQCHECKGOTO(ctx, nullptr, status, fail);

  ctx->collComm = cComm;
  ctx->nContexts = nContexts;
  ctx->nRanks = nranks;
  ctx->portNum = 1;  // single-port devices, matches the GDAKI assumption
  ctx->sqDepth = queueDepth > 0 ? (uint32_t)queueDepth : (uint32_t)ncclParamGinGdakiTroceQpDepth();
  ctx->cqDepth = 1; // IGNORE_OVERRUN collapsed CQ
  ctx->countersTable = countersTable;
  ctx->signalsTable = signalsTable;
  ctx->gpuCtx = gpuCtx;
  ctx->gpuQps = gpuQps;

  HGGCCHECKGOTO(hggcGetDevice(&ctx->hggcDev), status, fail);

  // troce's GPU-queue entry points live in libtroce and are resolved lazily so
  // PCCL still builds/runs on hosts without it.
  NCCLCHECKGOTO(ncclTroceFtableInit(&ctx->troce), status, fail);

  NCCLCHECKGOTO(gpuCtx->allocate(nContexts), status, fail);
  NCCLCHECKGOTO(gpuQps->allocate(nqps), status, fail);
  NCCLCHECKGOTO(countersTable->allocate(nCounters * nContexts, nranks), status, fail);
  NCCLCHECKGOTO(signalsTable->allocate(nSignals * nContexts, nranks), status, fail);

  NCCLCHECKGOTO(countersTable->registerMr(cComm->ib.pd), status, fail);
  NCCLCHECKGOTO(signalsTable->registerMr(cComm->ib.pd), status, fail);
  NCCLCHECKGOTO(countersTable->exchangeInfo(cComm), status, fail);
  NCCLCHECKGOTO(signalsTable->exchangeInfo(cComm), status, fail);

  // Sink buffer: receives the fetched value of remote atomics. Never consumed.
  // One 64-byte slot per QP: the slots are cheap and keep concurrent fetch-atomic
  // writebacks on different QPs from targeting the same address, mirroring
  // sailSHMEM's per-operation ibuf slots.
  NCCLCHECKGOTO(gdakiTroceAlignedHggcMalloc((void **)&sinkBuffer, (size_t)nqps * 64), status, fail);
  HGGCCHECKGOTO(hggcMemset(sinkBuffer, 0, (size_t)nqps * 64), status, fail);
  NCCLCHECKGOTO(gdakiTroceRegMr(&sinkMr, cComm->ib.pd, sinkBuffer, (size_t)nqps * 64,
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

  ctx->qps = (struct gdakiTroceQpHost *)calloc(nqps, sizeof(*ctx->qps));
  EQCHECKGOTO(ctx->qps, nullptr, status, fail);
  localExch = (struct gdakiTroceExchInfo *)calloc(nranks, sizeof(*localExch));
  EQCHECKGOTO(localExch, nullptr, status, fail);
  remoteExch = (struct gdakiTroceExchInfo *)calloc(nranks, sizeof(*remoteExch));
  EQCHECKGOTO(remoteExch, nullptr, status, fail);

  for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
    NCCLCHECKGOTO(gdakiTroceCreateQp(ctx, cComm->ib.pd, cComm->ib.context, &ctx->qps[qpIdx]), status,
                  fail);
  }

  // Full mesh: for every context, exchange one QP per peer and connect it.
  for (int ctxIdx = 0; ctxIdx < nContexts; ctxIdx++) {
    for (int peer = 0; peer < nranks; peer++) {
      gdakiTroceFillExchInfo(&localExch[peer], ctx, &portAttr, &gid,
                        ctx->qps[ctxIdx * nranks + peer].qp);
    }
    NCCLCHECKGOTO(cComm->allToAll(cComm, localExch, remoteExch, sizeof(struct gdakiTroceExchInfo)),
                  status, fail);
    for (int peer = 0; peer < nranks; peer++) {
      NCCLCHECKGOTO(gdakiTroceConnectQp(ctx, &ctx->qps[ctxIdx * nranks + peer], &remoteExch[peer]),
                    status, fail);
      INFO(NCCL_NET, "[%d] GIN/GDAKI_TROCE connected ctx=%d peer=%d qpn=%#x -> remote qpn=%#x tc=%u", rank,
           ctxIdx, peer, ctx->qps[ctxIdx * nranks + peer].qp->qp_num, remoteExch[peer].qpn,
           ctx->qps[ctxIdx * nranks + peer].trafficClass);
    }
  }

  // Mutable SQ/CQ indices live in device memory: resv/ready/prod/cons plus the
  // CQ progress lock, five uint64 per QP.
  HGGCCHECKGOTO(hggcMalloc((void **)&ctx->idxPool, (size_t)nqps * 5 * sizeof(uint64_t)), status,
                fail);
  HGGCCHECKGOTO(hggcMemset(ctx->idxPool, 0, (size_t)nqps * 5 * sizeof(uint64_t)), status, fail);
  // Pending-counter slots: one uint32 per WQEBB per QP, only when counters are in
  // play. The device CQ poll turns a non-zero slot into a local counter add.
  if (nCounters) {
    const size_t slots = (size_t)nqps * (ctx->qps[0].sqLen / TROCE_SEND_WQE_BB);
    HGGCCHECKGOTO(hggcMalloc((void **)&ctx->pendingCounterPool, slots * sizeof(uint32_t)), status,
                  fail);
    HGGCCHECKGOTO(hggcMemset(ctx->pendingCounterPool, 0, slots * sizeof(uint32_t)), status, fail);
  }

  // Build the device-visible view.
  for (int ctxIdx = 0; ctxIdx < nContexts; ctxIdx++) {
    struct pcclGinGdakiTroceGPUContext *devCtx = &gpuCtx->hostBuf[ctxIdx];
    unsigned int bufferStart = 0;

    devCtx->qps = gpuQps->gpuBuf + (size_t)ctxIdx * nranks;
    devCtx->nRanks = nranks;
    devCtx->sink_buffer = sinkBuffer;
    devCtx->sink_buffer_lkey = htobe32(sinkMr->lkey);

    if (nCounters) {
      NCCLCHECKGOTO(countersTable->allocateElements(nCounters, &bufferStart), status, fail);
      devCtx->counters_table.buffer = countersTable->gpuPtr + bufferStart;
      devCtx->counters_table.rkeys = countersTable->rkeysDev();
      devCtx->counters_table.lkey = htobe32(countersTable->mr->lkey);
      devCtx->counters_table.offset = bufferStart;
    }
    if (nSignals) {
      NCCLCHECKGOTO(signalsTable->allocateElements(nSignals, &bufferStart), status, fail);
      devCtx->signals_table.buffer = signalsTable->gpuPtr + bufferStart;
      devCtx->signals_table.rkeys = signalsTable->rkeysDev();
      devCtx->signals_table.lkey = htobe32(signalsTable->mr->lkey);
      devCtx->signals_table.offset = bufferStart;
    }

    for (int peer = 0; peer < nranks; peer++) {
      int qpIdx = ctxIdx * nranks + peer;
      struct pcclGinGdakiTroceQp *devQp = &gpuQps->hostBuf[qpIdx];
      devQp->qpn = ctx->qps[qpIdx].qp->qp_num;
      // Device-side WQEBB count is derived from the ring size, matching
      // sailSHMEM's ep->sq_cnt = sq_len / TROCE_SEND_WQE_BB. It is a power of two
      // because sqLen is, and the device masks WQE indices with nwqes-1.
      devQp->nwqes = (uint32_t)(ctx->qps[qpIdx].sqLen / TROCE_SEND_WQE_BB);
      devQp->cqn = ctx->qps[qpIdx].cqCtx.cqn;
      devQp->ncqes = 1; // the backing page is larger, but the CQ has one entry
      devQp->traffic_class = ctx->qps[qpIdx].trafficClass;
      // SQ/CQ rings are in device memory and the tail registers are mapped into
      // the GPU address space, so the device code owns the whole post path.
      devQp->sq_wqe = ctx->qps[qpIdx].sqBuf;
      devQp->sq_db = ctx->qps[qpIdx].sqDbGpu;
      devQp->cq_cqe = ctx->qps[qpIdx].cqBuf;
      devQp->cq_db = ctx->qps[qpIdx].cqDbGpu;
      devQp->resv_head = ctx->idxPool + (size_t)qpIdx * 5 + 0;
      devQp->ready_head = ctx->idxPool + (size_t)qpIdx * 5 + 1;
      devQp->prod_idx = ctx->idxPool + (size_t)qpIdx * 5 + 2;
      devQp->cons_idx = ctx->idxPool + (size_t)qpIdx * 5 + 3;
      devQp->progress_lock = ctx->idxPool + (size_t)qpIdx * 5 + 4;
      // Distinct 64-byte sink slot per QP (byte offset into the iova=0 sink MR),
      // so concurrent fetch-atomics on different QPs don't collide on writeback.
      devQp->sink_addr = (uint64_t)qpIdx * 64;
      // Per-WQE pending-counter slots, consumed by the device CQ poll. Only needed
      // when the caller asked for counters.
      devQp->pending_counter =
        nCounters ? (ctx->pendingCounterPool + (size_t)qpIdx * devQp->nwqes) : nullptr;
      devQp->counters_buf = nCounters ? devCtx->counters_table.buffer : nullptr;
    }
  }

  NCCLCHECKGOTO(gpuQps->copyHtoD(), status, fail);
  NCCLCHECKGOTO(gpuCtx->copyHtoD(), status, fail);

  NCCLCHECKGOTO(ncclCalloc(&devHandle, 1), status, fail);
  NCCLCHECKGOTO(ncclGinGdakiCreateHandle(PCCL_GIN_GDAKI_PROVIDER_TROCE, PCCL_GIN_TROCE_VERSION,
                                        (void*)gpuCtx->gpuBuf, devHandle), status, fail);
  devHandle->needsProxyProgress = 0;
  ctx->devHandle = devHandle;

  INFO(NCCL_INIT | NCCL_NET,
       "GIN/GDAKI_TROCE context ready: rank=%d nranks=%d nContexts=%d nSignals=%d nCounters=%d "
       "sqDepth=%u gidIndex=%d sqBuf=%p sqDb=%p cqBuf=%p cqDb=%p",
       rank, nranks, nContexts, nSignals, nCounters, ctx->sqDepth, ctx->gidIndex,
       ctx->qps[0].sqBuf, ctx->qps[0].sqDbGpu, ctx->qps[0].cqBuf, ctx->qps[0].cqDbGpu);

  free(localExch);
  free(remoteExch);
  *outGinCtx = ctx;
  *outDevHandle = devHandle;
  return ncclSuccess;

fail:
  if (localExch) free(localExch);
  if (remoteExch) free(remoteExch);
  if (devHandle) {
    if (devHandle->handle) hggcFree(devHandle->handle);
    free(devHandle);
  }
  if (ctx) {
    if (ctx->qps) {
      for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
        if (ctx->troce) gdakiTroceDestroyQp(ctx, &ctx->qps[qpIdx]);
      }
      free(ctx->qps);
    }
    if (ctx->idxPool) hggcFree(ctx->idxPool);
    if (ctx->pendingCounterPool) hggcFree(ctx->pendingCounterPool);
    free(ctx);
  }
  if (sinkMr) wrap_ibv_dereg_mr(sinkMr);
  if (sinkBuffer) hggcFree(sinkBuffer);
  countersTable->destroy();
  signalsTable->destroy();
  gpuCtx->deallocate();
  gpuQps->deallocate();
  delete countersTable;
  delete signalsTable;
  delete gpuCtx;
  delete gpuQps;
  return status;
}

ncclResult_t ncclGinGdakiTroceDestroyContext(void *ginCtx) {
  if (!ginCtx) return ncclInvalidArgument;
  struct gdakiTroceContext *ctx = (struct gdakiTroceContext *)ginCtx;
  const int nqps = ctx->nContexts * ctx->nRanks;

  if (ctx->qps) {
    for (int qpIdx = 0; qpIdx < nqps; qpIdx++) {
      gdakiTroceDestroyQp(ctx, &ctx->qps[qpIdx]);
    }
    free(ctx->qps);
    ctx->qps = nullptr;
  }

  if (ctx->countersTable) {
    NCCLCHECK(ctx->countersTable->destroy());
    delete ctx->countersTable;
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

  if (ctx->sinkBuffer.mr) NCCLCHECK(wrap_ibv_dereg_mr(ctx->sinkBuffer.mr));
  if (ctx->sinkBuffer.addr) HGGCCHECK(hggcFree(ctx->sinkBuffer.addr));
  if (ctx->idxPool) HGGCCHECK(hggcFree(ctx->idxPool));
  if (ctx->pendingCounterPool) HGGCCHECK(hggcFree(ctx->pendingCounterPool));
  if (ctx->devHandle) {
    HGGCCHECK(hggcFree(ctx->devHandle->handle));
    free(ctx->devHandle);
  }

  memset(ctx, 0, sizeof(*ctx));
  free(ctx);
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiTroceRegMrSym(void *collComm, void *data, size_t size, int type,
                                  uint64_t /*mr_flags*/, void **mhandle, void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  ncclResult_t status = ncclSuccess;
  struct ibv_mr *mr = nullptr;
  __be32 rkey;

  GdakiTroceHostGpuMem<struct pcclGinGdakiTroceMemHandle> *mhandleHd =
    new GdakiTroceHostGpuMem<struct pcclGinGdakiTroceMemHandle>();
  GdakiTroceHostGpuMem<__be32> *rkeysHd = new GdakiTroceHostGpuMem<__be32>();

  struct gdakiTroceMemHandle *handle = (struct gdakiTroceMemHandle *)calloc(1, sizeof(*handle));
  if (handle == nullptr) {
    status = ncclSystemError;
    goto fail;
  }

  NCCLCHECKGOTO(gdakiTroceRegMr(&mr, cComm->ib.pd, data, size,
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
  mhandleHd->deallocate();
  rkeysHd->deallocate();
  delete mhandleHd;
  delete rkeysHd;
  if (mr) wrap_ibv_dereg_mr(mr);
  if (handle) free(handle);
  return status;
}

ncclResult_t ncclGinGdakiTroceDeregMrSym(void * /*collComm*/, void *mhandle) {
  struct gdakiTroceMemHandle *handle = (struct gdakiTroceMemHandle *)mhandle;
  if (!handle) return ncclInvalidArgument;

  NCCLCHECK(wrap_ibv_dereg_mr(handle->mr));
  NCCLCHECK(handle->mhandleHd->deallocate());
  delete handle->mhandleHd;
  NCCLCHECK(handle->rkeysHd->deallocate());
  delete handle->rkeysHd;

  memset(handle, 0, sizeof(*handle));
  free(handle);
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiTroceProgress(void * /*ginCtx*/) {
  // Nothing to do: the GDAKI_TROCE path is fully GPU-initiated, there is no CPU-side
  // request translation. Kept so the backend table stays shaped like GDAKI's.
  return ncclSuccess;
}

ncclResult_t ncclGinGdakiTroceQueryLastError(void *ginCtx, bool *hasError) {
  struct gdakiTroceContext *ctx = (struct gdakiTroceContext *)ginCtx;
  bool err = false;
  const int nqps = ctx->nContexts * ctx->nRanks;

  // Querying every QP is expensive, so throttle it the same way GDAKI does.
  uint64_t now = clockNano();
  if ((now - ctx->lastErrorQueryTime) / 1e9 < ncclParamGinGdakiTroceErrorQuerySec()) {
    *hasError = false;
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
      WARN("GIN/GDAKI_TROCE: QP %#x is in ERROR state", ctx->qps[qpIdx].qp->qp_num);
      err = true;
    }
  }

  *hasError = err;
  return ncclSuccess;
}
