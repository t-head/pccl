/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "alloc.h"
#include <assert.h>
#include "pccl_compat.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <mutex>

#include "ibvwrap.h"
#include "gin/gin_host.h"
#include "gin_host_mlx5.h"
#include "param.h"

#include "doca_gpunetio_host.h"
#include "pccl_device/gin/providers/mlx5/gin_mlx5_device_host_common.h"
#include "../../common.h"
#include "../../gin.h"

#define DOCACHECK(call)                 \
  do {                                  \
    doca_error_t err = call;            \
    if (err != DOCA_SUCCESS) {          \
      WARN("DOCA failure %d", err);     \
      return ncclSystemError;           \
    }                                   \
  } while (0)

#define DOCACHECKGOTO(call, RES, label) \
  do {                                  \
    doca_error_t err = call;            \
    if (err != DOCA_SUCCESS) {          \
      WARN("DOCA failure %d", err);     \
      RES = ncclSystemError;            \
      goto label;                       \
    }                                   \
  } while (0)

#define MAX_PCI_ADDRESS_LEN 32U

NCCL_PARAM(GinGdakiMlxNicHandler, "GIN_GDAKI_NIC_HANDLER", 0);
NCCL_PARAM(GinGdakiMlxQpDepth, "GIN_GDAKI_QP_DEPTH", 128);
NCCL_PARAM(GinGdakiMlxMaxDestRdAtomic, "GIN_GDAKI_MAX_DEST_RD_ATOMIC", -2);
NCCL_PARAM(GinGdakiMlxMaxQpRdAtomic, "GIN_GDAKI_MAX_QP_RD_ATOMIC", -2);
NCCL_PARAM(GinGdakiMlxErrorQuerySec, "GIN_ERROR_QUERY_SEC", 10);
extern int64_t ncclParamIbTimeout();
extern int64_t ncclParamIbRetryCnt();
extern int64_t ncclParamIbPkey();
extern int64_t ncclParamIbSl();
extern int64_t ncclParamIbTc();
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamDmaBufEnable();
extern int64_t ncclParamIbPciRelaxedOrdering();

static inline bool gdakiMlxRelaxedOrderingEnabled() {
  static bool hasCheckedRelaxedOrdering = false;
  static bool relaxedOrderingEnabled = false;

  static std::mutex lockMutex;
  std::lock_guard<std::mutex> lock(lockMutex);

  if (!hasCheckedRelaxedOrdering) {
    int roMode = ncclParamIbPciRelaxedOrdering();
    ncclResult_t r = ncclInternalError;
    if (roMode == 1 || roMode == 2) {
      r = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
    }

    relaxedOrderingEnabled = (r != ncclInternalError);
    hasCheckedRelaxedOrdering = true;
  }
  return relaxedOrderingEnabled;
}

// Fetch a DMA-BUF FD for [addr, addr+length). Mirrors the proxy-GIN helper in
// src/gin/gin_host_proxy.cc::getDmaBufFd: when forceNonDataDirect=false and
// NCCL_IB_DATA_DIRECT is set, asks for the PCIe-mapped FD; otherwise asks for
// the default mapping. On success the caller owns the returned fd.
static ncclResult_t gdakiMlxGetDmaBufFd(void *addr, size_t length, int *fd,
                                        bool forceNonDataDirect = false) {
  if (ncclParamDmaBufEnable() == 0) return ncclInvalidUsage;

  static size_t hostPageSize = (size_t)sysconf(_SC_PAGESIZE);
  size_t aligned_size = ((length + hostPageSize - 1) / hostPageSize) * hostPageSize;

  if (ncclParamIbDataDirect() && !forceNonDataDirect) {
    HGresult st = hgMemGetHandleForAddressRange(
      (void *)fd, (HGdeviceptr)addr, aligned_size,
      HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      HG_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (st == HGGC_SUCCESS) return ncclSuccess;
  }

  HGresult st = hgMemGetHandleForAddressRange(
    (void *)fd, (HGdeviceptr)addr, aligned_size,
    HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (st == HGGC_SUCCESS) return ncclSuccess;

  return ncclInvalidUsage;
}

// DMA-BUF MR registration. Returns ncclInvalidUsage if the DMA-BUF path
// cannot be taken so the caller falls back to the peer-memory path.
// iova=0 is used so the GDAKI device kernel can keep treating WQE remote
// addresses as byte offsets within the MR.
static ncclResult_t gdakiMlxRegMrDmaBuf(struct ibv_mr** mr, struct ibv_pd* pd,
                                        void* addr, size_t length, int access) {
  int dmabuf_fd = -1;
  ncclResult_t fdRes = gdakiMlxGetDmaBufFd(addr, length, &dmabuf_fd);
  if (fdRes == ncclSuccess) {
    ncclResult_t ret = wrap_ibv_reg_dmabuf_mr(mr, pd, 0ULL /*offset*/, length, 0ULL /*iova*/,
                                              dmabuf_fd, access);
    (void)close(dmabuf_fd);
    if (ret == ncclSuccess) {
      INFO(NCCL_NET, "gdakiMlxRegMr: DMA-BUF addr=%p length=%zu lkey=0x%x rkey=0x%x",
           addr, length, (*mr)->lkey, (*mr)->rkey);
      return ncclSuccess;
    }
    // Fall through with a second try on the default mapping; matches the
    // proxy-GIN retry path that assumes mixed data-direct/non-data-direct is
    // unsafe so all-or-nothing per registration sweep.
  }

  dmabuf_fd = -1;
  fdRes = gdakiMlxGetDmaBufFd(addr, length, &dmabuf_fd, true /*forceNonDataDirect*/);
  if (fdRes != ncclSuccess) return ncclInvalidUsage;

  ncclResult_t ret = wrap_ibv_reg_dmabuf_mr(mr, pd, 0ULL /*offset*/, length, 0ULL /*iova*/,
                                            dmabuf_fd, access);
  (void)close(dmabuf_fd);
  if (ret != ncclSuccess) return ncclInvalidUsage;

  INFO(NCCL_NET, "gdakiMlxRegMr: DMA-BUF addr=%p length=%zu lkey=0x%x rkey=0x%x",
       addr, length, (*mr)->lkey, (*mr)->rkey);
  return ncclSuccess;
}

// MRs are registered with iova=0 (via ibv_reg_mr_iova2): the GDAKI device
// kernel passes the byte offset within the MR as the WQE remote address, so
// the MR's iova must start at 0 for raddr+rkey lookups to land inside it.
//
// mlx5 requires (iova ^ addr) & (PAGE_SIZE - 1) == 0; with iova=0 the buffer
// addr therefore has to be page-aligned. hggcMalloc bump-packs sub-page
// allocations, so internal callers (counters/signals/sink) must allocate at
// least PAGE_SIZE bytes via gdakiMlxAlignedHggcMalloc.
static ncclResult_t gdakiMlxRegMr(struct ibv_mr **mr, struct ibv_pd *pd, void *addr, size_t length,
                                  int access, bool force_strict_ordering = false) {
  if (!force_strict_ordering && gdakiMlxRelaxedOrderingEnabled())
    access |= IBV_ACCESS_RELAXED_ORDERING;

  ncclResult_t status = gdakiMlxRegMrDmaBuf(mr, pd, addr, length, access);
  if (status == ncclSuccess) return ncclSuccess;

  NCCLCHECK(wrap_ibv_reg_mr_iova2(mr, pd, addr, length, 0, (unsigned int)access));
  INFO(NCCL_NET, "gdakiMlxRegMr: iova=0 addr=%p length=%zu mr->addr=%p mr->length=%zu lkey=0x%x rkey=0x%x",
       addr, length, (*mr)->addr, (*mr)->length, (*mr)->lkey, (*mr)->rkey);
  return ncclSuccess;
}

// Page-aligned hggcMalloc. hggcMalloc bump-packs sub-page allocations from a
// shared slab, so requests smaller than PAGE_SIZE land at sub-page offsets.
// gdakiMlxRegMr passes iova=0 to ibv_reg_mr_iova2, which mlx5 rejects unless the
// addr's page-bits match (iova ^ addr) & (PAGE_SIZE-1) == 0. Rounding the
// allocation up to PAGE_SIZE forces the allocator to start a fresh
// page-aligned slab.
static constexpr size_t GDAKI_MLX_PAGE_SIZE = 4096;
static ncclResult_t gdakiMlxAlignedHggcMalloc(void **ptr, size_t size) {
  size_t aligned = (size + GDAKI_MLX_PAGE_SIZE - 1) & ~(GDAKI_MLX_PAGE_SIZE - 1);
  if (aligned == 0) aligned = GDAKI_MLX_PAGE_SIZE;
  HGGCCHECK(hggcMalloc(ptr, aligned));
  return ncclSuccess;
}

template <typename T>
class GdakiMlxHostGPUMemHandle {
 private:
  unsigned int num_elements;

 public:
  T *host_buf;
  T *gpu_buf;

  ncclResult_t allocate(unsigned int num_elements) {
    // Control collectives may register these host mirrors with verbs.
    NCCLCHECK(ncclIbMalloc((void**)&this->host_buf, (size_t)num_elements * sizeof(T)));

    HGGCCHECK(hggcMalloc((void **)&this->gpu_buf, (size_t)num_elements * sizeof(T)));

    this->num_elements = num_elements;

    return ncclSuccess;
  }

  ncclResult_t deallocate() {
    if (this->host_buf != nullptr) {
      NCCLCHECK(ncclFree(this->host_buf));
      this->host_buf = nullptr;
    }
    if (this->gpu_buf != nullptr) {
      HGGCCHECK(hggcFree(this->gpu_buf));
      this->gpu_buf = nullptr;
    }
    return ncclSuccess;
  }

  ncclResult_t copy_h_to_d() {
    HGGCCHECK(hggcMemcpy(this->gpu_buf, this->host_buf,
                         (size_t)this->num_elements * sizeof(T),
                         hggcMemcpyHostToDevice));
    return ncclSuccess;
  }

  ncclResult_t copy_d_to_h() {
    HGGCCHECK(hggcMemcpy(this->host_buf, this->gpu_buf,
                         (size_t)this->num_elements * sizeof(T),
                         hggcMemcpyDeviceToHost));
    return ncclSuccess;
  }

  GdakiMlxHostGPUMemHandle() : num_elements(0), host_buf(nullptr), gpu_buf(nullptr){};

  ~GdakiMlxHostGPUMemHandle() {
     // Only used on the error-cleanup path; we deliberately ignore the rc.
     this->deallocate();
  }
};

template <typename T>
class GdakiMlxGlobalGPUBufferTable {
 private:
  unsigned int num_elements;
  unsigned int next_unused_idx;
  unsigned int num_ranks;
  GdakiMlxHostGPUMemHandle<__be32> rkeys_hd_mhandle;

 public:
  T *gpu_ptr;
  struct ibv_mr *mr;

  ncclResult_t allocate(unsigned int num_elements, unsigned int num_ranks) {
    this->num_elements = num_elements;
    this->num_ranks = num_ranks;
    this->next_unused_idx = 0;
    if (num_elements == 0) return ncclSuccess;

    NCCLCHECK(gdakiMlxAlignedHggcMalloc((void **)&this->gpu_ptr, (size_t)num_elements * sizeof(T)));
    HGGCCHECK(hggcMemset(this->gpu_ptr, 0, (size_t)num_elements * sizeof(T)));
    NCCLCHECK(this->rkeys_hd_mhandle.allocate(num_ranks));
    return ncclSuccess;
  }

  ncclResult_t deallocate() {
    if (this->gpu_ptr != nullptr) {
      HGGCCHECK(hggcFree(this->gpu_ptr));
      this->gpu_ptr = nullptr;
    }
    return ncclSuccess;
  }

  ncclResult_t register_mr(struct ibv_pd *ib_pd, bool force_strict_ordering = false) {
    if (this->num_elements == 0) return ncclSuccess;
    NCCLCHECK(gdakiMlxRegMr(&this->mr, ib_pd, this->gpu_ptr, (size_t)this->num_elements * sizeof(T),
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                            IBV_ACCESS_REMOTE_ATOMIC,
                            force_strict_ordering));
    return ncclSuccess;
  }

  ncclResult_t deregister_mr() {
    if (this->mr != nullptr) {
      NCCLCHECK(wrap_ibv_dereg_mr(this->mr));
      this->mr = nullptr;
    }
    return ncclSuccess;
  }

  ncclResult_t exchange_info(struct ncclGinIbCollComm *cComm) {
    if (this->num_elements == 0) return ncclSuccess;
    __be32 rkey = htobe32(this->mr->rkey);
    NCCLCHECK(cComm->allGather(cComm, &rkey, this->rkeys_hd_mhandle.host_buf, sizeof(__be32)));
    NCCLCHECK(this->rkeys_hd_mhandle.copy_h_to_d());
    return ncclSuccess;
  }

  ncclResult_t allocate_elements(unsigned int num_elements, unsigned int *out_start_idx) {
    if (this->next_unused_idx + num_elements > this->num_elements) {
      WARN("Not enough space to get elements");
      return ncclInvalidUsage;
    }

    *out_start_idx = this->next_unused_idx;
    this->next_unused_idx += num_elements;

    return ncclSuccess;
  }

  void free_elements(unsigned int /*start_idx*/, unsigned int /*num_elements*/) {
    // No op for now as we don't allow reusing elements.
  }

  uint32_t *get_rkeys_d() { return this->rkeys_hd_mhandle.gpu_buf;
  }

  GdakiMlxGlobalGPUBufferTable()
    : num_elements(0), next_unused_idx(0), num_ranks(0), gpu_ptr(nullptr), mr(nullptr) {};
  ~GdakiMlxGlobalGPUBufferTable() {
     // Only used on the error-cleanup path; we deliberately ignore the rc.
     this->deregister_mr();
     this->deallocate();
  }
};

struct gdaki_mlx_mem_handle {
  int type;
  struct ibv_mr *mr;
  GdakiMlxHostGPUMemHandle<struct ncclGinGdakiMlxMemHandle> *gdaki_mhandle_hd_mhandle;
  GdakiMlxHostGPUMemHandle<__be32> *rkeys_hd_mhandle;
};

struct gdaki_mlx_exch_info {
  int lid;
  int qpn;
  union ibv_gid gid;
  struct doca_verbs_gid vgid;
  int gid_index;
  enum ibv_mtu mtu;
};

static unsigned ibvMtuToBytes(enum ibv_mtu mtu) {
  switch (mtu) {
    case IBV_MTU_256:  return 256;
    case IBV_MTU_512:  return 512;
    case IBV_MTU_1024: return 1024;
    case IBV_MTU_2048: return 2048;
    case IBV_MTU_4096: return 4096;
    default:           return 0;
  }
}

static enum doca_verbs_mtu_size ibvMtuToDocaMtu(enum ibv_mtu mtu) {
  switch (mtu) {
    case IBV_MTU_256:  return DOCA_VERBS_MTU_SIZE_256_BYTES;
    case IBV_MTU_512:  return DOCA_VERBS_MTU_SIZE_512_BYTES;
    case IBV_MTU_1024: return DOCA_VERBS_MTU_SIZE_1K_BYTES;
    case IBV_MTU_2048: return DOCA_VERBS_MTU_SIZE_2K_BYTES;
    case IBV_MTU_4096: return DOCA_VERBS_MTU_SIZE_4K_BYTES;
    default:           return DOCA_VERBS_MTU_SIZE_4K_BYTES;
  }
}

struct gdaki_mlx_context {
  int hggc_id;
  struct doca_gpu *gdev;
  struct ibv_device *ib_dev;
  struct doca_verbs_ah_attr *ah; /* DOCA Verbs address handle */
  struct ibv_device_attr ib_dev_attr;
  struct doca_verbs_gid gid;

  union ibv_gid rgid;
  struct ibv_port_attr port_attr;
  uint8_t port_num;
  int gid_index;
  enum doca_verbs_mtu_size path_mtu;

  uint32_t qp_rq_size;
  uint32_t qp_sq_size;
  struct doca_gpu_verbs_qp_group_hl **gqp_groups;
  struct doca_gpu_verbs_qp_hl **gqps;
  struct doca_gpu_verbs_qp_hl **companion_gqps;

  GdakiMlxGlobalGPUBufferTable<uint64_t> *counters_table;
  GdakiMlxGlobalGPUBufferTable<uint64_t> *signals_table;
  GdakiMlxHostGPUMemHandle<struct ncclGinGdakiMlxGPUContext> *gin_gdaki_gpu_ctx_hd_mhandle;
  struct {
    void *addr;
    struct ibv_mr *mr;
  } sink_buffer;
  uint64_t last_error_query_time;

  struct ncclGinIbCollComm *collComm;
  ncclNetDeviceHandle_t *devHandle;
  int nContexts;
};

template <typename T>
static inline T gdaki_mlx_round_up(T x, T y) {
  return ((x + y - 1) / y) * y;
}

static void gdakiMlxFillExchInfo(struct gdaki_mlx_exch_info *exch_info, struct gdaki_mlx_context *gdaki_ctx,
                                 struct doca_gpu_verbs_qp_hl *gqp) {
  exch_info->lid = gdaki_ctx->port_attr.lid;
  exch_info->qpn = doca_verbs_qp_get_qpn(gqp->qp);
  memcpy(exch_info->gid.raw, gdaki_ctx->rgid.raw, sizeof(union ibv_gid));
  memcpy(exch_info->vgid.raw, gdaki_ctx->rgid.raw, sizeof(union ibv_gid));
  exch_info->gid_index = gdaki_ctx->gid_index;
  exch_info->mtu = gdaki_ctx->port_attr.active_mtu;
}

static ncclResult_t gdakiMlxCreateVerbsAh(struct gdaki_mlx_context *ctx, struct ibv_context* ib_context,
                                          int ib_sl, int ib_tc, int ib_gid_index) {
  ncclResult_t status = ncclSuccess;

  DOCACHECK(doca_verbs_ah_attr_create(ib_context, &ctx->ah));

  if (ctx->port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_sl(ctx->ah, ib_sl), status, destroy_verbs_ah);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH),
                  status, destroy_verbs_ah);
  } else {
    DOCACHECKGOTO(doca_verbs_ah_attr_set_traffic_class(ctx->ah, ib_tc), status, destroy_verbs_ah);
    DOCACHECKGOTO(doca_verbs_ah_attr_set_addr_type(ctx->ah, DOCA_VERBS_ADDR_TYPE_IPv4),
                  status, destroy_verbs_ah);
  }

  DOCACHECKGOTO(doca_verbs_ah_attr_set_sgid_index(ctx->ah, ib_gid_index), status, destroy_verbs_ah);
  DOCACHECKGOTO(doca_verbs_ah_attr_set_hop_limit(ctx->ah, 255), status, destroy_verbs_ah);

  return ncclSuccess;

destroy_verbs_ah:
  DOCACHECK(doca_verbs_ah_attr_destroy(ctx->ah));
  return status;
}

// Set consistently on all ranks: QP multi-path mode is not negotiated in
// gdaki_mlx_exch_info. Only remote Ethernet main QPs request this mode below.
PCCL_PARAM(GinGdakiEnableMultiPath, "GDAKI_ENABLE_MULTI_PATH", 0);

static ncclResult_t gdakiMlxConnectQp(struct gdaki_mlx_context *ctx, struct doca_gpu_verbs_qp_hl *gqp,
                                      struct gdaki_mlx_exch_info *exch_info,
                                      bool is_remote_main_qp) {
  ncclResult_t status = ncclSuccess;
  struct doca_verbs_qp_attr *verbs_qp_attr = nullptr;
  int max_dest_rd_atomic = ncclParamGinGdakiMlxMaxDestRdAtomic() > 0
    ? ncclParamGinGdakiMlxMaxDestRdAtomic() : ctx->ib_dev_attr.max_qp_rd_atom;
  int max_qp_rd_atomic = ncclParamGinGdakiMlxMaxQpRdAtomic() > 0
    ? ncclParamGinGdakiMlxMaxQpRdAtomic() : ctx->ib_dev_attr.max_qp_rd_atom;

  DOCACHECK(doca_verbs_ah_attr_set_gid(ctx->ah, exch_info->vgid));
  DOCACHECK(doca_verbs_ah_attr_set_dlid(ctx->ah, exch_info->lid));
  DOCACHECK(doca_verbs_qp_attr_create(&verbs_qp_attr));
  if (is_remote_main_qp && ctx->port_attr.link_layer == IBV_LINK_LAYER_ETHERNET &&
      pcclParamGinGdakiEnableMultiPath())
    DOCACHECKGOTO(doca_verbs_qp_attr_set_multi_path(verbs_qp_attr, 1), status,
                  destroy_verbs_qp_attr);
  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_path_mtu(verbs_qp_attr, ctx->path_mtu),
    status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rq_psn(verbs_qp_attr, 0), status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_attr_set_sq_psn(verbs_qp_attr, 0), status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_port_num(verbs_qp_attr, ctx->port_num),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ack_timeout(verbs_qp_attr, ncclParamIbTimeout()),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_retry_cnt(verbs_qp_attr, ncclParamIbRetryCnt()),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_rnr_retry(verbs_qp_attr, 7), status,
                destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_min_rnr_timer(verbs_qp_attr, 12),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_INIT),
    status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_write(verbs_qp_attr, 1),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_read(verbs_qp_attr, 1),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_allow_remote_atomic(
                  verbs_qp_attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_ah_attr(verbs_qp_attr, ctx->ah),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_dest_qp_num(verbs_qp_attr, exch_info->qpn),
                status, destroy_verbs_qp_attr);
  DOCACHECKGOTO(doca_verbs_qp_attr_set_pkey_index(verbs_qp_attr, ncclParamIbPkey()),
                status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(
                  gqp->qp, verbs_qp_attr,
                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                    DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX |
                    DOCA_VERBS_QP_ATTR_PORT_NUM),
                status, destroy_verbs_qp_attr);


  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_max_dest_rd_atomic(verbs_qp_attr, max_dest_rd_atomic),
    status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTR),
    status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(
                  gqp->qp, verbs_qp_attr,
                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                    DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                    DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER | DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC),
                status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_max_rd_atomic(verbs_qp_attr, max_qp_rd_atomic),
    status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(
    doca_verbs_qp_attr_set_next_state(verbs_qp_attr, DOCA_VERBS_QP_STATE_RTS),
    status, destroy_verbs_qp_attr);

  DOCACHECKGOTO(doca_verbs_qp_modify(
                  gqp->qp, verbs_qp_attr,
                  DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                    DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                    DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC),
                status, destroy_verbs_qp_attr);

  DOCACHECK(doca_verbs_qp_attr_destroy(verbs_qp_attr));

  return ncclSuccess;

destroy_verbs_qp_attr:
  DOCACHECK(doca_verbs_qp_attr_destroy(verbs_qp_attr));
  return status;
}

NCCL_PARAM(GinGdakiMlxUseReliableDB, "GDAKI_USE_RELIABLE_DB", 0);

ncclResult_t ncclGinGdakiMlxCreateContext(void *collComm, int nSignals, int nCounters, int nContexts, int queueDepth,
                                          int trafficClass, void **outGinCtx, ncclNetDeviceHandle_t **outDevHandle) {
  ncclResult_t status = ncclSuccess;

  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;

  char pciBusId[MAX_PCI_ADDRESS_LEN];

  const int rank = cComm->rank;
  const int nranks = cComm->nranks;
  const int ncontexts = nContexts;
  const int nqps_per_rank = ncontexts;
  const int nqps_for_comm = nqps_per_rank * nranks;  // Number of QPs for communication
  const int ncompanion_qps = nqps_for_comm * 2;      // Number of companion QPs for communication
                                                     // Double because we connect to self.
  const int nqps =
    nqps_per_rank * (nranks + 1);  // +1 for the local rank.
                                   // The last group is the responder of the local rank.

  const int num_counters = nCounters;
  const int num_signals = nSignals;
  ncclNetProperties_t props;
  ncclNetDeviceHandle_t *devHandle = nullptr;
  struct gdaki_mlx_context *gdaki_ctx = nullptr;
  struct gdaki_mlx_exch_info *local_exch_info = nullptr;
  struct gdaki_mlx_exch_info *remote_exch_info = nullptr;

  struct doca_gpu_verbs_qp_init_attr_hl qp_init_attr;

  uint64_t *sink_buffer = nullptr;
  struct ibv_mr *sink_buffer_mr = nullptr;

  bool need_cpu_proxy = false;

  struct doca_gpu_verbs_qp **gverbs_qps = nullptr;

  GdakiMlxHostGPUMemHandle<struct ncclGinGdakiMlxGPUContext> *gin_gdaki_gpu_ctx_hd_mhandle =
    new GdakiMlxHostGPUMemHandle<struct ncclGinGdakiMlxGPUContext>();
  GdakiMlxGlobalGPUBufferTable<uint64_t> *counters_table = new GdakiMlxGlobalGPUBufferTable<uint64_t>();
  GdakiMlxGlobalGPUBufferTable<uint64_t> *signals_table = new GdakiMlxGlobalGPUBufferTable<uint64_t>();

  const int ib_sl = (ncclParamIbSl() != -1) ? ncclParamIbSl() : (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass : 0;
  const int ib_tc = (ncclParamIbTc() != -1) ? ncclParamIbTc() : (trafficClass != NCCL_NET_TRAFFIC_CLASS_UNDEF) ? trafficClass : 0;
  int ib_gid_index = 0;

  NCCLCHECKGOTO(cComm->getProperties(cComm->dev, &props), status, out);

  NCCLCHECKGOTO(gin_gdaki_gpu_ctx_hd_mhandle->allocate(ncontexts), status, out);
  NCCLCHECKGOTO(counters_table->allocate(num_counters * ncontexts, nranks), status, out);
  NCCLCHECKGOTO(signals_table->allocate(num_signals * ncontexts, nranks), status, out);

  gdaki_ctx = (struct gdaki_mlx_context *)calloc(1, sizeof(*gdaki_ctx));
  EQCHECKGOTO(gdaki_ctx, nullptr, status, out);

  devHandle = (ncclNetDeviceHandle_t *)calloc(1, sizeof(*devHandle));
  EQCHECKGOTO(devHandle, nullptr, status, out);

  gdaki_ctx->gqp_groups = (struct doca_gpu_verbs_qp_group_hl **)calloc(
    nqps_for_comm, sizeof(*gdaki_ctx->gqp_groups));
  EQCHECKGOTO(gdaki_ctx->gqp_groups, nullptr, status, out);

  // Main QP
  gdaki_ctx->gqps = (struct doca_gpu_verbs_qp_hl **)calloc(nqps, sizeof(*gdaki_ctx->gqps));
  EQCHECKGOTO(gdaki_ctx->gqps, nullptr, status, out);

  // Companion QP
  gdaki_ctx->companion_gqps =
    (struct doca_gpu_verbs_qp_hl **)calloc(ncompanion_qps, sizeof(*gdaki_ctx->companion_gqps));
  EQCHECKGOTO(gdaki_ctx->companion_gqps, nullptr, status, out);

  local_exch_info = (struct gdaki_mlx_exch_info *)calloc(nranks, sizeof(*local_exch_info));
  EQCHECKGOTO(local_exch_info, nullptr, status, out);

  remote_exch_info = (struct gdaki_mlx_exch_info *)calloc(nranks, sizeof(*remote_exch_info));
  EQCHECKGOTO(remote_exch_info, nullptr, status, out);

  HGGCCHECKGOTO(hggcGetDevice(&gdaki_ctx->hggc_id), status, out);
  HGGCCHECKGOTO(hggcDeviceGetPCIBusId(pciBusId, MAX_PCI_ADDRESS_LEN, gdaki_ctx->hggc_id), status, out);

  DOCACHECKGOTO(doca_gpu_create(pciBusId, &gdaki_ctx->gdev), status, out);

  NCCLCHECKGOTO(wrap_ibv_query_device(cComm->ib.context, &gdaki_ctx->ib_dev_attr), status, out);

  // Exchange counters and signals with peers
  NCCLCHECKGOTO(counters_table->register_mr(cComm->ib.pd, true), status, out);
  NCCLCHECKGOTO(signals_table->register_mr(cComm->ib.pd, true), status, out);

  NCCLCHECKGOTO(counters_table->exchange_info(cComm), status, out);
  NCCLCHECKGOTO(signals_table->exchange_info(cComm), status, out);

  gdaki_ctx->port_num = 1; // assume 1 for mlx5 devices
  NCCLCHECKGOTO(wrap_ibv_query_port(cComm->ib.context, gdaki_ctx->port_num, &gdaki_ctx->port_attr),
                status, out);

  // Get the GID index
  NCCLCHECKGOTO(cComm->getGidIndex(cComm->ib.context, gdaki_ctx->port_num, &gdaki_ctx->port_attr, &ib_gid_index), status, out);
  gdaki_ctx->gid_index = ib_gid_index;

  NCCLCHECKGOTO(wrap_ibv_query_gid(cComm->ib.context, 1, ib_gid_index, &gdaki_ctx->rgid), status,
                out);

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
    char gidStr[64];
    inet_ntop(AF_INET6, gdaki_ctx->rgid.raw, gidStr, sizeof(gidStr));
    INFO(NCCL_INIT, "GDAKI GID: rank=%d dev=%d gid_index=%d gid=%s", rank, cComm->dev, ib_gid_index, gidStr);
#endif

  NCCLCHECKGOTO(gdakiMlxCreateVerbsAh(gdaki_ctx, cComm->ib.context, ib_sl, ib_tc, ib_gid_index), status, out);

  gdaki_ctx->qp_rq_size = 0;
  gdaki_ctx->qp_sq_size = queueDepth > 0 ? queueDepth : ncclParamGinGdakiMlxQpDepth();

  memset(&qp_init_attr, 0, sizeof(qp_init_attr));
  qp_init_attr.gpu_dev = gdaki_ctx->gdev;
  qp_init_attr.ibpd = cComm->ib.pd;
  qp_init_attr.sq_nwqe = gdaki_ctx->qp_sq_size;
  qp_init_attr.nic_handler =
    (enum doca_gpu_dev_verbs_nic_handler)ncclParamGinGdakiMlxNicHandler();
  // DEFAULT lets DOCA probe for a dmabuf FD first and only falls back to
  // alixpu_peermem; pin the QP/CQ umem to peermem when DMA-BUF is disabled.
  qp_init_attr.mreg_type = ncclParamDmaBufEnable()
    ? DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT
    : DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM;
  if (ncclParamGinGdakiMlxUseReliableDB())
    qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW;
  else
    qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;

  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
retry_create_qp_group_hl:
    doca_error_t docaStatus = doca_gpu_verbs_create_qp_group_hl(&qp_init_attr, &gdaki_ctx->gqp_groups[qp_idx]);
    if (docaStatus != DOCA_SUCCESS) {
      if (qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW) {
        qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED;
        goto retry_create_qp_group_hl;
      }

      if ((qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED)
          && ncclParamGinGdakiMlxUseReliableDB() == 2) {
        qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
        goto retry_create_qp_group_hl;
      }

      WARN("DOCA Error %d", docaStatus);
      status = ncclSystemError;
      goto out;
    }

    gdaki_ctx->gqps[qp_idx] = &gdaki_ctx->gqp_groups[qp_idx]->qp_main;
    gdaki_ctx->companion_gqps[qp_idx] = &gdaki_ctx->gqp_groups[qp_idx]->qp_companion;

    const char *dbr_opt_str =
      qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_HW ? "HW"
      : qp_init_attr.send_dbr_mode_ext == DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_NO_DBR_SW_EMULATED
        ? "SW emulation"
        : "disabled";
    INFO(NCCL_NET,
         "[%d] Created a QP group: qp_idx=%d, main_qpn=%#x, companion_qpn=%#x, reliable_db=%s",
         rank, qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp),
         doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp), dbr_opt_str);
  }

  qp_init_attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
    DOCACHECKGOTO(doca_gpu_verbs_create_qp_hl(&qp_init_attr, &gdaki_ctx->gqps[qp_idx]),
                  status, out);
    INFO(NCCL_NET, "[%d] Created a self-loop peer QP: qp_idx=%d, qpn=%#x", rank, qp_idx,
         doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp));
  }

  for (int qp_idx = nqps_for_comm; qp_idx < ncompanion_qps; qp_idx++) {
    DOCACHECKGOTO(
      doca_gpu_verbs_create_qp_hl(&qp_init_attr, &gdaki_ctx->companion_gqps[qp_idx]),
      status, out);
    INFO(NCCL_NET, "[%d] Created a self-loop peer companion QP: qp_idx=%d, qpn=%#x", rank, qp_idx,
         doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp));
  }

  for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
    // Prepare information for exchange with peers
    for (int rank_idx = 0; rank_idx < nranks; rank_idx++) {
      int qp_idx = rank_idx + ctx_idx * nranks;
      gdakiMlxFillExchInfo(&local_exch_info[rank_idx], gdaki_ctx, gdaki_ctx->gqps[qp_idx]);
    }

    // Exchange information with peers
    NCCLCHECKGOTO(
      cComm->allToAll(cComm, local_exch_info, remote_exch_info, sizeof(struct gdaki_mlx_exch_info)),
      status, out);

    // On the first context exchange, compute the global minimum MTU across all ranks.
    if (ctx_idx == 0) {
      enum ibv_mtu min_mtu = gdaki_ctx->port_attr.active_mtu;
      for (int i = 0; i < nranks; i++) {
        if (remote_exch_info[i].mtu < min_mtu)
          min_mtu = remote_exch_info[i].mtu;
      }
      gdaki_ctx->path_mtu = ibvMtuToDocaMtu(min_mtu);
      INFO(NCCL_INIT, "GDAKI path MTU: rank=%d local=%u bytes, global_min=%u bytes",
           rank, ibvMtuToBytes(gdaki_ctx->port_attr.active_mtu), ibvMtuToBytes(min_mtu));
    }

    for (int rank_idx = 0; rank_idx < nranks; rank_idx++) {
      int qp_idx = rank_idx + ctx_idx * nranks;
      // Self-loop traffic never reaches an Ethernet path. Keep both ends of
      // that QP pair in the default ordered mode; only remote main QPs request
      // multi-path below.
      if (rank_idx == rank)
        gdakiMlxFillExchInfo(&remote_exch_info[rank_idx], gdaki_ctx,
                          gdaki_ctx->gqps[nqps_for_comm + ctx_idx]);

      NCCLCHECKGOTO(gdakiMlxConnectQp(gdaki_ctx, gdaki_ctx->gqps[qp_idx],
                                      &remote_exch_info[rank_idx], rank_idx != rank),
                    status, out);

      INFO(NCCL_NET,
           "[%d] Connected main QP: qp_idx=%d, main_qpn=%#x, remote_rank=%d, remote_qpn=%#x", rank,
           qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->gqps[qp_idx]->qp), rank_idx,
           remote_exch_info[rank_idx].qpn);
    }
  }

  for (int qp_idx = 0; qp_idx < nqps_per_rank; qp_idx++) {
    int peer_qp_idx = nqps_for_comm + qp_idx;
    struct gdaki_mlx_exch_info exch_info;
    gdakiMlxFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->gqps[qp_idx * nranks + rank]);
    NCCLCHECKGOTO(gdakiMlxConnectQp(gdaki_ctx, gdaki_ctx->gqps[peer_qp_idx], &exch_info, false),
                  status, out);
    INFO(NCCL_NET, "[%d] Connected self-loop peer QP: qp_idx=%d, qpn=%#x, main_qpn=%#x", rank,
         peer_qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->gqps[peer_qp_idx]->qp), exch_info.qpn);
  }

  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
    int peer_qp_idx = nqps_for_comm + qp_idx;
    struct gdaki_mlx_exch_info exch_info;
    gdakiMlxFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx]);
    NCCLCHECKGOTO(gdakiMlxConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx], &exch_info, false),
                  status, out);
    INFO(NCCL_NET,
         "[%d] Connected companion QP: qp_idx=%d, companion_qpn=%#x, peer_companion_qpn=%#x", rank,
         qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[qp_idx]->qp), exch_info.qpn);

    gdakiMlxFillExchInfo(&exch_info, gdaki_ctx, gdaki_ctx->companion_gqps[qp_idx]);
    NCCLCHECKGOTO(gdakiMlxConnectQp(gdaki_ctx, gdaki_ctx->companion_gqps[peer_qp_idx], &exch_info,
                                false),
                  status, out);
    INFO(NCCL_NET,
         "[%d] Connected self-loop peer companion QP: qp_idx=%d, peer_companion_qpn=%#x, "
         "companion_qpn=%#x",
         rank, peer_qp_idx, doca_verbs_qp_get_qpn(gdaki_ctx->companion_gqps[peer_qp_idx]->qp),
         exch_info.qpn);
  }

  NCCLCHECKGOTO(gdakiMlxAlignedHggcMalloc((void **)&sink_buffer, sizeof(uint64_t)), status, out);

  NCCLCHECKGOTO(gdakiMlxRegMr(&sink_buffer_mr, cComm->ib.pd, sink_buffer, sizeof(uint64_t),
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC),
                status, out);

  gverbs_qps = (struct doca_gpu_verbs_qp **)calloc(nranks, sizeof(struct doca_gpu_verbs_qp *));
  EQCHECKGOTO(gverbs_qps, nullptr, status, out);
  for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
    struct ncclGinGdakiMlxGPUContext *gin_gdaki_gpu_ctx =
      &gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];

    unsigned int buffer_start = 0;
    for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
      gverbs_qps[qp_idx] = gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
      need_cpu_proxy |= (gverbs_qps[qp_idx]->cpu_proxy);
    }
    DOCACHECKGOTO(doca_gpu_verbs_export_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks,
                                                      &gin_gdaki_gpu_ctx->gdqp),
                  status, out);

    for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
      gverbs_qps[qp_idx] = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
      need_cpu_proxy |= (gverbs_qps[qp_idx]->cpu_proxy);
    }
    DOCACHECKGOTO(doca_gpu_verbs_export_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks,
                                                      &gin_gdaki_gpu_ctx->companion_gdqp),
                  status, out);

    if (nCounters) {
      NCCLCHECKGOTO(counters_table->allocate_elements(num_counters, &buffer_start), status, out);
      gin_gdaki_gpu_ctx->counters_table.buffer = counters_table->gpu_ptr + buffer_start;
      gin_gdaki_gpu_ctx->counters_table.rkeys = counters_table->get_rkeys_d();
      gin_gdaki_gpu_ctx->counters_table.lkey = htobe32(counters_table->mr->lkey);
      gin_gdaki_gpu_ctx->counters_table.offset = buffer_start;
    }
    if (nSignals) {
      NCCLCHECKGOTO(signals_table->allocate_elements(num_signals, &buffer_start), status, out);
      gin_gdaki_gpu_ctx->signals_table.buffer = signals_table->gpu_ptr + buffer_start;
      gin_gdaki_gpu_ctx->signals_table.rkeys = signals_table->get_rkeys_d();
      gin_gdaki_gpu_ctx->signals_table.lkey = htobe32(signals_table->mr->lkey);
      gin_gdaki_gpu_ctx->signals_table.offset = buffer_start;
    }
    gin_gdaki_gpu_ctx->sink_buffer_lkey = htobe32(sink_buffer_mr->lkey);
  }

  NCCLCHECKGOTO(gin_gdaki_gpu_ctx_hd_mhandle->copy_h_to_d(), status, out);

  // Publish the context array to cleanup before the descriptor allocation can fail.
  gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle = gin_gdaki_gpu_ctx_hd_mhandle;
  NCCLCHECKGOTO(ncclGinGdakiCreateHandle(PCCL_GIN_GDAKI_PROVIDER_MLX5, PCCL_GIN_MLX5_VERSION,
                                         (void*)gin_gdaki_gpu_ctx_hd_mhandle->gpu_buf, devHandle), status, out);
  devHandle->needsProxyProgress = need_cpu_proxy;

  gdaki_ctx->counters_table = counters_table;
  gdaki_ctx->signals_table = signals_table;
  gdaki_ctx->sink_buffer.addr = sink_buffer;
  gdaki_ctx->sink_buffer.mr = sink_buffer_mr;
  gdaki_ctx->collComm = cComm;
  gdaki_ctx->devHandle = devHandle;
  gdaki_ctx->nContexts = ncontexts;

  *outDevHandle = devHandle;
  *outGinCtx = gdaki_ctx;

out:
  if (status != ncclSuccess) {
    if (gdaki_ctx) {
      // Clean up any allocated GPU memory. gverbs_qps may be NULL on cleanup
      // paths that fail before its calloc; guard the array accesses.
      if (gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle && gverbs_qps) {
        for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
          struct ncclGinGdakiMlxGPUContext *gin_gdaki_gpu_ctx =
            &gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];
          if (gin_gdaki_gpu_ctx->gdqp) {
            for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
              gverbs_qps[qp_idx] = gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
            }
            doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->gdqp);
            gin_gdaki_gpu_ctx->gdqp = nullptr;
          }
          if (gin_gdaki_gpu_ctx->companion_gdqp) {
            for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
              gverbs_qps[qp_idx] = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
            }
            doca_gpu_verbs_unexport_multi_qps_dev(gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->companion_gdqp);
            gin_gdaki_gpu_ctx->companion_gdqp = nullptr;
          }
        }
      }

      if (gdaki_ctx->gqp_groups) {
        for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
          if (gdaki_ctx->gqp_groups[qp_idx]) {
            doca_gpu_verbs_destroy_qp_group_hl(gdaki_ctx->gqp_groups[qp_idx]);
            gdaki_ctx->gqp_groups[qp_idx] = nullptr;
          }
        }
      }
      if (gdaki_ctx->gqps) {
        for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
          if (gdaki_ctx->gqps[qp_idx]) {
            doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->gqps[qp_idx]);
            gdaki_ctx->gqps[qp_idx] = nullptr;
          }
        }
      }
      if (gdaki_ctx->companion_gqps) {
        for (int qp_idx = nqps_for_comm; qp_idx < ncompanion_qps; qp_idx++) {
          if (gdaki_ctx->companion_gqps[qp_idx]) {
            doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->companion_gqps[qp_idx]);
            gdaki_ctx->companion_gqps[qp_idx] = nullptr;
          }
        }
      }

      if (gdaki_ctx->gqp_groups) {
        NCCLCHECK(ncclFree(gdaki_ctx->gqp_groups));
      }
      if (gdaki_ctx->gqps) {
        NCCLCHECK(ncclFree(gdaki_ctx->gqps));
      }
      if (gdaki_ctx->companion_gqps) {
        NCCLCHECK(ncclFree(gdaki_ctx->companion_gqps));
      }

      if (gdaki_ctx->ah) doca_verbs_ah_attr_destroy(gdaki_ctx->ah);

      if (gdaki_ctx->gdev) doca_gpu_destroy(gdaki_ctx->gdev);
    }

    if (devHandle) {
      if (devHandle->handle) {
        ncclResult_t cleanupResult = ncclHggcFree(devHandle->handle);
        if (cleanupResult != ncclSuccess) {
          WARN("Failed to release GDAKI descriptor: %d", cleanupResult);
        }
      }
      NCCLCHECK(ncclFree(devHandle));
    }

    if (sink_buffer_mr) wrap_ibv_dereg_mr(sink_buffer_mr);
    if (sink_buffer) hggcFree(sink_buffer);

    if (gin_gdaki_gpu_ctx_hd_mhandle) delete gin_gdaki_gpu_ctx_hd_mhandle;
    if (counters_table) delete counters_table;
    if (signals_table) delete signals_table;

    if (gdaki_ctx) {
      memset(gdaki_ctx, 0, sizeof(*gdaki_ctx));
      NCCLCHECK(ncclFree(gdaki_ctx));
    }
  }

  if (local_exch_info) {
    NCCLCHECK(ncclFree(local_exch_info));
  }

  if (remote_exch_info) {
    NCCLCHECK(ncclFree(remote_exch_info));
  }

  if (gverbs_qps) {
    NCCLCHECK(ncclFree(gverbs_qps));
  }

  return status;
}

ncclResult_t ncclGinGdakiMlxDestroyContext(void *ginCtx) {
  if (!ginCtx) return ncclInvalidArgument;

  ncclResult_t status = ncclSuccess;
  struct gdaki_mlx_context *gdaki_ctx = (struct gdaki_mlx_context *)ginCtx;
  struct ncclGinIbCollComm *cComm = gdaki_ctx->collComm;
  const int nranks = cComm->nranks;
  const int ncontexts = gdaki_ctx->nContexts;
  const int nqps_per_rank = ncontexts;
  const int nqps_for_comm = nqps_per_rank * nranks;  // Number of QPs for communication
  const int ncompanion_qps = nqps_for_comm * 2;      // Number of companion QPs for communication
                                                     // Double because we connect to self.
  const int nqps =
    nqps_per_rank * (nranks + 1);  // +1 for the local rank.
                                   // The last group is the responder of the local rank.

  if (gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle) {
    struct doca_gpu_verbs_qp **gverbs_qps = (struct doca_gpu_verbs_qp **)calloc(nranks, sizeof(struct doca_gpu_verbs_qp *));
    if (gverbs_qps != nullptr) {
      for (int ctx_idx = 0; ctx_idx < ncontexts; ctx_idx++) {
        struct ncclGinGdakiMlxGPUContext *gin_gdaki_gpu_ctx =
          &gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle->host_buf[ctx_idx];
        if (gin_gdaki_gpu_ctx->gdqp) {
          for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
            gverbs_qps[qp_idx] = gdaki_ctx->gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
          }
          doca_error_t docaStatus = doca_gpu_verbs_unexport_multi_qps_dev(
            gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->gdqp);
          if (docaStatus != DOCA_SUCCESS) {
            WARN("DOCA failure %d", docaStatus);
            if (status == ncclSuccess) status = ncclSystemError;
          } else {
            gin_gdaki_gpu_ctx->gdqp = nullptr;
          }
        }
        if (gin_gdaki_gpu_ctx->companion_gdqp) {
          for (int qp_idx = 0; qp_idx < nranks; qp_idx++) {
            gverbs_qps[qp_idx] = gdaki_ctx->companion_gqps[(ctx_idx * nranks) + qp_idx]->qp_gverbs;
          }
          doca_error_t docaStatus = doca_gpu_verbs_unexport_multi_qps_dev(
            gdaki_ctx->gdev, gverbs_qps, nranks, gin_gdaki_gpu_ctx->companion_gdqp);
          if (docaStatus != DOCA_SUCCESS) {
            WARN("DOCA failure %d", docaStatus);
            if (status == ncclSuccess) status = ncclSystemError;
          } else {
            gin_gdaki_gpu_ctx->companion_gdqp = nullptr;
          }
        }
      }
      NCCLCHECK(ncclFree(gverbs_qps));
    } else {
      WARN("Failed to allocate gverbs_qps array during DestroyContext; skipping per-context QP unexport");
      status = ncclSystemError;
    }
    NCCLCHECK(gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle->deallocate());
    delete gdaki_ctx->gin_gdaki_gpu_ctx_hd_mhandle;
  }

  for (int qp_idx = 0; qp_idx < nqps_for_comm; qp_idx++) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_group_hl(gdaki_ctx->gqp_groups[qp_idx]));
    gdaki_ctx->gqp_groups[qp_idx] = nullptr;
  }
  for (int qp_idx = nqps_for_comm; qp_idx < nqps; qp_idx++) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->gqps[qp_idx]));
    gdaki_ctx->gqps[qp_idx] = nullptr;
  }
  for (int qp_idx = nqps_for_comm; qp_idx < ncompanion_qps; qp_idx++) {
    DOCACHECK(doca_gpu_verbs_destroy_qp_hl(gdaki_ctx->companion_gqps[qp_idx]));
    gdaki_ctx->companion_gqps[qp_idx] = nullptr;
  }

  if (gdaki_ctx->gqp_groups) {
    NCCLCHECK(ncclFree(gdaki_ctx->gqp_groups));
  }
  if (gdaki_ctx->gqps) {
    NCCLCHECK(ncclFree(gdaki_ctx->gqps));
  }
  if (gdaki_ctx->companion_gqps) {
    NCCLCHECK(ncclFree(gdaki_ctx->companion_gqps));
  }

  if (gdaki_ctx->counters_table) {
    NCCLCHECK(gdaki_ctx->counters_table->deregister_mr());
    NCCLCHECK(gdaki_ctx->counters_table->deallocate());
    delete gdaki_ctx->counters_table;
  }
  if (gdaki_ctx->signals_table) {
    NCCLCHECK(gdaki_ctx->signals_table->deregister_mr());
    NCCLCHECK(gdaki_ctx->signals_table->deallocate());
    delete gdaki_ctx->signals_table;
  }

  if (gdaki_ctx->sink_buffer.mr) NCCLCHECK(wrap_ibv_dereg_mr(gdaki_ctx->sink_buffer.mr));
  if (gdaki_ctx->sink_buffer.addr) HGGCCHECK(hggcFree(gdaki_ctx->sink_buffer.addr));

  if (gdaki_ctx->ah) {
    DOCACHECK(doca_verbs_ah_attr_destroy(gdaki_ctx->ah));
  }

  if (gdaki_ctx->gdev) {
    DOCACHECK(doca_gpu_destroy(gdaki_ctx->gdev));
  }
  if (gdaki_ctx->devHandle) {
    NCCLCHECK(ncclHggcFree(gdaki_ctx->devHandle->handle));
    NCCLCHECK(ncclFree(gdaki_ctx->devHandle));
  }

  memset(gdaki_ctx, 0, sizeof(*gdaki_ctx));
  NCCLCHECK(ncclFree(gdaki_ctx));

  return status;
}

ncclResult_t ncclGinGdakiMlxRegMrSym(void *collComm, void *data, size_t size, int type, uint64_t mr_flags,
                                     void **mhandle, void **ginHandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  ncclResult_t status = ncclSuccess;

  struct ibv_mr *mr = nullptr;
  bool gdaki_mhandle_hd_allocated = false;
  bool rkeys_hd_allocated = false;
  __be32 rkey;

  GdakiMlxHostGPUMemHandle<struct ncclGinGdakiMlxMemHandle> *gdaki_mhandle_hd_mhandle =
    new GdakiMlxHostGPUMemHandle<struct ncclGinGdakiMlxMemHandle>();
  GdakiMlxHostGPUMemHandle<__be32> *rkeys_hd_mhandle =
    new GdakiMlxHostGPUMemHandle<__be32>();

  struct gdaki_mlx_mem_handle *gdaki_mhandle =
    (struct gdaki_mlx_mem_handle *)calloc(1, sizeof(*gdaki_mhandle));
  if (gdaki_mhandle == nullptr) { status = ncclSystemError; goto out; }

  {
    bool force_strict_ordering = (mr_flags & NCCL_NET_MR_FLAG_FORCE_SO);
    NCCLCHECKGOTO(gdakiMlxRegMr(&mr, cComm->ib.pd, data, size,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                                IBV_ACCESS_REMOTE_ATOMIC, force_strict_ordering),
                  status, out);
  }

  rkey = htobe32(mr->rkey);
  NCCLCHECKGOTO(rkeys_hd_mhandle->allocate(cComm->nranks), status, out);
  rkeys_hd_allocated = true;
  NCCLCHECKGOTO(cComm->allGather(cComm, &rkey, rkeys_hd_mhandle->host_buf, sizeof(__be32)), status, out);
  NCCLCHECKGOTO(rkeys_hd_mhandle->copy_h_to_d(), status, out);

  NCCLCHECKGOTO(gdaki_mhandle_hd_mhandle->allocate(1), status, out);
  gdaki_mhandle_hd_allocated = true;
  gdaki_mhandle_hd_mhandle->host_buf->rkeys = rkeys_hd_mhandle->gpu_buf;
  gdaki_mhandle_hd_mhandle->host_buf->lkey = htobe32(mr->lkey);
  NCCLCHECKGOTO(gdaki_mhandle_hd_mhandle->copy_h_to_d(), status, out);

  gdaki_mhandle->type = type;
  gdaki_mhandle->mr = mr;
  gdaki_mhandle->gdaki_mhandle_hd_mhandle = gdaki_mhandle_hd_mhandle;
  gdaki_mhandle->rkeys_hd_mhandle = rkeys_hd_mhandle;

  INFO(NCCL_NET, "[%d] Registered MR: data=%p, size=%zu, lkey(be32)=%#x, rkey(be32)=%#x",
       cComm->rank, data, size, htobe32(mr->lkey), htobe32(mr->rkey));

  *mhandle = (void *)gdaki_mhandle;
  *ginHandle = (void *)gdaki_mhandle_hd_mhandle->gpu_buf;
  return ncclSuccess;

out:
  if (gdaki_mhandle_hd_allocated) gdaki_mhandle_hd_mhandle->deallocate();
  if (rkeys_hd_allocated) rkeys_hd_mhandle->deallocate();
  delete gdaki_mhandle_hd_mhandle;
  delete rkeys_hd_mhandle;
  if (mr) wrap_ibv_dereg_mr(mr);
  if (gdaki_mhandle) {
    NCCLCHECK(ncclFree(gdaki_mhandle));
  }
  return status;
}

ncclResult_t ncclGinGdakiMlxDeregMrSym(void *collComm, void *mhandle) {
  struct ncclGinIbCollComm *cComm = (struct ncclGinIbCollComm *)collComm;
  struct gdaki_mlx_mem_handle *gdaki_mhandle = (struct gdaki_mlx_mem_handle *)mhandle;
  struct ibv_mr *mr = gdaki_mhandle->mr;

  INFO(NCCL_NET, "[%d] Unregistering MR: lkey(be32)=%#x, rkey(be32)=%#x", cComm->rank,
       htobe32(mr->lkey), htobe32(mr->rkey));

  NCCLCHECK(wrap_ibv_dereg_mr(mr));

  NCCLCHECK(gdaki_mhandle->gdaki_mhandle_hd_mhandle->deallocate());
  delete gdaki_mhandle->gdaki_mhandle_hd_mhandle;
  NCCLCHECK(gdaki_mhandle->rkeys_hd_mhandle->deallocate());
  delete gdaki_mhandle->rkeys_hd_mhandle;

  memset(gdaki_mhandle, 0, sizeof(*gdaki_mhandle));

  NCCLCHECK(ncclFree(gdaki_mhandle));

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiMlxProgress(void *ctx) {
  struct gdaki_mlx_context *gdakiCtx = (struct gdaki_mlx_context *)ctx;
  const int ncontexts = gdakiCtx->nContexts;
  const int nranks = gdakiCtx->collComm->nranks;
  const int nqpsPerRank = ncontexts;
  const int nqpsForComm = nqpsPerRank * nranks;  // Number of QPs for communication
  bool has_progressed = true;
  bool progressed;

  while (has_progressed) {
    has_progressed = false;
    for (int qpIdx = 0; qpIdx < nqpsForComm; qpIdx++) {
      struct doca_gpu_verbs_qp *qp = gdakiCtx->gqps[qpIdx]->qp_gverbs;
      if (qp->cpu_proxy) {
        DOCACHECK(doca_gpu_verbs_cpu_proxy_progress(qp, &progressed));
        has_progressed |= progressed;
      }

      qp = gdakiCtx->companion_gqps[qpIdx]->qp_gverbs;
      if (qp->cpu_proxy) {
        DOCACHECK(doca_gpu_verbs_cpu_proxy_progress(qp, &progressed));
        has_progressed |= progressed;
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclGinGdakiMlxQueryLastError(void *ginCtx, bool *hasError) {
  struct gdaki_mlx_context *gdakiCtx = (struct gdaki_mlx_context *)ginCtx;
  bool hasError_ = false;
  const int ncontexts = gdakiCtx->nContexts;
  const int nranks = gdakiCtx->collComm->nranks;
  const int nqpsPerRank = ncontexts;
  const int nqpsForComm = nqpsPerRank * nranks;  // Number of QPs for communication

  // We throttle the frequency of these queries since they can easily take 250us.
  uint64_t now = clockNano();
  if ((now - gdakiCtx->last_error_query_time) / 1e9 < ncclParamGinGdakiMlxErrorQuerySec()) {
    goto exit;
  }
  gdakiCtx->last_error_query_time = now;

  for (int qpIdx = 0; qpIdx < nqpsForComm; qpIdx++) {
    struct doca_gpu_verbs_qp *qp = gdakiCtx->gqps[qpIdx]->qp_gverbs;
    struct doca_gpu_verbs_qp_error_info errorInfo;
    DOCACHECK(doca_gpu_verbs_query_last_error(qp, &errorInfo));
    hasError_ |= errorInfo.has_error;
    if (hasError_) break;

    qp = gdakiCtx->companion_gqps[qpIdx]->qp_gverbs;
    DOCACHECK(doca_gpu_verbs_query_last_error(qp, &errorInfo));
    hasError_ |= errorInfo.has_error;
    if (hasError_) break;
  }
exit:
  *hasError = hasError_;
  return ncclSuccess;
}
