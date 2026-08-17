/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "ibvwrap.h"
#include <sys/types.h>
#include <unistd.h>

#include <dlfcn.h>
#include "core.h"

static enum { ibvUninitialized, ibvInitializing, ibvInitialized, ibvError } ibvState = ibvUninitialized;

/*Function Pointers*/
int (*ibv_internal_fork_init)(void);
struct ibv_device** (*ibv_internal_get_device_list)(int *num_devices);
void (*ibv_internal_free_device_list)(struct ibv_device **list);
const char * (*ibv_internal_get_device_name)(struct ibv_device *device);
struct ibv_context* (*ibv_internal_open_device)(struct ibv_device* device);
int (*ibv_internal_close_device)(struct ibv_context *context);
int (*ibv_internal_get_async_event)(struct ibv_context *context, struct ibv_async_event *event);
void (*ibv_internal_ack_async_event)(struct ibv_async_event *event);
int (*ibv_internal_query_device)(struct ibv_context *context, struct ibv_device_attr *device_attr);
int (*ibv_internal_query_port)(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr);
int (*ibv_internal_query_gid)(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid);
int (*ibv_internal_query_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr);
struct ibv_pd * (*ibv_internal_alloc_pd)(struct ibv_context *context);
int (*ibv_internal_dealloc_pd)(struct ibv_pd *pd);
struct ibv_mr * (*ibv_internal_reg_mr)(struct ibv_pd *pd, void *addr, size_t length, int access);
// Optional: only available with libibverbs >= 1.8 (IBVERBS_1.8 symver). Required by
// the GDAKI GIN MR path so MRs can be registered with iova=0; the device-side kernel
// uses raddr as an offset within the MR. Stays NULL on older libibverbs.
struct ibv_mr * (*ibv_internal_reg_mr_iova2)(struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, unsigned int access);
// Optional: only available with libibverbs >= 1.12 (IBVERBS_1.12 symver). Stays NULL
// if the running libibverbs is too old; callers (DMA-BUF probe / GIN MR path)
// check for NULL and fall back / report unsupported.
struct ibv_mr * (*ibv_internal_reg_dmabuf_mr)(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access);
int (*ibv_internal_dereg_mr)(struct ibv_mr *mr);
struct ibv_cq * (*ibv_internal_create_cq)(struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector);
int (*ibv_internal_destroy_cq)(struct ibv_cq *cq);
struct ibv_qp * (*ibv_internal_create_qp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr);
int (*ibv_internal_modify_qp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
int (*ibv_internal_destroy_qp)(struct ibv_qp *qp);
const char * (*ibv_internal_event_type_str)(enum ibv_event_type event);

// IBVERBS Library versioning
#define IBVERBS_VERSION "IBVERBS_1.1"

ncclResult_t wrap_ibv_symbols(void) {
  if (ibvState == ibvInitialized)
    return ncclSuccess;
  if (ibvState == ibvError)
    return ncclSystemError;

  if (__sync_bool_compare_and_swap(&ibvState, ibvUninitialized, ibvInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (ibvState == ibvInitializing) pthread_yield();
    return (ibvState == ibvInitialized) ? ncclSuccess : ncclSystemError;
  }

  static void* ibvhandle = NULL;
  void* tmp;
  void** cast;

  ibvhandle=dlopen("libibverbs.so", RTLD_NOW);
  if (!ibvhandle) {
    ibvhandle=dlopen("libibverbs.so.1", RTLD_NOW);
    if (!ibvhandle) {
      WARN("Failed to open libibverbs.so[.1]");
      goto teardown;
    }
  }

#define LOAD_SYM(handle, symbol, funcptr) do {         \
    cast = (void**)&funcptr;                             \
    tmp = dlvsym(handle, symbol, IBVERBS_VERSION);       \
    if (tmp == NULL) {                                   \
      WARN("dlvsym failed on %s - %s version %s", symbol, dlerror(), IBVERBS_VERSION);  \
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

  LOAD_SYM(ibvhandle, "ibv_get_device_list", ibv_internal_get_device_list);
  LOAD_SYM(ibvhandle, "ibv_free_device_list", ibv_internal_free_device_list);
  LOAD_SYM(ibvhandle, "ibv_get_device_name", ibv_internal_get_device_name);
  LOAD_SYM(ibvhandle, "ibv_open_device", ibv_internal_open_device);
  LOAD_SYM(ibvhandle, "ibv_close_device", ibv_internal_close_device);
  LOAD_SYM(ibvhandle, "ibv_get_async_event", ibv_internal_get_async_event);
  LOAD_SYM(ibvhandle, "ibv_ack_async_event", ibv_internal_ack_async_event);
  LOAD_SYM(ibvhandle, "ibv_query_device", ibv_internal_query_device);
  LOAD_SYM(ibvhandle, "ibv_query_port", ibv_internal_query_port);
  LOAD_SYM(ibvhandle, "ibv_query_gid", ibv_internal_query_gid);
  LOAD_SYM(ibvhandle, "ibv_query_qp", ibv_internal_query_qp);
  LOAD_SYM(ibvhandle, "ibv_alloc_pd", ibv_internal_alloc_pd);
  LOAD_SYM(ibvhandle, "ibv_dealloc_pd", ibv_internal_dealloc_pd);
  LOAD_SYM(ibvhandle, "ibv_reg_mr", ibv_internal_reg_mr);
  LOAD_SYM(ibvhandle, "ibv_dereg_mr", ibv_internal_dereg_mr);
  // ibv_reg_mr_iova2 is symver-tagged IBVERBS_1.8. dlvsym on IBVERBS_1.1 won't
  // find it; try the unversioned dlsym. NULL is acceptable (older libibverbs)
  // and is checked by callers (GDAKI GIN registers MRs with iova=0 via this).
  ibv_internal_reg_mr_iova2 = (struct ibv_mr* (*)(struct ibv_pd*, void*, size_t, uint64_t, unsigned int))
      dlsym(ibvhandle, "ibv_reg_mr_iova2");
  // ibv_reg_dmabuf_mr is symver-tagged IBVERBS_1.12. dlvsym on IBVERBS_1.1 won't
  // find it; try the unversioned dlsym so we get whatever the loader picks. NULL
  // is acceptable (older libibverbs) and is checked by callers.
  ibv_internal_reg_dmabuf_mr = (struct ibv_mr* (*)(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int))
      dlsym(ibvhandle, "ibv_reg_dmabuf_mr");
  LOAD_SYM(ibvhandle, "ibv_create_cq", ibv_internal_create_cq);
  LOAD_SYM(ibvhandle, "ibv_destroy_cq", ibv_internal_destroy_cq);
  LOAD_SYM(ibvhandle, "ibv_create_qp", ibv_internal_create_qp);
  LOAD_SYM(ibvhandle, "ibv_modify_qp", ibv_internal_modify_qp);
  LOAD_SYM(ibvhandle, "ibv_destroy_qp", ibv_internal_destroy_qp);
  LOAD_SYM(ibvhandle, "ibv_fork_init", ibv_internal_fork_init);
  LOAD_SYM(ibvhandle, "ibv_event_type_str", ibv_internal_event_type_str);

  ibvState = ibvInitialized;
  return ncclSuccess;

teardown:
  ibv_internal_get_device_list = NULL;
  ibv_internal_free_device_list = NULL;
  ibv_internal_get_device_name = NULL;
  ibv_internal_open_device = NULL;
  ibv_internal_close_device = NULL;
  ibv_internal_get_async_event = NULL;
  ibv_internal_ack_async_event = NULL;
  ibv_internal_query_device = NULL;
  ibv_internal_query_port = NULL;
  ibv_internal_query_gid = NULL;
  ibv_internal_query_qp = NULL;
  ibv_internal_alloc_pd = NULL;
  ibv_internal_dealloc_pd = NULL;
  ibv_internal_reg_mr = NULL;
  ibv_internal_reg_mr_iova2 = NULL;
  ibv_internal_reg_dmabuf_mr = NULL;
  ibv_internal_dereg_mr = NULL;
  ibv_internal_create_cq = NULL;
  ibv_internal_destroy_cq = NULL;
  ibv_internal_create_qp = NULL;
  ibv_internal_modify_qp = NULL;
  ibv_internal_destroy_qp = NULL;
  ibv_internal_fork_init = NULL;
  ibv_internal_event_type_str = NULL;

  if (ibvhandle != NULL) dlclose(ibvhandle);
  ibvState = ibvError;
  return ncclSystemError;
}

#define IBV_PTR_CHECK_ERRNO(name_internal, call, retval, error_retval, name) \
  if (name_internal == NULL) { \
     ERROR("lib wrapper not initialized: %s", name); \
     return ncclInternalError; \
  } \
  retval = call; \
  if (retval == error_retval) { \
    int err = errno; \
    ERROR("Call to %s failed: call=%s '%s'", name, #call, strerror(err)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_PTR_CHECK(name_internal, call, retval, error_retval, name) \
  if (name_internal == NULL) { \
     ERROR("lib wrapper not initialized: %s", name); \
     return ncclInternalError; \
  } \
  retval = call; \
  if (retval == error_retval) { \
    ERROR("Call to %s failed: call=%s retval=%p", name, #call, (void*)retval); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK_RET_ERRNO(name_internal, call, success_retval, name) \
  if (name_internal == NULL) { \
     ERROR("lib wrapper not initialized: %s", name); \
     return ncclInternalError; \
  } \
  int ret = call; \
  if (ret != success_retval) { \
    ERROR("Call to %s failed: call=%s '%s'", name, #call, strerror(ret)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_INT_CHECK_ERRNO(name_internal, call, error_retval, name) \
  if (name_internal == NULL) { \
     ERROR("lib wrapper not initialized: %s", name); \
     return ncclInternalError; \
  } \
  int ret = call; \
  if (ret == error_retval) { \
    int err = errno; \
    ERROR("Call to %s failed: call=%s '%s'", name, #call, strerror(err)); \
    return ncclSystemError; \
  } \
  return ncclSuccess;

#define IBV_PASSTHRU(name_internal, call) \
  if (name_internal == NULL) { \
     ERROR("lib wrapper not initialized."); \
     return ncclInternalError; \
  } \
  call; \
  return ncclSuccess;

ncclResult_t wrap_ibv_fork_init() {
  IBV_INT_CHECK_RET_ERRNO(ibv_internal_fork_init, ibv_internal_fork_init(), 0, "ibv_fork_init");
}

ncclResult_t wrap_ibv_get_device_list(struct ibv_device ***ret, int *num_devices) {
  *ret = ibv_internal_get_device_list(num_devices);
  if (*ret == NULL) *num_devices = 0;
  return ncclSuccess;
}

ncclResult_t wrap_ibv_free_device_list(struct ibv_device **list) {
  IBV_PASSTHRU(ibv_internal_free_device_list, ibv_internal_free_device_list(list));
}

const char *wrap_ibv_get_device_name(struct ibv_device *device) {
  if (ibv_internal_get_device_name == NULL) {
    WARN("lib wrapper not initialized.");
    exit(-1);
  }
  return ibv_internal_get_device_name(device);
}

ncclResult_t wrap_ibv_open_device(struct ibv_context **ret, struct ibv_device *device) { /*returns 0 on success, -1 on failure*/
  IBV_PTR_CHECK_ERRNO(ibv_internal_open_device, ibv_internal_open_device(device), *ret, NULL, "ibv_open_device");
}

ncclResult_t wrap_ibv_close_device(struct ibv_context *context) { /*returns 0 on success, -1 on failure*/
  IBV_INT_CHECK_ERRNO(ibv_internal_close_device, ibv_internal_close_device(context), -1, "ibv_close_device");
}

ncclResult_t wrap_ibv_get_async_event(struct ibv_context *context, struct ibv_async_event *event) { /*returns 0 on success, and -1 on error*/
  IBV_INT_CHECK_ERRNO(ibv_internal_get_async_event, ibv_internal_get_async_event(context, event), -1, "ibv_get_async_event");
}

ncclResult_t wrap_ibv_ack_async_event(struct ibv_async_event *event) {
  IBV_PASSTHRU(ibv_internal_ack_async_event, ibv_internal_ack_async_event(event));
}

ncclResult_t wrap_ibv_query_device(struct ibv_context *context, struct ibv_device_attr *device_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_internal_query_device, ibv_internal_query_device(context, device_attr), 0, "ibv_query_device");
}

ncclResult_t wrap_ibv_query_port(struct ibv_context *context, uint8_t port_num, struct ibv_port_attr *port_attr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_internal_query_port, ibv_internal_query_port(context, port_num, port_attr), 0, "ibv_query_port");
}

ncclResult_t wrap_ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index, union ibv_gid *gid) {
  IBV_INT_CHECK_ERRNO(ibv_internal_query_gid, ibv_internal_query_gid(context, port_num, index, gid), -1, "ibv_query_gid");
}

ncclResult_t wrap_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask, struct ibv_qp_init_attr *init_attr) {
  IBV_INT_CHECK_RET_ERRNO(ibv_internal_query_qp, ibv_internal_query_qp(qp, attr, attr_mask, init_attr), 0, "ibv_query_qp");
}

ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd **ret, struct ibv_context *context) {
  IBV_PTR_CHECK_ERRNO(ibv_internal_alloc_pd, ibv_internal_alloc_pd(context), *ret, NULL, "ibv_alloc_pd");
}

ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd *pd) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_internal_dealloc_pd, ibv_internal_dealloc_pd(pd), 0, "ibv_dealloc_pd");
}

ncclResult_t wrap_ibv_reg_mr(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, int access) {
  IBV_PTR_CHECK_ERRNO(ibv_internal_reg_mr, ibv_internal_reg_mr(pd, addr, length, access), *ret, NULL, "ibv_reg_mr");
}

struct ibv_mr * wrap_direct_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access) {
  if (ibv_internal_reg_mr == NULL) {
    ERROR("lib wrapper not initialized.");
    return NULL;
  }
  return ibv_internal_reg_mr(pd, addr, length, access);
}

ncclResult_t wrap_ibv_reg_mr_iova2(struct ibv_mr **ret, struct ibv_pd *pd, void *addr, size_t length, uint64_t iova, unsigned int access) {
  if (ibv_internal_reg_mr_iova2 == NULL) {
    // Probe-mode call: callers may invoke with all-zero args just to test availability.
    if (ret == NULL) return ncclInternalError;
    WARN("ibv_reg_mr_iova2 unavailable (libibverbs < 1.8)");
    return ncclSystemError;
  }
  if (ret == NULL) return ncclSuccess;  // probe-only call, symbol present
  *ret = ibv_internal_reg_mr_iova2(pd, addr, length, iova, access);
  if (*ret == NULL) {
    int e = errno;
    WARN("ibv_reg_mr_iova2 failed: pd=%p addr=%p length=%zu iova=0x%llx access=0x%x errno=%d (%s)",
         pd, addr, length, (unsigned long long)iova, access, e, strerror(e));
    errno = e;
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrap_ibv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
  IBV_PTR_CHECK_ERRNO(ibv_internal_reg_dmabuf_mr, ibv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access), *ret, NULL, "ibv_reg_dmabuf_mr");
}

struct ibv_mr * wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access) {
  if (ibv_internal_reg_dmabuf_mr == NULL) {
    errno = EOPNOTSUPP;
    return NULL;
  }
  return ibv_internal_reg_dmabuf_mr(pd, offset, length, iova, fd, access);
}

ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr *mr) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_internal_dereg_mr, ibv_internal_dereg_mr(mr), 0, "ibv_dereg_mr");
}

ncclResult_t wrap_ibv_create_cq(struct ibv_cq **ret, struct ibv_context *context, int cqe, void *cq_context, struct ibv_comp_channel *channel, int comp_vector) {
  IBV_PTR_CHECK_ERRNO(ibv_internal_create_cq, ibv_internal_create_cq(context, cqe, cq_context, channel, comp_vector), *ret, NULL, "ibv_create_cq");
}

ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq *cq) {
  IBV_INT_CHECK_RET_ERRNO(ibv_internal_destroy_cq, ibv_internal_destroy_cq(cq), 0, "ibv_destroy_cq");
}

ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp *qp) {
  IBV_INT_CHECK_RET_ERRNO(ibv_internal_destroy_qp, ibv_internal_destroy_qp(qp), 0, "ibv_destroy_qp");
}

ncclResult_t wrap_ibv_create_qp(struct ibv_qp **ret, struct ibv_pd *pd, struct ibv_qp_init_attr *qp_init_attr) {
  IBV_PTR_CHECK_ERRNO(ibv_internal_create_qp, ibv_internal_create_qp(pd, qp_init_attr), *ret, NULL, "ibv_create_qp");
}

ncclResult_t wrap_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask) { /*returns 0 on success, or the value of errno on failure (which indicates the failure reason)*/
  IBV_INT_CHECK_RET_ERRNO(ibv_internal_modify_qp, ibv_internal_modify_qp(qp, attr, attr_mask), 0, "ibv_modify_qp");
}

ncclResult_t wrap_ibv_event_type_str(char **ret, enum ibv_event_type event) {
  *ret = (char *) ibv_internal_event_type_str(event);
  return ncclSuccess;
}

// Helper function to convert IB work completion status to string
const char* ibvWcStatusStr(enum ibv_wc_status status) {
  switch (status) {
    case IBV_WC_SUCCESS:            return "IBV_WC_SUCCESS";
    case IBV_WC_LOC_LEN_ERR:        return "IBV_WC_LOC_LEN_ERR";
    case IBV_WC_LOC_QP_OP_ERR:      return "IBV_WC_LOC_QP_OP_ERR";
    case IBV_WC_LOC_EEC_OP_ERR:     return "IBV_WC_LOC_EEC_OP_ERR";
    case IBV_WC_LOC_PROT_ERR:       return "IBV_WC_LOC_PROT_ERR";
    case IBV_WC_WR_FLUSH_ERR:       return "IBV_WC_WR_FLUSH_ERR";
    case IBV_WC_MW_BIND_ERR:        return "IBV_WC_MW_BIND_ERR";
    case IBV_WC_BAD_RESP_ERR:       return "IBV_WC_BAD_RESP_ERR";
    case IBV_WC_LOC_ACCESS_ERR:     return "IBV_WC_LOC_ACCESS_ERR";
    case IBV_WC_REM_INV_REQ_ERR:    return "IBV_WC_REM_INV_REQ_ERR";
    case IBV_WC_REM_ACCESS_ERR:     return "IBV_WC_REM_ACCESS_ERR";
    case IBV_WC_REM_OP_ERR:         return "IBV_WC_REM_OP_ERR";
    case IBV_WC_RETRY_EXC_ERR:      return "IBV_WC_RETRY_EXC_ERR";
    case IBV_WC_RNR_RETRY_EXC_ERR:  return "IBV_WC_RNR_RETRY_EXC_ERR";
    case IBV_WC_LOC_RDD_VIOL_ERR:   return "IBV_WC_LOC_RDD_VIOL_ERR";
    case IBV_WC_REM_INV_RD_REQ_ERR: return "IBV_WC_REM_INV_RD_REQ_ERR";
    case IBV_WC_REM_ABORT_ERR:      return "IBV_WC_REM_ABORT_ERR";
    case IBV_WC_INV_EECN_ERR:       return "IBV_WC_INV_EECN_ERR";
    case IBV_WC_INV_EEC_STATE_ERR:  return "IBV_WC_INV_EEC_STATE_ERR";
    case IBV_WC_FATAL_ERR:          return "IBV_WC_FATAL_ERR";
    case IBV_WC_RESP_TIMEOUT_ERR:   return "IBV_WC_RESP_TIMEOUT_ERR";
    case IBV_WC_GENERAL_ERR:        return "IBV_WC_GENERAL_ERR";
    default:                        return "UNKNOWN_STATUS";
  }
}

// Helper function to convert IB work completion opcode to string
const char* ibvWcOpcodeStr(enum ibv_wc_opcode opcode) {
  switch (opcode) {
    case IBV_WC_SEND:               return "IBV_WC_SEND";
    case IBV_WC_RDMA_WRITE:         return "IBV_WC_RDMA_WRITE";
    case IBV_WC_RDMA_READ:          return "IBV_WC_RDMA_READ";
    case IBV_WC_COMP_SWAP:          return "IBV_WC_COMP_SWAP";
    case IBV_WC_FETCH_ADD:          return "IBV_WC_FETCH_ADD";
    case IBV_WC_BIND_MW:            return "IBV_WC_BIND_MW";
    case IBV_WC_RECV:               return "IBV_WC_RECV";
    case IBV_WC_RECV_RDMA_WITH_IMM: return "IBV_WC_RECV_RDMA_WITH_IMM";
    default:                        return "UNKNOWN_OPCODE";
  }
}

const char* ibvWrOpcodeStr(enum ibv_wr_opcode opcode) {
  switch (opcode) {
    case IBV_WR_RDMA_WRITE:          return "IBV_WR_RDMA_WRITE";
    case IBV_WR_RDMA_WRITE_WITH_IMM: return "IBV_WR_RDMA_WRITE_WITH_IMM";
    case IBV_WR_SEND:                return "IBV_WR_SEND";
    case IBV_WR_SEND_WITH_IMM:       return "IBV_WR_SEND_WITH_IMM";
    case IBV_WR_RDMA_READ:           return "IBV_WR_RDMA_READ";
    case IBV_WR_ATOMIC_CMP_AND_SWP:  return "IBV_WR_ATOMIC_CMP_AND_SWP";
    case IBV_WR_ATOMIC_FETCH_AND_ADD: return "IBV_WR_ATOMIC_FETCH_AND_ADD";
    default:                          return "UNKNOWN_OPCODE";
  }
}