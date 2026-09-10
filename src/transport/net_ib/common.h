/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_IB_COMMON_H_
#define NET_IB_COMMON_H_

#include "pccl_internal.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "utils.h"
#include "param.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>

#include "ibvwrap.h"
#include "mlx5/mlx5dvwrap.h"

#define MAXNAMESIZE 64
extern char ncclIbIfName[MAX_IF_NAME_SIZE+1];
extern union ncclSocketAddress ncclIbIfAddr;

struct ncclIbStats {
    int fatalErrorCount;
};

extern int ncclNIbDevs;
extern int ncclIbRelaxedOrderingEnabled;
enum ncclIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
};
struct ncclIbDev {
    int device;
    uint64_t guid;
    uint8_t port;
    uint8_t link;
    int speed;
    ibv_context* context;
    char devName[MAXNAMESIZE];
    char ifName[MAX_IF_NAME_SIZE+1];
    char* pciPath;
    int realPort;
    int maxQp;
    int gidIndex;
    int ar;
    uint32_t oooRqSize;
    struct ncclIbStats stats;
    enum ncclIbProvider ibProvider;
    struct ibv_pd* pd;
    int pdRefs;
    std::mutex mutex;
    // -1 = not probed, 0 = probed and unsupported, 1 = probed and supported.
    int dmaBufSupported;
};

#define MAX_IB_DEVS 32
extern struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
extern int ncclGinIbGdakiDevIndexes[MAX_IB_DEVS];

#define NCCL_NET_IB_MAX_RECVS 8

#define NCCL_NET_IB_REQ_UNUSED 0
#define NCCL_NET_IB_REQ_SEND 1
#define NCCL_NET_IB_REQ_RECV 2
#define NCCL_NET_IB_REQ_FLUSH 3
#define NCCL_NET_IB_REQ_GIN_IPUT 4
#define NCCL_NET_IB_REQ_GIN_IGET 5

extern const char* ncclIbReqTypeStr[];

struct ncclNetCommConfig_t {
  int trafficClass;
};
#define NCCL_NET_TRAFFIC_CLASS_UNDEF -1

// Maximal number of QPs a communicator can have for data transfers
#define NCCL_IB_MAX_QPS 128

struct ncclIbRequest {
  struct ncclIbNetCommBase* base;
  int type;
  int events;
  struct ncclIbNetCommDevBase* devBase;
  struct ncclSocket* sock;
  int nreqs;
  union {
    struct {
    int size;
    void* data;
    uint32_t lkey;
    int offset;
    } send;
    struct {
      int sizes[NCCL_NET_IB_MAX_RECVS];
    } recv;
  };
  // GIN proxy bookkeeping (used only for GIN iput/iget/iflush requests).
  void* ginProxyCtx;
  uint32_t ginPeerRank;
};

static inline void ncclIbAddEvent(struct ncclIbRequest* req, int /*devIndex*/) {
  req->events++;
}

struct ncclIbNetCommDevBase {
  int ibDevN;
    struct ibv_pd* pd;
    struct ibv_cq* cq;
};

enum ncclIbCommState {
    ncclIbCommStateStart = 0,
    ncclIbCommStateConnect = 1,
    ncclIbCommStateAccept = 3,
    ncclIbCommStateSend = 4,
    ncclIbCommStateRecv = 5,
    ncclIbCommStateConnected = 6,
};

enum ncclIbRecvMatchingScheme {
  BY_INDEX = 0,
  BY_ID = 1,
};

struct ncclIbCommStage {
    enum ncclIbCommState state;
    int offset;
    void* buffer;
    void* comm;
};

struct ncclIbListenComm {
  int dev;
  struct ncclSocket sock;
  struct ncclIbCommStage stage;
};

struct ncclIbSendFifo {
    uint64_t addr;
    int      size;
    uint32_t rkey;
    // TODO: Match the upstream NCCL CTS FIFO layout after porting its shared
    // logical request-ID generation and request completion records. Those
    // features remove the need for reqIdx/reserved and these narrowed fields.
    uint8_t  nreqs;
    uint8_t  reqIdx;
    uint16_t reserved;
    uint32_t tag;
    uint64_t idx;
};

// A per-dev struct for netIbSendComm
struct alignas(8) ncclIbSendCommDev {
    struct ncclIbNetCommDevBase base;
    // Scratchpad MR used as the local SGE of FETCH_AND_ADD atomic ops issued
    // by the GIN proxy iputSignal path.
    struct ibv_mr* putSignalScratchpadMr;
};

// We need to support NCCL_NET_MAX_REQUESTS for each concurrent receive
#define NET_IB_MAX_REQUESTS (NCCL_NET_MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS)
static_assert(NET_IB_MAX_REQUESTS <= 256, "request id are encoded in wr_id and we need up to 8 requests ids per completion");

struct alignas(32) ncclIbNetCommBase {
    bool isSend;
    struct ncclIbRequest reqs[NET_IB_MAX_REQUESTS];
    struct ibv_qp* qps[NCCL_IB_MAX_QPS];
    uint64_t fifoHead;
    int nqps;
    int qpIndex;
    int splitDataOnQps;
    struct ncclSocket sock;
    int ready;
    uint32_t remOooRqSize;
    int oooRq;
    int recvMatchingScheme;
    // statistics about the comm
    struct ncclIbStats stats;
};

struct ncclIbNetCommDevBase* ncclIbGetNetCommDevBase(ncclIbNetCommBase* base, int devIndex);

// Each request is transfered over all devices, and depending on the
// "splitDataOnQps" configuration parameter, a request may be transffered over
// a single QP per device or on all QPs of each device.
static inline int ncclIbCommBaseGetNqpsPerRequest(struct ncclIbNetCommBase* baseComm) {
  if (baseComm->nqps <= 0) {
    ERROR("NET/IB: %s: Invalid nqps=%d (comm=%p)", __func__, baseComm->nqps, baseComm);
    return ncclInternalError;
  }
  return (baseComm->splitDataOnQps) ? baseComm->nqps : 1;
}

struct ncclIbSendComm {
    struct ncclIbNetCommBase base;
    // Start with CTS FIFO and ibv structs as they have alignment restrictions

    // CTS FIFO from which the sender reads the Clear-to-Send (CTS) messages that
    // are written by the receiver (The receiver side writes into it upon
    // issuing a (multi-)receive request). Each row in the 2D array corresponds
    // to a single CTS message but can describe multiple recv-requests issued
    // on the receiver side.
    struct ncclIbSendFifo ctsFifo[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
    struct ibv_sge sges[NCCL_NET_IB_MAX_RECVS];
    struct ibv_send_wr wrs[NCCL_NET_IB_MAX_RECVS + 1];

    struct ncclIbSendCommDev dev;
    // Array of pointers to store the send requests for faster access. The
    // pointers are pointing into requests stored in ncclIbNetCommBase::reqs[]
    // array. The requests are inserted to this array based on the "slot" they
    // are associated with.
    struct ncclIbRequest* fifoReqs[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];

    struct ibv_mr* fifoMr;
    int netDev;
    int ar; // Use adaptive routing when enabled on the selected device
    // Local scratchpad target for the GIN proxy iputSignal atomic FETCH_AND_ADD.
    uint64_t putSignalScratchpad;
};
// The SendFifo needs to be 32-byte aligned and each element needs
// to be a 32-byte multiple, so that an entry does not get split and
// written out of order when IB Relaxed Ordering is enabled
static_assert((sizeof(struct ncclIbNetCommBase) % 32) == 0, "ncclIbNetCommBase size must be 32-byte multiple to ensure ctsFifo is at proper offset");
static_assert((offsetof(struct ncclIbSendComm, ctsFifo) % 32) == 0, "ncclIbSendComm ctsFifo must be 32-byte aligned");
static_assert(NCCL_NET_IB_MAX_RECVS <= UINT8_MAX, "nreqs must fit in ncclIbSendFifo::nreqs");
static_assert(NET_IB_MAX_REQUESTS <= UINT8_MAX + 1, "request index must fit in ncclIbSendFifo::reqIdx");
static_assert(sizeof(struct ncclIbSendFifo) == 32, "ncclIbSendFifo element size must be 32 bytes");
static_assert((offsetof(struct ncclIbSendComm, sges) % 32) == 0, "sges must be 32-byte aligned");
static_assert((offsetof(struct ncclIbSendComm, wrs) % 32) == 0, "wrs must be 32-byte aligned");

struct ncclIbGpuFlush {
    int enabled;
    int useWrite;  // 1 = use RDMA Write when the NIC does not support RDMA Read
    int hostMem;
    struct ibv_mr* hostMr;
    struct ibv_sge sge;
    struct ibv_qp* qp;
};

// This structure describes the FIFO which the receiver uses when it sends CTS
// messages to the sender.
struct ncclIbRemCtsFifo {
    // A "shadow" structure of the sender's CTS FIFO in which the receiver tracks
    // the CTS FIFO locally on its side. Receiver uses this memory to place the
    // CTS messages and populates the RDMA message "gather address" with the
    // memory of the CTS message that is sent.
    struct ncclIbSendFifo elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
    uint64_t fifoTail;
    uint64_t addr;
    uint32_t rkey;
    uint32_t tail;
    uint32_t flags;
    struct ibv_mr* mr;
    struct ibv_sge sge;
};

struct alignas(16) ncclIbRecvCommDev {
    struct ncclIbNetCommDevBase base;
};

struct ncclIbRecvComm {
    struct ncclIbNetCommBase base;
    struct ncclIbRecvCommDev dev;
    // Structure to hold all the related structures regarding the CTS FIFO
    // structure.
    struct ncclIbRemCtsFifo remCtsFifo;
    struct ibv_recv_wr ibRecvWorkRequest;
    struct ncclIbRequest* recvReqs[NET_IB_MAX_REQUESTS];
    struct ncclIbGpuFlush gpuFlush;
};
static_assert((offsetof(struct ncclIbRecvComm, remCtsFifo) % 32) == 0, "ncclIbSendComm fifo must be 32-byte aligned");

ncclResult_t ncclIbBaseCommInit(struct ncclIbNetCommBase* baseComm, bool isSend);
ncclResult_t ncclIbRecvCommInit(struct ncclIbRecvComm* recvComm);
ncclResult_t ncclIbSendCommInit(struct ncclIbSendComm* sendComm);

static ncclResult_t ncclIbStatsInit(struct ncclIbStats* stat) {
  __atomic_store_n(&stat->fatalErrorCount, 0, std::memory_order_relaxed);
  return ncclSuccess;
}
static void ncclIbStatsFatalError(struct ncclIbStats* stat){
  __atomic_fetch_add(&stat->fatalErrorCount, 1, std::memory_order_relaxed);
}
static void ncclIbQpFatalError(struct ibv_qp* qp) {
  ncclIbStatsFatalError((struct ncclIbStats*)qp->qp_context);
}
static void ncclIbCqFatalError(struct ibv_cq* cq) {
  ncclIbStatsFatalError((struct ncclIbStats*)cq->cq_context);
}
static void ncclIbDevFatalError(struct ncclIbDev* dev) {
  ncclIbStatsFatalError(&dev->stats);
}
ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName);

extern pthread_t ncclIbAsyncThread;
void* ncclIbAsyncThreadMain(void* args);

ncclResult_t ncclIbGdrSupport();
ncclResult_t ncclIbPeerMemSupport();
ncclResult_t ncclIbDmaBufSupport(int dev);

// GIN bridge entry points (wrappers around the existing IB transport surface).
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t logFunction, void* config);
ncclResult_t ncclIbFinalizeDevices();
ncclResult_t ncclIbSetTrafficClass(void* ctx, int trafficClass);
ncclResult_t ncclIbGetPhysProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle);

ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum, int gidTblLen, int *gidIndex);
int ncclIbGidIndexFromDevice(int netDev);
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req);
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r);


// Net IB plugin entry functions.
ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction);
ncclResult_t ncclIbDevices(int* ndev);
ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props);
ncclResult_t ncclIbListen(int dev, void* opaqueHandle, void** listenComm);
ncclResult_t ncclIbConnect(int dev, void* opaqueHandle, void** sendComm);
ncclResult_t ncclIbAccept(void* listenComm, void** recvComm);
ncclResult_t ncclIbRegMr(void* comm, void* data, int size, int type, void** mhandle);
ncclResult_t ncclIbRegMrInternal(void* comm, void* data, size_t size, int type, uint64_t mrFlags,
                                 void** mhandle);
ncclResult_t ncclIbDeregMr(void* comm, void* mhandle);
ncclResult_t ncclIbIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
ncclResult_t ncclIbTest(void* request, int* done, int* sizes);
ncclResult_t ncclIbCloseSend(void* sendComm);
ncclResult_t ncclIbCloseRecv(void* recvComm);
ncclResult_t ncclIbCloseListen(void* listenComm);


#endif
