/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "devcomm.h"
#include "graph.h"
#include "nvmlwrap.h"
#include "core.h"
#include "nccl_device/core.h"

#define NTRANSPORTS 4
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2
#define TRANSPORT_COLLNET 3
#define TRANSPORT_PROFILER 4

#include "proxy.h"
#include "bootstrap.h"
extern struct ncclTransport p2pTransport;
extern struct ncclTransport shmTransport;
extern struct ncclTransport netTransport;
extern struct ncclTransport collNetTransport;
extern struct ncclTransport profilerTransport;

extern struct ncclTransport* ncclTransports[];

// Forward declarations
struct ncclRing;
struct ncclConnector;

struct ncclPeerInfo {
  int rank;
  int ppuid;
  int hostid;
  int hggcDev;
  int hgmlDev;
  int64_t netId;
  int gdrSupport;
  uint64_t hostHash;
  uint64_t pidHash;
  dev_t shmDev;
  int64_t busId;
  ncclComm_t comm;
  int hggcCompCap;
  size_t totalGlobalMem;
  int strideScale;
  // MNNVL support
  hgmlGpuFabricInfoV_t fabricInfo;
  int cuMemSupport;
  int version;
  ncclGinType_t supportedGinType;
  bool crossNicSupport;
  bool cuMemGdrSupport;
};

#define PCCL_SAME_PID(MYINFO, PEERINFO) \
  ((MYINFO)->hostHash == (PEERINFO)->hostHash && (MYINFO)->pidHash == (PEERINFO)->pidHash)

#define CONNECT_SIZE 128
#define NCCL_MAX_PAGE_SIZE (512L * 1024L * 1024L)
#define NCCL_REC_PAGE_SIZE (2L * 1024L * 1024L)
struct ncclConnect {
  char data[CONNECT_SIZE];
};

struct ncclTransportComm {
  ncclResult_t (*setup)(ncclComm_t comm, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, int channelId, int connIndex);
  ncclResult_t (*connect)(ncclComm_t comm, struct ncclConnect*, int nranks, int rank, struct ncclConnector*);
  ncclResult_t (*free)(struct ncclConnector*);
  ncclResult_t (*proxySharedInit)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels);
  ncclResult_t (*proxySetup)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyConnect)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyFree)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState);
  ncclResult_t (*proxyProgress)(struct ncclProxyState* proxyState, struct ncclProxyArgs*);
  ncclResult_t (*proxyRegister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done);
  ncclResult_t (*proxyDeregister)(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, int* done);
};

struct ncclTransport {
  const char name[8];
  ncclResult_t (*canConnect)(int*, ncclComm_t, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

ncclResult_t ncclTransportP2pConnect(ncclComm_t comm, struct ncclChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend, int connIndex);
ncclResult_t ncclTransportP2pSetup(ncclComm_t comm, struct ncclTopoGraph* graph, int connIndex);
ncclResult_t ncclConnTypeCollect(ncclComm_t comm, struct ncclTopoGraph* graph);
ncclResult_t ncclTransportCheckP2pType(ncclComm_t comm, bool* localP2pSupport, bool* directMode, bool* globalP2pSupport);

ncclResult_t ncclTransportRingConnect(ncclComm_t comm);
ncclResult_t ncclTransportTreeConnect(ncclComm_t comm);
ncclResult_t ncclTransportCollnetConnect(ncclComm_t comm);
ncclResult_t ncclTransportFcConnect(ncclComm_t comm);

enum { collNetRecv=0, collNetSend=1 };
int ncclTransportCollNetSetup(ncclComm_t comm, struct ncclTopoGraph* collNetGraph, struct ncclChannel* channel, int masterRank, int masterPeer, int collNetGraphChannelId, int type);
ncclResult_t ncclTransportCollNetCheck(ncclComm_t comm, int collNetSetupFail);
ncclResult_t ncclTransportCollNetFree(ncclComm_t comm);
ncclResult_t ncclCollnetDeregBuffer(ncclComm_t comm, struct ncclProxyConnector* proxyconn, void* handle);

ncclResult_t ncclNetDeregBuffer(ncclComm_t comm, struct ncclProxyConnector* proxyConn, void* handle);

ncclResult_t ncclRegisterCollBuffers(ncclComm_t comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS], void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS], struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);
ncclResult_t ncclRegisterP2pIpcBuffer(ncclComm_t comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue);


#endif
