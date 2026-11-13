//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "core.h"
#include "comm.h"
#include "checks.h"
#include "collectives.h"
#include "transport.h"
#include "pccl_internal.h"
#include "comm_dumper.h"
#include "net.h"
#include "p2p.h"
#include "ras/ras_wire.h"

#define PCCL_DUMP_VERSION_MAJOR 0x02
#define PCCL_DUMP_VERSION_MINOR 0x02
#define PCCL_DUMP_VERSION ((PCCL_DUMP_VERSION_MAJOR<<16) + PCCL_DUMP_VERSION_MINOR)

PCCL_PARAM(DevKernelArgsFifoDepth, "DEV_KERNEL_ARGS_FIFO_DEPTH", 2048);

int64_t pcclDevKernelArgsFifoDepth() {
  return pcclParamDevKernelArgsFifoDepth();
}

static bool isNetConnection(ncclConn_t type) {
  return type == CONN_GDRDMA || type == CONN_RDMA;
}

static bool isRingHeadTransportSupported(const ncclConnector& connector, ncclConn_t type, bool isSend) {
  if (!connector.connected) return false;

  const ncclTransportComm* p2pTransport =
      isSend ? &ncclTransports[TRANSPORT_P2P]->send : &ncclTransports[TRANSPORT_P2P]->recv;
  if (connector.transportComm == p2pTransport) {
    return type == CONN_ICN && !usingMemcpy();
  }

  // Ring connector 0 has dedicated buffers, including PXN. NET consumption
  // is read at the destination; the send proxy need not belong to this rank.
  const ncclTransportComm* netTransport =
      isSend ? &ncclTransports[TRANSPORT_NET]->send : &ncclTransports[TRANSPORT_NET]->recv;
  bool isNetDevicePath = isNetConnection(type) || (isSend && type == CONN_ICN);
  return isNetDevicePath && ncclNet == &ncclNetIb && connector.transportComm == netTransport;
}

// Conservative scope: one retained in-args work per channel. FIFO/persistent
// storage and multi-work execution require additional lifetime/identity checks.
void readRingOperation(ncclComm_t comm, int channelId, rasDiagRingOperation* info) {
  *info = {};
  info->sendProxyRank = -1;
  info->sendProxyTpRank = -1;
  // Require retained kernel arguments and a valid channel before reading its kernel index.
  if (comm->kernelInfoMonitorPtr == nullptr || channelId < 0 || channelId >= comm->nChannels) return;
  uint64_t kernelIndex = comm->channelKernelIdx[channelId];
  int64_t fifoDepth = pcclDevKernelArgsFifoDepth();
  // No submitted kernel or no retention capacity means there is no operation to inspect.
  if (kernelIndex == 0 || fifoDepth <= 0) return;

  KernelInfoMonitor savedMonitor;
  memcpy(&savedMonitor, &comm->kernelInfoMonitorPtr[(kernelIndex - 1) % fifoDepth], sizeof(savedMonitor));
  const ncclDevKernelArgs1K& savedArgs = savedMonitor.kernelArgs;
  const ncclDevKernelArgs& args = savedArgs.args;
  // Reject a reused monitor slot or a channel that advanced while we copied its arguments.
  // Persistent/FIFO work requires lifetime and operation-identity checks beyond this
  // saved argument copy. The kernel must contain this channel to locate its batch correctly.
  if (args.kernelIdx != kernelIndex || comm->channelKernelIdx[channelId] != kernelIndex || args.persistent ||
      args.workStorageType != ncclDevWorkStorageTypeArgs || !(args.channelMask & (1ULL << channelId))) {
    return;
  }

  size_t blockIndex = __builtin_popcountll(args.channelMask & ((1ULL << channelId) - 1));
  size_t batchOffset = sizeof(ncclDevKernelArgs) + blockIndex * sizeof(ncclDevWorkBatch);
  // The channel's batch descriptor must fit entirely in the retained argument copy.
  if (batchOffset + sizeof(ncclDevWorkBatch) > sizeof(savedArgs)) return;
  ncclDevWorkBatch batch;
  memcpy(&batch, (const char*)&savedArgs + batchOffset, sizeof(batch));
  // Exactly one set bit means that this batch contains exactly one work.
  bool hasSingleWork = batch.offsetBitset != 0 && (batch.offsetBitset & (batch.offsetBitset - 1)) == 0;
  // Head/step attribution requires one native collective work for this channel.
  // Linked or extended batches can contain additional work; P2P work/functions use
  // a different layout and must not be decoded as a collective operation.
  if (batch.nextJump || batch.nextExtends || batch.workType != ncclDevWorkTypeColl || !hasSingleWork ||
      batch.funcId < NUM_P2P_FUNC) {
    return;
  }

  // Decode FUNC_INDEX in reverse order: protocol, algorithm, type, red-op, collective.
  int functionIndex = batch.funcId - NUM_P2P_FUNC;
  int protocol = functionIndex % NCCL_NUM_PROTOCOLS;
  functionIndex /= NCCL_NUM_PROTOCOLS;
  int algorithm = functionIndex % NCCL_NUM_ALGORITHMS;
  functionIndex /= NCCL_NUM_ALGORITHMS;
  functionIndex /= ncclNumTypes;
  functionIndex /= ncclNumDevRedOps;
  int collective = functionIndex;
  // The forwarding model covers the five native Ring collectives only; other
  // algorithms and function kinds do not follow the same predecessor/successor rules.
  if (collective < ncclFuncBroadcast || collective > ncclFuncAllReduce || algorithm != NCCL_ALGO_RING) return;

  size_t workOffset = (size_t)batch.offsetBase + __builtin_ctzll(batch.offsetBitset) * sizeof(ncclDevWorkColl);
  size_t batchesEnd = sizeof(ncclDevKernelArgs) + __builtin_popcountll(args.channelMask) * sizeof(ncclDevWorkBatch);
  // Do not interpret the argument header/batch descriptors as work, or read past the snapshot.
  if (workOffset < batchesEnd || workOffset + sizeof(ncclDevWorkColl) > sizeof(savedArgs)) return;
  ncclDevWorkColl work;
  memcpy(&work, (const char*)&savedArgs + workOffset, sizeof(work));
  // Require a valid root and nonempty work with usable channel geometry. The modulo
  // check handles a participating channel range that wraps around commNChannels;
  // this channel must belong to that range before its counters can represent this work.
  if (work.root >= (uint32_t)comm->nRanks || work.count <= 0 || !work.nChannels || !work.commNChannels ||
      channelId >= work.commNChannels || work.startChannelId >= work.commNChannels ||
      (channelId + work.commNChannels - work.startChannelId) % work.commNChannels >= work.nChannels) {
    return;
  }

  // Channel indices may skip globally submitted kernels. Find the preceding
  // kernel that actually used this channel; do not assume kernelIndex - 1.
  uint64_t previousKernelIndex = 0;
  for (uint64_t candidateIndex = kernelIndex - 1; candidateIndex > 0; --candidateIndex) {
    // Once history has been overwritten, the preceding operation cannot be established safely.
    if (kernelIndex - candidateIndex >= (uint64_t)fifoDepth) return;
    const ncclDevKernelArgs& previousArgs =
        comm->kernelInfoMonitorPtr[(candidateIndex - 1) % fifoDepth].kernelArgs.args;
    // A mismatched slot is not evidence that the candidate kernel skipped this channel.
    if (previousArgs.kernelIdx != candidateIndex) return;
    if (previousArgs.channelMask & (1ULL << channelId)) {
      previousKernelIndex = candidateIndex;
      break;
    }
  }

  ncclChannel& channel = comm->channels[channelId];
  const ncclRing& ring = channel.ring;
  // The operation is usable only if both Ring connectors support the head model.
  // Keep one eligibility flag instead of exposing partially valid capture states.
  if (ring.prev < 0 || ring.prev >= comm->nRanks || ring.next < 0 || ring.next >= comm->nRanks ||
      channel.peers == nullptr || channel.peers[ring.prev] == nullptr || channel.peers[ring.next] == nullptr) return;
  const ncclConnector& sendConnector = channel.peers[ring.next]->send[0];
  const ncclConnector& recvConnector = channel.peers[ring.prev]->recv[0];
  if (!isRingHeadTransportSupported(sendConnector, ring.sendConnType, true) ||
      !isRingHeadTransportSupported(recvConnector, ring.recvConnType, false)) return;

  info->kernelIndex = kernelIndex;
  info->workOffset = workOffset;
  for (int i = 0; i < savedMonitor.nWorkSequences; i++) {
    if (savedMonitor.workSequences[i].workOffset == workOffset) {
      info->seqNumber = savedMonitor.workSequences[i].seqNumber;
      info->seqNumberValid = 1;
      break;
    }
  }
  info->count = work.count;
  info->root = work.root;
  info->lastChunkSize = work.lastChunkSize;
  info->funcId = batch.funcId;
  info->nThreads = work.nThreads;
  info->nChannels = work.nChannels;
  info->previousKernelIndex = previousKernelIndex;
  info->collective = collective;
  info->protocol = protocol;
  info->sendIsNet = sendConnector.transportComm == &ncclTransports[TRANSPORT_NET]->send;
  if (info->sendIsNet) {
    // Shared connectors retain their creating communicator's rank numbering.
    // Translate through the top parent; an external proxy has no current rank.
    info->sendProxyTpRank = sendConnector.proxyConn.tpRank;
    for (int rank = 0; rank < comm->nRanks; rank++) {
      if (comm->topParentRanks[rank] == info->sendProxyTpRank) {
        info->sendProxyRank = rank;
        break;
      }
    }
  }
  info->valid = 1;
}

const char* pcclCommDumpMaskStr[] = {"DUMP_ROUGH", "DUMP_RAW", "DUMP_BUFFER", "DUMP_HWINFO"};

// pcclCommDumpMask: bit control for dump subs;
// default: DUMP_NONE;
// env init(only once): highes priority
// file pipe control: second priority
pcclCommDumpMaskType globalPcclCommDumpMask = (pcclCommDumpMaskType)(DUMP_ROUGH);

const char* getConnTypeStr(int peer, ncclConn_t type) {
  if (peer<0 || (int)type<CONN_NORMAL || (int)type>=CONN_NUM) return "NONE";
  else return rasConnTypeStr[type];
}

const char* indentLevelStr[] = {"", "  ", "    ", "      ", "        "};
const char* getIndentStr(int indentLevel) {
  int level = 0;
  if (indentLevel >= 0 && indentLevel < sizeof(indentLevelStr)/sizeof(char*)) level = indentLevel;
  return indentLevelStr[level];
}

const char* generateDumpFileNamePrefix() {
  static char* dumpFileName = NULL;
  if (dumpFileName != NULL) return dumpFileName;

  const char* dumpFileEnvName = pcclGetEnvWithDualPrefix("PCCL_COMM_DUMP_FILE");
  const char* dumpFileDefaultName = "pccl_comm_dump";
  if (dumpFileEnvName) {
    dumpFileName = (char*)dumpFileEnvName;
    INFO(NCCL_DUMP, "PCCL_COMM_DUMP_FILE set by environment to %s", dumpFileName);
  } else {
    dumpFileName = (char*)dumpFileDefaultName;
    INFO(NCCL_DUMP, "use default comm dump name %s", dumpFileName);
  }
  return dumpFileName;
}

ncclResult_t dumpRingToFile(FILE* file, struct ncclRing* ring, int indentLevel) {
  const char *indentStr = getIndentStr(indentLevel);
  fprintf(file, "%sRing: prev:%d(%s), next:%d(%s)\n", indentStr, ring->prev, getConnTypeStr(ring->prev, ring->recvConnType), ring->next, getConnTypeStr(ring->next, ring->sendConnType));
  fflush(file);
  return ncclSuccess;
}

ncclResult_t dumpTreeToFile(FILE* file, struct ncclTree* tree, int indentLevel) {
  const char *indentStr = getIndentStr(indentLevel);
  fprintf(file, "%sTree: up:%d(%s), ", indentStr, tree->up, getConnTypeStr(tree->up, tree->upConnType));
  for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) {
    fprintf(file, "down[%d]:%d(%s), ", i, tree->down[i], getConnTypeStr(tree->down[i], tree->downConnType));
  }
  fprintf(file, "\n");
  fflush(file);
  return ncclSuccess;
}

ncclResult_t dumpP2pWorkToFile(FILE* file, struct ncclDevWorkP2p* work, uint16_t index, int indentLevel) {
  const char* indentStr = getIndentStr(indentLevel);
  fprintf(file, "%s%d: ", indentStr, index);
  fprintf(file, "sendRegUsed:%d, recvRegUsed:%d sendbuff %p recvbuff %p ",
    work->sendRegUsed, work->recvRegUsed, work->sendbuff, work->recvbuff);
  fprintf(file, "p2p, ");
  fprintf(file, "sendRank:%d, recvRank:%d, sendCount:%ld, recvCount:%ld, sendChunkSize:%d, recvChunkSize:%d, ",
    work->sendRank, work->recvRank, work->sendCount, work->recvCount, work->sendChunkSize, work->recvChunkSize);
  fprintf(file, "\n");
  return ncclSuccess;
}

ncclResult_t dumpCollWorkToFile(FILE* file, struct ncclDevWorkColl* work, uint16_t index, int indentLevel) {

  const char* indentStr = getIndentStr(indentLevel);
  fprintf(file, "%s%d: ", indentStr, index);
  fprintf(file, "work nThreads:%d ", work->nThreads);
  fprintf(file, "sendRegUsed:%d, recvRegUsed:%d sendbuff %p recvbuff %p ",
    work->sendRegUsed, work->recvRegUsed, work->sendbuff, work->recvbuff);
  fprintf(file, "coll, count:%ld, lastChunkSize:%ld, root:%d, startChannelId:%d, nChannels:%d, commNChannels:%d, ",
    work->count, work->lastChunkSize, work->root, work->startChannelId, work->nChannels, work->commNChannels);
  fprintf(file, "\n");

  return ncclSuccess;
}

ncclResult_t dumpCollRegWorkToFile(FILE* file, struct ncclDevWorkCollReg* work, uint16_t index, int indentLevel) {
  return ncclSuccess;
}

ncclResult_t dumpWorkToFile(FILE* file, uint32_t workType, uint64_t offsetBitset, const struct DevWorkStorageRef& storage, int indentLevel) {
  int workIdx = 0;;
  alignas(16) char localWork[sizeof(struct ncclDevWorkCollReg)];
  const size_t workSize = ncclDevWorkSize((enum ncclDevWorkType)workType);
  for (uint16_t i = 0; i < (NCCL_MAX_BIT_IDX_IN_OFFSET_BITSET + 1); i++) {
    if (offsetBitset & (1L << i)) {
      uint32_t workOffset = storage.offsetBase + i * workSize;
      char* workPtr = copyDevWorkFromStorage(localWork, storage, workOffset, workSize);
      if (workPtr == nullptr) continue;
      switch (workType) {
        case ncclDevWorkTypeP2p: {
          ncclDevWorkP2p* p2pWork = (ncclDevWorkP2p*)workPtr;
          NCCLCHECK(dumpP2pWorkToFile(file, p2pWork, workIdx, indentLevel));
          break;
        }
        case ncclDevWorkTypeColl: {
          ncclDevWorkColl* collWork = (ncclDevWorkColl*)workPtr;
          NCCLCHECK(dumpCollWorkToFile(file, collWork, workIdx, indentLevel));
          break;
        }
        case ncclDevWorkTypeCollReg:
        default: {
          ncclDevWorkCollReg* collRegWork = (ncclDevWorkCollReg*)workPtr;
          NCCLCHECK(dumpCollRegWorkToFile(file, collRegWork, workIdx, indentLevel));
          break;
        }
      }
      workIdx++;
    }
  }
  return ncclSuccess;
}

ncclResult_t dumpChannelWorkBatchToFile(FILE* file, ncclComm_t comm, ncclDevKernelArgs* kernelArgs, uint16_t blkIdx, uint16_t channelId, int indentLevel) {
  const char* indentStr = getIndentStr(indentLevel);
  fprintf(file, "%sblkIdx %d channelId %d: ", indentStr, blkIdx, channelId);
  uint16_t batchIx = blkIdx;
  while (true) {
    struct ncclDevWorkBatch batch = ((struct ncclDevWorkBatch*)(kernelArgs+1))[batchIx];
    char kernelName[1024];
    getKerFuncStr(batch.funcId, kernelName, NULL, NULL);
    fprintf(file, "workType:%u, funcId:%u(%s), nextJump:%u, nextExtends:%u ", batch.workType, batch.funcId, kernelName, batch.nextJump, batch.nextExtends);
    fprintf(file, "offsetBase:%x, offsetBitset:%lx ", batch.offsetBase, batch.offsetBitset);
    fprintf(file, "\n");

    // kernelArgs->workBuf is device addr if comm->workFifoBuf/comm->workFifoBufDev is allocated by ncclGdrCudaCalloc
    // kernelArgs->workBuf is equal to comm->workFifoBufDev if workStorageType is ncclDevWorkStorageTypeFifo.
    // The device masks per-work offsets with workMask for Fifo/Persistent storage
    // (offsetBase can exceed workFifoBytes once the circular Fifo wraps).
    char* workPtrBase = (char*)comm->workFifoBuf;
    if (kernelArgs->workStorageType == ncclDevWorkStorageTypeArgs) {
      workPtrBase = (char*)kernelArgs;
    }

    if (kernelArgs->workStorageType != ncclDevWorkStorageTypePersistent) {
      struct DevWorkStorageRef storage = { workPtrBase, batch.offsetBase, kernelArgs->workMask, kernelArgs->workStorageType };
      NCCLCHECK(dumpWorkToFile(file, batch.workType, batch.offsetBitset, storage, indentLevel + 1));
    }

    if (batch.nextExtends) {
      batchIx += batch.nextJump;
    } else {
      if (batch.nextJump == 0) {
        break;
      }
      batchIx += batch.nextJump;
      fprintf(file, "\n");
    }
    fprintf(file, "\n");
  }
  fprintf(file, "\n");
  return ncclSuccess;
}

ncclResult_t dumpKernelInfoToFile(FILE* file, ncclComm_t comm, KernelInfoMonitor* kernelInfoMonitor, uint64_t index, int indentLevel) {
  struct ncclDevKernelArgs* kernelArgs = &(kernelInfoMonitor->kernelArgs.args);
  const char* indentStr = getIndentStr(indentLevel);
  fprintf(file, "%skernelIdx%ld: ", indentStr, kernelArgs->kernelIdx);
  if(index + 1 != kernelArgs->kernelIdx) {
    ERROR("index(%d) + 1 should be equal to kernelIdx(%d)", index, kernelArgs->kernelIdx);
    return ncclInternalError;
  }

  fprintf(file, "channelMask:%lx, workStorageType:%d, workMask:%x ", kernelArgs->channelMask, kernelArgs->workStorageType, kernelArgs->workMask);
  fprintf(file, "\n");

  uint16_t blkIdx = 0;

  for (uint16_t i = 0; i < 64; i++) {
    if (kernelArgs->channelMask & (1L << i)) {
      // coverity[overrun-buffer-arg]
      NCCLCHECK(dumpChannelWorkBatchToFile(file, comm, kernelArgs, blkIdx, i, indentLevel + 1));
      blkIdx++;
    }
  }

  fprintf(file, "\n");
  return ncclSuccess;
}

static ncclResult_t dumpKernelInfoFifoToFile(ncclComm_t comm, int stateMonitorLevel, FILE* file, int indentLevel) {
  const char* indentStr = getIndentStr(indentLevel);
  fprintf(file, "%sDevKernelArgs: \n", indentStr);
  uint64_t maxKernelNum = comm->kernelNum > pcclDevKernelArgsFifoDepth() ? pcclDevKernelArgsFifoDepth() : comm->kernelNum;
  uint64_t startIdx = comm->kernelNum > pcclDevKernelArgsFifoDepth() ? comm->kernelNum - pcclDevKernelArgsFifoDepth() : 0;
  for (uint64_t i = startIdx; i < startIdx + maxKernelNum; i++) {
    NCCLCHECK(dumpKernelInfoToFile(file, comm, comm->kernelInfoMonitorPtr + i % pcclDevKernelArgsFifoDepth(), i, indentLevel+1));
  }
  fflush(file);
  return ncclSuccess;
}

template<typename T>
ncclResult_t dumpLockPtr(T* devPtr, T** cpuPtr, T* constZeroPtr) {
  hggcEitInterface& hggcEit = hggcEitInterface::getInstance();
  ncclResult_t res = ncclSuccess;
  int currentProcessDevPtr = 0;
  NCCLCHECKGOTO(checkPtrLockable(devPtr, &currentProcessDevPtr), res, set_zero);
  if (currentProcessDevPtr == 0) goto set_zero;
  if (currentProcessDevPtr == 2) {
    *cpuPtr = devPtr;
  } else {
    HGGCCHECKGOTO(hggcEit.wrapHggcMemLock((HGdeviceptr)devPtr, (void**)cpuPtr), res, set_zero);
  }
  return ncclSuccess;

 set_zero:
  *cpuPtr = constZeroPtr;
  return res;
}


ncclResult_t dumpPeerInfoToFile(FILE* file, int peer, struct ncclChannelPeer* hostPeer, struct ncclDevChannelPeer* devPeer, int indentLevel,
                                pcclCommDiagCollector* collector, ncclComm_t collectorComm, int collectorChannelId) {
  hggcEitInterface& hggcEit = hggcEitInterface::getInstance();
  uint64_t constZero = 0;
  if (hostPeer == nullptr || devPeer == nullptr) {
    fprintf(file, "%suninitialized peer %d, no valid peers head&tail.\n", getIndentStr(indentLevel), peer);
    return ncclSuccess;
  }
  for (int c = 0; c < NCCL_MAX_CONNS; c++) {
    struct ncclConnector *sendConn = &hostPeer->send[c];
    struct ncclConnector *recvConn = &hostPeer->recv[c];
    uint64_t *sendHeadPtr = &constZero;
    uint64_t *sendTailPtr = &constZero;
    uint64_t *sendStep = &constZero;
    uint64_t *recvHeadPtr = &constZero;
    uint64_t *recvTailPtr = &constZero;
    uint64_t *recvStep = &constZero;

    uint64_t *srcPtrs[] = {
            sendConn->conn.head,
            sendConn->conn.tail,
            &devPeer->send[c].step,
            recvConn->conn.head,
            recvConn->conn.tail,
            &devPeer->recv[c].step
    };
    uint64_t **accessiblePtrs[] = {
            &sendHeadPtr,
            &sendTailPtr,
            &sendStep,
            &recvHeadPtr,
            &recvTailPtr,
            &recvStep,
    };

    static_assert(sizeof(srcPtrs) == sizeof(accessiblePtrs), "sizeof(srcPtrs) should equal sizeof(accessiblePtrs)");
    const int num = sizeof(srcPtrs) / sizeof(uint64_t *);
    for (int i = 0; i < num; i++) {
      NCCLVERIFY(dumpLockPtr<uint64_t>(srcPtrs[i], accessiblePtrs[i], &constZero));
    }

    uint64_t values[6];
    for (int i=0; i<6; i++) values[i] = *(volatile uint64_t*)*accessiblePtrs[i];
    const char *indentStr = getIndentStr(indentLevel);
    fprintf(file, "%s%d: send connect(%d), recv connect(%d)\n", indentStr, peer, sendConn->connected,
            recvConn->connected);
    const char *subIndentStr = getIndentStr(indentLevel + 1);
    if (sendConn->connected == 1) {
      uint8_t sendReadValid = (sendHeadPtr != &constZero ? RAS_DIAG_PEER_HEAD_VALID : 0) |
                              (sendStep != &constZero ? RAS_DIAG_PEER_STEP_VALID : 0) |
                              (sendTailPtr != &constZero ? RAS_DIAG_PEER_TAIL_VALID : 0);
      fprintf(file, "%ssend step:%20ld, head:%20ld, tail: %20ld, headPtr:%p, tailPtr: %p, connector: %s, readValid:%u\n",
              subIndentStr, (int64_t)values[2], (int64_t)values[0], (int64_t)values[1], sendConn->conn.head, sendConn->conn.tail,
              c == 0 ? "coll" : "p2p", sendReadValid);
      if (collector && collector->collectPeerConn) {
        collector->collectPeerConn(collector->ctx, collectorComm, collectorChannelId, peer,
                          (uint8_t)c, /*direction=send*/0,
                          values[0], values[1], values[2],
                          sendReadValid);
      }
    }
    if (recvConn->connected == 1) {
      uint8_t recvReadValid = (recvHeadPtr != &constZero ? RAS_DIAG_PEER_HEAD_VALID : 0) |
                              (recvStep != &constZero ? RAS_DIAG_PEER_STEP_VALID : 0) |
                              (recvTailPtr != &constZero ? RAS_DIAG_PEER_TAIL_VALID : 0);
      fprintf(file, "%srecv step:%20ld, head:%20ld, tail: %20ld, headPtr:%p, tailPtr: %p, connector: %s, readValid:%u\n",
              subIndentStr, (int64_t)values[5], (int64_t)values[3], (int64_t)values[4], recvConn->conn.head, recvConn->conn.tail,
              c == 0 ? "coll" : "p2p", recvReadValid);
      if (collector && collector->collectPeerConn) {
        collector->collectPeerConn(collector->ctx, collectorComm, collectorChannelId, peer,
                          (uint8_t)c, /*direction=recv*/1,
                          values[3], values[4], values[5],
                          recvReadValid);
      }
    }

    for (int i = 0; i < num; i++) {
      if (*accessiblePtrs[i] != &constZero && *accessiblePtrs[i] != srcPtrs[i])
        HGGCCHECKWITHWARN(hggcEit.wrapHggcMemUnlock((HGdeviceptr) srcPtrs[i]));
    }
  }
  return ncclSuccess;
}

static ncclResult_t dumpChannelToFile(ncclComm_t comm, int channelId, int nRanks, int stateMonitorLevel, FILE* file, int indentLevel,
                                      pcclCommDiagCollector* collector) {
  struct ncclChannel* channel = comm->channels + channelId;
  fprintf(file, "channelId:%d\n", channel->id);
  const char* indentStr = getIndentStr(indentLevel);

  // TODO: to make the logic simpler, we would dump hostDevIndex after watch dog.
  fprintf(file, "%schannelKernelIdx:%lu ringExcluded:%d ", indentStr,
          comm->channelKernelIdx[channelId], channelId >= comm->nChannels);
  bool unknown = false;
  // Two distinct kernel indices captured for the collector:
  //   hangReported = hostDevIndexLevel1 (written by primitives.h checkAbort when
  //     the kernel detects it has been spinning; level-agnostic, sticky 0).
  //   completed    = devIndexLevel1 (level==1) or hostDevIndexLevel2 (level==2),
  //     written at kernel FINI; ground truth of how many kernels actually ran.
  // Each has its own `valid` bool so 0 (legitimate "never written") is distinct
  // from a failed device read.
  uint64_t hangReportedIdx = 0; bool hangReportedValid = false;
  uint64_t completedIdx = 0;    bool completedValid = false;
  hggcEitInterface& hggcEit = hggcEitInterface::getInstance();
  if (hggcEit.checkExtTable()) {
    uint64_t currentDevIndex;
    // hangReported is the same field (hostDevIndexLevel1) regardless of level.
    // Always print the read outcome (including a successful zero) so the
    // offline parser can tell "no self-reported hang" from a failed read.
    if (hggcEit.wrapHggcMemDump(&currentDevIndex, (HGdeviceptr)(channel->hostDevIndexLevel1), sizeof(uint64_t)) == hggcSuccess) {
      hangReportedIdx = currentDevIndex;
      hangReportedValid = true;
      fprintf(file, " lastTimeoutKernelIndex:%lu", currentDevIndex);
    } else {
      fprintf(file, " unknownTimeoutKernelIndex");
    }
    if (stateMonitorLevel == 1) {
      if (hggcEit.wrapHggcMemDump(&currentDevIndex, (HGdeviceptr)(channel->devIndexLevel1), sizeof(uint64_t)) == hggcSuccess) {
        completedIdx = currentDevIndex;
        completedValid = true;
        fprintf(file, " completeKernelIndex:%lu", currentDevIndex);
      } else {
        unknown = true;
      }
      if (hggcEit.wrapHggcMemDump(&currentDevIndex, (HGdeviceptr)(channel->devPerKernelBatchCounter), sizeof(uint64_t)) == hggcSuccess) {
        fprintf(file, " completePerKernelBatchCounter:%lu", currentDevIndex);
      }
    } else if (stateMonitorLevel == 2) {
      if (hggcEit.wrapHggcMemDump(&currentDevIndex, (HGdeviceptr)(channel->hostDevIndexLevel2), sizeof(uint64_t)) == hggcSuccess) {
        completedIdx = currentDevIndex;
        completedValid = true;
        fprintf(file, " completeCount:%lu", currentDevIndex);
      } else {
        unknown = true;
      }
    } else {
      unknown = true;
    }
  } else {
    unknown = true;
  }
  if (unknown) fprintf(file, " unknownDevIndex");
  fprintf(file, "\n");

  // Aligned with populateDiagChannel's hasRing precondition: channels beyond
  // comm->nChannels (FC clones via a raw struct memcpy in connect.cc) and
  // channels whose ring is not up yet (abort during init) both carry a
  // zeroed/copied ring the online snapshot never reports.
  if (channelId < comm->nChannels && channel->ring.userRanks != nullptr) {
    NCCLVERIFY(dumpRingToFile(file, &channel->ring, indentLevel));
    NCCLVERIFY(dumpTreeToFile(file, &channel->tree, indentLevel));
  }

  // Fire collectChannelState BEFORE the peer loop so collectors that anchor per-channel
  // state (e.g. a "current rasDiagChannel" pointer) are positioned correctly when
  // collectPeerConn fires.
  rasDiagRingOperation info{};
  readRingOperation(comm, channelId, &info);
  if (collector && collector->collectChannelState) {
    collector->collectChannelState(collector->ctx, comm, channelId,
                         hangReportedIdx, hangReportedValid,
                         completedIdx,    completedValid, &info);
  }

  if (hggcEit.checkExtTable()) {
    if (channel->peers == nullptr || channel->devPeersHostPtr == nullptr) {
      // Same init race as the per-peer null check: the channel exists but its
      // peer arrays are not up yet (abort dump during initChannel).
      fprintf(file, "%suninitialized channel, no valid peers head&tail.\n", indentStr);
    } else {
      fprintf(file, "%speers: \n", indentStr);
      for (int p=0; p<nRanks; p++) {
        NCCLVERIFY(dumpPeerInfoToFile(file, p, channel->peers[p], channel->devPeersHostPtr[p], indentLevel+1,
                                       collector, comm, channelId));
      }
    }
  } else {
    fprintf(file, "invalid hggc eit, no valid peers head&tail.\n");
  }

  fprintf(file, "%sringHead valid:%u kernel:%lu previous:%lu coll:%u proto:%u funcId:%u root:%d count:%lu "
                "lastChunkSize:%lu nChannels:%u nThreads:%u sendIsNet:%u sendProxyRank:%d sendProxyTpRank:%d "
                "workOffset:%u seqNumber:%lu seqNumberValid:%u\n",
          indentStr, info.valid, info.kernelIndex, info.previousKernelIndex,
          info.collective, info.protocol, info.funcId, info.root, info.count,
          info.lastChunkSize, info.nChannels, info.nThreads, info.sendIsNet, info.sendProxyRank, info.sendProxyTpRank,
          info.workOffset, info.seqNumber, info.seqNumberValid);

  return ncclSuccess;
}

static ncclResult_t dumpRoughComm(ncclComm_t comm, char* commDumpRoughFile, pcclCommDiagCollector* collector) {
  FILE* roughDumpFile = NULL;
  INFO(NCCL_DUMP, "dump comm rough to file:%s", commDumpRoughFile);
  roughDumpFile = fopen(commDumpRoughFile, "w");
  if (roughDumpFile == NULL) {
    WARN("counld not open pccl dump file:%s", commDumpRoughFile);
    return ncclSuccess;
  }

  int indentLevel = 0;
  const char* indentStr = getIndentStr(indentLevel);
  int stateMonitorLevel = comm->hostDevComm.comm.stateMonitorLevel;
  fprintf(roughDumpFile, "DUMP VERSION: %x\n", PCCL_DUMP_VERSION);
  fprintf(roughDumpFile, "%srank:%d, nRanks:%d, hggcDev:%d, commHash:%lx, comm:%p, devComm:%p stateMonitorLevel:%d\n", indentStr, comm->rank, comm->nRanks, comm->hggcDev, comm->commHash, comm, comm->devComm, stateMonitorLevel);

  NCCLVERIFY(dumpKernelInfoFifoToFile(comm, stateMonitorLevel, roughDumpFile, indentLevel+1));

  if (collector && collector->initRankDiagHeader) collector->initRankDiagHeader(collector->ctx, comm);

  fprintf(roughDumpFile, "%stotalKernelNum:%lu ", indentStr, comm->kernelNum);
  // Match the online diag snapshot (rasDiagSnapshotInto): P2P may initialize
  // channels beyond comm->nChannels (ncclTopoComputeP2pChannels), so dump
  // every initialized channel instead of only the coll ones.
  for (int c=0; c<MAXCHANNELS; c++) {
    if (comm->channels[c].id < 0) continue;
    NCCLVERIFY(dumpChannelToFile(comm, c, comm->nRanks, stateMonitorLevel, roughDumpFile, indentLevel+1, collector));
  }

  if (collector && collector->finalizeRankDiagHeader) collector->finalizeRankDiagHeader(collector->ctx, comm);

  fclose(roughDumpFile);
  return ncclSuccess;
}

template<typename T>
ncclResult_t dumpHostObj(T* obj, FILE* file, int num=1) {
  size_t size = num*sizeof(T);
  fwrite(reinterpret_cast<char*>(&obj), sizeof(char*), 1, file);
  fwrite(reinterpret_cast<char*>(&size), sizeof(size), 1, file);
  fwrite(obj, sizeof(T), num, file);
  return ncclSuccess;
}

template<typename T>
ncclResult_t dumpDevObj(FILE* file, T* cpuPtr, T* devPtr,  int num=1){
  hggcEitInterface& hggcEit = hggcEitInterface::getInstance();
  if (devPtr != NULL && cpuPtr != NULL && num > 0) {
    // check devPtr attribute type attribute
    hggcPointerAttributes ptrAttr;

    hggcError_t getPtrAttrErr = hggcPointerGetAttributes(&ptrAttr, reinterpret_cast<const void*>(devPtr));
    if (getPtrAttrErr != hggcSuccess) {
      WARN("Fail to get pointer attribute! Ignore dump devPtr:%p", devPtr);
      return ncclUnhandledHggcError;
    }
    if (ptrAttr.type != hggcMemoryTypeDevice && ptrAttr.type != hggcMemoryTypeManaged && ptrAttr.type != hggcMemoryTypeHost) {
      WARN("devPtr(%p)'s type(%d) is unregistered or unknown! Ignore dump", devPtr, ptrAttr.type);
      return ncclUnhandledHggcError;
    }

    HGpointerIpc_attribute ptrIpcAttr;
    HGGCCHECKWITHWARN(hggcEit.wrapHggcPointerGetIpcAttribute((HGdeviceptr)devPtr, &ptrIpcAttr));

    if (ptrIpcAttr == HG_POINTER_IPC_ATTRIBUTE_IMPORTED) {
      INFO(NCCL_DUMP, "skip ipc imported pointer(%p) dump", devPtr);
      return ncclSuccess;
    }

    size_t size = num*sizeof(T);
    if (ptrAttr.type == hggcMemoryTypeDevice) {
      HGGCCHECKWITHWARN(hggcEit.wrapHggcMemDump(cpuPtr, (HGdeviceptr)devPtr, size));
    }
    fwrite(reinterpret_cast<char*>(&devPtr), sizeof(char*), 1, file);
    fwrite(reinterpret_cast<char*>(&size), sizeof(size), 1, file);
    if (ptrAttr.type == hggcMemoryTypeDevice) {
      fwrite(cpuPtr, sizeof(T), num, file);
    } else {
      fwrite(devPtr, sizeof(T), num, file);
    }
  }
  return ncclSuccess;
}

ncclResult_t dumpRawComm(ncclComm_t comm, char* commDumpRawFile, pcclCommDumpMaskType dumpMask) {
  ncclResult_t res = ncclSuccess;
  // struct ncclWork* workFifoDump = NULL;
  struct ncclDevChannelPeer** devPeersDump = NULL;
  char* buffDump[NCCL_NUM_PROTOCOLS] = {NULL};
  char* sendbuffDump = NULL;
  char* recvbuffDump = NULL;
  struct ncclKernelComm devCommDump;
  constexpr int dumpProtoNum = 3;
  const uint64_t commDumpVersion = (static_cast<uint64_t>(PCCL_VERSION_CODE) << 32) | 0x00000001 ;

  hggcEitInterface& hggcEit = hggcEitInterface::getInstance();

  FILE* rawDumpFile = NULL;

  INFO(NCCL_DUMP, "dump comm raw to file:%s", commDumpRawFile);
  rawDumpFile = fopen(commDumpRawFile, "wb");
  if (!rawDumpFile) {
    WARN("counld not open pccl dump file:%s", commDumpRawFile);
    goto clean;
  }
  fwrite(reinterpret_cast<const char*>(&commDumpVersion), sizeof(commDumpVersion), 1, rawDumpFile);

  // dump host accesible
  NCCLVERIFY(dumpHostObj((pcclComm*)comm, rawDumpFile));
  for (int c=0; c<comm->nChannels; c++) {
    NCCLVERIFY(dumpHostObj(comm->channels + c, rawDumpFile));
    NCCLVERIFY(dumpHostObj(comm->channels[c].devPeersHostPtr, rawDumpFile, comm->nRanks));
  }
  NCCLVERIFY(dumpHostObj(comm->kernelInfoMonitorPtr, rawDumpFile, pcclDevKernelArgsFifoDepth()));
  NCCLVERIFY(dumpHostObj((char*)(comm->workFifoBuf), rawDumpFile, comm->workFifoBytes));

  // dump dev mem
  // TODO: maybe we can unify the host & device dump code. opt later.
  if (hggcEit.checkExtTable()) {
    NCCLVERIFY(dumpDevObj(rawDumpFile, &devCommDump, comm->devComm));

    // prepare temporary dump buffer
    NCCLCHECKGOTO(ncclCalloc(&devPeersDump, comm->nRanks), res, clean);
    if ((dumpMask & DUMP_BUFFER) != 0) {
      for (int b=0; b<dumpProtoNum; b++) {
        NCCLCHECKGOTO(ncclCalloc(&buffDump[b], comm->buffSizes[b]), res, clean);
      }
    }

    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel channelDump;
      NCCLVERIFY(dumpDevObj(rawDumpFile, &channelDump, devCommDump.channels+c));
      NCCLVERIFY(dumpDevObj(rawDumpFile, devPeersDump, channelDump.devPeers, comm->nRanks));

      for (int p=0; p<comm->nRanks; p++) {
        uint64_t sendHead=-1, sendTail=-1, recvHead=-1, recvTail=-1;
        for (int c = 0; c < NCCL_MAX_CONNS; c++) {
          if (devPeersDump[p]->send[c].tail) {
            NCCLVERIFY(dumpDevObj(rawDumpFile, &sendTail, devPeersDump[p]->send[c].tail));
          }

          if (devPeersDump[p]->send[c].head) {
            NCCLVERIFY(dumpDevObj(rawDumpFile, &sendHead, devPeersDump[p]->send[c].head));
          }

          if (devPeersDump[p]->recv[c].tail) {
            NCCLVERIFY(dumpDevObj(rawDumpFile, &recvTail, devPeersDump[p]->recv[c].tail));
          }

          if (devPeersDump[p]->recv[c].head) {
            NCCLVERIFY(dumpDevObj(rawDumpFile, &recvHead, devPeersDump[p]->recv[c].head));
          }

          if ((dumpMask & DUMP_BUFFER) != 0) {
            for (int b = 0; b < dumpProtoNum; b++) {
              if (devPeersDump[p]->send[c].buffs[b]) {
                NCCLVERIFY(dumpDevObj(rawDumpFile, buffDump[b], devPeersDump[p]->send[c].buffs[b], comm->buffSizes[b]));
              }
              if (devPeersDump[p]->recv[c].buffs[b]) {
                NCCLVERIFY(dumpDevObj(rawDumpFile, buffDump[b], devPeersDump[p]->recv[c].buffs[b], comm->buffSizes[b]));
              }
            }
          }
        }
      }
    }
  } else {
    WARN("invalid hggc eit, skip raw dev mem dump");
  }

 clean:
  NCCLCHECK(ncclFree(devPeersDump));
  for (int b=0; b<dumpProtoNum; b++) {
    NCCLCHECK(ncclFree(buffDump[b]));
  }
  NCCLCHECK(ncclFree(sendbuffDump));
  NCCLCHECK(ncclFree(recvbuffDump));
  if (rawDumpFile) fclose(rawDumpFile);

  return res;
}

ncclResult_t pcclCommHwDump(ncclComm_t comm, char* dumpFileName) {
  INFO(NCCL_DUMP, "dump comm hw core to file:%s", dumpFileName);
  hggcEitInterface& hggcEit = hggcEitInterface::getInstance();
  // lightweight dump for now.
  HGGCCHECKWITHWARN(hggcEit.wrapHggcCoreDumpGenerate(comm->hggcDev, dumpFileName, HG_COREDUMP_SKIP_GLOBAL_MEMORY));
  return ncclSuccess;
}

ncclResult_t dumpParseList(const char* str, const char* elems[], int nelems, int* list) {
  int def, set;
  if (str[0] == '^') {
    def = 1; set = 0; str++;
  } else {
    def = 0; set = 1;
  }
  for (int i=0; i<nelems; i++) list[i] = def;
  char* tokStr = strdup(str);
  char* tmpStr;
  char* token = strtok_r(tokStr, ",", &tmpStr);
  while (token) {
    for (int i=0; i<nelems; i++)
      if (strcasecmp(token, elems[i]) == 0) list[i] = set;
    token = strtok_r(NULL, ",", &tmpStr);
  }
  NCCLCHECK(ncclFree(tokStr));
  return ncclSuccess;
}

ncclResult_t setCommDumpMaskByStr(const char* usrMaskStr, pcclCommDumpMaskType* dumpMask) {
  int pcclCommDumpEnableBit[COMM_DUMP_MASK_BITS] = {0};
  NCCLCHECK(dumpParseList(usrMaskStr, pcclCommDumpMaskStr, COMM_DUMP_MASK_BITS, pcclCommDumpEnableBit));
  *dumpMask = DUMP_NONE;
  for (int i=0; i<COMM_DUMP_MASK_BITS; i++) {
    *dumpMask = static_cast<pcclCommDumpMaskType>(static_cast<int>(*dumpMask) | (pcclCommDumpEnableBit[i]<<i));
  }
  return ncclSuccess;
}

ncclResult_t setCommDumpMaskByVal(const uint32_t usrMaskVal, pcclCommDumpMaskType* dumpMask) {
  *dumpMask = static_cast<pcclCommDumpMaskType>(usrMaskVal & 0x1f);
  return ncclSuccess;
}


static bool globalPcclCommDumpMaskInited = false;
ncclResult_t getGlobalCommDumpMask(pcclCommDumpMaskType* dumpMask) {
  // init pcclCommDumpMaskOnce
  if (!globalPcclCommDumpMaskInited) {
    const char *maskStr = pcclGetEnvWithDualPrefix("PCCL_COMM_DUMP_LEVEL");
    if (maskStr) {
      NCCLCHECK(setCommDumpMaskByStr(maskStr, &globalPcclCommDumpMask));
      INFO(NCCL_ENV, "PCCL_COMM_DUMP_CFG set by environment to %s(%x)", maskStr, globalPcclCommDumpMask);
    }
    globalPcclCommDumpMaskInited = true;
  }
  *dumpMask = globalPcclCommDumpMask;
  return ncclSuccess;
}

ncclResult_t pcclCommDump(ncclComm_t comm, pcclCommDumpMaskType dumpMask, pcclCommDiagCollector* collector) {
  if (dumpMask == DUMP_NONE) {
    INFO(NCCL_DUMP, "dump Mask:%x, skip dump.", dumpMask);
    return ncclSuccess;
  }
  if (comm == NULL) {
    WARN("dump null comm:%p!", comm);
    return ncclSuccess;
  }
  INFO(NCCL_DUMP, "pccl comm:%p dump start.", comm);

  int offset = 0;
  char dumpFileNamePrefix[PATH_MAX+1];
  if (createOrFillDumpDir(dumpFileNamePrefix, PATH_MAX, "comm_dump", &offset) != ncclSuccess) {
    WARN("pccl comm: %p dump failed because dump file can't be created.", comm);
    return ncclInternalError;
  }
  offset += fillConfigName(dumpFileNamePrefix+offset, PATH_MAX-offset, comm->commHash, comm->rank, comm->nRanks, comm->hggcDev, comm->hgmlDev, generateDumpFileNamePrefix());

  int savedDev = -1;
  ncclResult_t res = ncclSuccess;
  HGGCCHECKGOTO(hggcGetDevice(&savedDev), res, clean);
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), res, clean);

  if ((dumpMask & DUMP_ROUGH) > 0) {
    char commDumpRoughFile[PATH_MAX+1];
    strncpy(commDumpRoughFile, dumpFileNamePrefix, offset);
    if (PATH_MAX-offset < sizeof(".txt")) offset = PATH_MAX-sizeof(".txt");
    snprintf(commDumpRoughFile+offset, PATH_MAX-offset, ".txt");
    dumpRoughComm(comm, commDumpRoughFile, collector);

    if (comm->extKernel) {
      comm->extKernel->kernelPluginDump(comm->extKernelPluginWorkspace.extKernelPluginParams, commDumpRoughFile);
    }

    ncclProfilerPrintEvent(comm, dumpFileNamePrefix);
  }

  // DUMP_RAW: dump host&dev to binary.
  // DUMP_BUFFER: DUMP_RAW with buffer
  if ((dumpMask & (DUMP_RAW|DUMP_BUFFER)) > 0) {
    char commDumpRawFile[PATH_MAX+1];
    strncpy(commDumpRawFile, dumpFileNamePrefix, offset);
    if (PATH_MAX-offset < sizeof(".bin")) offset = PATH_MAX-sizeof(".bin");
    snprintf(commDumpRawFile+offset, PATH_MAX-offset, ".bin");
    dumpRawComm(comm, commDumpRawFile, dumpMask);
  }

  if ((dumpMask & (DUMP_HWINFO)) > 0) {
    char commHwDumpFile[PATH_MAX+1];
    strncpy(commHwDumpFile, dumpFileNamePrefix, offset);
    if (PATH_MAX-offset < sizeof(".core")) offset = PATH_MAX-sizeof(".core");
    snprintf(commHwDumpFile+offset, PATH_MAX-offset, ".core");
    NCCLCHECK(pcclCommHwDump(comm, commHwDumpFile));
  }

 clean:
  if (savedDev != -1) HGGCCHECK(hggcSetDevice(savedDev));
  INFO(NCCL_DUMP, "pccl comm:%p dump finished.", comm);
  return res;
}
