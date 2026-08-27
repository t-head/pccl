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

#define PCCL_DUMP_VERSION_MAJOR 0x02
#define PCCL_DUMP_VERSION_MINOR 0x01
#define PCCL_DUMP_VERSION ((PCCL_DUMP_VERSION_MAJOR<<16) + PCCL_DUMP_VERSION_MINOR)

PCCL_PARAM(DevKernelArgsFifoDepth, "DEV_KERNEL_ARGS_FIFO_DEPTH", 2048);

int64_t pcclDevKernelArgsFifoDepth() { return pcclParamDevKernelArgsFifoDepth(); }


const char* pcclCommDumpMaskStr[] = {"DUMP_ROUGH", "DUMP_RAW", "DUMP_BUFFER", "DUMP_HWINFO"};

// pcclCommDumpMask: bit control for dump subs;
// default: DUMP_NONE;
// env init(only once): highes priority
// file pipe control: second priority
pcclCommDumpMaskType globalPcclCommDumpMask = (pcclCommDumpMaskType)(DUMP_ROUGH);

const char* ncclConnTypeStr[] = {"CONN_NORMAL", "CONN_ICN", "CONN_PCIE", "CONN_SHM", "CONN_GDRDMA", "CONN_RDMA", "CONN_SOCKET", "CONN_NET_EXT", "CONN_NUM"};
const char* getConnTypeStr(int peer, ncclConn_t type) {
  if (peer<0 || (int)type<CONN_NORMAL || (int)type>=CONN_NUM) return "NONE";
  else return ncclConnTypeStr[type];
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

ncclResult_t dumpWorkToFile(FILE* file, uint32_t workType, uint64_t offsetBitset, char* workPtr, int indentLevel) {
  int workIdx = 0;;
  for (uint16_t i = 0; i < (NCCL_MAX_BIT_IDX_IN_OFFSET_BITSET + 1); i++) {
    if (offsetBitset & (1L << i)) {
      int workSize;
      switch (workType) {
        case ncclDevWorkTypeP2p: {
          workSize = sizeof(struct ncclDevWorkP2p);
          ncclDevWorkP2p* p2pWork = (ncclDevWorkP2p*)(workPtr + i * workSize);
          NCCLCHECK(dumpP2pWorkToFile(file, p2pWork, workIdx, indentLevel));
          break;
        }
        case ncclDevWorkTypeColl: {
          workSize = sizeof(struct ncclDevWorkColl);
          ncclDevWorkColl* collWork = (ncclDevWorkColl*)(workPtr + i * workSize);
          NCCLCHECK(dumpCollWorkToFile(file, collWork, workIdx, indentLevel));
          break;
        }
        case ncclDevWorkTypeCollReg:
        default: {
          workSize = sizeof(struct ncclDevWorkCollReg);
          ncclDevWorkCollReg* collRegWork = (ncclDevWorkCollReg*)(workPtr + i * workSize);
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
    // kernelArgs->workBuf is equal to comm->workFifoBufDev if workStorageType is ncclDevWorkStorageTypeFifo
    char* workPtr = (char*)comm->workFifoBuf + batch.offsetBase;
    if (kernelArgs->workStorageType == ncclDevWorkStorageTypeArgs) {
      workPtr = (char*)kernelArgs + batch.offsetBase;
    }

    if (kernelArgs->workStorageType != ncclDevWorkStorageTypePersistent) {
      NCCLCHECK(dumpWorkToFile(file, batch.workType, batch.offsetBitset, workPtr, indentLevel + 1));
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

    const char *indentStr = getIndentStr(indentLevel);
    fprintf(file, "%s%d: send connect(%d), recv connect(%d)\n", indentStr, peer, sendConn->connected,
            recvConn->connected);
    const char *subIndentStr = getIndentStr(indentLevel + 1);
    if (sendConn->connected == 1) {
      fprintf(file, "%ssend step:%20ld, head:%20ld, tail: %20ld, headPtr:%p, tailPtr: %p, connector: %s\n",
              subIndentStr, (int64_t)*sendStep, (int64_t)*sendHeadPtr, (int64_t)*sendTailPtr, sendConn->conn.head, sendConn->conn.tail, c == 0 ? "coll" : "p2p");
      if (collector && collector->collectPeerConn) {
        collector->collectPeerConn(collector->ctx, collectorComm, collectorChannelId, peer,
                          (uint8_t)c, /*direction=send*/0,
                          *sendHeadPtr, *sendTailPtr, *sendStep);
      }
    }
    if (recvConn->connected == 1) {
      fprintf(file, "%srecv step:%20ld, head:%20ld, tail: %20ld, headPtr:%p, tailPtr: %p, connector: %s\n",
              subIndentStr, (int64_t)*recvStep, (int64_t)*recvHeadPtr, (int64_t)*recvTailPtr, recvConn->conn.head, recvConn->conn.tail, c == 0 ? "coll" : "p2p");
      if (collector && collector->collectPeerConn) {
        collector->collectPeerConn(collector->ctx, collectorComm, collectorChannelId, peer,
                          (uint8_t)c, /*direction=recv*/1,
                          *recvHeadPtr, *recvTailPtr, *recvStep);
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
  fprintf(file, "%schannelKernelIdx:%lu ", indentStr, comm->channelKernelIdx[channelId]);
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
    if (hggcEit.wrapHggcMemDump(&currentDevIndex, (HGdeviceptr)(channel->hostDevIndexLevel1), sizeof(uint64_t)) == hggcSuccess) {
      hangReportedIdx = currentDevIndex;
      hangReportedValid = true;
      if (currentDevIndex > 0) fprintf(file, " lastTimeoutKernelIndex:%lu", currentDevIndex);
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

  NCCLVERIFY(dumpRingToFile(file, &channel->ring, indentLevel));
  NCCLVERIFY(dumpTreeToFile(file, &channel->tree, indentLevel));

  // Fire collectChannelState BEFORE the peer loop so collectors that anchor per-channel
  // state (e.g. a "current rasDiagChannel" pointer) are positioned correctly when
  // collectPeerConn fires.
  if (collector && collector->collectChannelState) {
    collector->collectChannelState(collector->ctx, comm, channelId,
                         hangReportedIdx, hangReportedValid,
                         completedIdx,    completedValid);
  }

  if (hggcEit.checkExtTable()) {
    fprintf(file, "%speers: \n", indentStr);
    for (int p=0; p<nRanks; p++) {
      NCCLVERIFY(dumpPeerInfoToFile(file, p, channel->peers[p], channel->devPeersHostPtr[p], indentLevel+1,
                                     collector, comm, channelId));
    }
  } else {
    fprintf(file, "invalid hggc eit, no valid peers head&tail.\n");
  }

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
  for (int c=0; c<comm->nChannels; c++) {
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
