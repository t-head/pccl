//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "checks.h"
#include "utils.h"
#include "ras_internal.h"
#include "drv_extension.h"
#include "hggc_runtime.h"
#include "comm_dumper.h"

static int stopMonitor = 0;
static int eitEnable = -1;
extern int ncclParamStateMonitorLevelFlag;

PCCL_PARAM(StateMonitorLogEveryMs, "STATE_MONITOR_LOG_EVERY_MS", 0 /* disabled */);
PCCL_PARAM(StateMonitorDumpWhenExceptionDisable, "STATE_MONITOR_DUMP_WHEN_EXCEPTION_DISABLE", 0 /* dump */);

struct rasStateMonitorStatus {
  int timeout;
  uint64_t commHash;
  uint64_t seq;
};

void rasStateMonitorAtHost();
void rasStateMonitorAtDevice();

void rasStateMonitor() {
  if (eitEnable < 0) {
    eitEnable = hggcEitInterface::getInstance().checkExtTable();
  }
  if (stopMonitor || eitEnable == 0) return;

  std::lock_guard<std::recursive_mutex> lock(ncclCommsMutex);
  if (ncclParamStateMonitorLevelFlag == 1) {
    rasStateMonitorAtDevice();
  } else if (ncclParamStateMonitorLevelFlag == 2) {
    rasStateMonitorAtHost();
  }
}

void rasStateMonitorInit(rasNcclComm *rasComm, ncclComm_t comm) {
  rasComm->comm = comm;
  rasComm->timeout = (STATE_MONITOR_TIMEOUT);
  rasComm->lastTimeoutIndex = 0;
  rasComm->hangFlag = 0;
  rasComm->abortFlag = 0;
  rasComm->lastUpdateTime = clockNano();
  memset(rasComm->lastHostIndex, 0, MAXCHANNELS*sizeof(uint64_t));
  memset(rasComm->lastDevIndex, 0, MAXCHANNELS*sizeof(uint64_t));
}

#define STATE_DUMP_LINE_LENGTH 1024

static inline void rasStateMonitorRecord(ncclComm_t comm, const char* info) {
  WARN(info);
  int offset = 0;
  char dumpFileName[PATH_MAX+1];
  if (createOrFillDumpDir(dumpFileName, PATH_MAX, "state_monitor_dump", &offset) != ncclSuccess) {
    WARN("RAS record state failed because dump file can't be created.");
    return;
  }
  offset += fillConfigName(dumpFileName+offset, PATH_MAX-offset, comm->commHash, comm->rank, comm->nRanks, comm->hggcDev, comm->hgmlDev, "pccl_state_monitor_dump", "txt", false);

  FILE* stateDumpFile = fopen(dumpFileName, "a+");
  if (stateDumpFile == NULL) {
    WARN("RAS could not open pccl state dump file %s", dumpFileName);
  } else {
    fprintf(stateDumpFile, "%s\n", info);
    fclose(stateDumpFile);
  }
}

void rasStateMonitorDump(const struct rasCollRequest* req, bool *pDone) {
  *pDone = true;
  if (pcclParamStateMonitorDumpWhenExceptionDisable() != 0) return;
  std::lock_guard<std::recursive_mutex> lock(ncclCommsMutex);
  for (int i = 0; i < nNcclComms; i++) {
    struct rasNcclComm* rasComm = ncclComms + i;
    ncclComm_t comm = rasComm->comm;
    if (comm && (req->dump.dumpAllComms || (comm->commHash == req->dump.commHash && rasComm->lastTimeoutIndex < req->dump.seq))) {
      *pDone = false;
      rasComm->lastTimeoutIndex = req->dump.seq;
      if (req->dump.dumpAllComms) {
        WARN("RAS trigger debugging dump by client commHash %lx", comm->commHash);
      } else {
        WARN("RAS trigger debugging dump by monitor commHash %lx seq %lu", comm->commHash, req->dump.seq);
      }
      pcclCommDumpMaskType dumpMask;
      NCCLVERIFY(getGlobalCommDumpMask(&dumpMask));
      if (req->dump.dumpLevel > 0) setCommDumpMaskByVal(req->dump.dumpLevel, &dumpMask);
      pcclCommDump(comm, dumpMask);
      if (comm->nNodes > 1) pcclDumpProxyState(comm->proxyState);
    }
  }
}

void rasStateMonitorAbort(const struct rasCollRequest* req, bool *pDone) {
  std::lock_guard<std::recursive_mutex> lock(ncclCommsMutex);
  *pDone = true;
  for (int i = 0; i < nNcclComms; i++) {
    struct rasNcclComm* rasComm = ncclComms + i;
    ncclComm_t comm = rasComm->comm;
    if (comm && !rasComm->abortFlag) {
      WARN("RAS call pcclCommAbort commHash %lx rank %d", comm->commHash, comm->rank);
      *pDone = false;
      rasComm->abortFlag = 1;
      pcclCommAbort(comm);
    }
  }
}

void rasStateMonitorBroadcastTimeout(const struct rasStateMonitorStatus& status) {
  struct rasCollRequest bCast;
  rasCollReqInit(&bCast);
  bCast.type = RAS_BC_DUMP;
  bCast.dump.dumpAllComms = false;
  bCast.dump.dumpLevel = 0;
  bCast.dump.commHash = status.commHash;
  bCast.dump.seq = status.seq;
  (void)rasNetSendCollReq(&bCast);
}

void rasStateMonitorAtDevice() {
  VERBOSE(NCCL_RAS, "rasStateMonitorAtDevice stopMonitor %d eitEnable %d", stopMonitor, eitEnable);

  char timeStr[64];
  char infoStr[STATE_DUMP_LINE_LENGTH];
  for (int i = 0; i < nNcclComms; i++) {
    struct rasNcclComm* rasComm = ncclComms + i;
    ncclComm_t comm = rasComm->comm;
    if (comm == nullptr) continue;

    struct ncclChannel* channels = comm->channels;
    int nChannels = 0;
    uint64_t currentMaxDevIndex = 0;
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (channels[c].id < 0) continue;

      uint64_t index;
      hggcError_t err = hggcEitInterface::getInstance().wrapHggcMemDump(&index, (HGdeviceptr)(channels[c].hostDevIndexLevel1), sizeof(uint64_t));
      if (err != hggcSuccess) {
        INFO(NCCL_RAS, "RAS detect driver error %s.", hggcGetErrorString(err));
        stopMonitor = 1;
        return;
      }
      // Record all channels reported timeout
      if (index > rasComm->lastDevIndex[c]) {
        snprintf(infoStr, STATE_DUMP_LINE_LENGTH-1, "RAS kernel begins at %zu may hang in polling, comm %p, rank %d, nranks %d, commhash %lx, channel %d, time %s",
                  index-1, comm, comm->rank, comm->nRanks, comm->commHash, c, getDateTime(timeStr));
        rasStateMonitorRecord(comm, infoStr);
        currentMaxDevIndex = std::max(currentMaxDevIndex, index);
        rasComm->lastDevIndex[c] = index;
      }
      nChannels++;
    }

    if (currentMaxDevIndex > rasComm->lastTimeoutIndex) {
      struct rasStateMonitorStatus status;
      // detect hang
      snprintf(infoStr, STATE_DUMP_LINE_LENGTH-1, "RAS broadcast kernel timeout status, comm %p, rank %d, nranks %d, commhash %lx, seq %ld, time %s",
                comm, comm->rank, comm->nRanks, comm->commHash, currentMaxDevIndex, getDateTime(timeStr));
      rasStateMonitorRecord(comm, infoStr);
      status.timeout = 1;
      status.commHash = comm->commHash;
      status.seq = currentMaxDevIndex;
      rasStateMonitorBroadcastTimeout(status);
    }
  }
}

void rasStateMonitorAtHost() {
  uint64_t currentHostIndex[MAXCHANNELS];
  uint64_t currentDevIndex[MAXCHANNELS];
  static uint64_t lastLogTime = clockNano();
  uint64_t currentTime = clockNano();
  bool showCurrentState = (STATE_MONITOR_LOG_PERIOD > 0 && (currentTime-lastLogTime >= STATE_MONITOR_LOG_PERIOD));

  for (int i = 0; i < nNcclComms; i++) {
    struct rasNcclComm* rasComm = ncclComms + i;
    ncclComm_t comm = rasComm->comm;
    if (comm == nullptr) continue;

    int nChannels = 0;
    struct ncclChannel* channels = comm->channels;
    memset(currentHostIndex, 0, MAXCHANNELS*sizeof(uint64_t));
    memset(currentDevIndex, 0, MAXCHANNELS*sizeof(uint64_t));

    // query current host and device index
    char timeStr[64];
    char infoStr[STATE_DUMP_LINE_LENGTH];
    uint64_t currentMaxDevIndex = 0;
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (channels[c].id < 0) continue;

      hggcError_t err = hggcEitInterface::getInstance().wrapHggcMemDump(currentDevIndex+c, (HGdeviceptr)(channels[c].hostDevIndexLevel2), sizeof(uint64_t));
      if (err != hggcSuccess) {
        INFO(NCCL_RAS, "RAS detect driver error %s.", hggcGetErrorString(err));
        return;
      }
      currentHostIndex[c] = comm->channelKernelIdx[c];
      nChannels++;
    }

    // check status
    bool detectTimeout = false;
    for (int c = 0; c < nChannels; c++) {
      if (showCurrentState) {
        INFO(NCCL_RAS, "RAS current status, comm %p, rank %d, nranks %d, commhash %lx, channel %d, lastHostIndex %ld, currentHostIndex %ld, lastDevIndex %ld, currentDevIndex %ld",
              comm, comm->rank, comm->nRanks, comm->commHash, c, rasComm->lastHostIndex[c], currentHostIndex[c], rasComm->lastDevIndex[c], currentDevIndex[c]);
      }

      if (/* enqueue work to idle channel */
          (rasComm->lastHostIndex[c] == rasComm->lastDevIndex[c] && currentHostIndex[c] != rasComm->lastHostIndex[c]) ||
          /* pending work(s) completed */
          currentDevIndex[c] != rasComm->lastDevIndex[c]) {
        rasComm->lastUpdateTime = currentTime;
        if ((rasComm->hangFlag & (1 << c)) > 0) {
          // clear hang flag cause current channel got back to normal from timeout status (maybe after a long work)
          rasComm->hangFlag &= ~(1 << c);
        }
        if (currentDevIndex[c] != rasComm->lastDevIndex[c]) {
          VERBOSE(NCCL_RAS, "RAS %ld work(s) newly completed, comm %p, rank %d, nranks %d, commhash %lx, channel %d, equeued %ld, completed %ld",
                  currentDevIndex[c]-rasComm->lastDevIndex[c], comm, comm->rank, comm->nRanks, comm->commHash, c, currentHostIndex[c], currentDevIndex[c]);
        } else {
          VERBOSE(NCCL_RAS, "RAS %ld work(s) newly enqueued, comm %p, rank %d, nranks %d, commhash %lx, channel %d, equeued %ld, completed %ld",
                  currentHostIndex[c]-rasComm->lastHostIndex[c], comm, comm->rank, comm->nRanks, comm->commHash, c, currentHostIndex[c], currentDevIndex[c]);
        }
      }

      // timeout
      if ((currentHostIndex[c] != currentDevIndex[c]) && (currentDevIndex[c] == rasComm->lastDevIndex[c]) && (currentTime-rasComm->lastUpdateTime > rasComm->timeout) && ((rasComm->hangFlag & (1 << c)) == 0)) {
        snprintf(infoStr, STATE_DUMP_LINE_LENGTH-1, "RAS no pending work(s) completed over %.2f seconds, comm %p, rank %d, nranks %d, commhash %lx, channel %d, enqueued %ld, completed %ld, datetime %s",
                  (currentTime-rasComm->lastUpdateTime)/1e9, comm, comm->rank, comm->nRanks, comm->commHash, c, currentHostIndex[c], currentDevIndex[c], getDateTime(timeStr));
        rasStateMonitorRecord(comm, infoStr);
        currentMaxDevIndex = std::max(currentMaxDevIndex, currentDevIndex[c]+1);
        rasComm->hangFlag |= (1 << c);
        detectTimeout = true;
      }
    }
    // broadcast hang status and trigger hang dump when a larger timeout index detected
    if (detectTimeout) {
      if (currentMaxDevIndex > rasComm->lastTimeoutIndex) {
        struct rasStateMonitorStatus status;
        snprintf(infoStr, STATE_DUMP_LINE_LENGTH-1, "RAS broadcast kernel timeout status, comm %p, rank %d, nranks %d, commhash %lx, seq %ld, time %s",
                  comm, comm->rank, comm->nRanks, comm->commHash, currentMaxDevIndex, getDateTime(timeStr));
        rasStateMonitorRecord(comm, infoStr);
        status.timeout = 1;
        status.commHash = comm->commHash;
        status.seq = currentMaxDevIndex;
        rasStateMonitorBroadcastTimeout(status);
      }
      rasComm->timeout += std::min(rasComm->timeout, UINT64_MAX-rasComm->timeout);
    }

    // update last host and device index
    for (int c = 0; c < nChannels; c++) {
      rasComm->lastHostIndex[c] = currentHostIndex[c];
      rasComm->lastDevIndex[c] = currentDevIndex[c];
    }

    // update log timestamp
    if (showCurrentState) {
      lastLogTime = currentTime;
    }
  }
}
