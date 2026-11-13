//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

/*************************************************************************
 * Runtime state collection, monitor triggers and report persistence.
 * Dump sampling and on-demand sampling share the same collector callbacks.
 ************************************************************************/

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "transport.h"
#include "utils.h"
#include "param.h"
#include "ras_internal.h"
#include "analyzer.h"
#include "drv_extension.h"
#include "hggc_runtime.h"
#include "comm_dumper.h"
#include "diagnostics.h"
#include "diagnostics_runtime.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

static_assert(RAS_DIAG_RUNTIME_MAX_CONNS == NCCL_MAX_CONNS, "runtime connector limit must match transport");

// Callers hold ncclCommsMutex and select the communicator's device before sampling.

static uint64_t readDevHangReportedKernelIdx(struct ncclChannel* channel, bool* outValid) {
  return readDevU64(channel->hostDevIndexLevel1, outValid);
}

static uint64_t readDevCompletedKernelIdx(struct ncclChannel* channel, uint8_t monitorLevel, bool* outValid) {
  uint64_t* devPtr = (monitorLevel == 2 ? channel->hostDevIndexLevel2 : channel->devIndexLevel1);
  return readDevU64(devPtr, outValid);
}

static void populateDiagChannel(struct rasDiagChannel* dc, ncclComm_t comm, int channelId) {
  struct ncclChannel* channel = &comm->channels[channelId];
  memset(dc, 0, sizeof(*dc));
  dc->channelId = channelId;
  dc->workFifoTail = comm->channelKernelIdx[channelId];

  bool hasRing = (channelId < comm->nChannels) && (channel->ring.userRanks != nullptr);
  if (hasRing) {
    int prev = channel->ring.prev, next = channel->ring.next;
    bool collEstablished = false;
    if (channel->peers != nullptr) {
      if (next >= 0 && next < comm->nRanks && channel->peers[next] != nullptr
          && channel->peers[next]->send[0].connected == 1) collEstablished = true;
      if (prev >= 0 && prev < comm->nRanks && channel->peers[prev] != nullptr
          && channel->peers[prev]->recv[0].connected == 1) collEstablished = true;
    }
    if (!collEstablished) hasRing = false;
  }
  bool hasTree = hasRing;

  if (hasRing) {
    dc->ringPrev = channel->ring.prev;
    dc->ringNext = channel->ring.next;
    dc->ringSendConn = (channel->ring.next < 0) ? RAS_DIAG_CONN_NONE : (uint8_t)channel->ring.sendConnType;
    dc->ringRecvConn = (channel->ring.prev < 0) ? RAS_DIAG_CONN_NONE : (uint8_t)channel->ring.recvConnType;
  } else {
    dc->ringPrev = -1;
    dc->ringNext = -1;
    dc->ringSendConn = RAS_DIAG_CONN_NONE;
    dc->ringRecvConn = RAS_DIAG_CONN_NONE;
  }

  if (hasTree) {
    dc->treeUp = channel->tree.up;
    dc->treeUpConn = (channel->tree.up < 0) ? RAS_DIAG_CONN_NONE : (uint8_t)channel->tree.upConnType;
    bool hasTreeDown = false;
    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
      dc->treeDown[i] = channel->tree.down[i];
      if (channel->tree.down[i] >= 0) hasTreeDown = true;
    }
    dc->treeDownConn = hasTreeDown ? (uint8_t)channel->tree.downConnType : RAS_DIAG_CONN_NONE;
  } else {
    dc->treeUp = -1;
    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) dc->treeDown[i] = -1;
    dc->treeUpConn = RAS_DIAG_CONN_NONE;
    dc->treeDownConn = RAS_DIAG_CONN_NONE;
  }

  dc->fcConnType = (channel->fc.nPeers > 0) ? (uint8_t)channel->fc.connType : RAS_DIAG_CONN_NONE;
}

///////////////////////////////////////////////////////////////////////////////
// Shared collector for dump callbacks and on-demand device sampling.        //
///////////////////////////////////////////////////////////////////////////////

namespace {
struct rasDiagCollectorEntry {
  std::vector<rasDiagnosticsRuntimeRecord> records;
  int channelRecords[MAXCHANNELS];
  ncclResult_t result = ncclSuccess;
  bool finalized = false;
};

struct rasDiagCollectorImpl {
  std::unordered_map<ncclComm_t, rasDiagCollectorEntry> entries;
  pcclCommDiagCollector iface;
};

static bool rasDiagAppendRecord(rasDiagCollectorEntry& entry, const rasDiagnosticsRuntimeRecord& record) {
  if (entry.result != ncclSuccess || entry.finalized) return false;
  if (entry.records.size() >= INT_MAX / sizeof(record)) {
    entry.result = ncclInternalError;
    return false;
  }
  try {
    entry.records.push_back(record);
    return true;
  } catch (const std::bad_alloc&) {
    entry.result = ncclSystemError;
  } catch (...) {
    entry.result = ncclInternalError;
  }
  return false;
}

static void rasDiagInitRankDiagHeader(void* ctx, ncclComm_t comm) {
  if (ctx == nullptr || comm == nullptr) return;
  auto* impl = (rasDiagCollectorImpl*)ctx;
  // Re-sampling replaces any earlier result, including a partially collected dump.
  impl->entries.erase(comm);
  if (!__atomic_load_n(&comm->peerInfoValid, __ATOMIC_ACQUIRE) || comm->peerInfo == nullptr ||
      (comm->abortFlag && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) ||
      comm->nRanks <= 0 || comm->rank < 0 || comm->rank >= comm->nRanks) return;

  try {
    rasDiagCollectorEntry entry;
    for (int c = 0; c < MAXCHANNELS; c++) entry.channelRecords[c] = -1;
    rasDiagnosticsRuntimeRecord record = {};
    record.rank.commId.commHash = comm->commHash;
    record.rank.commId.hostHash = comm->peerInfo[0].hostHash;
    record.rank.commId.pidHash = comm->peerInfo[0].pidHash;
    record.rank.commRank = comm->rank;
    record.rank.commNRanks = comm->nRanks;
    record.type = RAS_DIAG_RUNTIME_RANK;
    record.channelId = -1;
    record.value.rankData.version = RAS_DIAG_RUNTIME_WIRE_VERSION;
    record.value.rankData.availability = RAS_DIAG_RUNTIME_UNAVAILABLE;
    record.value.rankData.hggcDev = comm->hggcDev;
    record.value.rankData.hgmlDev = comm->hgmlDev;
    if (rasDiagAppendRecord(entry, record)) impl->entries.emplace(comm, std::move(entry));
  } catch (...) {
    // The dump callbacks cannot propagate allocation failures to their caller.
    impl->entries.erase(comm);
  }
}

static void rasDiagCollectChannelState(void* ctx, ncclComm_t comm, int channelId,
                                      uint64_t devHangReportedKernelIdx, bool devHangReportedKernelIdxValid,
                                      uint64_t devCompletedKernelIdx, bool devCompletedKernelIdxValid) {
  if (ctx == nullptr || comm == nullptr) return;
  auto* impl = (rasDiagCollectorImpl*)ctx;
  auto it = impl->entries.find(comm);
  if (it == impl->entries.end()) return;
  rasDiagCollectorEntry& entry = it->second;
  if (entry.result != ncclSuccess || entry.finalized) return;
  if (channelId < 0 || channelId >= MAXCHANNELS || entry.channelRecords[channelId] >= 0) {
    entry.result = ncclInternalError;
    return;
  }

  rasDiagnosticsRuntimeRecord record = {};
  record.rank = entry.records[0].rank;
  record.type = RAS_DIAG_RUNTIME_CHANNEL;
  record.channelId = channelId;
  rasDiagChannel& channel = record.value.channel;
  populateDiagChannel(&channel, comm, channelId);
  channel.devHangReportedKernelIdx = devHangReportedKernelIdx;
  channel.devHangReportedKernelIdxValid = devHangReportedKernelIdxValid ? 1 : 0;
  channel.devCompletedKernelIdx = devCompletedKernelIdx;
  channel.devCompletedKernelIdxValid = devCompletedKernelIdxValid ? 1 : 0;
  int recordIndex = (int)entry.records.size();
  if (rasDiagAppendRecord(entry, record)) {
    entry.channelRecords[channelId] = recordIndex;
    entry.records[0].value.rankData.nChannels++;
  }
}

static void rasDiagCollectPeerConn(void* ctx, ncclComm_t comm, int channelId, int peerRank,
                                   uint8_t connector, uint8_t direction,
                                   uint64_t head, uint64_t tail, uint64_t devStep) {
  if (ctx == nullptr || comm == nullptr) return;
  auto* impl = (rasDiagCollectorImpl*)ctx;
  auto it = impl->entries.find(comm);
  if (it == impl->entries.end()) return;
  rasDiagCollectorEntry& entry = it->second;
  if (entry.result != ncclSuccess || entry.finalized) return;
  if (channelId < 0 || channelId >= MAXCHANNELS || entry.channelRecords[channelId] < 0 ||
      peerRank < 0 || peerRank >= comm->nRanks || connector >= NCCL_MAX_CONNS || direction > 1) {
    entry.result = ncclInternalError;
    return;
  }

  rasDiagnosticsRuntimeRecord record = {};
  record.rank = entry.records[0].rank;
  record.type = RAS_DIAG_RUNTIME_CONNECTION;
  record.channelId = channelId;
  rasDiagPeer& peer = record.value.connection;
  peer.peerRank = peerRank;
  peer.connector = connector;
  peer.direction = direction;
  peer.connected = 1;
  peer.head = head;
  peer.tail = tail;
  peer.devStep = devStep;
  if (rasDiagAppendRecord(entry, record)) {
    // Store indices, not pointers: adding a connection can reallocate the vector.
    entry.records[entry.channelRecords[channelId]].value.channel.nPeerEntries++;
  }
}

static void rasDiagFinalizeRankDiagHeader(void* ctx, ncclComm_t comm) {
  if (ctx == nullptr || comm == nullptr) return;
  auto* impl = (rasDiagCollectorImpl*)ctx;
  auto it = impl->entries.find(comm);
  if (it == impl->entries.end()) return;
  rasDiagCollectorEntry& entry = it->second;
  if (entry.result != ncclSuccess || entry.finalized) return;
  if (comm->abortFlag && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
    entry.result = ncclInternalError;
    return;
  }
  // A dump may skip peer reads while still invoking finalize. Do not expose
  // such a partial sample as a complete runtime record set.
  for (int c = 0; c < MAXCHANNELS; c++) {
    ncclChannel* channel = &comm->channels[c];
    if (channel->id < 0) continue;
    int index = entry.channelRecords[c];
    if (index < 0) {
      entry.result = ncclInternalError;
      return;
    }
    size_t nConnections = 0;
    if (channel->peers != nullptr) {
      for (int p = 0; p < comm->nRanks; p++) {
        ncclChannelPeer* peer = channel->peers[p];
        if (peer == nullptr) continue;
        for (int conn = 0; conn < NCCL_MAX_CONNS; conn++) {
          nConnections += peer->send[conn].connected == 1;
          nConnections += peer->recv[conn].connected == 1;
        }
      }
    }
    if (nConnections != (size_t)entry.records[index].value.channel.nPeerEntries) {
      entry.result = ncclInternalError;
      return;
    }
  }
  entry.records[0].value.rankData.availability = RAS_DIAG_RUNTIME_AVAILABLE;
  entry.finalized = true;
}

static ncclResult_t rasDiagCollectorResult(pcclCommDiagCollector* collector, ncclComm_t comm) {
  auto* impl = (rasDiagCollectorImpl*)collector->ctx;
  auto it = impl->entries.find(comm);
  return it == impl->entries.end() ? ncclInternalError : it->second.result;
}
} // namespace

struct pcclCommDiagCollector* rasDiagCollectorCreate() {
  auto* impl = new (std::nothrow) rasDiagCollectorImpl();
  if (impl == nullptr) return nullptr;
  impl->iface.ctx = impl;
  impl->iface.initRankDiagHeader = rasDiagInitRankDiagHeader;
  impl->iface.collectChannelState = rasDiagCollectChannelState;
  impl->iface.collectPeerConn = rasDiagCollectPeerConn;
  impl->iface.finalizeRankDiagHeader = rasDiagFinalizeRankDiagHeader;
  // impl escapes via iface.ctx and is freed by rasDiagCollectorDestroy.
  // coverity[leaked_storage]
  return &impl->iface;
}

const struct rasDiagnosticsRuntimeRecord* rasDiagCollectorGetRuntime(struct pcclCommDiagCollector* collector,
                                                                    ncclComm_t comm, int* nRecords) {
  if (nRecords != nullptr) *nRecords = 0;
  if (collector == nullptr || comm == nullptr) return nullptr;
  auto* impl = (rasDiagCollectorImpl*)collector->ctx;
  auto it = impl->entries.find(comm);
  if (it == impl->entries.end()) return nullptr;
  const rasDiagCollectorEntry& entry = it->second;
  if (entry.result != ncclSuccess || !entry.finalized || entry.records.empty()) return nullptr;
  if (nRecords != nullptr) *nRecords = (int)entry.records.size();
  return entry.records.data();
}

void rasDiagCollectorDestroy(struct pcclCommDiagCollector* collector) {
  if (collector == nullptr) return;
  delete (rasDiagCollectorImpl*)collector->ctx;
}

ncclResult_t rasDiagCollectRuntimeSnapshot(ncclComm_t comm, struct pcclCommDiagCollector* collector) {
  if (comm == nullptr || collector == nullptr) return ncclInvalidArgument;
  collector->initRankDiagHeader(collector->ctx, comm);
  NCCLCHECK(rasDiagCollectorResult(collector, comm));
  uint8_t monitorLevel = comm->hostDevComm.comm.stateMonitorLevel;

  for (int c = 0; c < MAXCHANNELS; c++) {
    struct ncclChannel* channel = &comm->channels[c];
    if (channel->id < 0) continue;
    bool hangValid = false, completedValid = false;
    uint64_t hang = readDevHangReportedKernelIdx(channel, &hangValid);
    uint64_t completed = readDevCompletedKernelIdx(channel, monitorLevel, &completedValid);
    collector->collectChannelState(collector->ctx, comm, c, hang, hangValid, completed, completedValid);
    NCCLCHECK(rasDiagCollectorResult(collector, comm));

    if (channel->peers == nullptr) continue;
    for (int p = 0; p < comm->nRanks; p++) {
      struct ncclChannelPeer* hostPeer = channel->peers[p];
      struct ncclDevChannelPeer* devPeer = channel->devPeersHostPtr ? channel->devPeersHostPtr[p] : nullptr;
      if (hostPeer == nullptr) continue;
      for (int c0 = 0; c0 < NCCL_MAX_CONNS; c0++) {
        if (hostPeer->send[c0].connected == 1) {
          uint64_t head = safeReadDevU64(hostPeer->send[c0].conn.head);
          uint64_t tail = safeReadDevU64(hostPeer->send[c0].conn.tail);
          uint64_t step = devPeer ? safeReadDevU64(&devPeer->send[c0].step) : 0;
          collector->collectPeerConn(collector->ctx, comm, c, p, (uint8_t)c0, 0, head, tail, step);
          NCCLCHECK(rasDiagCollectorResult(collector, comm));
        }
        if (hostPeer->recv[c0].connected == 1) {
          uint64_t head = safeReadDevU64(hostPeer->recv[c0].conn.head);
          uint64_t tail = safeReadDevU64(hostPeer->recv[c0].conn.tail);
          uint64_t step = devPeer ? safeReadDevU64(&devPeer->recv[c0].step) : 0;
          collector->collectPeerConn(collector->ctx, comm, c, p, (uint8_t)c0, 1, head, tail, step);
          NCCLCHECK(rasDiagCollectorResult(collector, comm));
        }
      }
    }
  }
  collector->finalizeRankDiagHeader(collector->ctx, comm);
  return rasDiagCollectorResult(collector, comm);
}


///////////////////////////////////////////////////////////////////////////////
// Monitor-originated trigger and report writer.                             //
///////////////////////////////////////////////////////////////////////////////

namespace {
// Local dedup table.  RAS thread is single-threaded, so no lock needed.
static std::unordered_map<uint64_t, int64_t> g_dedupExpiry;

static uint64_t dedupKey(uint64_t commHash, uint64_t seq) {
  return commHash ^ (seq * 0x9E3779B97F4A7C15ULL);
}

static bool dedupShouldSkip(uint64_t commHash, uint64_t seq, int64_t now) {
  for (auto it = g_dedupExpiry.begin(); it != g_dedupExpiry.end(); ) {
    if (it->second < now) it = g_dedupExpiry.erase(it);
    else ++it;
  }
  uint64_t k = dedupKey(commHash, seq);
  if (g_dedupExpiry.count(k)) return true;
  g_dedupExpiry[k] = now + 2 * STATE_MONITOR_TIMEOUT;
  return false;
}
}  // anon namespace

void rasDiagTriggerFromMonitor(uint64_t commHash, uint64_t seq) {
  struct rasDiagnosticsContext ctx = {};
  ncclResult_t ret = ncclInvalidArgument;

  // Copy the communicator identity while it is protected; the asynchronous
  // request must not retain a pointer to the live communicator.
  {
    std::lock_guard<std::recursive_mutex> lock(ncclCommsMutex);
    for (int i = 0; i < nNcclComms; i++) {
      ncclComm_t comm = ncclComms[i].comm;
      if (comm == nullptr || !__atomic_load_n(&comm->peerInfoValid, __ATOMIC_ACQUIRE) ||
          comm->commHash != commHash) continue;
      ret = rasDiagnosticsContextInit(&ctx, comm);
      break;
    }
  }
  if (ret != ncclSuccess) {
    WARN("RAS diagnostics could not find communicator 0x%lx for sequence %lu",
         (unsigned long)commHash, (unsigned long)seq);
    return;
  }

  int64_t now = clockNano();
  if (dedupShouldSkip(commHash, seq, now)) {
    INFO(NCCL_RAS, "RAS diag: dedup skip (commHash 0x%lx, seq %lu)",
         (unsigned long)commHash, (unsigned long)seq);
    return;
  }

  ctx.triggerLocalDump = true;
  ctx.diagSeq = seq;
  ctx.persistReport = true;
  ctx.summaryOnly = true;
  ret = rasLocalHandleRunDiag(&ctx);
  if (ret != ncclSuccess) {
    g_dedupExpiry.erase(dedupKey(commHash, seq));
    WARN("RAS diagnostics trigger returned %d for comm 0x%lx", ret, (unsigned long)commHash);
    return;
  }
  INFO(NCCL_RAS, "RAS diagnostics triggered (commHash 0x%lx, seq %lu)",
       (unsigned long)commHash, (unsigned long)seq);
}

ncclResult_t rasDiagWriteRuntimeReport(const struct rasDiagnosticsRuntimeRecord* records,
                                     int nRecords, int nLegTimeouts, char* pathOut, size_t pathLen) {
  if (pathOut != nullptr && pathLen > 0) pathOut[0] = '\0';
  if (records == nullptr || nRecords <= 0) {
    WARN("RAS diag: no runtime records; nothing to report");
    return ncclInvalidArgument;
  }

  int dirOffset = 0;
  char dir[PATH_MAX + 1];
  if (createOrFillDumpDir(dir, PATH_MAX, "analyze", &dirOffset) != ncclSuccess) {
    WARN("RAS diag: createOrFillDumpDir failed");
    return ncclSystemError;
  }
  dir[dirOffset] = '\0';

  bool rawDataSaved = analyzerDumpRuntimeData(dir, records, nRecords, nLegTimeouts) == 0;
  if (!rawDataSaved)
    WARN("RAS diag: analyzerDumpRuntimeData failed");
  int nWritten = analyzerDumpRuntimeReport(dir, records, nRecords, nLegTimeouts);
  if (nWritten <= 0) {
    WARN("RAS diag: failed to write any per-category report");
    return ncclSystemError;
  }

  if (pathOut != nullptr && pathLen > 0) {
    strncpy(pathOut, dir, pathLen - 1);
    pathOut[pathLen - 1] = '\0';
  }
  WARN("RAS diag: %d per-category report(s) written under %s (runtime records %s)",
       nWritten, dir, rawDataSaved ? "saved" : "not saved");
  return ncclSuccess;
}
