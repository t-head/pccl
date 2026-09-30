//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

/*************************************************************************
 * Online hang analyzer implementation (merged-COMMS design).
 *
 * Snapshot logic adapted from the reference analyzer tool and the reference
 * 0001-pccl-online-debug.patch.  Cross-rank checks are run against the
 * merged rasCollComms blob (which carries per-comm diag tiers when the
 * RAS_COMMS_FLAG_INCLUDE_DIAG bit was set on the request).
 ************************************************************************/

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "utils.h"
#include "param.h"
#include "ras_internal.h"
#include "analyzer.h"
#include "drv_extension.h"
#include "hggc_runtime.h"
#include "comm_dumper.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>


///////////////////////////////////////////////////////////////////////////////
// Local snapshot size + serialization.                                      //
// Caller must hold ncclCommsMutex.                                          //
///////////////////////////////////////////////////////////////////////////////

static int countPopulatedChannels(ncclComm_t comm) {
  int n = 0;
  for (int c = 0; c < MAXCHANNELS; c++) if (comm->channels[c].id >= 0) n++;
  return n;
}

static int countConnectedPeers(struct ncclChannel* channel, int nRanks) {
  int n = 0;
  if (channel->peers == nullptr) return 0;
  for (int p = 0; p < nRanks; p++) {
    struct ncclChannelPeer* hostPeer = channel->peers[p];
    if (hostPeer == nullptr) continue;
    for (int c0 = 0; c0 < NCCL_MAX_CONNS; c0++) {
      if (hostPeer->send[c0].connected == 1) n++;
      if (hostPeer->recv[c0].connected == 1) n++;
    }
  }
  return n;
}

static uint64_t readDevHangReportedKernelIdx(struct ncclChannel* channel, bool* outValid) {
  return readDevU64(channel->hostDevIndexLevel1, outValid);
}

static uint64_t readDevCompletedKernelIdx(struct ncclChannel* channel, uint8_t monitorLevel, bool* outValid) {
  uint64_t* devPtr = (monitorLevel == 2 ? channel->hostDevIndexLevel2 : channel->devIndexLevel1);
  return readDevU64(devPtr, outValid);
}

size_t rasDiagTotalSnapshotDataSize(ncclComm_t comm) {
  if (comm == nullptr) return 0;
  if (comm->abortFlag && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0) return 0;
  size_t totalBytes = sizeof(struct rasDiagRankHeader);
  for (int c = 0; c < MAXCHANNELS; c++) {
    struct ncclChannel* channel = &comm->channels[c];
    if (channel->id < 0) continue;
    totalBytes += sizeof(struct rasDiagChannel);
    totalBytes += (size_t)countConnectedPeers(channel, comm->nRanks) * sizeof(struct rasDiagPeer);
  }
  return totalBytes;
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

size_t rasDiagSnapshotInto(char* buf, size_t bufLen, ncclComm_t comm) {
  if (comm == nullptr || buf == nullptr) return 0;
  if (comm->abortFlag && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0) return 0;
  size_t need = rasDiagTotalSnapshotDataSize(comm);
  if (need == 0 || bufLen < need) return 0;

  uint8_t monitorLevel = comm->hostDevComm.comm.stateMonitorLevel;

  struct rasDiagRankHeader* rh = (struct rasDiagRankHeader*)buf;
  rh->wireVersion = RAS_DIAG_WIRE_VERSION;
  rh->commRank = comm->rank;
  rh->nChannels = countPopulatedChannels(comm);
  char* cursor = buf + sizeof(*rh);

  for (int c = 0; c < MAXCHANNELS; c++) {
    struct ncclChannel* channel = &comm->channels[c];
    if (channel->id < 0) continue;

    struct rasDiagChannel* dc = (struct rasDiagChannel*)cursor;
    populateDiagChannel(dc, comm, c);
    bool valid = false;
    dc->devHangReportedKernelIdx = readDevHangReportedKernelIdx(channel, &valid);
    dc->devHangReportedKernelIdxValid = valid ? 1 : 0;
    dc->devCompletedKernelIdx = readDevCompletedKernelIdx(channel, monitorLevel, &valid);
    dc->devCompletedKernelIdxValid = valid ? 1 : 0;
    cursor += sizeof(*dc);

    int nPeerEntries = 0;
    if (channel->peers != nullptr) {
      for (int p = 0; p < comm->nRanks; p++) {
        struct ncclChannelPeer* hostPeer = channel->peers[p];
        struct ncclDevChannelPeer* devPeer = (channel->devPeersHostPtr ? channel->devPeersHostPtr[p] : nullptr);
        if (hostPeer == nullptr) continue;
        for (int c0 = 0; c0 < NCCL_MAX_CONNS; c0++) {
          if (hostPeer->send[c0].connected == 1) {
            struct rasDiagPeer* dp = (struct rasDiagPeer*)cursor;
            cursor += sizeof(*dp);
            dp->peerRank = p;
            dp->connector = (uint8_t)c0;
            dp->direction = 0;
            dp->connected = 1;
            dp->_pad = 0;
            dp->head = safeReadDevU64(hostPeer->send[c0].conn.head);
            dp->tail = safeReadDevU64(hostPeer->send[c0].conn.tail);
            dp->devStep = (devPeer ? safeReadDevU64(&devPeer->send[c0].step) : 0);
            nPeerEntries++;
          }
          if (hostPeer->recv[c0].connected == 1) {
            struct rasDiagPeer* dp = (struct rasDiagPeer*)cursor;
            cursor += sizeof(*dp);
            dp->peerRank = p;
            dp->connector = (uint8_t)c0;
            dp->direction = 1;
            dp->connected = 1;
            dp->_pad = 0;
            dp->head = safeReadDevU64(hostPeer->recv[c0].conn.head);
            dp->tail = safeReadDevU64(hostPeer->recv[c0].conn.tail);
            dp->devStep = (devPeer ? safeReadDevU64(&devPeer->recv[c0].step) : 0);
            nPeerEntries++;
          }
        }
      }
    }
    dc->nPeerEntries = nPeerEntries;
  }

  size_t written = (size_t)(cursor - buf);
  rh->totalBytes = (uint32_t)written;
  return written;
}

///////////////////////////////////////////////////////////////////////////////
// Dump-path collector: piggyback on pcclCommDump's device reads to fill the //
// diag buffer.  Avoids the second device-IO pass that rasDiagSnapshot*      //
// would otherwise do when the monitor path runs dump + diag together.       //
///////////////////////////////////////////////////////////////////////////////

namespace {
struct rasDiagCollectorEntry {
  char* scratch;
  size_t capacity;
  size_t cursor;
  struct rasDiagChannel* currentChannel;
};

struct rasDiagCollectorImpl {
  std::unordered_map<ncclComm_t, rasDiagCollectorEntry> entries;
  pcclCommDiagCollector iface;
};

static void rasDiagInitRankDiagHeader(void* ctx, ncclComm_t comm) {
  if (comm == nullptr) return;
  auto* impl = (rasDiagCollectorImpl*)ctx;
  size_t need = rasDiagTotalSnapshotDataSize(comm);
  if (need < sizeof(struct rasDiagRankHeader)) return;
  rasDiagCollectorEntry e{};
  if (ncclCalloc(&e.scratch, need) != ncclSuccess) return;
  e.capacity = need;
  auto* rh = (struct rasDiagRankHeader*)e.scratch;
  rh->wireVersion = RAS_DIAG_WIRE_VERSION;
  rh->commRank = comm->rank;
  rh->nChannels = comm->nChannels;
  rh->totalBytes = (uint32_t)sizeof(*rh);
  e.cursor = sizeof(*rh);
  e.currentChannel = nullptr;
  impl->entries[comm] = e;
}

static void rasDiagCollectChannelState(void* ctx, ncclComm_t comm, int channelId,
                                   uint64_t devHangReportedKernelIdx, bool devHangReportedKernelIdxValid,
                                   uint64_t devCompletedKernelIdx,    bool devCompletedKernelIdxValid) {
  if (comm == nullptr) return;
  auto* impl = (rasDiagCollectorImpl*)ctx;
  auto it = impl->entries.find(comm);
  if (it == impl->entries.end()) return;
  rasDiagCollectorEntry& e = it->second;
  if (e.scratch == nullptr) return;
  if (e.cursor + sizeof(struct rasDiagChannel) > e.capacity) { e.currentChannel = nullptr; return; }
  auto* dc = (struct rasDiagChannel*)(e.scratch + e.cursor);
  e.cursor += sizeof(*dc);
  populateDiagChannel(dc, comm, channelId);
  dc->devHangReportedKernelIdx = devHangReportedKernelIdx;
  dc->devHangReportedKernelIdxValid = devHangReportedKernelIdxValid ? 1 : 0;
  dc->devCompletedKernelIdx = devCompletedKernelIdx;
  dc->devCompletedKernelIdxValid = devCompletedKernelIdxValid ? 1 : 0;
  e.currentChannel = dc;
}

static void rasDiagCollectPeerConn(void* ctx, ncclComm_t comm, int /*channelId*/, int peerRank,
                                uint8_t connector, uint8_t direction,
                                uint64_t head, uint64_t tail, uint64_t devStep) {
  if (comm == nullptr) return;
  auto* impl = (rasDiagCollectorImpl*)ctx;
  auto it = impl->entries.find(comm);
  if (it == impl->entries.end()) return;
  rasDiagCollectorEntry& e = it->second;
  if (e.currentChannel == nullptr) return;
  if (e.cursor + sizeof(struct rasDiagPeer) > e.capacity) return;
  auto* dp = (struct rasDiagPeer*)(e.scratch + e.cursor);
  e.cursor += sizeof(*dp);
  dp->peerRank = peerRank;
  dp->connector = connector;
  dp->direction = direction;
  dp->connected = 1;
  dp->_pad = 0;
  dp->head = head;
  dp->tail = tail;
  dp->devStep = devStep;
  e.currentChannel->nPeerEntries++;
}

static void rasDiagFinalizeRankDiagHeader(void* ctx, ncclComm_t comm) {
  if (comm == nullptr) return;
  auto* impl = (rasDiagCollectorImpl*)ctx;
  auto it = impl->entries.find(comm);
  if (it == impl->entries.end()) return;
  rasDiagCollectorEntry& e = it->second;
  if (e.scratch == nullptr) return;
  auto* rh = (struct rasDiagRankHeader*)e.scratch;
  rh->totalBytes = (uint32_t)e.cursor;
}
}  // anon namespace

struct pcclCommDiagCollector* rasDiagCollectorCreate() {
  auto* impl = new (std::nothrow) rasDiagCollectorImpl();
  if (impl == nullptr) return nullptr;
  impl->iface.ctx = impl;
  impl->iface.initRankDiagHeader = rasDiagInitRankDiagHeader;
  impl->iface.collectChannelState = rasDiagCollectChannelState;
  impl->iface.collectPeerConn = rasDiagCollectPeerConn;
  impl->iface.finalizeRankDiagHeader = rasDiagFinalizeRankDiagHeader;
  // impl escapes via the returned iface (iface.ctx == impl) and is freed by rasDiagCollectorDestroy; not leaked.
  // coverity[leaked_storage]
  return &impl->iface;
}

const char* rasDiagCollectorGet(struct pcclCommDiagCollector* collector,
                                     ncclComm_t comm, size_t* pBytes) {
  if (pBytes != nullptr) *pBytes = 0;
  if (collector == nullptr || comm == nullptr) return nullptr;
  auto* impl = (rasDiagCollectorImpl*)collector->ctx;
  auto it = impl->entries.find(comm);
  if (it == impl->entries.end()) return nullptr;
  const rasDiagCollectorEntry& e = it->second;
  if (e.scratch == nullptr || e.cursor == 0) return nullptr;
  if (pBytes != nullptr) *pBytes = e.cursor;
  return e.scratch;
}

void rasDiagCollectorDestroy(struct pcclCommDiagCollector* collector) {
  if (collector == nullptr) return;
  auto* impl = (rasDiagCollectorImpl*)collector->ctx;
  for (auto& kv : impl->entries) ncclFree(kv.second.scratch);
  delete impl;
}


///////////////////////////////////////////////////////////////////////////////
// Monitor-originated trigger + onComplete report writer.                    //
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
  int64_t now = clockNano();
  if (dedupShouldSkip(commHash, seq, now)) {
    INFO(NCCL_RAS, "RAS diag: dedup skip (commHash 0x%lx, seq %lu)",
         (unsigned long)commHash, (unsigned long)seq);
    return;
  }

  struct rasCollRequest req;
  memset(&req, 0, sizeof(req));
  rasCollReqInit(&req);
  req.type = RAS_COLL_COMMS;
  req.timeout = RAS_COLLECTIVE_LEG_TIMEOUT;
  req.comms.flags = RAS_COMMS_FLAG_INCLUDE_DIAG | RAS_COMMS_FLAG_TRIGGER_LOCAL_DUMP;
  req.comms.diagOnlyCommHash = commHash;
  req.comms.diagSeq = seq;
  req.comms.nSkipMissingRanksComms = 0;

  ncclResult_t r = rasNetSendCollReq(&req, /*pAllDone*/nullptr, /*pColl*/nullptr,
                                     /*fromConn*/nullptr,
                                     /*onComplete*/rasDiagOnComplete);
  if (r != ncclSuccess) {
    WARN("RAS diag: rasNetSendCollReq returned %d", r);
    return;
  }
  INFO(NCCL_RAS, "RAS diag: triggered COMMS+DIAG+TRIGGER_LOCAL_DUMP (commHash 0x%lx, seq %lu)",
       (unsigned long)commHash, (unsigned long)seq);
}

ncclResult_t rasDiagWriteFullReport(const struct rasCollComms* blob,
                                         int nBlobBytes, int nLegTimeouts,
                                         char* pathOut, size_t pathLen) {
  if (pathOut != nullptr && pathLen > 0) pathOut[0] = '\0';
  if (blob == nullptr || blob->nComms == 0) {
    WARN("RAS diag: no comm data; nothing to report");
    return ncclInvalidArgument;
  }

  try {
    int dirOffset = 0;
    char dir[PATH_MAX + 1];
    if (createOrFillDumpDir(dir, PATH_MAX, "analyze", &dirOffset) != ncclSuccess) {
      WARN("RAS diag: createOrFillDumpDir failed");
      return ncclSystemError;
    }
    dir[dirOffset] = '\0';

    if (nBlobBytes > 0 && analyzerDumpData(dir, blob, nBlobBytes, nLegTimeouts) != 0)
      WARN("RAS diag: analyzerDumpData failed");

    int nWritten = analyzerDumpReport(dir, blob, nLegTimeouts);
    if (nWritten <= 0) {
      WARN("RAS diag: failed to write any per-category report");
      return ncclSystemError;
    }

    if (pathOut != nullptr && pathLen > 0) {
      strncpy(pathOut, dir, pathLen - 1);
      pathOut[pathLen - 1] = '\0';
    }
    WARN("RAS diag: %d per-category report(s) + diag blob written under %s",
         nWritten, dir);
    return ncclSuccess;
  } catch (...) {
    WARN("RAS diag: internal error (exception caught in WriteFullReport)");
    return ncclInternalError;
  }
}

static void warnSummaryLines(const struct rasCollComms* blob) {
  if (blob == nullptr) return;
  char* buf = nullptr;
  size_t bufLen = 0;
  FILE* mem = open_memstream(&buf, &bufLen);
  if (mem == nullptr) return;
  analyzerFormatSummary(mem, blob);
  fclose(mem);
  if (buf == nullptr || buf[0] == '\0') { free(buf); return; }

  size_t len = strlen(buf);
  char* p = buf;
  while (p < buf + len) {
    char* nl = (char*)memchr(p, '\n', (buf + len) - p);
    size_t n = nl ? (size_t)(nl - p) : (size_t)(buf + len - p);
    if (n > 0) WARN("RAS diag: %.*s", (int)n, p);
    if (!nl) break;
    p = nl + 1;
  }
  free(buf);
}

void rasDiagOnComplete(struct rasCollective* coll) {
  if (coll == nullptr) return;
  try {
    struct rasCollComms* blob = (struct rasCollComms*)coll->data;
    warnSummaryLines(blob);
    (void)rasDiagWriteFullReport(blob, coll->nData, coll->nLegTimeouts, /*pathOut*/nullptr, /*pathLen*/0);
  } catch (...) {
    WARN("RAS diag: internal error (exception caught in onComplete)");
  }
  rasCollFree(coll);
}
