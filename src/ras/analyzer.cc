//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "analyzer.h"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <unistd.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define PCCL_RAS_ANALYZER_MAGIC        0x50434844u
#define PCCL_RAS_ANALYZER_MAX_BLOB_SIZE (256 * 1024 * 1024)

#ifndef NCCL_DEVICE_H_
typedef enum { CONN_NORMAL=0, CONN_ICN, CONN_PCIE, CONN_SHM, CONN_GDRDMA,
               CONN_RDMA, CONN_SOCKET, CONN_NET_EXT, CONN_NUM } ncclConn_t;
#endif

// These indexes own only small descriptors. Channel/connection state stays in
// the caller's record array.
struct analyzerChannelView {
  const rasDiagChannel* state;
  std::vector<const rasDiagPeer*> peers;
};
struct analyzerRankView {
  int commRank;
  int nChannels;
  std::vector<analyzerChannelView> channels;
};
struct analyzerCommView {
  rasCommId id;
  int nRanks;
  size_t nRecords;
  std::vector<analyzerRankView> ranks;
  // Excluded snapshots do not prevent analysis of the valid ranks above.
  bool malformed = false;
  bool unavailable = false;
};
struct analyzerRankEntry {
  int rank;
  const analyzerRankView* rh;
};

static int analyzerCommIdCompare(const rasCommId& a, const rasCommId& b) {
  if (a.commHash != b.commHash) return a.commHash < b.commHash ? -1 : 1;
  if (a.hostHash != b.hostHash) return a.hostHash < b.hostHash ? -1 : 1;
  return a.pidHash == b.pidHash ? 0 : (a.pidHash < b.pidHash ? -1 : 1);
}

static bool analyzerRecordLess(const rasDiagnosticsRuntimeRecord* a,
                               const rasDiagnosticsRuntimeRecord* b) {
  int cmp = analyzerCommIdCompare(a->rank.commId, b->rank.commId);
  if (cmp) return cmp < 0;
  if (a->rank.commRank != b->rank.commRank) return a->rank.commRank < b->rank.commRank;
  if (a->channelId != b->channelId) return a->channelId < b->channelId;
  if (a->type != b->type) return a->type < b->type;
  if (a->type != RAS_DIAG_RUNTIME_CONNECTION) return false;
  const auto& ap = a->value.connection;
  const auto& bp = b->value.connection;
  if (ap.peerRank != bp.peerRank) return ap.peerRank < bp.peerRank;
  if (ap.connector != bp.connector) return ap.connector < bp.connector;
  return ap.direction < bp.direction;
}

static bool analyzerBuildRankView(const rasDiagnosticsRuntimeRecord* const* records,
                                  size_t count, analyzerRankView* out, bool* unavailable) {
  if (count == 0 || records[0]->type != RAS_DIAG_RUNTIME_RANK || records[0]->channelId != -1) return false;
  const auto& header = records[0]->value.rankData;
  if (header.version != RAS_DIAG_RUNTIME_WIRE_VERSION ||
      (header.availability != RAS_DIAG_RUNTIME_AVAILABLE && header.availability != RAS_DIAG_RUNTIME_ABORTED &&
       header.availability != RAS_DIAG_RUNTIME_UNAVAILABLE) ||
      header.nChannels < 0 || header.nChannels > MAXCHANNELS) return false;
  out->commRank = records[0]->rank.commRank;
  out->nChannels = header.nChannels;
  if (header.availability != RAS_DIAG_RUNTIME_AVAILABLE) {
    *unavailable = true;
    return count == 1 && header.nChannels == 0;
  }
  size_t i = 1;
  int previousChannel = -1;
  while (i < count) {
    const auto* record = records[i++];
    if (record->type != RAS_DIAG_RUNTIME_CHANNEL || record->channelId <= previousChannel ||
        record->channelId >= MAXCHANNELS) return false;
    const auto& channel = record->value.channel;
    if (channel.channelId != record->channelId || channel.nPeerEntries < 0 ||
        (size_t)channel.nPeerEntries > count - i || channel.devHangReportedKernelIdxValid > 1 ||
        channel.devCompletedKernelIdxValid > 1) return false;
    previousChannel = record->channelId;
    analyzerChannelView view = {&channel, {}};
    for (int p = 0; p < channel.nPeerEntries; p++, i++) {
      const auto* connection = records[i];
      const auto& peer = connection->value.connection;
      if (connection->type != RAS_DIAG_RUNTIME_CONNECTION || connection->channelId != channel.channelId ||
          peer.peerRank < 0 || peer.peerRank >= record->rank.commNRanks || peer.connector >= RAS_DIAG_RUNTIME_MAX_CONNS ||
          peer.direction > 1 || peer.connected != 1) return false;
      if (p > 0 && !analyzerRecordLess(records[i - 1], connection)) return false;
      view.peers.push_back(&peer);
    }
    out->channels.push_back(std::move(view));
  }
  return out->channels.size() == (size_t)header.nChannels;
}

static bool analyzerBuildRuntimeViews(const rasDiagnosticsRuntimeRecord* data, int nRecords,
                                      std::vector<analyzerCommView>* comms) {
  if (nRecords < 0 || (nRecords > 0 && data == nullptr) ||
      (size_t)nRecords > (size_t)INT_MAX / sizeof(*data)) return false;
  std::vector<const rasDiagnosticsRuntimeRecord*> records;
  records.reserve(nRecords);
  for (int i = 0; i < nRecords; i++) records.push_back(data + i);
  std::sort(records.begin(), records.end(), analyzerRecordLess);
  for (size_t begin = 0; begin < records.size();) {
    size_t end = begin + 1;
    while (end < records.size() &&
           analyzerCommIdCompare(records[begin]->rank.commId, records[end]->rank.commId) == 0) end++;
    analyzerCommView comm = {};
    comm.id = records[begin]->rank.commId;
    comm.nRecords = end - begin;
    for (size_t rankBegin = begin; rankBegin < end;) {
      size_t rankEnd = rankBegin + 1;
      while (rankEnd < end && records[rankEnd]->rank.commRank == records[rankBegin]->rank.commRank) rankEnd++;
      int nRanks = records[rankBegin]->rank.commNRanks;
      bool valid = nRanks > 0;
      for (size_t i = rankBegin; i < rankEnd; i++) {
        const auto& record = *records[i];
        if (record.rank.commNRanks != nRanks || record.rank.commRank < 0 ||
            record.rank.commRank >= nRanks || record.type > RAS_DIAG_RUNTIME_CONNECTION) valid = false;
      }
      analyzerRankView rank = {};
      bool unavailable = false;
      if (!valid || !analyzerBuildRankView(records.data() + rankBegin, rankEnd - rankBegin,
                                          &rank, &unavailable) ||
          (comm.nRanks != 0 && comm.nRanks != nRanks)) {
        comm.malformed = true;
      } else {
        comm.nRanks = nRanks;
        if (unavailable) comm.unavailable = true;
        // Rules may dereference every advertised channel/peer; never publish a partial rank view.
        else comm.ranks.push_back(std::move(rank));
      }
      rankBegin = rankEnd;
    }
    comms->push_back(std::move(comm));
    begin = end;
  }
  return true;
}

static bool analyzerCommUsable(FILE* fp, const analyzerCommView& comm) {
  if (comm.ranks.empty()) {
    fprintf(fp, "[INFO] Runtime state: no usable snapshots for comm 0x%lx; analysis skipped\n",
            (unsigned long)comm.id.commHash);
    return false;
  }
  if (comm.malformed || comm.unavailable || comm.ranks.size() != (size_t)comm.nRanks)
    fprintf(fp, "[INFO] Runtime state: partial snapshots for comm 0x%lx (%zu/%d usable ranks); "
                "PASS applies only to checked data\n",
            (unsigned long)comm.id.commHash, comm.ranks.size(), comm.nRanks);
  return true;
}

static std::vector<analyzerRankEntry> analyzerRankIndex(const analyzerCommView& comm) {
  std::vector<analyzerRankEntry> result;
  result.reserve(comm.ranks.size());
  for (const auto& rank : comm.ranks) result.push_back({rank.commRank, &rank});
  return result;
}

static const analyzerChannelView* findChannelInRank(const analyzerRankView* rank, int channelId) {
  for (const auto& channel : rank->channels) if (channel.state->channelId == channelId) return &channel;
  return nullptr;
}

static const rasDiagPeer* findPeerInChannel(const analyzerChannelView* channel, int peerRank,
                                           uint8_t direction, uint8_t connector) {
  for (const auto* peer : channel->peers)
    if (peer->peerRank == peerRank && peer->direction == direction && peer->connector == connector) return peer;
  return nullptr;
}

struct analyzerSummaryMessage {
  uint64_t commHash;
  std::string brief;
  bool isWarn;
};

typedef enum {
  ANALYZER_CAT_CONN_TYPE = 0,
  ANALYZER_CAT_CONSISTENCY,
  ANALYZER_CAT_RING_GRAPH,
  ANALYZER_CAT_P2P_CONNECT,
  ANALYZER_CAT_HANG_REPORTED,
  ANALYZER_CAT_COUNT
} analyzerCategory_t;

typedef enum {
  analyzerSuccess = 0,
  analyzerInternalError,
  analyzerDataMissing,
} analyzerResult_t;

static const char* analyzerConnTypeToString(uint8_t t) {
  if (t == RAS_DIAG_CONN_NONE) return "NONE";
  if (t >= CONN_NUM) return "UNKNOWN";
  return rasConnTypeStr[t];
}

int analyzerConnTypeFromString(const char* s, int peer) {
  if (s == nullptr || peer < 0) return RAS_DIAG_CONN_NONE;
  if (strcmp(s, "NONE") == 0) return CONN_NUM;
  for (int i = 0; i < CONN_NUM; i++) {
    if (strcmp(s, rasConnTypeStr[i]) == 0) return i;
  }
  fprintf(stderr, "WARN: unrecognized conn type token \"%s\" in comm dump, treating as NONE\n", s);
  return RAS_DIAG_CONN_NONE;
}

static bool analyzerConnTypeIsValid(uint8_t t) {
  if (t == RAS_DIAG_CONN_NONE) return true;
  return (t == CONN_ICN || t == CONN_GDRDMA || t == CONN_RDMA || t == CONN_NET_EXT);
}

static const char* const kCategoryNames[] = {
  "conntype", "consistency", "ringgraph", "p2pconnect", "hangreported"
};
static const char* analyzerCategoryName(int c) {
  if (c >= 0 && c < ANALYZER_CAT_COUNT) return kCategoryNames[c];
  return "unknown";
}

static int rankIndexCompare(const void* a, const void* b) {
  return ((const analyzerRankEntry*)a)->rank - ((const analyzerRankEntry*)b)->rank;
}

static bool formatCompactRankList(const std::vector<int>& sortedRanks, char* out, size_t outLen) {
  if (sortedRanks.empty()) { if (outLen) out[0] = '\0'; return false; }
  const int MAX_SEGMENTS = 8;
  std::vector<std::pair<int,int>> segs;
  int s = sortedRanks[0], e = sortedRanks[0];
  for (size_t i = 1; i < sortedRanks.size(); i++) {
    if (sortedRanks[i] == e + 1) { e = sortedRanks[i]; continue; }
    segs.push_back({s, e}); s = e = sortedRanks[i];
  }
  segs.push_back({s, e});

  size_t pos = 0;
  bool truncated = false;
  auto emit = [&](int sv, int ev, bool first) {
    if (pos >= outLen - 1) return;
    int n;
    if (!first) { n = snprintf(out + pos, outLen - pos, ","); pos += (n > 0 ? (size_t)n : 0); }
    if (sv == ev) n = snprintf(out + pos, outLen - pos, "%d", sv);
    else          n = snprintf(out + pos, outLen - pos, "%d-%d", sv, ev);
    pos += (n > 0 ? (size_t)n : 0);
  };

  if ((int)segs.size() <= MAX_SEGMENTS) {
    for (size_t i = 0; i < segs.size(); i++) emit(segs[i].first, segs[i].second, i == 0);
  } else {
    truncated = true;
    for (int i = 0; i < MAX_SEGMENTS - 1; i++) emit(segs[i].first, segs[i].second, i == 0);
    int n = snprintf(out + pos, outLen - pos, ",...");
    pos += (n > 0 ? (size_t)n : 0);
    emit(segs.back().first, segs.back().second, false);
  }
  return truncated;
}

///////////////////////////////////////////////////////////////////////////////
// Common edge check util                                                    //
///////////////////////////////////////////////////////////////////////////////

enum { EDGE_OK = 0, EDGE_ERROR = 1, EDGE_WARN = 2, EDGE_INFO = 3 };

static int classifyEdge(uint64_t producerStep, uint64_t consumerStep,
                        uint64_t producerHead, uint64_t consumerTail) {
  if (producerStep != consumerStep) return EDGE_ERROR;
  if (producerHead != 0 && consumerTail != 0 && producerHead != consumerTail) return EDGE_WARN;
  return EDGE_OK;
}

template<typename EnsureHeaderFn>
static int checkEdgeStepHeadTail(FILE* fp, EnsureHeaderFn ensureHeader,
                                  int srcRank, int dstRank,
                                  uint64_t producerStep, uint64_t consumerStep,
                                  uint64_t producerHead, uint64_t consumerTail,
                                  const char* prodLabel, const char* consLabel) {
  if (producerStep != consumerStep) {
    ensureHeader();
    long delta = (long)((int64_t)consumerStep - (int64_t)producerStep);
    fprintf(fp, "    [ERROR] rank%d -> rank%d : %s=%ld %s=%ld (delta=%+ld)\n",
            srcRank, dstRank, prodLabel, (long)producerStep,
            consLabel, (long)consumerStep, delta);
    return EDGE_ERROR;
  }
  // Use signed comparison: head/tail can be negative (initialized to -NCCL_STEPS
  // in shared mode for flow control credit accounting).
  bool headActive = ((int64_t)producerHead > (int64_t)consumerStep);
  bool tailActive = ((int64_t)consumerTail > (int64_t)producerStep);
  if (headActive && tailActive && producerHead != consumerTail) {
    ensureHeader();
    fprintf(fp, "    [WARN]  rank%d -> rank%d : head/tail mismatch (head=%ld tail=%ld), "
                "kernel in-flight, possible anomaly\n",
            srcRank, dstRank, (long)producerHead, (long)consumerTail);
    return EDGE_WARN;
  }
  if (headActive) {
    ensureHeader();
    fprintf(fp, "    [INFO]  rank%d -> rank%d : producer.head=%ld > consumer.step=%ld, "
                "kernel in-flight (consumer receiving)\n",
            srcRank, dstRank, (long)producerHead, (long)consumerStep);
    return EDGE_INFO;
  }
  if (tailActive) {
    ensureHeader();
    fprintf(fp, "    [INFO]  rank%d -> rank%d : consumer.tail=%ld > producer.step=%ld, "
                "kernel in-flight (producer sending)\n",
            srcRank, dstRank, (long)consumerTail, (long)producerStep);
    return EDGE_INFO;
  }
  return EDGE_OK;
}

///////////////////////////////////////////////////////////////////////////////
// Check functions                                                           //
///////////////////////////////////////////////////////////////////////////////

static analyzerResult_t analyzerCheckConnType(FILE* fp, const analyzerRankEntry* ranks, int nRanks) {
  if (fp == nullptr || ranks == nullptr || nRanks <= 0) return analyzerDataMissing;
  fprintf(fp, "\n[Check: ConnType]\n");
  int nBad = 0;
  for (int r = 0; r < nRanks; r++) {
    const analyzerRankView* rh = ranks[r].rh;
    const analyzerChannelView* dc = rh->channels.data();
    bool printedRank = false;
    for (int c = 0; c < rh->nChannels; c++) {
      struct { const char* name; uint8_t t; bool isFc; } legs[] = {
        {"ring_send", dc->state->ringSendConn, false},
        {"ring_recv", dc->state->ringRecvConn, false},
        {"tree_up",   dc->state->treeUpConn,   false},
        {"tree_down", dc->state->treeDownConn, false},
        {"fc",        dc->state->fcConnType,   true},
      };
      bool printedChannel = false;
      for (auto& leg : legs) {
        bool valid = (leg.t == RAS_DIAG_CONN_NONE) ||
                     (leg.isFc ? (leg.t == CONN_ICN) : analyzerConnTypeIsValid(leg.t));
        if (!valid) {
          if (!printedRank) { fprintf(fp, "  rank %d:\n", rh->commRank); printedRank = true; }
          if (!printedChannel) { fprintf(fp, "    channel %d:\n", dc->state->channelId); printedChannel = true; }
          fprintf(fp, "      [ERROR] %s = %s (invalid)\n", leg.name, analyzerConnTypeToString(leg.t));
          nBad++;
        }
      }
      dc++;
    }
  }
  if (nBad == 0) fprintf(fp, "  PASS (no invalid connection types)\n");
  else            fprintf(fp, "  FAIL (%d invalid entries)\n", nBad);
  return analyzerSuccess;
}

static bool analyzerConsistencyStateValid(const analyzerChannelView* dc) {
  return dc != nullptr && dc->state->devCompletedKernelIdxValid && dc->state->devHangReportedKernelIdxValid;
}

static analyzerResult_t analyzerCheckConsistency(FILE* fp, const analyzerRankEntry* ranks, int nRanks,
                                               int expectedRanks) {
  if (fp == nullptr || ranks == nullptr || nRanks <= 0) return analyzerDataMissing;
  fprintf(fp, "\n[Check: Consistency (per-channel device kernel index across ranks; kernel idx is 1-based, 0 = none yet)]\n");

  // Check nChannels agreement across ranks
  int minN = ranks[0].rh->nChannels, maxN = minN;
  for (int r = 1; r < nRanks; r++) {
    int n = ranks[r].rh->nChannels;
    if (n < minN) minN = n;
    if (n > maxN) maxN = n;
  }
  if (minN != maxN) {
    std::unordered_map<int, std::vector<int>> byNChannel;
    for (int r = 0; r < nRanks; r++) byNChannel[ranks[r].rh->nChannels].push_back(ranks[r].rank);
    fprintf(fp, "  [WARN] ranks have inconsistent nChannels:\n");
    char rankBuf[320];
    for (auto& kv : byNChannel) {
      std::sort(kv.second.begin(), kv.second.end());
      formatCompactRankList(kv.second, rankBuf, sizeof(rankBuf));
      fprintf(fp, "    nChannels=%d: ranks [%s]\n", kv.first, rankBuf);
    }
  }

  // Build per-channel rank count for iteration
  std::unordered_map<int, int> channelCount;
  for (int r = 0; r < nRanks; r++) {
    const analyzerChannelView* dc = ranks[r].rh->channels.data();
    for (int c = 0; c < ranks[r].rh->nChannels; c++) { channelCount[dc->state->channelId]++; dc++; }
  }
  // Group ranks by (completedIdx, hangIdx) state per channel
  struct GroupInfo {
    std::vector<int> ranksInGroup;
    uint64_t minWft, maxWft;
  };
  auto stateKey = [](uint64_t cmpl, uint64_t hang) {
    return std::pair<uint64_t, uint64_t>(cmpl, hang);
  };
  struct PairHash {
    size_t operator()(const std::pair<uint64_t,uint64_t>& p) const noexcept {
      return std::hash<uint64_t>()(p.first) * 31ULL ^ std::hash<uint64_t>()(p.second);
    }
  };

  int nBad = 0;
  size_t nUnavailable = 0;
  for (auto& kv : channelCount) {
    int channelId = kv.first;
    std::unordered_map<std::pair<uint64_t,uint64_t>, GroupInfo, PairHash> groups;
    int nSkipped = std::max(0, expectedRanks - nRanks);
    for (int r = 0; r < nRanks; r++) {
      const analyzerChannelView* dc = findChannelInRank(ranks[r].rh, channelId);
      if (!analyzerConsistencyStateValid(dc)) { nSkipped++; continue; }
      auto key = stateKey(dc->state->devCompletedKernelIdx, dc->state->devHangReportedKernelIdx);
      auto it = groups.find(key);
      if (it == groups.end()) { // new state group
        GroupInfo gi;
        gi.minWft = gi.maxWft = dc->state->workFifoTail;
        gi.ranksInGroup.push_back(ranks[r].rank);
        groups.emplace(key, std::move(gi));
      } else { // merge into existing group
        GroupInfo& gi = it->second;
        gi.ranksInGroup.push_back(ranks[r].rank);
        if (dc->state->workFifoTail < gi.minWft) gi.minWft = dc->state->workFifoTail;
        if (dc->state->workFifoTail > gi.maxWft) gi.maxWft = dc->state->workFifoTail;
      }
    }

    // Report divergent groups sorted by completedIdx↑ then hangIdx↓
    if (groups.size() > 1) {
      typedef const std::pair<const std::pair<uint64_t,uint64_t>, GroupInfo>* GroupPtr;
      std::vector<GroupPtr> ordered;
      ordered.reserve(groups.size());
      for (auto& g : groups) ordered.push_back(&g);
      std::sort(ordered.begin(), ordered.end(), [](GroupPtr a, GroupPtr b) {
        if (a->first.first != b->first.first) return a->first.first < b->first.first;
        return a->first.second > b->first.second;
      });

      fprintf(fp, "  [ERROR] channel %d divergent (%zu state(s) across %d rank(s)):\n",
              channelId, groups.size(), nRanks);
      struct AlignedLine { std::string left, ranks; size_t nRanks; bool truncated; };
      std::vector<AlignedLine> alines;
      int maxLeftLen = 0;
      char rankBuf[320], wftBuf[64], hangIdxBuf[32], leftBuf[256];
      for (auto* g : ordered) {
        auto& info = const_cast<GroupInfo&>(g->second);
        std::sort(info.ranksInGroup.begin(), info.ranksInGroup.end());
        bool trunc = formatCompactRankList(info.ranksInGroup, rankBuf, sizeof(rankBuf));
        if (info.minWft == info.maxWft) snprintf(wftBuf, sizeof(wftBuf), "%lu", (unsigned long)info.minWft);
        else                            snprintf(wftBuf, sizeof(wftBuf), "%lu..%lu",
                                                  (unsigned long)info.minWft, (unsigned long)info.maxWft);
        if (g->first.second == 0) snprintf(hangIdxBuf, sizeof(hangIdxBuf), "None");
        else                      snprintf(hangIdxBuf, sizeof(hangIdxBuf), "%lu", (unsigned long)g->first.second);
        snprintf(leftBuf, sizeof(leftBuf),
                 "      hostWorkFifoTail=%s devCompletedKernelIdx=%lu devHangReportedKernelIdx=%s",
                 wftBuf, (unsigned long)g->first.first, hangIdxBuf);
        AlignedLine al;
        al.left = leftBuf; al.ranks = rankBuf; al.nRanks = info.ranksInGroup.size(); al.truncated = trunc;
        int len = (int)al.left.size();
        if (len > maxLeftLen) maxLeftLen = len;
        alines.push_back(std::move(al));
      }
      for (auto& al : alines) {
        fprintf(fp, "%-*s - %zu rank(s) [%s%s]\n",
                maxLeftLen, al.left.c_str(), al.nRanks, al.ranks.c_str(), al.truncated ? " truncated" : "");
      }
      nBad++;
    }
    nUnavailable += nSkipped;
    if (nSkipped > 0) {
      fprintf(fp, "  [WARN] channel %d: %d rank(s) with missing channel data or an unreadable device index (unknown)\n",
              channelId, nSkipped);
    }
  }
  if (nBad > 0) fprintf(fp, "  FAIL (%d channel(s) divergent)\n", nBad);
  else if (channelCount.empty()) fprintf(fp, "  INCONCLUSIVE (no channel state available)\n");
  else if (nUnavailable > 0)
    fprintf(fp, "  INCONCLUSIVE (%zu rank-channel state(s) unavailable; agreement cannot be established)\n", nUnavailable);
  else fprintf(fp, "  PASS (all ranks agree on completed and hang-reported kernel idx per channel)\n");
  return analyzerSuccess;
}

static analyzerResult_t analyzerCheckRingGraph(FILE* fp, const analyzerRankEntry* ranks, int nRanks) {
  if (fp == nullptr || ranks == nullptr || nRanks <= 0) return analyzerDataMissing;
  fprintf(fp, "\n[Check: RingGraph (per ring edge: prev.send.devStep vs my.recv.devStep; "
              "secondary inflight check via head/tail)]\n");
  std::unordered_map<int, int> channelCount;
  for (int r = 0; r < nRanks; r++) {
    const analyzerChannelView* dc = ranks[r].rh->channels.data();
    for (int c = 0; c < ranks[r].rh->nChannels; c++) { channelCount[dc->state->channelId]++; dc++; }
  }
  std::vector<int> sortedChannels;
  sortedChannels.reserve(channelCount.size());
  for (auto& kv : channelCount) sortedChannels.push_back(kv.first);
  std::sort(sortedChannels.begin(), sortedChannels.end());

  int nBad = 0;
  int nInflightWarn = 0;
  for (int channelId : sortedChannels) {
    int nSkippedNoPrevRank = 0, nSkippedNoPrevChannel = 0, nSkippedNoPeerEntry = 0;
    bool printedHeader = false;
    auto ensureHeader = [&]() {
      if (!printedHeader) { fprintf(fp, "  channel %d:\n", channelId); printedHeader = true; }
    };
    for (int r = 0; r < nRanks; r++) {
      const analyzerChannelView* myChannel = findChannelInRank(ranks[r].rh, channelId);
      if (myChannel == nullptr) continue;
      int myRank = ranks[r].rank;
      int prev = myChannel->state->ringPrev;
      if (prev < 0) continue;
      analyzerRankEntry key{prev, nullptr};
      analyzerRankEntry* found = static_cast<analyzerRankEntry*>(bsearch(&key, ranks, nRanks, sizeof(*ranks), rankIndexCompare));
      if (found == nullptr)       { nSkippedNoPrevRank++; continue; }
      const analyzerChannelView* prevChannel = findChannelInRank(found->rh, channelId);
      if (prevChannel == nullptr)    { nSkippedNoPrevChannel++; continue; }
      const rasDiagPeer* prevSend = findPeerInChannel(prevChannel, myRank, /*send*/0, /*coll*/0);
      const rasDiagPeer* myRecv  = findPeerInChannel(myChannel,   prev,   /*recv*/1, /*coll*/0);
      if (prevSend == nullptr || myRecv == nullptr) { nSkippedNoPeerEntry++; continue; }
      int edgeResult = checkEdgeStepHeadTail(fp, ensureHeader, prev, myRank,
                          prevSend->devStep, myRecv->devStep,
                          prevSend->head, myRecv->tail,
                          "prev.send.devStep", "my.recv.devStep");
      if (edgeResult == EDGE_ERROR) { nBad++; continue; }
      if (edgeResult == EDGE_WARN) nInflightWarn++;
    }
    if (nSkippedNoPrevRank || nSkippedNoPrevChannel || nSkippedNoPeerEntry) {
      ensureHeader();
      fprintf(fp, "    [WARN]  skipped");
      if (nSkippedNoPrevRank)  fprintf(fp, " %d edge(s) [prev rank has no diag block]", nSkippedNoPrevRank);
      if (nSkippedNoPrevChannel)  fprintf(fp, " %d edge(s) [prev rank missing this channel]", nSkippedNoPrevChannel);
      if (nSkippedNoPeerEntry) fprintf(fp, " %d edge(s) [coll send/recv peer entry not found]", nSkippedNoPeerEntry);
      fprintf(fp, "\n");
    }
  }
  if (nBad > 0) {
    fprintf(fp, "  FAIL (%d ring edge(s) with prev.send.devStep != my.recv.devStep%s)\n",
            nBad, nInflightWarn ? "; additional inflight imbalances also detected" : "");
  } else if (nInflightWarn > 0) {
    fprintf(fp, "  WARNING: %d edge(s) with mid-kernel inflight imbalance "
                "(head != tail); may be a real mid-kernel hang or LL protocol leaving stale tail\n",
            nInflightWarn);
  } else {
    fprintf(fp, "  PASS (prev.send.devStep == my.recv.devStep on every checked ring edge)\n");
  }
  return analyzerSuccess;
}

static analyzerResult_t analyzerCheckP2pConnect(FILE* fp, const analyzerRankEntry* ranks, int nRanks) {
  if (fp == nullptr || ranks == nullptr || nRanks <= 0) return analyzerDataMissing;
  fprintf(fp, "\n[Check: P2pConnect (per p2p send/recv pair: devStep, head/tail)]\n");
  std::unordered_map<int, int> channelCount;
  for (int r = 0; r < nRanks; r++) {
    const analyzerChannelView* dc = ranks[r].rh->channels.data();
    for (int c = 0; c < ranks[r].rh->nChannels; c++) { channelCount[dc->state->channelId]++; dc++; }
  }
  std::vector<int> sortedChannels;
  sortedChannels.reserve(channelCount.size());
  for (auto& kv : channelCount) sortedChannels.push_back(kv.first);
  std::sort(sortedChannels.begin(), sortedChannels.end());

  int nBad = 0;
  int nInflightWarn = 0;
  int nPairsChecked = 0;
  for (int channelId : sortedChannels) {
    int nSkippedNoPeerRank = 0, nSkippedNoPeerChannel = 0, nSkippedNoRecvEntry = 0;
    bool printedHeader = false;
    auto ensureHeader = [&]() {
      if (!printedHeader) { fprintf(fp, "  channel %d:\n", channelId); printedHeader = true; }
    };
    for (int r = 0; r < nRanks; r++) {
      const analyzerChannelView* myChannel = findChannelInRank(ranks[r].rh, channelId);
      if (myChannel == nullptr) continue;
      int myRank = ranks[r].rank;
      const rasDiagPeer* const* dp = myChannel->peers.data();
      for (int i = 0; i < myChannel->state->nPeerEntries; i++) {
        if (!dp[i]->connected || dp[i]->direction != 0) continue;
        if (dp[i]->connector != 1) continue;
        int peer = dp[i]->peerRank;
        if (peer <= myRank) continue;
        analyzerRankEntry key{peer, nullptr};
        analyzerRankEntry* found = static_cast<analyzerRankEntry*>(bsearch(&key, ranks, nRanks, sizeof(*ranks), rankIndexCompare));
        if (!found)              { nSkippedNoPeerRank++; continue; }
        const analyzerChannelView* peerChannel = findChannelInRank(found->rh, channelId);
        if (!peerChannel)           { nSkippedNoPeerChannel++; continue; }
        const rasDiagPeer* peerRecv = findPeerInChannel(peerChannel, myRank, /*recv*/1, dp[i]->connector);
        if (!peerRecv)           { nSkippedNoRecvEntry++; continue; }
        nPairsChecked++;
        int edgeResult = checkEdgeStepHeadTail(fp, ensureHeader, myRank, peer,
                            dp[i]->devStep, peerRecv->devStep,
                            dp[i]->head, peerRecv->tail,
                            "send.devStep", "recv.devStep");
        if (edgeResult == EDGE_ERROR) { nBad++; continue; }
        if (edgeResult == EDGE_WARN) nInflightWarn++;
      }
    }
    if (nSkippedNoPeerRank || nSkippedNoPeerChannel || nSkippedNoRecvEntry) {
      ensureHeader();
      fprintf(fp, "    [WARN]  skipped");
      if (nSkippedNoPeerRank)  fprintf(fp, " %d pair(s) [peer rank has no diag block]", nSkippedNoPeerRank);
      if (nSkippedNoPeerChannel)  fprintf(fp, " %d pair(s) [peer rank missing this channel]", nSkippedNoPeerChannel);
      if (nSkippedNoRecvEntry) fprintf(fp, " %d pair(s) [peer's matching recv entry not found]", nSkippedNoRecvEntry);
      fprintf(fp, "\n");
    }
  }
  if (nPairsChecked == 0) {
    fprintf(fp, "  PASS (no comparable p2p (connector=1) pairs in received snapshots; "
                "coll connector pairs are covered by RingGraph)\n");
  } else if (nBad > 0) {
    fprintf(fp, "  FAIL (%d p2p pair(s) with devStep mismatch%s; %d pair(s) checked)\n",
            nBad, nInflightWarn ? "; additional inflight imbalances also detected" : "", nPairsChecked);
  } else if (nInflightWarn > 0) {
    fprintf(fp, "  WARNING: %d p2p pair(s) with inflight head/tail mismatch; %d pair(s) checked\n",
            nInflightWarn, nPairsChecked);
  } else {
    fprintf(fp, "  PASS (devStep matched on every checked p2p pair; %d pair(s) checked)\n", nPairsChecked);
  }
  return analyzerSuccess;
}

static analyzerResult_t analyzerCheckHangReported(FILE* fp, const analyzerRankEntry* ranks, int nRanks) {
  if (fp == nullptr || ranks == nullptr || nRanks <= 0) return analyzerDataMissing;
  fprintf(fp, "\n[Check: HangReportedKernelIdx (kernel idx is 1-based; idx=N means the N-th kernel ever submitted)]\n");
  std::unordered_map<uint64_t, std::unordered_set<int>> byIdx;
  std::unordered_set<int> uniqReportingRanks;
  int nUnknownChannels = 0;
  for (int r = 0; r < nRanks; r++) {
    const analyzerRankView* rh = ranks[r].rh;
    const analyzerChannelView* dc = rh->channels.data();
    for (int c = 0; c < rh->nChannels; c++) {
      if (!dc->state->devHangReportedKernelIdxValid) {
        nUnknownChannels++;
      } else if (dc->state->devHangReportedKernelIdx != 0) {
        byIdx[dc->state->devHangReportedKernelIdx].insert(rh->commRank);
        uniqReportingRanks.insert(rh->commRank);
      }
      dc++;
    }
  }
  int nReporting = (int)uniqReportingRanks.size();
  fprintf(fp, "  ranks self-reporting a hung kernel: %d / %d\n", nReporting, nRanks);
  std::vector<uint64_t> idxs;
  idxs.reserve(byIdx.size());
  for (auto& kv : byIdx) idxs.push_back(kv.first);
  std::sort(idxs.begin(), idxs.end());
  char rankBuf[320];
  for (uint64_t idx : idxs) {
    std::vector<int> sortedRanks(byIdx[idx].begin(), byIdx[idx].end());
    std::sort(sortedRanks.begin(), sortedRanks.end());
    bool truncated = formatCompactRankList(sortedRanks, rankBuf, sizeof(rankBuf));
    fprintf(fp, "  devHangReportedKernelIdx %lu: %zu rank(s) [%s%s]\n",
            (unsigned long)idx, sortedRanks.size(), rankBuf, truncated ? " truncated" : "");
  }
  if (nUnknownChannels > 0)
    fprintf(fp, "  [WARN] %d channel(s) had an unreadable hang-reported index (hang status unknown there)\n",
            nUnknownChannels);
  if (nReporting > 0)
    fprintf(fp, "  FAIL (%d rank(s) self-reported hang)\n", nReporting);
  else if (nUnknownChannels > 0)
    fprintf(fp, "  INCONCLUSIVE (no readable channel self-reported hang, but %d channel(s) unreadable)\n",
            nUnknownChannels);
  else
    fprintf(fp, "  PASS (no received rank self-reported hang)\n");
  return analyzerSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Renderers                                                                 //
///////////////////////////////////////////////////////////////////////////////

int analyzerFormatRuntimeReport(FILE* fp, const rasDiagnosticsRuntimeRecord* records, int nRecords,
                                int nLegTimeouts, int singleCategory, const char* title) {
  if (fp == nullptr) return -1;
  std::vector<analyzerCommView> comms;
  try {
    if (!analyzerBuildRuntimeViews(records, nRecords, &comms)) return -1;
  } catch (...) {
    return -1;
  }
  time_t now = time(nullptr);
  char ts[64];
  strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", localtime(&now));
  fprintf(fp, "=== %s ===\n", title != nullptr ? title : "PCCL Online Hang Analyzer Report");
  fprintf(fp, "wall time      : %s\n", ts);
  fprintf(fp, "leg timeouts   : %d%s\n", nLegTimeouts, nLegTimeouts > 0 ? " (incomplete data!)" : "");
  fprintf(fp, "wire version   : 0x%x\n", RAS_DIAG_RUNTIME_WIRE_VERSION);
  if (singleCategory >= 0 && singleCategory < ANALYZER_CAT_COUNT)
    fprintf(fp, "category       : %s\n", analyzerCategoryName(singleCategory));
  int analyzed = 0;
  for (const auto& comm : comms) {
    if (!analyzerCommUsable(fp, comm)) continue;
    fprintf(fp, "\n---- comm 0x%lx (%zu/%d ranks, %zu runtime records) ----\n",
            (unsigned long)comm.id.commHash, comm.ranks.size(), comm.nRanks, comm.nRecords);
    try {
      auto ranksVec = analyzerRankIndex(comm);
      const auto* ranks = ranksVec.data();
      int count = (int)ranksVec.size();
      if (singleCategory == -1 || singleCategory == ANALYZER_CAT_CONN_TYPE) analyzerCheckConnType(fp, ranks, count);
      if (singleCategory == -1 || singleCategory == ANALYZER_CAT_CONSISTENCY)
        analyzerCheckConsistency(fp, ranks, count, comm.nRanks);
      if (singleCategory == -1 || singleCategory == ANALYZER_CAT_RING_GRAPH) analyzerCheckRingGraph(fp, ranks, count);
      if (singleCategory == -1 || singleCategory == ANALYZER_CAT_P2P_CONNECT) analyzerCheckP2pConnect(fp, ranks, count);
      if (singleCategory == -1 || singleCategory == ANALYZER_CAT_HANG_REPORTED)
        analyzerCheckHangReported(fp, ranks, count);
    } catch (...) {
      return -1;
    }
    analyzed++;
  }
  if (analyzed == 0) fprintf(fp, "\n(no usable runtime snapshots)\n");
  fprintf(fp, "\n=== End of Report ===\n");
  return ferror(fp) ? -1 : 0;
}

///////////////////////////////////////////////////////////////////////////////
// Summary                                                                   //
///////////////////////////////////////////////////////////////////////////////

static void buildChannelCount(const analyzerRankEntry* ranks, int nRanks,
                           std::unordered_map<int, int>& channelCount) {
  for (int r = 0; r < nRanks; r++) {
    const analyzerChannelView* dc = ranks[r].rh->channels.data();
    for (int c = 0; c < ranks[r].rh->nChannels; c++) { channelCount[dc->state->channelId]++; dc++; }
  }
}

static void analyzerCollectConnType(uint64_t commHash, const analyzerRankEntry* ranks, int nRanks,
                            std::vector<analyzerSummaryMessage>& out) {
  int nInvalid = 0;
  std::unordered_map<int, int> badRanks, badChannels;
  for (int r = 0; r < nRanks; r++) {
    const analyzerRankView* rh = ranks[r].rh;
    const analyzerChannelView* dc = rh->channels.data();
    for (int c = 0; c < rh->nChannels; c++) {
      int chanBad = (!analyzerConnTypeIsValid(dc->state->ringSendConn))
                  + (!analyzerConnTypeIsValid(dc->state->ringRecvConn))
                  + (!analyzerConnTypeIsValid(dc->state->treeUpConn))
                  + (!analyzerConnTypeIsValid(dc->state->treeDownConn))
                  + (!analyzerConnTypeIsValid(dc->state->fcConnType));
      if (chanBad) { nInvalid += chanBad; badRanks[rh->commRank]++; badChannels[dc->state->channelId]++; }
      dc++;
    }
  }
  if (nInvalid == 0) return;
  char buf[256];
  snprintf(buf, sizeof(buf), "%d invalid conn type(s) across %zu rank(s) / %zu channel(s)",
           nInvalid, badRanks.size(), badChannels.size());
  out.push_back({commHash, buf, false});
}

static void analyzerCollectConsistency(uint64_t commHash, const analyzerRankEntry* ranks, int nRanks,
                               int expectedRanks, std::vector<analyzerSummaryMessage>& out) {
  int minN = ranks[0].rh->nChannels, maxN = minN;
  for (int r = 1; r < nRanks; r++) {
    int n = ranks[r].rh->nChannels;
    if (n < minN) minN = n;
    if (n > maxN) maxN = n;
  }
  if (minN != maxN) {
    char buf[256];
    snprintf(buf, sizeof(buf), "ranks have inconsistent nChannels (range %d..%d)", minN, maxN);
    out.push_back({commHash, buf, true});
  }

  std::unordered_map<int, int> channelCount;
  buildChannelCount(ranks, nRanks, channelCount);
  int nDivergent = 0;
  size_t nUnavailable = 0;
  for (auto& kv : channelCount) {
    int channelId = kv.first;
    nUnavailable += std::max(0, expectedRanks - nRanks);
    uint64_t expectedCmpl = 0, expectedHang = 0;
    bool initialized = false, diverged = false;
    for (int r = 0; r < nRanks; r++) {
      const analyzerChannelView* dc = findChannelInRank(ranks[r].rh, channelId);
      if (!analyzerConsistencyStateValid(dc)) { nUnavailable++; continue; }
      if (!initialized) {
        expectedCmpl = dc->state->devCompletedKernelIdx;
        expectedHang = dc->state->devHangReportedKernelIdx;
        initialized = true;
        continue;
      }
      if (dc->state->devCompletedKernelIdx != expectedCmpl || dc->state->devHangReportedKernelIdx != expectedHang) {
        diverged = true;
      }
    }
    if (diverged) nDivergent++;
  }
  if (channelCount.empty()) {
    out.push_back({commHash, "INCONCLUSIVE: no channel state available", true});
  } else if (nUnavailable > 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), "INCONCLUSIVE coverage: %zu rank-channel state(s) missing or unreadable", nUnavailable);
    out.push_back({commHash, buf, true});
  }
  if (nDivergent == 0) return;
  char buf[256];
  snprintf(buf, sizeof(buf), "%d channel(s) divergent in completed-or-hang-reported kernel idx across ranks", nDivergent);
  out.push_back({commHash, buf, false});
}

static void analyzerCollectRingGraph(uint64_t commHash, const analyzerRankEntry* ranks, int nRanks,
                             std::vector<analyzerSummaryMessage>& out) {
  std::unordered_map<int, int> channelCount;
  buildChannelCount(ranks, nRanks, channelCount);
  int nMismatch = 0, nInflightWarn = 0;
  std::unordered_map<int, int> errorChannels, warnChannels;
  for (auto& kv : channelCount) {
    int channelId = kv.first;
    for (int r = 0; r < nRanks; r++) {
      const analyzerChannelView* myChannel = findChannelInRank(ranks[r].rh, channelId);
      if (myChannel == nullptr) continue;
      int myRank = ranks[r].rank;
      int prev = myChannel->state->ringPrev;
      if (prev < 0) continue;
      analyzerRankEntry key{prev, nullptr};
      analyzerRankEntry* found = static_cast<analyzerRankEntry*>(bsearch(&key, ranks, nRanks, sizeof(*ranks), rankIndexCompare));
      if (found == nullptr) continue;
      const analyzerChannelView* prevChannel = findChannelInRank(found->rh, channelId);
      if (prevChannel == nullptr) continue;
      const rasDiagPeer* prevSend = findPeerInChannel(prevChannel, myRank, 0, 0);
      const rasDiagPeer* myRecv = findPeerInChannel(myChannel, prev, 1, 0);
      if (prevSend == nullptr || myRecv == nullptr) continue;
      int cls = classifyEdge(prevSend->devStep, myRecv->devStep, prevSend->head, myRecv->tail);
      if (cls == EDGE_ERROR) { nMismatch++; errorChannels[channelId]++; }
      else if (cls == EDGE_WARN) { nInflightWarn++; warnChannels[channelId]++; }
    }
  }
  if (nMismatch > 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%d ring edge(s) with step mismatch across %zu channel(s)",
             nMismatch, errorChannels.size());
    out.push_back({commHash, buf, false});
  }
  if (nInflightWarn > 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%d ring edge(s) with head/tail imbalance across %zu channel(s)",
             nInflightWarn, warnChannels.size());
    out.push_back({commHash, buf, true});
  }
}

static void analyzerCollectP2pConnect(uint64_t commHash, const analyzerRankEntry* ranks, int nRanks,
                              std::vector<analyzerSummaryMessage>& out) {
  std::unordered_map<int, int> channelCount;
  buildChannelCount(ranks, nRanks, channelCount);
  int nMismatch = 0, nInflightWarn = 0;
  std::unordered_map<int, int> errorChannels, warnChannels;
  for (auto& kv : channelCount) {
    int channelId = kv.first;
    for (int r = 0; r < nRanks; r++) {
      const analyzerChannelView* myChannel = findChannelInRank(ranks[r].rh, channelId);
      if (myChannel == nullptr) continue;
      int myRank = ranks[r].rank;
      const rasDiagPeer* const* dp = myChannel->peers.data();
      for (int i = 0; i < myChannel->state->nPeerEntries; i++) {
        if (!dp[i]->connected || dp[i]->direction != 0) continue;
        if (dp[i]->connector != 1) continue;
        int peer = dp[i]->peerRank;
        if (peer <= myRank) continue;
        analyzerRankEntry key{peer, nullptr};
        analyzerRankEntry* found = static_cast<analyzerRankEntry*>(bsearch(&key, ranks, nRanks, sizeof(*ranks), rankIndexCompare));
        if (!found) continue;
        const analyzerChannelView* peerChannel = findChannelInRank(found->rh, channelId);
        if (!peerChannel) continue;
        const rasDiagPeer* peerRecv = findPeerInChannel(peerChannel, myRank, 1, dp[i]->connector);
        if (!peerRecv) continue;
        int cls = classifyEdge(dp[i]->devStep, peerRecv->devStep, dp[i]->head, peerRecv->tail);
        if (cls == EDGE_ERROR) { nMismatch++; errorChannels[channelId]++; }
        else if (cls == EDGE_WARN) { nInflightWarn++; warnChannels[channelId]++; }
      }
    }
  }
  if (nMismatch > 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%d p2p pair(s) with step mismatch across %zu channel(s)",
             nMismatch, errorChannels.size());
    out.push_back({commHash, buf, false});
  }
  if (nInflightWarn > 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%d p2p pair(s) with head/tail imbalance across %zu channel(s)",
             nInflightWarn, warnChannels.size());
    out.push_back({commHash, buf, true});
  }
}

static void analyzerCollectHangReported(uint64_t commHash, const analyzerRankEntry* ranks, int nRanks,
                                         std::vector<analyzerSummaryMessage>& out) {
  std::unordered_map<uint64_t, std::unordered_map<int, int>> agg;
  int nUnknown = 0;
  for (int r = 0; r < nRanks; r++) {
    const analyzerRankView* rh = ranks[r].rh;
    const analyzerChannelView* dc = rh->channels.data();
    for (int c = 0; c < rh->nChannels; c++) {
      if (!dc->state->devHangReportedKernelIdxValid) {
        nUnknown++;
      } else if (dc->state->devHangReportedKernelIdx != 0) {
        agg[dc->state->devHangReportedKernelIdx][rh->commRank]++;
      }
      dc++;
    }
  }
  if (nUnknown > 0) {
    char ubuf[128];
    snprintf(ubuf, sizeof(ubuf), "%d channel(s) with unreadable hang-reported index (hang status unknown)",
             nUnknown);
    out.push_back({commHash, ubuf, true});
  }
  if (agg.empty()) return;
  std::unordered_map<int, int> allRanks;
  for (auto& kv : agg) for (auto& r : kv.second) allRanks[r.first]++;
  char buf[256];
  snprintf(buf, sizeof(buf), "%zu distinct hang-reported kernel idx value(s) from %zu/%d rank(s)",
           agg.size(), allRanks.size(), nRanks);
  out.push_back({commHash, buf, false});
}

int analyzerFormatRuntimeSummary(FILE* fp, const rasDiagnosticsRuntimeRecord* records, int nRecords) {
  if (fp == nullptr) return -1;
  std::vector<analyzerCommView> comms;
  std::vector<analyzerSummaryMessage> byCategory[ANALYZER_CAT_COUNT];
  int errorComms[ANALYZER_CAT_COUNT] = {};
  int warnComms[ANALYZER_CAT_COUNT] = {};

  int analyzed = 0;
  try {
    if (!analyzerBuildRuntimeViews(records, nRecords, &comms)) return -1;
    for (const auto& comm : comms) {
      if (analyzerCommUsable(fp, comm)) {
        auto ranksVec = analyzerRankIndex(comm);
        const auto* ranks = ranksVec.data();
        int nRanks = (int)ranksVec.size();
        if (nRanks > 0) {
          size_t previousSizes[ANALYZER_CAT_COUNT];
          for (int cat = 0; cat < ANALYZER_CAT_COUNT; cat++) previousSizes[cat] = byCategory[cat].size();
          uint64_t h = comm.id.commHash;
          analyzerCollectConnType(h, ranks, nRanks, byCategory[ANALYZER_CAT_CONN_TYPE]);
          analyzerCollectConsistency(h, ranks, nRanks, comm.nRanks, byCategory[ANALYZER_CAT_CONSISTENCY]);
          analyzerCollectRingGraph(h, ranks, nRanks, byCategory[ANALYZER_CAT_RING_GRAPH]);
          analyzerCollectP2pConnect(h, ranks, nRanks, byCategory[ANALYZER_CAT_P2P_CONNECT]);
          analyzerCollectHangReported(h, ranks, nRanks, byCategory[ANALYZER_CAT_HANG_REPORTED]);
          // Classify per collected entry: an entry counts as ERROR only if it
          // produced an error-severity finding; same-hash comms stay separate.
          for (int cat = 0; cat < ANALYZER_CAT_COUNT; cat++) {
            size_t prev = previousSizes[cat];
            size_t now = byCategory[cat].size();
            if (now <= prev) continue;
            bool hasErrorFinding = false;
            for (size_t i = prev; i < now; i++) {
              if (!byCategory[cat][i].isWarn) hasErrorFinding = true;
            }
            if (hasErrorFinding) errorComms[cat]++;
            else warnComms[cat]++;
          }
          analyzed++;
        }
      }
    }
  } catch (...) {
    return -1;
  }

  if (analyzed == 0) {
    fprintf(fp, "(no usable runtime snapshots)\n\n");
    return 0;
  }
  for (int cat = 0; cat < ANALYZER_CAT_COUNT; cat++) {
    auto& findings = byCategory[cat];
    bool hasError = false, hasWarn = false;
    for (auto& f : findings) { if (f.isWarn) hasWarn = true; else hasError = true; }
    if (hasError) {
      fprintf(fp, "# %s %d/%d comms ERROR\n", analyzerCategoryName(cat), errorComms[cat], analyzed);
    } else if (hasWarn) {
      fprintf(fp, "# %s %d/%d comms WARNING\n", analyzerCategoryName(cat), warnComms[cat], analyzed);
    } else {
      fprintf(fp, "# %s PASS\n", analyzerCategoryName(cat));
      continue;
    }
    for (auto& f : findings) {
      fprintf(fp, "  0x%lx : %s\n", (unsigned long)f.commHash, f.brief.c_str());
    }
  }
  fprintf(fp, "\n");
  return ferror(fp) ? -1 : 0;
}

struct analyzerFileHeader {
  uint32_t magic;
  uint32_t version;
  int32_t nData;
  int32_t nLegTimeouts;
};
static_assert(sizeof(analyzerFileHeader) == 16, "stable analyzer file header");

static int analyzerBuildPath(char* buf, int n, const char* dir,
                             const char* prefix, const char* suffix) {
  char hostname[64] = {0};
  gethostname(hostname, sizeof(hostname));
  hostname[sizeof(hostname) - 1] = '\0';
  for (char* p = hostname; *p; p++) if (*p == '.') { *p = '\0'; break; }
  time_t t = time(nullptr);
  struct tm tm;
  localtime_r(&t, &tm);
  char ts[32];
  strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm);
  return snprintf(buf, n, "%s/%s_%s_pid-%d_%s.%s",
                  dir, prefix, hostname, (int)getpid(), ts, suffix);
}

int analyzerDumpRuntimeData(const char* dir, const rasDiagnosticsRuntimeRecord* records,
                            int nRecords, int nLegTimeouts) {
  if (!dir || nLegTimeouts < 0 || nRecords < 0 ||
      (size_t)nRecords > PCCL_RAS_ANALYZER_MAX_BLOB_SIZE / sizeof(*records)) return -1;
  try {
    std::vector<analyzerCommView> comms;
    if (!analyzerBuildRuntimeViews(records, nRecords, &comms)) return -1;
    for (const auto& comm : comms) if (comm.malformed) return -1;
  } catch (...) {
    return -1;
  }
  char path[PATH_MAX + 1];
  analyzerBuildPath(path, sizeof(path), dir, "pccl_diag_data", "bin");
  FILE* fp = fopen(path, "wb");
  if (!fp) return -1;
  const size_t bytes = (size_t)nRecords * sizeof(*records);
  analyzerFileHeader header = {PCCL_RAS_ANALYZER_MAGIC, 2, (int32_t)bytes, nLegTimeouts};
  bool ok = fwrite(&header, sizeof(header), 1, fp) == 1;
  if (ok && bytes) ok = fwrite(records, 1, bytes, fp) == bytes;
  if (fclose(fp) != 0) ok = false;
  return ok ? 0 : -1;
}

rasDiagnosticsRuntimeRecord* analyzerLoadRuntimeData(FILE* fp, int* outNRecords, int* outLegTimeouts) {
  if (outNRecords) *outNRecords = 0;
  if (outLegTimeouts) *outLegTimeouts = 0;
  analyzerFileHeader header;
  if (!fp || fread(&header, sizeof(header), 1, fp) != 1 || header.magic != PCCL_RAS_ANALYZER_MAGIC) return nullptr;
  if (header.version != 2) {
    fprintf(stderr, "analyzerLoadRuntimeData: unsupported file version %u; expected version 2. "
                    "Read historical files with the matching PCCL version.\n", (unsigned)header.version);
    return nullptr;
  }
  if (header.nData < 0 || header.nData > PCCL_RAS_ANALYZER_MAX_BLOB_SIZE || header.nLegTimeouts < 0) return nullptr;
  std::unique_ptr<char, decltype(&free)> data((char*)malloc(header.nData ? header.nData : 1), free);
  if (!data) return nullptr;
  size_t bytesRead = fread(data.get(), 1, header.nData, fp);
  if (bytesRead != (size_t)header.nData || ferror(fp))
    fprintf(stderr, "analyzerLoadRuntimeData: short read; using %zu of %d payload bytes\n", bytesRead, header.nData);
  else if (fgetc(fp) != EOF)
    fprintf(stderr, "analyzerLoadRuntimeData: ignoring trailing file data\n");
  int count = bytesRead / sizeof(rasDiagnosticsRuntimeRecord);
  if (bytesRead % sizeof(rasDiagnosticsRuntimeRecord) != 0)
    fprintf(stderr, "analyzerLoadRuntimeData: ignoring an incomplete final record\n");
  if (header.nData > 0 && count == 0) return nullptr;
  try {
    std::vector<analyzerCommView> comms;
    if (!analyzerBuildRuntimeViews(reinterpret_cast<const rasDiagnosticsRuntimeRecord*>(data.get()), count, &comms))
      return nullptr;
    bool readable = count == 0, damaged = false;
    for (const auto& comm : comms) {
      if (comm.nRanks > 0) readable = true;
      if (comm.malformed) damaged = true;
    }
    if (damaged) fprintf(stderr, "analyzerLoadRuntimeData: malformed rank snapshots will be skipped\n");
    if (!readable) return nullptr;
  } catch (...) {
    return nullptr;
  }
  if (outNRecords) *outNRecords = count;
  if (outLegTimeouts) *outLegTimeouts = header.nLegTimeouts;
  return reinterpret_cast<rasDiagnosticsRuntimeRecord*>(data.release());
}

int analyzerDumpRuntimeReport(const char* dir, const rasDiagnosticsRuntimeRecord* records,
                              int nRecords, int nLegTimeouts) {
  if (!dir) return -1;
  int written = 0;
  for (int category = 0; category < ANALYZER_CAT_COUNT; category++) {
    char path[PATH_MAX + 1], prefix[64];
    snprintf(prefix, sizeof(prefix), "pccl_diag_report_%s", analyzerCategoryName(category));
    analyzerBuildPath(path, sizeof(path), dir, prefix, "log");
    FILE* fp = fopen(path, "w");
    if (!fp) continue;
    int result = analyzerFormatRuntimeReport(fp, records, nRecords, nLegTimeouts, category, nullptr);
    if (fclose(fp) != 0) result = -1;
    if (result == 0) written++;
  }
  return written;
}
