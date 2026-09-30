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

struct analyzerRankEntry {
  int rank;
  struct rasDiagRankHeader* rh;
};

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

static int analyzerCategoryCount(void) { return ANALYZER_CAT_COUNT; }

static const char* const kRasConnTypeStr[] = {
  "CONN_NORMAL", "CONN_ICN", "CONN_PCIE", "CONN_SHM", "CONN_GDRDMA",
  "CONN_RDMA", "CONN_SOCKET", "CONN_NET_EXT", "CONN_NUM"
};

static const char* analyzerConnTypeToString(uint8_t t) {
  if (t == RAS_DIAG_CONN_NONE) return "NONE";
  if (t >= CONN_NUM) return "UNKNOWN";
  return kRasConnTypeStr[t];
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

static struct rasDiagChannel* findChannelInRank(struct rasDiagRankHeader* rh, int channelId) {
  struct rasDiagChannel* dc = rasDiagFirstChannel(rh);
  for (int c = 0; c < rh->nChannels; c++) {
    if (dc->channelId == channelId) return dc;
    dc = rasDiagNextChannel(dc);
  }
  return nullptr;
}

static struct rasDiagPeer* findPeerInChannel(struct rasDiagChannel* dc, int peerRank, uint8_t direction, uint8_t connector) {
  struct rasDiagPeer* dp = rasDiagFirstPeer(dc);
  for (int i = 0; i < dc->nPeerEntries; i++) {
    if (dp[i].peerRank == peerRank && dp[i].direction == direction && dp[i].connector == connector) {
      return &dp[i];
    }
  }
  return nullptr;
}

static int analyzerBuildDiagIndex(const struct rasCollComms::comm* comm, analyzerRankEntry* idx, int maxIdx) {
  if (comm == nullptr || comm->nDiagBytes <= 0 || idx == nullptr || maxIdx <= 0) return 0;
  const char* base = rasCollCommsDiagBytes(comm);
  const char* end = base + comm->nDiagBytes;
  int n = 0;
  const char* p = base;
  while (p + sizeof(struct rasDiagRankHeader) <= end && n < maxIdx) {
    struct rasDiagRankHeader* rh = (struct rasDiagRankHeader*)p;
    if (rh->totalBytes < sizeof(*rh) || p + rh->totalBytes > end) break;
    if (rh->nChannels < 0 || rh->nChannels > MAXCHANNELS) break;
    if (rh->wireVersion == RAS_DIAG_WIRE_VERSION) {
      idx[n].rank = rh->commRank;
      idx[n].rh = rh;
      n++;
    } else {
      fprintf(stderr, "WARN: RAS hang-check: skipping rank block with wire version 0x%x (expected 0x%x)\n",
              rh->wireVersion, RAS_DIAG_WIRE_VERSION);
    }
    p = (char*)rasDiagNextRank(rh);
  }
  qsort(idx, n, sizeof(*idx), rankIndexCompare);
  return n;
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
    struct rasDiagRankHeader* rh = ranks[r].rh;
    struct rasDiagChannel* dc = rasDiagFirstChannel(rh);
    bool printedRank = false;
    for (int c = 0; c < rh->nChannels; c++) {
      struct { const char* name; uint8_t t; bool isFc; } legs[] = {
        {"ring_send", dc->ringSendConn, false},
        {"ring_recv", dc->ringRecvConn, false},
        {"tree_up",   dc->treeUpConn,   false},
        {"tree_down", dc->treeDownConn, false},
        {"fc",        dc->fcConnType,   true},
      };
      bool printedChannel = false;
      for (auto& leg : legs) {
        bool valid = (leg.t == RAS_DIAG_CONN_NONE) ||
                     (leg.isFc ? (leg.t == CONN_ICN) : analyzerConnTypeIsValid(leg.t));
        if (!valid) {
          if (!printedRank) { fprintf(fp, "  rank %d:\n", rh->commRank); printedRank = true; }
          if (!printedChannel) { fprintf(fp, "    channel %d:\n", dc->channelId); printedChannel = true; }
          fprintf(fp, "      [ERROR] %s = %s (invalid)\n", leg.name, analyzerConnTypeToString(leg.t));
          nBad++;
        }
      }
      dc = rasDiagNextChannel(dc);
    }
  }
  if (nBad == 0) fprintf(fp, "  PASS (no invalid connection types)\n");
  else            fprintf(fp, "  FAIL (%d invalid entries)\n", nBad);
  return analyzerSuccess;
}

static analyzerResult_t analyzerCheckConsistency(FILE* fp, const analyzerRankEntry* ranks, int nRanks) {
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
    fprintf(fp, "  [WARN] ranks have inconsistent nChannels:");
    char rankBuf[320];
    for (auto& kv : byNChannel) {
      std::sort(kv.second.begin(), kv.second.end());
      formatCompactRankList(kv.second, rankBuf, sizeof(rankBuf));
      fprintf(fp, " nChannels=%d: ranks [%s]", kv.first, rankBuf);
    }
    fprintf(fp, "\n");
  }

  // Build per-channel rank count for iteration
  std::unordered_map<int, int> channelCount;
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagChannel* dc = rasDiagFirstChannel(ranks[r].rh);
    for (int c = 0; c < ranks[r].rh->nChannels; c++) { channelCount[dc->channelId]++; dc = rasDiagNextChannel(dc); }
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
  for (auto& kv : channelCount) {
    int channelId = kv.first;
    std::unordered_map<std::pair<uint64_t,uint64_t>, GroupInfo, PairHash> groups;
    int nSkipped = 0;
    for (int r = 0; r < nRanks; r++) {
      struct rasDiagChannel* dc = findChannelInRank(ranks[r].rh, channelId);
      if (dc == nullptr) continue;
      if (!dc->devCompletedKernelIdxValid && !dc->devHangReportedKernelIdxValid) { nSkipped++; continue; }
      auto key = stateKey(dc->devCompletedKernelIdx, dc->devHangReportedKernelIdx);
      auto it = groups.find(key);
      if (it == groups.end()) { // new state group
        GroupInfo gi;
        gi.minWft = gi.maxWft = dc->workFifoTail;
        gi.ranksInGroup.push_back(ranks[r].rank);
        groups.emplace(key, std::move(gi));
      } else { // merge into existing group
        GroupInfo& gi = it->second;
        gi.ranksInGroup.push_back(ranks[r].rank);
        if (dc->workFifoTail < gi.minWft) gi.minWft = dc->workFifoTail;
        if (dc->workFifoTail > gi.maxWft) gi.maxWft = dc->workFifoTail;
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
    if (nSkipped > 0) {
      fprintf(fp, "  [WARN] channel %d skipped %d rank(s) with both device reads failing\n",
              channelId, nSkipped);
    }
  }
  if (nBad == 0) fprintf(fp, "  PASS (all ranks agree on completed and hang-reported kernel idx per channel)\n");
  else           fprintf(fp, "  FAIL (%d channel(s) divergent)\n", nBad);
  return analyzerSuccess;
}

static analyzerResult_t analyzerCheckRingGraph(FILE* fp, const analyzerRankEntry* ranks, int nRanks) {
  if (fp == nullptr || ranks == nullptr || nRanks <= 0) return analyzerDataMissing;
  fprintf(fp, "\n[Check: RingGraph (per ring edge: prev.send.devStep vs my.recv.devStep; "
              "secondary inflight check via head/tail)]\n");
  std::unordered_map<int, int> channelCount;
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagChannel* dc = rasDiagFirstChannel(ranks[r].rh);
    for (int c = 0; c < ranks[r].rh->nChannels; c++) { channelCount[dc->channelId]++; dc = rasDiagNextChannel(dc); }
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
      struct rasDiagChannel* myChannel = findChannelInRank(ranks[r].rh, channelId);
      if (myChannel == nullptr) continue;
      int myRank = ranks[r].rank;
      int prev = myChannel->ringPrev;
      if (prev < 0) continue;
      analyzerRankEntry key{prev, nullptr};
      analyzerRankEntry* found = static_cast<analyzerRankEntry*>(bsearch(&key, ranks, nRanks, sizeof(*ranks), rankIndexCompare));
      if (found == nullptr)       { nSkippedNoPrevRank++; continue; }
      struct rasDiagChannel* prevChannel = findChannelInRank(found->rh, channelId);
      if (prevChannel == nullptr)    { nSkippedNoPrevChannel++; continue; }
      struct rasDiagPeer* prevSend = findPeerInChannel(prevChannel, myRank, /*send*/0, /*coll*/0);
      struct rasDiagPeer* myRecv  = findPeerInChannel(myChannel,   prev,   /*recv*/1, /*coll*/0);
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
    fprintf(fp, "  PASS (prev.send.devStep == my.recv.devStep on every ring edge)\n");
  }
  return analyzerSuccess;
}

static analyzerResult_t analyzerCheckP2pConnect(FILE* fp, const analyzerRankEntry* ranks, int nRanks) {
  if (fp == nullptr || ranks == nullptr || nRanks <= 0) return analyzerDataMissing;
  fprintf(fp, "\n[Check: P2pConnect (per p2p send/recv pair: devStep, head/tail)]\n");
  std::unordered_map<int, int> channelCount;
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagChannel* dc = rasDiagFirstChannel(ranks[r].rh);
    for (int c = 0; c < ranks[r].rh->nChannels; c++) { channelCount[dc->channelId]++; dc = rasDiagNextChannel(dc); }
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
      struct rasDiagChannel* myChannel = findChannelInRank(ranks[r].rh, channelId);
      if (myChannel == nullptr) continue;
      int myRank = ranks[r].rank;
      struct rasDiagPeer* dp = rasDiagFirstPeer(myChannel);
      for (int i = 0; i < myChannel->nPeerEntries; i++) {
        if (!dp[i].connected || dp[i].direction != 0) continue;
        if (dp[i].connector != 1) continue;
        int peer = dp[i].peerRank;
        if (peer <= myRank) continue;
        analyzerRankEntry key{peer, nullptr};
        analyzerRankEntry* found = static_cast<analyzerRankEntry*>(bsearch(&key, ranks, nRanks, sizeof(*ranks), rankIndexCompare));
        if (!found)              { nSkippedNoPeerRank++; continue; }
        struct rasDiagChannel* peerChannel = findChannelInRank(found->rh, channelId);
        if (!peerChannel)           { nSkippedNoPeerChannel++; continue; }
        struct rasDiagPeer* peerRecv = findPeerInChannel(peerChannel, myRank, /*recv*/1, dp[i].connector);
        if (!peerRecv)           { nSkippedNoRecvEntry++; continue; }
        nPairsChecked++;
        int edgeResult = checkEdgeStepHeadTail(fp, ensureHeader, myRank, peer,
                            dp[i].devStep, peerRecv->devStep,
                            dp[i].head, peerRecv->tail,
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
    fprintf(fp, "  PASS (no p2p (connector=1) pairs to check — coll connector pairs are covered by RingGraph)\n");
  } else if (nBad > 0) {
    fprintf(fp, "  FAIL (%d p2p pair(s) with devStep mismatch%s; %d pair(s) checked)\n",
            nBad, nInflightWarn ? "; additional inflight imbalances also detected" : "", nPairsChecked);
  } else if (nInflightWarn > 0) {
    fprintf(fp, "  WARNING: %d p2p pair(s) with inflight head/tail mismatch; %d pair(s) checked\n",
            nInflightWarn, nPairsChecked);
  } else {
    fprintf(fp, "  PASS (devStep matched on every p2p pair; %d pair(s) checked)\n", nPairsChecked);
  }
  return analyzerSuccess;
}

static analyzerResult_t analyzerCheckHangReported(FILE* fp, const analyzerRankEntry* ranks, int nRanks) {
  if (fp == nullptr || ranks == nullptr || nRanks <= 0) return analyzerDataMissing;
  fprintf(fp, "\n[Check: HangReportedKernelIdx (kernel idx is 1-based; idx=N means the N-th kernel ever submitted)]\n");
  std::unordered_map<uint64_t, std::unordered_set<int>> byIdx;
  std::unordered_set<int> uniqReportingRanks;
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagRankHeader* rh = ranks[r].rh;
    struct rasDiagChannel* dc = rasDiagFirstChannel(rh);
    for (int c = 0; c < rh->nChannels; c++) {
      if (dc->devHangReportedKernelIdxValid && dc->devHangReportedKernelIdx != 0) {
        byIdx[dc->devHangReportedKernelIdx].insert(rh->commRank);
        uniqReportingRanks.insert(rh->commRank);
      }
      dc = rasDiagNextChannel(dc);
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
  if (nReporting > 0)
    fprintf(fp, "  FAIL (%d rank(s) self-reported hang)\n", nReporting);
  else
    fprintf(fp, "  PASS (no rank self-reported hang)\n");
  return analyzerSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Renderers                                                                 //
///////////////////////////////////////////////////////////////////////////////

int analyzerFormatReport(FILE* fp, const struct rasCollComms* blob, int nLegTimeouts,
                     int singleCategory) {
  if (fp == nullptr || blob == nullptr) return -1;
  time_t now = time(nullptr);
  char ts[64];
  strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", localtime(&now));
  fprintf(fp, "=== PCCL Online Hang Analyzer Report ===\n");
  fprintf(fp, "wall time      : %s\n", ts);
  fprintf(fp, "leg timeouts   : %d%s\n", nLegTimeouts,
          nLegTimeouts > 0 ? " (incomplete data!)" : "");
  fprintf(fp, "wire version   : 0x%x\n", RAS_DIAG_WIRE_VERSION);
  if (singleCategory >= 0 && singleCategory < ANALYZER_CAT_COUNT) {
    fprintf(fp, "category       : %s\n", analyzerCategoryName(singleCategory));
  }

  const struct rasCollComms::comm* comm = blob->comms;
  int analyzed = 0;
  for (int i = 0; i < blob->nComms; i++) {
    if (comm->nDiagBytes > 0) {
      fprintf(fp, "\n---- comm 0x%lx (%d/%d ranks, %d diag bytes) ----\n",
              (unsigned long)comm->commId.commHash, comm->nValidRanks, comm->nRanks, comm->nDiagBytes);
      std::vector<analyzerRankEntry> ranksVec(comm->nRanks);
      analyzerRankEntry* ranks = ranksVec.data();
      int nRanks = analyzerBuildDiagIndex(comm, ranks, comm->nRanks);
      if (nRanks == 0) {
        fprintf(fp, "  (no usable rank blocks; check wire version)\n");
      } else {
        if (singleCategory == -1 || singleCategory == ANALYZER_CAT_CONN_TYPE)   analyzerCheckConnType(fp, ranks, nRanks);
        if (singleCategory == -1 || singleCategory == ANALYZER_CAT_CONSISTENCY) analyzerCheckConsistency(fp, ranks, nRanks);
        if (singleCategory == -1 || singleCategory == ANALYZER_CAT_RING_GRAPH)  analyzerCheckRingGraph(fp, ranks, nRanks);
        if (singleCategory == -1 || singleCategory == ANALYZER_CAT_P2P_CONNECT) analyzerCheckP2pConnect(fp, ranks, nRanks);
        if (singleCategory == -1 || singleCategory == ANALYZER_CAT_HANG_REPORTED) analyzerCheckHangReported(fp, ranks, nRanks);
      }
      analyzed++;
    }
    comm = rasCollCommsCommNext(comm);
  }
  if (analyzed == 0) {
    fprintf(fp, "\n(no comms carried diag tier; INCLUDE_DIAG flag may not have been set)\n");
  }
  fprintf(fp, "\n=== End of Report ===\n");
  return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Summary                                                                   //
///////////////////////////////////////////////////////////////////////////////

static void buildChannelCount(const analyzerRankEntry* ranks, int nRanks,
                           std::unordered_map<int, int>& channelCount) {
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagChannel* dc = rasDiagFirstChannel(ranks[r].rh);
    for (int c = 0; c < ranks[r].rh->nChannels; c++) { channelCount[dc->channelId]++; dc = rasDiagNextChannel(dc); }
  }
}

static void analyzerCollectConnType(uint64_t commHash, const analyzerRankEntry* ranks, int nRanks,
                            std::vector<analyzerSummaryMessage>& out) {
  int nInvalid = 0;
  std::unordered_map<int, int> badRanks, badChannels;
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagRankHeader* rh = ranks[r].rh;
    struct rasDiagChannel* dc = rasDiagFirstChannel(rh);
    for (int c = 0; c < rh->nChannels; c++) {
      int chanBad = (!analyzerConnTypeIsValid(dc->ringSendConn))
                  + (!analyzerConnTypeIsValid(dc->ringRecvConn))
                  + (!analyzerConnTypeIsValid(dc->treeUpConn))
                  + (!analyzerConnTypeIsValid(dc->treeDownConn))
                  + (!analyzerConnTypeIsValid(dc->fcConnType));
      if (chanBad) { nInvalid += chanBad; badRanks[rh->commRank]++; badChannels[dc->channelId]++; }
      dc = rasDiagNextChannel(dc);
    }
  }
  if (nInvalid == 0) return;
  char buf[256];
  snprintf(buf, sizeof(buf), "%d invalid conn type(s) across %zu rank(s) / %zu channel(s)",
           nInvalid, badRanks.size(), badChannels.size());
  out.push_back({commHash, buf, false});
}

static void analyzerCollectConsistency(uint64_t commHash, const analyzerRankEntry* ranks, int nRanks,
                               std::vector<analyzerSummaryMessage>& out) {
  int minN = ranks[0].rh->nChannels, maxN = minN;
  for (int r = 1; r < nRanks; r++) {
    int n = ranks[r].rh->nChannels;
    if (n < minN) minN = n;
    if (n > maxN) maxN = n;
  }
  if (minN != maxN) {
    char buf[256];
    snprintf(buf, sizeof(buf), "ranks have inconsistent nChannels (range %d..%d)", minN, maxN);
    out.push_back({commHash, buf, false});
  }

  std::unordered_map<int, int> channelCount;
  buildChannelCount(ranks, nRanks, channelCount);
  int nDivergent = 0;
  for (auto& kv : channelCount) {
    int channelId = kv.first;
    uint64_t expectedCmpl = 0, expectedHang = 0;
    bool initialized = false, diverged = false;
    for (int r = 0; r < nRanks; r++) {
      struct rasDiagChannel* dc = findChannelInRank(ranks[r].rh, channelId);
      if (dc == nullptr) continue;
      if (!dc->devCompletedKernelIdxValid && !dc->devHangReportedKernelIdxValid) continue;
      if (!initialized) {
        expectedCmpl = dc->devCompletedKernelIdx;
        expectedHang = dc->devHangReportedKernelIdx;
        initialized = true;
        continue;
      }
      if (dc->devCompletedKernelIdx != expectedCmpl || dc->devHangReportedKernelIdx != expectedHang) {
        diverged = true;
        break;
      }
    }
    if (diverged) nDivergent++;
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
      struct rasDiagChannel* myChannel = findChannelInRank(ranks[r].rh, channelId);
      if (myChannel == nullptr) continue;
      int myRank = ranks[r].rank;
      int prev = myChannel->ringPrev;
      if (prev < 0) continue;
      analyzerRankEntry key{prev, nullptr};
      analyzerRankEntry* found = static_cast<analyzerRankEntry*>(bsearch(&key, ranks, nRanks, sizeof(*ranks), rankIndexCompare));
      if (found == nullptr) continue;
      struct rasDiagChannel* prevChannel = findChannelInRank(found->rh, channelId);
      if (prevChannel == nullptr) continue;
      struct rasDiagPeer* prevSend = findPeerInChannel(prevChannel, myRank, 0, 0);
      struct rasDiagPeer* myRecv = findPeerInChannel(myChannel, prev, 1, 0);
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
      struct rasDiagChannel* myChannel = findChannelInRank(ranks[r].rh, channelId);
      if (myChannel == nullptr) continue;
      int myRank = ranks[r].rank;
      struct rasDiagPeer* dp = rasDiagFirstPeer(myChannel);
      for (int i = 0; i < myChannel->nPeerEntries; i++) {
        if (!dp[i].connected || dp[i].direction != 0) continue;
        if (dp[i].connector != 1) continue;
        int peer = dp[i].peerRank;
        if (peer <= myRank) continue;
        analyzerRankEntry key{peer, nullptr};
        analyzerRankEntry* found = static_cast<analyzerRankEntry*>(bsearch(&key, ranks, nRanks, sizeof(*ranks), rankIndexCompare));
        if (!found) continue;
        struct rasDiagChannel* peerChannel = findChannelInRank(found->rh, channelId);
        if (!peerChannel) continue;
        struct rasDiagPeer* peerRecv = findPeerInChannel(peerChannel, myRank, 1, dp[i].connector);
        if (!peerRecv) continue;
        int cls = classifyEdge(dp[i].devStep, peerRecv->devStep, dp[i].head, peerRecv->tail);
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
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagRankHeader* rh = ranks[r].rh;
    struct rasDiagChannel* dc = rasDiagFirstChannel(rh);
    for (int c = 0; c < rh->nChannels; c++) {
      if (dc->devHangReportedKernelIdxValid && dc->devHangReportedKernelIdx != 0) {
        agg[dc->devHangReportedKernelIdx][rh->commRank]++;
      }
      dc = rasDiagNextChannel(dc);
    }
  }
  if (agg.empty()) return;
  std::unordered_map<int, int> allRanks;
  for (auto& kv : agg) for (auto& r : kv.second) allRanks[r.first]++;
  char buf[256];
  snprintf(buf, sizeof(buf), "%zu distinct hang-reported kernel idx value(s) from %zu/%d rank(s)",
           agg.size(), allRanks.size(), nRanks);
  out.push_back({commHash, buf, false});
}

int analyzerFormatSummary(FILE* fp, const struct rasCollComms* blob) {
  if (fp == nullptr || blob == nullptr) return -1;
  std::vector<analyzerSummaryMessage> byCategory[ANALYZER_CAT_COUNT];

  const struct rasCollComms::comm* comm = blob->comms;
  int analyzed = 0;
  for (int i = 0; i < blob->nComms; i++) {
    if (comm->nDiagBytes > 0) {
      std::vector<analyzerRankEntry> ranksVec(comm->nRanks);
      analyzerRankEntry* ranks = ranksVec.data();
      int nRanks = analyzerBuildDiagIndex(comm, ranks, comm->nRanks);
      if (nRanks > 0) {
        uint64_t h = comm->commId.commHash;
        analyzerCollectConnType(h, ranks, nRanks, byCategory[ANALYZER_CAT_CONN_TYPE]);
        analyzerCollectConsistency(h, ranks, nRanks, byCategory[ANALYZER_CAT_CONSISTENCY]);
        analyzerCollectRingGraph(h, ranks, nRanks, byCategory[ANALYZER_CAT_RING_GRAPH]);
        analyzerCollectP2pConnect(h, ranks, nRanks, byCategory[ANALYZER_CAT_P2P_CONNECT]);
        analyzerCollectHangReported(h, ranks, nRanks, byCategory[ANALYZER_CAT_HANG_REPORTED]);
        analyzed++;
      }
    }
    comm = rasCollCommsCommNext(comm);
  }

  if (analyzed == 0) {
    fprintf(fp, "(no comms carried diag tier; INCLUDE_DIAG flag may not have been set)\n\n");
    return 0;
  }
  for (int cat = 0; cat < ANALYZER_CAT_COUNT; cat++) {
    auto& findings = byCategory[cat];
    bool hasError = false, hasWarn = false;
    for (auto& f : findings) { if (f.isWarn) hasWarn = true; else hasError = true; }

    if (hasError) {
      fprintf(fp, "# %s %d/%d comms ERROR\n", analyzerCategoryName(cat), (int)findings.size(), analyzed);
    } else if (hasWarn) {
      fprintf(fp, "# %s %d/%d comms WARNING\n", analyzerCategoryName(cat), (int)findings.size(), analyzed);
    } else {
      fprintf(fp, "# %s PASS\n", analyzerCategoryName(cat));
      continue;
    }
    for (auto& f : findings) {
      fprintf(fp, "  0x%lx : %s\n", (unsigned long)f.commHash, f.brief.c_str());
    }
  }
  fprintf(fp, "\n");
  return 0;
}

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

int analyzerDumpData(const char* dir, const void* blob, int nBlobBytes, int nLegTimeouts) {
  if (dir == nullptr || blob == nullptr || nBlobBytes <= 0) return -1;
  char path[PATH_MAX + 1];
  analyzerBuildPath(path, sizeof(path), dir, "pccl_diag_data", "bin");
  FILE* fp = fopen(path, "wb");
  if (fp == nullptr) {
    fprintf(stderr, "analyzerDumpData: cannot open %s\n", path);
    return -1;
  }
  struct { uint32_t magic; uint32_t version; int32_t nData; int32_t nLegTimeouts; } hdr =
    { PCCL_RAS_ANALYZER_MAGIC, 1, (int32_t)nBlobBytes, (int32_t)nLegTimeouts };
  int ok = (fwrite(&hdr, sizeof(hdr), 1, fp) == 1 &&
            fwrite(blob, 1, nBlobBytes, fp) == (size_t)nBlobBytes) ? 0 : -1;
  fclose(fp);
  return ok;
}

struct rasCollComms* analyzerLoadData(FILE* fp, int* outBlobBytes, int* outLegTimeouts) {
  if (fp == nullptr) return nullptr;
  struct { uint32_t magic; uint32_t version; int32_t nData; int32_t nLegTimeouts; } hdr;
  if (fread(&hdr, sizeof(hdr), 1, fp) != 1) {
    fprintf(stderr, "analyzerLoadData: failed to read header\n");
    return nullptr;
  }
  if (hdr.magic != PCCL_RAS_ANALYZER_MAGIC) {
    fprintf(stderr, "analyzerLoadData: bad magic 0x%08x (expected 0x%08x)\n", hdr.magic, PCCL_RAS_ANALYZER_MAGIC);
    return nullptr;
  }
  if (hdr.version != 1) {
    fprintf(stderr, "analyzerLoadData: unsupported version %u\n", hdr.version);
    return nullptr;
  }
  if (hdr.nData <= 0 || hdr.nData > PCCL_RAS_ANALYZER_MAX_BLOB_SIZE) {
    fprintf(stderr, "analyzerLoadData: invalid blob size %d\n", hdr.nData);
    return nullptr;
  }
  char* buf = (char*)malloc(hdr.nData);
  if (buf == nullptr) {
    fprintf(stderr, "analyzerLoadData: out of memory (%d bytes)\n", hdr.nData);
    return nullptr;
  }
  if (fread(buf, 1, (size_t)hdr.nData, fp) != (size_t)hdr.nData) {
    fprintf(stderr, "analyzerLoadData: short read\n");
    free(buf);
    return nullptr;
  }
  if (outBlobBytes) *outBlobBytes = hdr.nData;
  if (outLegTimeouts) *outLegTimeouts = hdr.nLegTimeouts;
  return reinterpret_cast<struct rasCollComms*>(buf);
}

int analyzerDumpReport(const char* dir, const struct rasCollComms* blob, int nLegTimeouts) {
  if (dir == nullptr || blob == nullptr) return -1;
  int nCats = analyzerCategoryCount();
  int nWritten = 0;
  for (int cat = 0; cat < nCats; cat++) {
    char* buf = nullptr;
    size_t len = 0;
    FILE* mem = open_memstream(&buf, &len);
    if (mem == nullptr) {
      fprintf(stderr, "analyzerDumpReport: open_memstream failed for category %s\n", analyzerCategoryName(cat));
      continue;
    }
    analyzerFormatReport(mem, blob, nLegTimeouts, cat);
    fclose(mem);
    if (buf == nullptr || len == 0) { free(buf); continue; }
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "pccl_diag_report_%s", analyzerCategoryName(cat));
    char path[PATH_MAX + 1];
    analyzerBuildPath(path, sizeof(path), dir, prefix, "log");
    FILE* fp = fopen(path, "w");
    if (fp == nullptr) {
      fprintf(stderr, "analyzerDumpReport: cannot open %s\n", path);
      free(buf);
      continue;
    }
    fputs(buf, fp);
    fclose(fp);
    free(buf);
    nWritten++;
  }
  return nWritten;
}
