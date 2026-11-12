//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "analyzer.h"
#include "ring_backtrace.h"

#include <algorithm>
#include <climits>
#include <map>
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

// Standalone pcclras does not link init.cc; keep these names identical to
// ncclFuncStr/ncclProtoStr there, like the connection-name table above.
static const char* const kRasFuncStr[] = {
  "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv"
};
static_assert(sizeof(kRasFuncStr) / sizeof(kRasFuncStr[0]) == NCCL_NUM_FUNCTIONS_WITH_P2P,
              "kRasFuncStr must match NCCL_NUM_FUNCTIONS_WITH_P2P");
static const char* const kRasProtoStr[] = {
  "LL", "LL128", "Simple", "Signal", "SimpleBulk", "SignalBulk", "SimpleBulkAR"
};
static_assert(sizeof(kRasProtoStr) / sizeof(kRasProtoStr[0]) == NCCL_NUM_PROTOCOLS,
              "kRasProtoStr must match NCCL_NUM_PROTOCOLS");

static const char* analyzerFuncToString(uint8_t func) {
  return func < NCCL_NUM_FUNCTIONS_WITH_P2P ? kRasFuncStr[func] : "Unknown";
}
static const char* analyzerProtoToString(uint8_t proto) {
  return proto < NCCL_NUM_PROTOCOLS ? kRasProtoStr[proto] : "Unknown";
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

static int analyzerBuildDiagIndex(const struct rasCollComms::comm* comm, analyzerRankEntry* idx, int maxIdx) {
  if (comm == nullptr || comm->nDiagBytes <= 0 || idx == nullptr || maxIdx <= 0) return 0;
  const char* base = rasCollCommsDiagBytes(comm);
  const char* end = base + comm->nDiagBytes;
  int n = 0;
  const char* p = base;
  while ((size_t)(end - p) >= sizeof(struct rasDiagRankHeader) && n < maxIdx) {
    struct rasDiagRankHeader* rh = (struct rasDiagRankHeader*)p;
    if (rh->totalBytes < sizeof(*rh) || rh->totalBytes > (size_t)(end - p)) break;
    if (rh->nChannels < 0 || rh->nChannels > MAXCHANNELS) break;
    if (rh->wireVersion == RAS_DIAG_WIRE_VERSION) {
      const char* channel = p + sizeof(*rh);
      size_t remaining = rh->totalBytes - sizeof(*rh);
      bool valid = true;
      for (int c = 0; c < rh->nChannels; c++) {
        if (remaining < sizeof(struct rasDiagChannel)) { valid = false; break; }
        const auto* dc = (const struct rasDiagChannel*)channel;
        remaining -= sizeof(*dc);
        if (dc->nPeerEntries < 0 || (size_t)dc->nPeerEntries > remaining / sizeof(struct rasDiagPeer)) {
          valid = false;
          break;
        }
        size_t peerBytes = (size_t)dc->nPeerEntries * sizeof(struct rasDiagPeer);
        channel += sizeof(*dc) + peerBytes;
        remaining -= peerBytes;
      }
      if (!valid) {
        fprintf(stderr, "WARN: RAS hang-check: skipping rank %d with incomplete channel/peer records\n", rh->commRank);
        p += rh->totalBytes;
        continue;
      }
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
                        uint64_t producerHead, uint64_t consumerTail, bool headTailValid = true) {
  if (producerStep != consumerStep) return EDGE_ERROR;
  if (!headTailValid) return EDGE_OK;
  bool headActive = (int64_t)producerHead > (int64_t)consumerStep;
  bool tailActive = (int64_t)consumerTail > (int64_t)producerStep;
  if (headActive && tailActive && producerHead != consumerTail) return EDGE_WARN;
  return EDGE_OK;
}

template<typename EnsureHeaderFn>
static int checkEdgeStepHeadTail(FILE* fp, EnsureHeaderFn ensureHeader,
                                  int srcRank, int dstRank,
                                  uint64_t producerStep, uint64_t consumerStep,
                                  uint64_t producerHead, uint64_t consumerTail,
                                  const char* prodLabel, const char* consLabel, bool headTailValid = true) {
  int classification = classifyEdge(producerStep, consumerStep, producerHead, consumerTail, headTailValid);
  if (fp == nullptr) return classification;
  if (classification == EDGE_ERROR) {
    ensureHeader();
    long delta = (long)((int64_t)consumerStep - (int64_t)producerStep);
    fprintf(fp, "    [ERROR] rank%d -> rank%d : %s=%ld %s=%ld (delta=%+ld)\n",
            srcRank, dstRank, prodLabel, (long)producerStep,
            consLabel, (long)consumerStep, delta);
    return EDGE_ERROR;
  }
  if (!headTailValid) return EDGE_OK;
  // Use signed comparison: head/tail can be negative (initialized to -NCCL_STEPS
  // in shared mode for flow control credit accounting).
  bool headActive = ((int64_t)producerHead > (int64_t)consumerStep);
  bool tailActive = ((int64_t)consumerTail > (int64_t)producerStep);
  if (classification == EDGE_WARN) {
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

static bool analyzerConsistencyStateValid(const struct rasDiagChannel* dc) {
  return dc != nullptr && dc->devCompletedKernelIdxValid && dc->devHangReportedKernelIdxValid;
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
  size_t nUnavailable = 0;
  for (auto& kv : channelCount) {
    int channelId = kv.first;
    std::unordered_map<std::pair<uint64_t,uint64_t>, GroupInfo, PairHash> groups;
    int nSkipped = std::max(0, expectedRanks - nRanks);
    for (int r = 0; r < nRanks; r++) {
      struct rasDiagChannel* dc = findChannelInRank(ranks[r].rh, channelId);
      if (!analyzerConsistencyStateValid(dc)) { nSkipped++; continue; }
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

static void reportRingTrace(FILE* fp, const ringTraceResult& trace, const analyzerRankEntry* ranks) {
  if (trace.status != RING_TRACE_SUSPECT) return;
  // Print the operands from existing peer records and the gaps already computed
  // by traceRingHead. Do not reconstruct the diagnostic rule in the formatter.
  const char* stepPrefix = trace.stepAlignment > 1 ? "ROUNDUP(" : "";
  std::string stepSuffix = trace.stepAlignment > 1 ? ", " + std::to_string(trace.stepAlignment) + ")" : "";
  bool usesIcnRecvHead = false;
  bool usesNetSendHead = false;
  auto printComparison = [&](int rank, int64_t gap, const char* relation) {
    auto* channel = findChannelInRank(ranks[rank].rh, trace.channelId);
    int previousRank = channel->ringPrev;
    int nextRank = channel->ringNext;
    const auto* previousSend = findPeerInChannel(findChannelInRank(ranks[previousRank].rh, trace.channelId),
                                               rank, /*send*/ 0, /*coll*/ 0);
    const auto* recvPeer = findPeerInChannel(channel, previousRank, /*recv*/ 1, /*coll*/ 0);
    const auto* sendPeer = findPeerInChannel(channel, nextRank, /*send*/ 0, /*coll*/ 0);
    const auto* nextRecv = findPeerInChannel(findChannelInRank(ranks[nextRank].rh, trace.channelId),
                                           rank, /*recv*/ 1, /*coll*/ 0);
    bool incomingIsNet = isNetConnection(channel->ringRecvConn);
    bool outgoingIsNet = channel->ringOperation.sendIsNet;
    // ICN recv.head aliases prev.send.head, read from the sender's local mapping.
    // NET send.logicalHead denotes next.recv.head, not send-proxy completion.
    usesIcnRecvHead |= !incomingIsNet;
    usesNetSendHead |= outgoingIsNet;
    fprintf(fp, "rank%d.forwardingGap=(rank%d.recv.head=%lu - %srank%d.recv.devStep=%lu%s) - ",
            rank, rank, incomingIsNet ? recvPeer->head : previousSend->head,
            stepPrefix, rank, recvPeer->devStep, stepSuffix.c_str());
    fprintf(fp, "(rank%d.send.%s=%lu - %srank%d.send.devStep=%lu%s)=%ld %s 0",
            rank, outgoingIsNet ? "logicalHead" : "head", outgoingIsNet ? nextRecv->head : sendPeer->head,
            stepPrefix, rank, sendPeer->devStep, stepSuffix.c_str(), (long)gap, relation);
  };
  auto printRingEdge = [&](int rank) {
    const auto* channel = findChannelInRank(ranks[rank].rh, trace.channelId);
    const auto& operation = channel->ringOperation;
    if (operation.sendIsNet && operation.sendProxyRank != rank) {
      if (operation.sendProxyRank >= 0) fprintf(fp, "->(%d)", operation.sendProxyRank);
      else if (operation.sendProxyTpRank >= 0) fprintf(fp, "->(tpRank=%d)", operation.sendProxyTpRank);
      else fprintf(fp, "->(?)");
    }
    fprintf(fp, "->%d", channel->ringNext);
  };
  fprintf(fp, "  [SUSPECT] HeadRingTrace channel %d seq=%lu: collective=%s protocol=%s candidates=",
          trace.channelId, trace.seqNumber, analyzerFuncToString(trace.collective), analyzerProtoToString(trace.protocol));
  for (size_t i = 0; i < trace.candidates.size(); i++) {
    const auto& candidate = trace.candidates[i];
    fprintf(fp, "%s%d->%d", i ? "," : "", candidate.sourceRank, candidate.destinationRank);
  }
  fprintf(fp, "\n");
  for (const auto& candidate : trace.candidates) {
    const auto* sourceChannel = findChannelInRank(ranks[candidate.sourceRank].rh, trace.channelId);
    // PXN keeps an ICN device send type; report this edge's logical NET transport.
    uint8_t transport = sourceChannel->ringOperation.sendIsNet
                            ? findChannelInRank(ranks[candidate.destinationRank].rh, trace.channelId)->ringRecvConn
                            : sourceChannel->ringSendConn;
    fprintf(fp, "    rank%d -> rank%d: ring=%d", candidate.sourceRank, candidate.destinationRank, sourceChannel->ringPrev);
    printRingEdge(sourceChannel->ringPrev);
    printRingEdge(candidate.sourceRank);
    if (candidate.destinationIsForwarder) printRingEdge(candidate.destinationRank);
    fprintf(fp, "; rank%d.kernelIndex=%lu rank%d.kernelIndex=%lu; transport=%s; ",
            candidate.sourceRank, sourceChannel->ringOperation.kernelIndex, candidate.destinationRank,
            findChannelInRank(ranks[candidate.destinationRank].rh, trace.channelId)->ringOperation.kernelIndex,
            analyzerConnTypeToString(transport));
    printComparison(candidate.sourceRank, candidate.forwardingGap, ">");
    if (candidate.destinationIsForwarder) {
      fprintf(fp, "; ");
      printComparison(candidate.destinationRank, candidate.nextForwardingGap,
                      trace.collective < ncclFuncAllGather ? "==" : "<=");
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "    ");
  if (usesIcnRecvHead) fprintf(fp, "ICN recv.head read via prev.send.head; ");
  if (usesNetSendHead) fprintf(fp, "NET/PXN send.logicalHead=next.recv.head; ");
  fprintf(fp,
          "single snapshot; counter stability not checked; device timeout observed; %d positive forwarding lower-bound gap(s). "
          "%s; transient backlog possible; cross-rank snapshot is not atomic, not proof of physical cause or a unique fault.\n",
          trace.positiveGapCount,
          trace.collective < ncclFuncAllGather ? "First boundary from open-chain terminal"
                                               : "Terminal boundaries of closed-ring backlog segments");
}

// Shared Ring result for detailed reports and summaries. Counters remain in the
// original records; only diagnostic results and coverage are retained here.
enum ringGraphStatus { RING_GRAPH_PASS, RING_GRAPH_SKIPPED, RING_GRAPH_INCONCLUSIVE, RING_GRAPH_WARNING, RING_GRAPH_ERROR };

static const char* ringGraphStatusName(ringGraphStatus status) {
  static const char* const names[] = {"PASS", "SKIPPED", "INCONCLUSIVE", "WARNING", "ERROR"};
  return names[status];
}

struct ringGraphResult {
  std::vector<ringTraceResult> traces;
  int nonRingChannels = 0;
  int completed = 0, skipped = 0, inconclusive = 0, suspect = 0;
  int stepMismatches = 0, inflightWarnings = 0, uncheckedEdges = 0;

  ringGraphStatus status() const {
    if (stepMismatches) return RING_GRAPH_ERROR;
    if (suspect || inflightWarnings) return RING_GRAPH_WARNING;
    if (inconclusive || uncheckedEdges || (traces.empty() && !nonRingChannels)) return RING_GRAPH_INCONCLUSIVE;
    if (skipped || traces.empty()) return RING_GRAPH_SKIPPED;
    return RING_GRAPH_PASS;
  }
};

// Coalesce expected skips and incomplete diagnoses by reason. Completed channels
// need only one count, while suspect channels retain their detailed operands.
static std::vector<std::string> ringGraphReasons(const ringGraphResult& result) {
  std::map<std::pair<ringTraceStatus, std::string>, std::vector<int>> groups;
  for (const auto& trace : result.traces) {
    if (trace.status == RING_TRACE_SKIPPED || trace.status == RING_TRACE_INCONCLUSIVE)
      groups[{trace.status, trace.reason}].push_back(trace.channelId);
  }
  std::vector<std::string> messages;
  for (auto& group : groups) {
    char channels[320];
    formatCompactRankList(group.second, channels, sizeof(channels));
    const char* status = group.first.first == RING_TRACE_SKIPPED ? "SKIPPED" : "INCONCLUSIVE";
    messages.push_back(std::string("[") + status + "] HeadRingTrace channels [" + channels + "]: " + group.first.second);
  }
  if (result.nonRingChannels)
    messages.push_back("[SKIPPED] " + std::to_string(result.nonRingChannels) + " non-Ring channel(s)");
  if (result.traces.empty() && !result.nonRingChannels)
    messages.push_back("[INCONCLUSIVE] no usable Ring channel snapshots");
  return messages;
}

// A null output stream collects the identical results without printing details.
static ringGraphResult analyzeRingGraph(FILE* fp, const analyzerRankEntry* ranks, int nRanks, int expectedRanks) {
  ringGraphResult result;
  std::map<int, int> channelCount, nonRingCount;
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagChannel* dc = rasDiagFirstChannel(ranks[r].rh);
    for (int c = 0; c < ranks[r].rh->nChannels; c++) {
      channelCount[dc->channelId]++;
      if (dc->ringExcluded == 1) nonRingCount[dc->channelId]++;
      dc = rasDiagNextChannel(dc);
    }
  }
  for (const auto& channel : channelCount) {
    int channelId = channel.first;
    // Exclude only explicitly non-Ring channels. Negative Ring peers alone may
    // mean incomplete initialization; missing ranks or conflicting snapshots stay inconclusive.
    if (nRanks == expectedRanks && nonRingCount[channelId] == channel.second) {
      result.nonRingChannels++;
      continue;
    }
    result.traces.push_back(traceRingHead(ranks, nRanks, expectedRanks, channelId));
    const auto& trace = result.traces.back();
    switch (trace.status) {
      case RING_TRACE_COMPLETED: result.completed++; break;
      case RING_TRACE_SKIPPED: result.skipped++; break;
      case RING_TRACE_INCONCLUSIVE: result.inconclusive++; break;
      case RING_TRACE_SUSPECT: result.suspect++; break;
    }
    if (fp) reportRingTrace(fp, trace, ranks);
    bool printedHeader = false;
    auto ensureHeader = [&]() {
      if (!printedHeader) { fprintf(fp, "  channel %d:\n", channelId); printedHeader = true; }
    };
    for (int r = 0; r < nRanks; r++) {
      auto* myChannel = findChannelInRank(ranks[r].rh, channelId);
      if (!myChannel) { result.uncheckedEdges++; continue; }
      int myRank = ranks[r].rank;
      int prev = myChannel->ringPrev;
      if (prev < 0) { result.uncheckedEdges++; continue; }
      analyzerRankEntry key{prev, nullptr};
      auto* found = static_cast<analyzerRankEntry*>(bsearch(&key, ranks, nRanks, sizeof(*ranks), rankIndexCompare));
      auto* prevChannel = found ? findChannelInRank(found->rh, channelId) : nullptr;
      auto* prevSend = prevChannel ? findPeerInChannel(prevChannel, myRank, 0, 0) : nullptr;
      auto* myRecv = findPeerInChannel(myChannel, prev, 1, 0);
      if (!prevSend || !myRecv || !prevSend->connected || !myRecv->connected ||
          !(prevSend->readValid & RAS_DIAG_PEER_STEP_VALID) || !(myRecv->readValid & RAS_DIAG_PEER_STEP_VALID)) {
        result.uncheckedEdges++;
        continue;
      }
      // Still compare valid saved steps when either head/tail read failed.
      // Count the secondary check as incomplete instead of interpreting placeholders.
      bool headTailValid = (prevSend->readValid & RAS_DIAG_PEER_HEAD_VALID) &&
                           (myRecv->readValid & RAS_DIAG_PEER_TAIL_VALID);
      if (!headTailValid) result.uncheckedEdges++;
      int edgeResult = checkEdgeStepHeadTail(fp, ensureHeader, prev, myRank,
                           prevSend->devStep, myRecv->devStep, prevSend->head, myRecv->tail,
                           "prev.send.devStep", "my.recv.devStep", headTailValid);
      if (edgeResult == EDGE_ERROR) result.stepMismatches++;
      if (edgeResult == EDGE_WARN) result.inflightWarnings++;
    }
  }
  return result;
}

static analyzerResult_t analyzerCheckRingGraph(FILE* fp, const analyzerRankEntry* ranks, int nRanks, int expectedRanks) {
  if (fp == nullptr) return analyzerDataMissing;
  fprintf(fp, "\n[Check: RingGraph (saved steps, head/tail and consumption-head attribution)]\n");
  auto result = analyzeRingGraph(fp, ranks, nRanks, expectedRanks);
  if (result.stepMismatches)
    fprintf(fp, "  Legacy: ERROR (%d ring edge(s) with step mismatch)\n", result.stepMismatches);
  else if (result.inflightWarnings)
    fprintf(fp, "  Legacy: WARNING (%d edge(s) with mid-kernel head/tail imbalance)\n", result.inflightWarnings);
  else if (result.traces.empty() && result.nonRingChannels)
    fprintf(fp, "  Legacy: SKIPPED (no Ring channels)\n");
  else
    fprintf(fp, "  Legacy: %s\n", result.uncheckedEdges || result.traces.empty() ? "INCOMPLETE" : "PASS");
  if (result.uncheckedEdges)
    fprintf(fp, "  [INCONCLUSIVE] Legacy: %d edge check(s) incomplete; missing peer data or unreadable head/tail/step\n",
            result.uncheckedEdges);
  for (const auto& reason : ringGraphReasons(result)) fprintf(fp, "  %s\n", reason.c_str());
  fprintf(fp, "  HeadRingTrace: completed=%d, suspect=%d, inconclusive=%d, skipped=%d\n",
          result.completed, result.suspect, result.inconclusive, result.skipped);
  fprintf(fp, "  Result: %s\n", ringGraphStatusName(result.status()));
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
  int nUnknownChannels = 0;
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagRankHeader* rh = ranks[r].rh;
    struct rasDiagChannel* dc = rasDiagFirstChannel(rh);
    for (int c = 0; c < rh->nChannels; c++) {
      if (!dc->devHangReportedKernelIdxValid) {
        nUnknownChannels++;
      } else if (dc->devHangReportedKernelIdx != 0) {
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
  if (nUnknownChannels > 0)
    fprintf(fp, "  [WARN] %d channel(s) had an unreadable hang-reported index (hang status unknown there)\n",
            nUnknownChannels);
  if (nReporting > 0)
    fprintf(fp, "  FAIL (%d rank(s) self-reported hang)\n", nReporting);
  else if (nUnknownChannels > 0)
    fprintf(fp, "  INCONCLUSIVE (no readable channel self-reported hang, but %d channel(s) unreadable)\n",
            nUnknownChannels);
  else
    fprintf(fp, "  PASS (no rank self-reported hang)\n");
  return analyzerSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// Renderers                                                                 //
///////////////////////////////////////////////////////////////////////////////

int analyzerFormatReport(FILE* fp, const struct rasCollComms* blob, int nLegTimeouts,
                     int singleCategory, const char* title) {
  if (fp == nullptr || blob == nullptr) return -1;
  time_t now = time(nullptr);
  char ts[64];
  strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", localtime(&now));
  fprintf(fp, "=== %s ===\n", title != nullptr ? title : "PCCL Online Hang Analyzer Report");
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
    if (comm->nDiagBytes > 0 || singleCategory == -1 || singleCategory == ANALYZER_CAT_RING_GRAPH) {
      fprintf(fp, "\n---- comm 0x%lx (%d/%d ranks, %d diag bytes) ----\n",
              (unsigned long)comm->commId.commHash, comm->nValidRanks, comm->nRanks, comm->nDiagBytes);
      std::vector<analyzerRankEntry> ranksVec(comm->nRanks);
      analyzerRankEntry* ranks = ranksVec.data();
      int nRanks = analyzerBuildDiagIndex(comm, ranks, comm->nRanks);
      if (nRanks == 0) {
        fprintf(fp, comm->nDiagBytes > 0 ? "  (no usable rank blocks; check wire version)\n"
                                      : "  (no rank diagnostic snapshots)\n");
        if (singleCategory == -1 || singleCategory == ANALYZER_CAT_RING_GRAPH)
          analyzerCheckRingGraph(fp, ranks, nRanks, comm->nRanks);
      } else {
        if (singleCategory == -1 || singleCategory == ANALYZER_CAT_CONN_TYPE)   analyzerCheckConnType(fp, ranks, nRanks);
        if (singleCategory == -1 || singleCategory == ANALYZER_CAT_CONSISTENCY) analyzerCheckConsistency(fp, ranks, nRanks, comm->nRanks);
        if (singleCategory == -1 || singleCategory == ANALYZER_CAT_RING_GRAPH)  analyzerCheckRingGraph(fp, ranks, nRanks, comm->nRanks);
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
      struct rasDiagChannel* dc = findChannelInRank(ranks[r].rh, channelId);
      if (!analyzerConsistencyStateValid(dc)) { nUnavailable++; continue; }
      if (!initialized) {
        expectedCmpl = dc->devCompletedKernelIdx;
        expectedHang = dc->devHangReportedKernelIdx;
        initialized = true;
        continue;
      }
      if (dc->devCompletedKernelIdx != expectedCmpl || dc->devHangReportedKernelIdx != expectedHang) {
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

static ringGraphStatus analyzerCollectRingGraph(uint64_t commHash, const analyzerRankEntry* ranks, int nRanks, int expectedRanks,
                                               std::vector<analyzerSummaryMessage>& out) {
  auto result = analyzeRingGraph(nullptr, ranks, nRanks, expectedRanks);
  for (const auto& trace : result.traces) {
    if (trace.status != RING_TRACE_SUSPECT) continue;
    const auto& firstCandidate = trace.candidates.front();
    char buf[320];
    snprintf(buf, sizeof(buf), "HeadRingTrace channel %d seq=%lu suspect rank%d -> rank%d: collective=%s protocol=%s; "
                              "%zu terminal boundary candidate(s), first shown; see detailed report; not physical root cause",
             trace.channelId, trace.seqNumber, firstCandidate.sourceRank, firstCandidate.destinationRank,
             analyzerFuncToString(trace.collective), analyzerProtoToString(trace.protocol), trace.candidates.size());
    out.push_back({commHash, buf, true});
  }
  for (const auto& reason : ringGraphReasons(result)) out.push_back({commHash, reason, true});
  if (result.stepMismatches)
    out.push_back({commHash, std::to_string(result.stepMismatches) + " ring edge(s) with step mismatch", false});
  if (result.inflightWarnings)
    out.push_back({commHash, std::to_string(result.inflightWarnings) + " ring edge(s) with head/tail imbalance", true});
  if (result.uncheckedEdges)
    out.push_back({commHash, std::to_string(result.uncheckedEdges) +
                            " legacy edge check(s) incomplete: missing peer data or unreadable head/tail/step", true});
  return result.status();
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
  int nUnknown = 0;
  for (int r = 0; r < nRanks; r++) {
    struct rasDiagRankHeader* rh = ranks[r].rh;
    struct rasDiagChannel* dc = rasDiagFirstChannel(rh);
    for (int c = 0; c < rh->nChannels; c++) {
      if (!dc->devHangReportedKernelIdxValid) {
        nUnknown++;
      } else if (dc->devHangReportedKernelIdx != 0) {
        agg[dc->devHangReportedKernelIdx][rh->commRank]++;
      }
      dc = rasDiagNextChannel(dc);
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

int analyzerSelectComm(struct rasCollComms* blob, int nBlobBytes, uint64_t commHash) {
  if (blob == nullptr || nBlobBytes < (int)sizeof(*blob) || blob->nComms < 0) return -1;
  if (commHash == 0) return nBlobBytes;
  size_t readOffset = sizeof(*blob), writeOffset = sizeof(*blob);
  int selected = 0;
  for (int i = 0; i < blob->nComms; i++) {
    if (sizeof(rasCollComms::comm) > (size_t)nBlobBytes - readOffset) return -1;
    auto* comm = (struct rasCollComms::comm*)((char*)blob + readOffset);
    if (comm->nValidRanks < 0 || comm->nMissingRanks < 0 || comm->nDiagBytes < 0) return -1;
    size_t bytes = rasCollCommsCommSize(comm);
    if (bytes > (size_t)nBlobBytes - readOffset) return -1;
    if (comm->commId.commHash == commHash) {
      // Keep selected entries even when their diagnostic tier is missing.
      memmove((char*)blob + writeOffset, comm, bytes);
      writeOffset += bytes;
      selected++;
    }
    readOffset += bytes;
  }
  blob->nComms = selected;
  return (int)writeOffset;
}

int analyzerFormatSummary(FILE* fp, const struct rasCollComms* blob) {
  if (fp == nullptr || blob == nullptr) return -1;
  std::vector<analyzerSummaryMessage> byCategory[ANALYZER_CAT_COUNT];
  int errorComms[ANALYZER_CAT_COUNT] = {};
  int warnComms[ANALYZER_CAT_COUNT] = {};
  std::vector<ringGraphStatus> ringStatuses;

  const struct rasCollComms::comm* comm = blob->comms;
  int analyzed = 0;
  for (int i = 0; i < blob->nComms; i++) {
    if (comm->nDiagBytes > 0) {
      std::vector<analyzerRankEntry> ranksVec(comm->nRanks);
      analyzerRankEntry* ranks = ranksVec.data();
      int nRanks = analyzerBuildDiagIndex(comm, ranks, comm->nRanks);
      uint64_t h = comm->commId.commHash;
      ringStatuses.push_back(analyzerCollectRingGraph(h, ranks, nRanks, comm->nRanks, byCategory[ANALYZER_CAT_RING_GRAPH]));
      if (nRanks > 0) {
        size_t previousSizes[ANALYZER_CAT_COUNT];
        for (int cat = 0; cat < ANALYZER_CAT_COUNT; cat++) previousSizes[cat] = byCategory[cat].size();
        analyzerCollectConnType(h, ranks, nRanks, byCategory[ANALYZER_CAT_CONN_TYPE]);
        analyzerCollectConsistency(h, ranks, nRanks, comm->nRanks, byCategory[ANALYZER_CAT_CONSISTENCY]);
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
    } else {
      ringStatuses.push_back(RING_GRAPH_INCONCLUSIVE);
      byCategory[ANALYZER_CAT_RING_GRAPH].push_back({comm->commId.commHash, "[INCONCLUSIVE] no rank diagnostic snapshots", true});
    }
    comm = rasCollCommsCommNext(comm);
  }

  if (analyzed == 0 && ringStatuses.empty()) {
    fprintf(fp, "(no comms carried diag tier; INCLUDE_DIAG flag may not have been set)\n\n");
    return 0;
  }
  for (int cat = 0; cat < ANALYZER_CAT_COUNT; cat++) {
    auto& findings = byCategory[cat];
    if (cat == ANALYZER_CAT_RING_GRAPH) {
      ringGraphStatus status = RING_GRAPH_PASS;
      for (auto commStatus : ringStatuses) status = std::max(status, commStatus);
      int affectedComms = std::count(ringStatuses.begin(), ringStatuses.end(), status);
      if (status == RING_GRAPH_PASS) fprintf(fp, "# ringgraph PASS\n");
      else fprintf(fp, "# ringgraph %d/%zu comms %s\n", affectedComms, ringStatuses.size(), ringGraphStatusName(status));
      for (const auto& finding : findings)
        fprintf(fp, "  0x%lx : %s\n", (unsigned long)finding.commHash, finding.brief.c_str());
      continue;
    }
    if (analyzed == 0) continue;
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
