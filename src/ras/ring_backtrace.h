/**************************************************************************
* SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
*
* SPDX-License-Identifier: Apache-2.0
*
* See LICENSE.txt for more license information
**************************************************************************/

#ifndef PCCL_RING_BACKTRACE_H_
#define PCCL_RING_BACKTRACE_H_

#include "analyzer_internal.h"
#include "pccl_common.h"

#include <string>
#include <vector>

// Uses capture-only existing head counters. A forwarding boundary is not proof
// of its underlying physical cause.
enum ringTraceStatus {
  RING_TRACE_COMPLETED,
  RING_TRACE_SKIPPED,
  RING_TRACE_INCONCLUSIVE,
  RING_TRACE_SUSPECT
};

struct ringTraceCandidate {
  int sourceRank;
  int destinationRank;
  int64_t forwardingGap;
  int64_t nextForwardingGap;
  bool destinationIsForwarder;
};

struct ringTraceResult {
  ringTraceStatus status = RING_TRACE_INCONCLUSIVE;
  int channelId = -1;
  uint64_t kernelIndex = 0;
  int stepAlignment = 1;
  int positiveGapCount = 0;
  uint8_t collective = 0;
  uint8_t protocol = 0;
  std::vector<ringTraceCandidate> candidates;
  std::string reason;
};

// Borrow existing channel/peer records; keep all lookups for a rank together.
struct ringTraceRank {
  const rasDiagChannel* channel;
  const rasDiagPeer* sendPeer;
  const rasDiagPeer* recvPeer;
};

static bool isNetConnection(uint8_t type) {
  return type == CONN_GDRDMA || type == CONN_RDMA;
}

// Compare operation parameters across ranks. Work offsets are rank-local;
// completion and transport are validated separately.
static inline bool isSameRingOperation(const rasDiagRingOperation& first, const rasDiagRingOperation& second) {
  return first.kernelIndex == second.kernelIndex && first.count == second.count &&
         first.lastChunkSize == second.lastChunkSize && first.root == second.root &&
         first.funcId == second.funcId && first.nThreads == second.nThreads && first.nChannels == second.nChannels &&
         first.collective == second.collective && first.protocol == second.protocol;
}

// Resolve existing records and reject inputs outside the Ring head model.
// A null result means the caller can use rankStates and ring directly.
static const char* prepareRingTrace(const analyzerRankEntry* ranks, int nRanks, int expectedRanks, int channelId,
                                   std::vector<ringTraceRank>& rankStates, std::vector<int>& ring, ringTraceStatus& status) {
  if (nRanks != expectedRanks) return "missing ranks";
  if (nRanks < 3) {
    status = RING_TRACE_SKIPPED;
    return "no intermediate forwarding rank";
  }
  // analyzerBuildDiagIndex sorts communicator ranks. A complete set is 0..nRanks-1,
  // so index existing records directly instead of rebuilding a rank hash table.
  rankStates.resize(nRanks);
  for (int rank = 0; rank < nRanks; rank++) {
    auto* channel = findChannelInRank(ranks[rank].rh, channelId);
    if (ranks[rank].rank != rank || !channel) return "missing/duplicate rank or channel";
    if ((unsigned)channel->ringPrev >= (unsigned)nRanks || (unsigned)channel->ringNext >= (unsigned)nRanks) {
      return "broken ring";
    }
    rankStates[rank] = {channel, findPeerInChannel(channel, channel->ringNext, /*send*/ 0, /*coll*/ 0),
                                findPeerInChannel(channel, channel->ringPrev, /*recv*/ 1, /*coll*/ 0)};
  }
  const auto& referenceOperation = rankStates[0].channel->ringOperation;
  if (!referenceOperation.valid || !referenceOperation.kernelIndex) return "no supported single-work native Ring metadata";
  if (referenceOperation.collective >= NCCL_NUM_FUNCTIONS || referenceOperation.protocol >= NCCL_NUM_PROTOCOLS) {
    return "unknown collective/protocol";
  }
  bool hasIncompleteRank = false;
  bool hasDeviceTimeout = false;
  bool timeoutReadsValid = true;
  for (int rankIndex = 0; rankIndex < nRanks; rankIndex++) {
    const ringTraceRank& rankState = rankStates[rankIndex];
    const rasDiagChannel* channel = rankState.channel;
    const auto& operationInfo = channel->ringOperation;
    const auto* send = rankState.sendPeer;
    const auto* recv = rankState.recvPeer;
    bool recvIsNet = isNetConnection(channel->ringRecvConn);
    const uint8_t requiredSendReads = RAS_DIAG_PEER_STEP_VALID | (operationInfo.sendIsNet ? 0 : RAS_DIAG_PEER_HEAD_VALID);
    const uint8_t requiredRecvReads = RAS_DIAG_PEER_STEP_VALID | (recvIsNet ? RAS_DIAG_PEER_HEAD_VALID : 0);
    if (!send || !recv || !send->connected || !recv->connected ||
        (send->readValid & requiredSendReads) != requiredSendReads ||
        (recv->readValid & requiredRecvReads) != requiredRecvReads) {
      return "unreadable head/step snapshot";
    }
    // All ranks must refer to this work; each channel has either completed it
    // or completed the immediately preceding work that used that channel.
    if (!operationInfo.valid || !isSameRingOperation(operationInfo, referenceOperation) ||
        channel->workFifoTail != referenceOperation.kernelIndex || !channel->devCompletedKernelIdxValid ||
        operationInfo.previousKernelIndex >= referenceOperation.kernelIndex ||
        (channel->devCompletedKernelIdx != referenceOperation.kernelIndex &&
         channel->devCompletedKernelIdx != operationInfo.previousKernelIndex)) {
      return "mixed/stale operation metadata or preceding kernel unfinished";
    }
    // PXN keeps an ICN device send type while its actual connector is NET.
    bool sendTransportSupported =
        channel->ringSendConn == CONN_ICN || (operationInfo.sendIsNet && isNetConnection(channel->ringSendConn));
    bool recvTransportSupported = channel->ringRecvConn == CONN_ICN || recvIsNet;
    if (!sendTransportSupported || !recvTransportSupported) return "transport is not native ICN or built-in IB";
    bool hasNetConnection = operationInfo.sendIsNet || recvIsNet;
    if (hasNetConnection &&
        (operationInfo.protocol == NCCL_PROTO_SIGNAL || operationInfo.protocol == NCCL_PROTO_SIGNAL_BULK)) {
      return "Signal protocol over NET is unsupported";
    }
    // Other SimpleBulkAR collectives instantiate an ICN-only send primitive;
    // only AllReduce dispatches its send path by the actual connection type.
    if (hasNetConnection && operationInfo.protocol == NCCL_PROTO_SIMPLE_BULK_AR &&
        operationInfo.collective != ncclFuncAllReduce) {
      return "SimpleBulkAR over NET requires AllReduce";
    }
    hasIncompleteRank |= channel->devCompletedKernelIdx < referenceOperation.kernelIndex;
    auto* otherChannel = rasDiagFirstChannel(ranks[rankIndex].rh);
    for (int c = 0; c < ranks[rankIndex].rh->nChannels; c++, otherChannel = rasDiagNextChannel(otherChannel)) {
      // A kernel can contain distinct works on different channels, even with
      // identical parameters. Borrow a timeout only from the same rank-local work.
      const auto& otherOperation = otherChannel->ringOperation;
      if (otherChannel != channel &&
          (!otherOperation.valid || !operationInfo.workOffset ||
           otherOperation.kernelIndex != operationInfo.kernelIndex ||
           otherOperation.workOffset != operationInfo.workOffset)) continue;
      bool channelStateValid = otherChannel->devHangReportedKernelIdxValid && otherChannel->devCompletedKernelIdxValid;
      timeoutReadsValid &= channelStateValid;
      hasDeviceTimeout |= channelStateValid &&
                          otherChannel->devHangReportedKernelIdx == referenceOperation.kernelIndex &&
                          otherChannel->devCompletedKernelIdx < referenceOperation.kernelIndex;
    }
  }
  if (!hasIncompleteRank) {
    status = RING_TRACE_COMPLETED;
    return "operation completed";
  }
  if (!hasDeviceTimeout) {
    if (!timeoutReadsValid) return "unreadable device timeout/completion snapshot";
    status = RING_TRACE_SKIPPED;
    return "no device timeout for this operation";
  }
  if ((unsigned)referenceOperation.root >= (unsigned)nRanks) return "missing root";
  // Build the logical Ring in traversal order and verify reciprocal edges.
  int source = referenceOperation.collective == ncclFuncReduce
                   ? rankStates[referenceOperation.root].channel->ringNext
                   : referenceOperation.root;
  int rank = source;
  for (int i = 0; i < nRanks; i++) {
    const auto* sourceChannel = rankStates[rank].channel;
    int next = sourceChannel->ringNext;
    const auto* destinationChannel = rankStates[next].channel;
    // Reciprocal edges form cycles. Excluding an early return to source makes
    // this nRanks-step traversal a single complete Ring; no visited set is needed.
    if ((i > 0 && rank == source) || destinationChannel->ringPrev != rank) return "broken ring";
    ring.push_back(rank);
    bool matchingEdgeTransport = sourceChannel->ringOperation.sendIsNet ? isNetConnection(destinationChannel->ringRecvConn)
                                                                        : destinationChannel->ringRecvConn == CONN_ICN;
    if (!matchingEdgeTransport) return "inconsistent edge transport types";
    rank = next;
  }
  return nullptr;
}

// Locate candidate forwarding stalls for one channel from captured single-work snapshots:
// 1. Require all ranks to describe the same supported native Ring operation, with
//    readable counters and valid completion state. At least one rank
//    must be unfinished on this channel, and a device timeout must exist for the operation.
// 2. Follow ringNext to build the Ring order; verify closure, reciprocal neighbors and
//    matching transport types on every edge.
// 3. Read consumption from source.send.head for ICN, or destination.recv.head for NET/PXN.
//    These are existing snapshot fields; this function performs no device reads.
// 4. Subtract the local saved recv/send steps, aligned to the protocol's chunk step count,
//    from incoming/outgoing consumption. Their difference is the forwarding gap;
//    reject gaps outside the algorithm and transport bounds.
// 5. For Broadcast/Reduce, walk backward from the chain terminal to the first positive
//    gap. For AllGather/ReduceScatter/AllReduce, select each positive-gap rank followed
//    by a non-positive-gap rank. Report the outgoing edge of each selected rank.
// Selected edges mark downstream boundaries of forwarding backlog.
// Counter stability is not checked, and snapshots are not atomic across ranks.
// Candidates can include transient backlog and do not prove a unique physical cause.
static ringTraceResult traceRingHead(const analyzerRankEntry* ranks, int nRanks, int expectedRanks, int channelId) {
  ringTraceResult result;
  result.channelId = channelId;
  std::vector<ringTraceRank> rankStates;
  std::vector<int> ring;
  if (const char* reason = prepareRingTrace(ranks, nRanks, expectedRanks, channelId, rankStates, ring, result.status)) {
    result.reason = reason;
    return result;
  }
  const auto& operation = rankStates[0].channel->ringOperation;
  result.kernelIndex = operation.kernelIndex;
  result.collective = operation.collective;
  result.protocol = operation.protocol;
  const bool isClosedRing = operation.collective >= ncclFuncAllGather;
  // LL/LL128: one step per GenericOp. Simple-family: BC/Reduce use 1,
  // AG/RS/AR use two slices of two steps (one chunk = 4).
  const int chunkSteps = operation.protocol < NCCL_PROTO_SIMPLE || !isClosedRing ? 1 : 4;
  result.stepAlignment = chunkSteps;
  // In an open chain (Broadcast/Reduce), every intermediate recv has a
  // corresponding send before head is returned. In closed AG/RS/AR, the
  // initial send and terminal recv mean completed recv is only a LOWER BOUND
  // on forwarded sends. This avoids pretending every rank is a BC forwarder.
  // Both directions advance equally over this single work, so a paired
  // destructor update preserves their difference. Normalize Simple's initial
  // ROUNDUP before subtracting; constructor credit return is not a data send.
  auto alignedStepBase = [chunkSteps](uint64_t step) -> __int128 {
    return (((__int128)step + chunkSteps - 1) / chunkSteps) * chunkSteps;
  };
  auto consumedHead = [&](int sourceRank) -> uint64_t {
    const ringTraceRank& sourceState = rankStates[sourceRank];
    const rasDiagChannel* channel = sourceState.channel;
    // ICN head is receiver credit in sender-local storage. NET sender head
    // is proxy completion instead: use the destination's local recv head.
    return channel->ringOperation.sendIsNet ? rankStates[channel->ringNext].recvPeer->head : sourceState.sendPeer->head;
  };
  const int icnBufferSteps = 8;
  const int netBufferSteps = 2 * icnBufferSteps;
  std::vector<int64_t> forwardingGaps(nRanks, 0);
  int firstForwarder = isClosedRing ? 0 : 1;
  int forwarderEnd = isClosedRing ? nRanks : nRanks - 1;
  for (int i = firstForwarder; i < forwarderEnd; i++) {
    int previousIndex = (i + nRanks - 1) % nRanks;
    const ringTraceRank& rankState = rankStates[ring[i]];
    // Offsets can be negative after a primitive saves its final step. Only
    // their difference is used, so paired destructor updates cancel out.
    __int128 recvHeadOffset =
        (__int128)consumedHead(ring[previousIndex]) - alignedStepBase(rankState.recvPeer->devStep);
    __int128 sendHeadOffset = (__int128)consumedHead(ring[i]) - alignedStepBase(rankState.sendPeer->devStep);
    __int128 forwardingGap = recvHeadOffset - sendHeadOffset;
    // Closed rings may legitimately lead their own recv by one full chunk.
    // Dedicated IB has two <=8-step windows: GPU->send proxy and posted
    // receive->GPU. IB Isend requires a posted receive (CTS). PXN changes
    // ownership/mapping of the send buffer, without adding another queue.
    // External plugins are rejected at capture rather than assigned this bound.
    const int maxForwardingGap = rankState.channel->ringOperation.sendIsNet ? netBufferSteps : icnBufferSteps;
    if (forwardingGap < (isClosedRing ? -chunkSteps : 0) || forwardingGap > maxForwardingGap) {
      result.reason = "inconsistent head/step snapshots or baseline alignment";
      return result;
    }
    forwardingGaps[i] = (int64_t)forwardingGap;
    result.positiveGapCount += forwardingGap > 0;
  }
  // Keep each boundary's computed evidence for both report formats.
  auto selectBoundary = [&](int i) {
    int next = (i + 1) % nRanks;
    result.candidates.push_back({ring[i], ring[next], forwardingGaps[i], forwardingGaps[next],
                                 isClosedRing || next < nRanks - 1});
  };
  if (!isClosedRing) {
    // BC terminal = root.prev; Reduce terminal = root. The closing edge is unused.
    for (int i = nRanks - 2; i >= 1; i--) {
      if (forwardingGaps[i] > 0) {
        selectBoundary(i);
        break;
      }
    }
  } else {
    // There is no fixed tail. Each non-positive gap ends a positive backlog
    // segment. Select these boundaries directly.
    for (int i = 0; i < nRanks; i++) {
      if (forwardingGaps[i] > 0 && forwardingGaps[(i + 1) % nRanks] <= 0) selectBoundary(i);
    }
  }
  if (result.candidates.empty())
    result.reason =
        "no forwarding boundary; initial-send/partial-step or cyclic wait unresolved (consult legacy step checks)";
  else
    result.status = RING_TRACE_SUSPECT;
  return result;
}

#endif  // PCCL_RING_BACKTRACE_H_
