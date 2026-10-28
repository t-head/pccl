/**************************************************************************
* SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
*
* SPDX-License-Identifier: Apache-2.0
*
* See LICENSE.txt for more license information
**************************************************************************/

/*************************************************************************
 * Wire-format types shared between libpccl.so and the hang-analyzer
 * plugin / pcclras offline tool.
 *
 * Server builds (libpccl.so) include this after socket.h, so
 * ncclSocketAddress is already defined (NCCL_SOCKET_H_).  Standalone
 * client builds (pcclras, ext-hang-analyzer) include this directly and
 * get a self-contained ncclSocketAddress fallback.
 ************************************************************************/

#ifndef NCCL_RAS_WIRE_H_
#define NCCL_RAS_WIRE_H_

#include <stdint.h>
#include <stddef.h>

// Client-visible protocol constants (shared between server and client builds).
#define NCCL_RAS_CLIENT_PORT 28028
#define NCCL_RAS_CLIENT_PROTOCOL 2
#define RAS_COLLECTIVE_LEG_TIMEOUT_SEC 5
#define RAS_COLLECTIVE_EXTRA_TIMEOUT_SEC 5

#ifndef NCCL_SOCKET_H_
#include <sys/socket.h>
#include <arpa/inet.h>
union ncclSocketAddress {
  struct sockaddr sa;
  struct sockaddr_in sin;
  struct sockaddr_in6 sin6;
};
#endif

// Wire-format version of the diag block embedded in rasCollComms.  Mismatched versions are
// dropped at the analyzer (with a WARN) so a rolling upgrade across the RAS network is safe.
#define RAS_COMMS_DIAG_WIRE_VERSION 0x00010003u
#define RAS_DIAG_WIRE_VERSION RAS_COMMS_DIAG_WIRE_VERSION
#define RAS_DIAG_CONN_NONE   ((uint8_t)0xFF)

#define RAS_CONN_TYPE_STR_COUNT 8
static const char* const rasConnTypeStr[RAS_CONN_TYPE_STR_COUNT] = {
  "CONN_NORMAL", "CONN_ICN", "CONN_PCIE", "CONN_SHM", "CONN_GDRDMA",
  "CONN_RDMA", "CONN_SOCKET", "CONN_NET_EXT"
};

// Unique communicator identifier.  commHash by itself is definitely not guaranteed to be unique.
// Combined with the two other hashes, the chance is much better...
// All three fields are used for sorting.
struct rasCommId {
  uint64_t commHash;
  uint64_t hostHash, pidHash; // These are the hashes of the *first* rank (comm->peerInfo[0]).
};

// Collective data in RAS_COLL_COMMS responses.
struct rasCollComms {
  int nComms;
  struct comm {
    struct rasCommId commId;
    int nRanks; // >= nValidRanks + nMissingRanks
    int nValidRanks; // Number of elements in the ranks array below, *not* in the communicator.
    int nMissingRanks; // Number of elements in the missingRanks array below.
    int nDiagBytes; // Length in bytes of the optional diag tier following missingRanks (0 when
                    // RAS_COMMS_FLAG_INCLUDE_DIAG was not set on the request).  The diag tier is
                    // a concatenation of per-rank blocks:
                    //   rasDiagRankHeader + N × (rasDiagChannel + M × rasDiagPeer)
    struct rank {
      int commRank;
      int peerIdx; // Index within rasCollective->peers, *not* rasPeers.
      uint64_t collOpCount;
      struct {
        uint32_t initState:4;
        uint32_t asyncError:4;
        uint32_t finalizeCalled:1;
        uint32_t destroyFlag:1;
        uint32_t abortFlag:1;
      } status;
      char hggcDev;
      char hgmlDev;
    } ranks[0]; // Variable length. Sorted by commRank.  Optimized for 1 GPU/process.
    // The ranks array is followed by:
    // struct rasCollCommsMissingRank missingRanks[0]; // Variable length.  Sorted by commRank.
    // alignas(int64_t) char diag[0];                  // nDiagBytes long, opaque to rasCollComms.
  } comms[0]; // Variable length.  Sorted by commId.
};

// Provides info about missing ranks.  An array of these structures can be part of struct rasCollComms above.
// Because the arrays are of variable length, we can't describe them in C.  Aligned to int64_t so that the
// optional diag tier that follows (whose blocks are alignas(int64_t)) lands on an 8-byte boundary.
struct alignas(int64_t) rasCollCommsMissingRank {
  int commRank;
  union ncclSocketAddress addr;
  char hggcDev;
  char hgmlDev;
};

// Walk helpers for the variable-length rasCollComms::comm stream.  The header + ranks +
// missingRanks layout is unchanged; nDiagBytes adds an opaque trailer that only the next-comm
// stride needs to account for.  Anyone reading missingRanks should still use
// (rasCollCommsMissingRank*)(comm->ranks + comm->nValidRanks) -- only walking past a comm requires
// rasCollCommsCommNext().
static inline size_t rasCollCommsCommSize(const struct rasCollComms::comm* c) {
  int nr = (c->nValidRanks >= 0) ? c->nValidRanks : 0;
  int nm = (c->nMissingRanks >= 0) ? c->nMissingRanks : 0;
  int nd = (c->nDiagBytes >= 0) ? c->nDiagBytes : 0;
  return sizeof(*c)
       + (size_t)nr * sizeof(*c->ranks)
       + (size_t)nm * sizeof(struct rasCollCommsMissingRank)
       + (size_t)nd;
}
static inline struct rasCollComms::comm* rasCollCommsCommNext(const struct rasCollComms::comm* c) {
  return (struct rasCollComms::comm*)((char*)c + rasCollCommsCommSize(c));
}
static inline const char* rasCollCommsDiagBytes(const struct rasCollComms::comm* c) {
  return (const char*)(c+1)
       + (size_t)c->nValidRanks * sizeof(*c->ranks)
       + (size_t)c->nMissingRanks * sizeof(struct rasCollCommsMissingRank);
}
static inline char* rasCollCommsDiagBytesMut(struct rasCollComms::comm* c) {
  return (char*)(c+1)
       + (size_t)c->nValidRanks * sizeof(*c->ranks)
       + (size_t)c->nMissingRanks * sizeof(struct rasCollCommsMissingRank);
}

///////////////////////////////////////////////////////////////////////////////
// Per-rank diagnostic tier                                                  //
///////////////////////////////////////////////////////////////////////////////

#ifndef MAXCHANNELS
#define MAXCHANNELS 64
#endif
#ifndef NCCL_MAX_TREE_ARITY
#define NCCL_MAX_TREE_ARITY 3
#endif

// Per-rank diag block header.  Stream is a concatenation of these, each
// followed by nChannels * rasDiagChannel (each chan followed by nPeerEntries
// * rasDiagPeer).  totalBytes spans the entire block (header + chans + peers).
struct alignas(int64_t) rasDiagRankHeader {
  uint32_t totalBytes;
  uint32_t wireVersion;
  int commRank;
  int nChannels;
};

struct alignas(int64_t) rasDiagChannel {
  int channelId;
  int nPeerEntries;
  uint64_t workFifoTail;
  uint64_t devHangReportedKernelIdx;
  uint64_t devCompletedKernelIdx;
  int ringPrev;
  int ringNext;
  int treeUp;
  int treeDown[NCCL_MAX_TREE_ARITY];
  uint8_t ringSendConn;
  uint8_t ringRecvConn;
  uint8_t treeUpConn;
  uint8_t treeDownConn;
  uint8_t fcConnType;
  uint8_t devHangReportedKernelIdxValid;
  uint8_t devCompletedKernelIdxValid;
  uint8_t _pad0;
};

struct alignas(int64_t) rasDiagPeer {
  int peerRank;
  uint8_t connector;     // 0 = coll connector, 1 = p2p connector.
  uint8_t direction;     // 0 = send, 1 = recv.
  uint8_t connected;
  uint8_t _pad;
  uint64_t head;
  uint64_t tail;
  uint64_t devStep;
};

// Walk helpers for the diag stream (used by both writer and reader sides).
static inline struct rasDiagChannel* rasDiagFirstChannel(struct rasDiagRankHeader* rh) {
  return (struct rasDiagChannel*)((char*)rh + sizeof(*rh));
}
static inline struct rasDiagChannel* rasDiagNextChannel(struct rasDiagChannel* dc) {
  int np = (dc->nPeerEntries >= 0) ? dc->nPeerEntries : 0;
  return (struct rasDiagChannel*)((char*)dc + sizeof(*dc) + (size_t)np * sizeof(struct rasDiagPeer));
}
static inline struct rasDiagPeer* rasDiagFirstPeer(struct rasDiagChannel* dc) {
  return (struct rasDiagPeer*)((char*)dc + sizeof(*dc));
}
static inline struct rasDiagRankHeader* rasDiagNextRank(struct rasDiagRankHeader* rh) {
  size_t stride = (rh->totalBytes >= sizeof(*rh)) ? rh->totalBytes : sizeof(*rh);
  return (struct rasDiagRankHeader*)((char*)rh + stride);
}

#endif // NCCL_RAS_WIRE_H_
