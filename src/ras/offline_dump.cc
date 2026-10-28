//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "analyzer.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#define OFFLINE_COMM_DUMP_VERSION_KNOWN 0x20001

namespace {

struct offlinePeer {
  int peerRank;
  uint8_t connector; // 0 = coll, 1 = p2p
  uint8_t direction; // 0 = send, 1 = recv
  uint64_t head;
  uint64_t tail;
  uint64_t devStep;
};

struct offlineChannel {
  int channelId = -1;
  uint64_t workFifoTail = 0;
  uint64_t hangReportedIdx = 0;
  uint8_t hangReportedValid = 0;
  uint64_t completedIdx = 0;
  uint8_t completedValid = 0;
  int ringPrev = -1, ringNext = -1;
  int treeUp = -1;
  int treeDown[NCCL_MAX_TREE_ARITY] = {-1, -1, -1};
  uint8_t ringSendConn = RAS_DIAG_CONN_NONE;
  uint8_t ringRecvConn = RAS_DIAG_CONN_NONE;
  uint8_t treeUpConn = RAS_DIAG_CONN_NONE;
  uint8_t treeDownConn = RAS_DIAG_CONN_NONE;
  std::vector<offlinePeer> peers;
};

struct offlineRank {
  int rank = -1;
  int nRanks = 0;
  int hggcDev = 0;
  bool peersUnavailable = false; // "invalid hggc eit" or "uninitialized channel/peer" seen.
  std::vector<offlineChannel> channels;
};

struct offlineRankFile {
  long long stamp = -1; // Parsed from the name's trailing _YYYYMMDD-HHMMSS; -1 = absent/malformed.
  offlineRank data;
};

struct offlineCommGroup {
  int nRanks = 0; // Comm size as reported by the rank files (max when they disagree).
  std::map<int, std::vector<offlineRankFile>> byRank; // Per rank, oldest dump first.
};

static const char* skipSpace(const char* p) {
  while (*p == ' ' || *p == '\t') p++;
  return p;
}

// Filename stamp: "..._pid-2797788_20260729-152716.txt" -> 20260729152716.
static long long parseStampFromName(const std::string& name) {
  size_t dot = name.rfind('.');
  std::string stem = (dot == std::string::npos) ? name : name.substr(0, dot);
  size_t us = stem.rfind('_');
  if (us == std::string::npos) return -1;
  std::string digits;
  for (char c : stem.substr(us + 1)) {
    if (c >= '0' && c <= '9') digits += c;
    else if (c != '-') return -1;
  }
  if (digits.size() != 14) return -1;
  return atoll(digits.c_str());
}

// Accepts "YYYYMMDD-HHMMSS" and shorter prefixes (down to the bare date);
// shorter inputs are zero-padded, so "20260729-1527" targets 15:27:00.
static bool parseTimeArg(const char* s, long long* out) {
  if (s == nullptr) return false;
  std::string digits;
  for (const char* p = s; *p != '\0'; p++) {
    if (*p >= '0' && *p <= '9') digits += *p;
    else if (*p != '-' && *p != ':' && *p != ' ') return false;
  }
  if (digits.size() < 8 || digits.size() > 14) return false;
  while (digits.size() < 14) digits += '0';
  *out = atoll(digits.c_str());
  return true;
}

static void formatStamp(long long stamp, char* buf, size_t n) {
  snprintf(buf, n, "%08lld-%06lld", stamp / 1000000LL, stamp % 1000000LL);
}

// "rank:0, nRanks:1152, hggcDev:0, commHash:b7fc683c18db46d9, comm:0x..., devComm:0x... stateMonitorLevel:1"
static bool parseRankLine(const char* p, offlineRank* r, uint64_t* commHash) {
  unsigned long long hash = 0;
  int rank = -1, nRanks = 0, hggcDev = 0;
  if (sscanf(p, "rank:%d, nRanks:%d, hggcDev:%d, commHash:%llx",
             &rank, &nRanks, &hggcDev, &hash) != 4)
    return false;
  if (rank < 0 || nRanks <= 0) return false;
  r->rank = rank;
  r->nRanks = nRanks;
  r->hggcDev = hggcDev;
  *commHash = (uint64_t)hash;
  return true;
}

// "1: send connect(1), recv connect(1)" — one line per (peer, connector); the
// matching send/recv step lines below it carry the connector identity.  The
// writer prints a step line for every connected leg, so nConnected is the
// number of step lines this connect line promises.
static bool parsePeerConnectLine(const char* p, int* peerRank, int* nConnected) {
  int sendConn, recvConn;
  bool ok = sscanf(p, "%d: send connect(%d), recv connect(%d)", peerRank, &sendConn, &recvConn) == 3;
  if (ok) *nConnected = (sendConn == 1) + (recvConn == 1);
  return ok;
}

// "step:                   2, head:                   2, tail:                    0, headPtr:..., tailPtr: ..., connector: coll"
static bool parseStepLine(const char* p, offlinePeer* peer) {
  long long step = 0, head = 0, tail = 0;
  if (sscanf(p, "step:%lld, head:%lld, tail:%lld", &step, &head, &tail) != 3) return false;
  const char* conn = strstr(p, "connector:");
  if (conn == nullptr) return false;
  conn += strlen("connector:");
  while (*conn == ' ') conn++;
  if (strncmp(conn, "coll", 4) == 0)      peer->connector = 0;
  else if (strncmp(conn, "p2p", 3) == 0)   peer->connector = 1;
  else return false;
  peer->devStep = (uint64_t)step;
  peer->head = (uint64_t)head;
  peer->tail = (uint64_t)tail;
  return true;
}

// "channelKernelIdx:5  lastTimeoutKernelIndex:5 completeKernelIndex:4 completePerKernelBatchCounter:0"
// Variants: "completeCount:N" (stateMonitorLevel 2), "unknownDevIndex" (completed-index
// read failed) and "unknownTimeoutKernelIndex" (hang-reported read failed).
static bool parseChannelStateLine(const char* p, offlineChannel* ch) {
  const char* value = skipSpace(p + strlen("channelKernelIdx:"));
  if (*value < '0' || *value > '9') return false;
  char* end = nullptr;
  errno = 0;
  unsigned long long v = strtoull(value, &end, 10);
  if (errno == ERANGE || (*end != '\0' && *end != ' ' && *end != '\t' && *end != '\n' && *end != '\r'))
    return false;
  ch->workFifoTail = v;
  const char* q;
  // lastTimeoutKernelIndex is printed for every successful read (including
  // zero); a failed read prints "unknownTimeoutKernelIndex" instead.  Absence
  // (older writers) means "no hang OR unreadable".
  if ((q = strstr(p, "lastTimeoutKernelIndex:")) != nullptr &&
      sscanf(q, "lastTimeoutKernelIndex:%llu", &v) == 1) {
    ch->hangReportedIdx = v;
    ch->hangReportedValid = 1;
  }
  if (strstr(p, "unknownTimeoutKernelIndex") != nullptr) ch->hangReportedValid = 0;
  if ((q = strstr(p, "completeKernelIndex:")) != nullptr &&
      sscanf(q, "completeKernelIndex:%llu", &v) == 1) {
    ch->completedIdx = v;
    ch->completedValid = 1;
  } else if ((q = strstr(p, "completeCount:")) != nullptr &&
             sscanf(q, "completeCount:%llu", &v) == 1) {
    ch->completedIdx = v;
    ch->completedValid = 1;
  }
  if (strstr(p, "unknownDevIndex") != nullptr) ch->completedValid = 0;
  return true;
}

// "Ring: prev:1089(CONN_GDRDMA), next:63(CONN_ICN)"
static bool parseRingLine(const char* p, offlineChannel* ch) {
  int prev = -1, next = -1, consumed = 0;
  char prevConn[32] = {0}, nextConn[32] = {0};
  // The trailing %n is the truncation check: a partially written line stops at
  // the last literal ')' of the format, %n is never written, and sscanf's
  // return value alone cannot tell because literal matches do not count in it.
  if (sscanf(p, "Ring: prev:%d(%31[^)]), next:%d(%31[^)])%n",
             &prev, prevConn, &next, nextConn, &consumed) != 4 || consumed == 0)
    return false;
  ch->ringPrev = prev;
  ch->ringNext = next;
  ch->ringRecvConn = (uint8_t)analyzerConnTypeFromString(prevConn, prev);
  ch->ringSendConn = (uint8_t)analyzerConnTypeFromString(nextConn, next);
  return true;
}

// "Tree: up:-1(NONE), down[0]:1(CONN_GDRDMA), down[1]:1024(CONN_GDRDMA), down[2]:-1(NONE), "
static bool parseTreeLine(const char* p, offlineChannel* ch) {
  int up = -1, consumed = 0;
  char upConn[32] = {0};
  if (sscanf(p, "Tree: up:%d(%31[^)])%n", &up, upConn, &consumed) != 2 || consumed == 0)
    return false;
  ch->treeUp = up;
  ch->treeUpConn = (uint8_t)analyzerConnTypeFromString(upConn, up);
  for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
    ch->treeDown[i] = -1;
    char tag[16];
    snprintf(tag, sizeof(tag), "down[%d]:", i);
    const char* q = strstr(p, tag);
    if (q == nullptr) return false;
    int down = -1;
    consumed = 0;
    char downConn[32] = {0};
    if (sscanf(q, "down[%*d]:%d(%31[^)])%n", &down, downConn, &consumed) != 2 || consumed == 0)
      return false;
    ch->treeDown[i] = down;
    // dumpTreeToFile prints the single tree->downConnType for every child;
    // take it from the first child that is actually set (populateDiagChannel parity).
    if (down >= 0 && ch->treeDownConn == RAS_DIAG_CONN_NONE) {
      ch->treeDownConn = (uint8_t)analyzerConnTypeFromString(downConn, down);
    }
  }
  return true;
}

static void pushChannel(std::vector<offlineChannel>* channels, offlineChannel* cur, bool* inChannel) {
  if (*inChannel && cur->channelId >= 0) channels->push_back(std::move(*cur));
  *cur = offlineChannel();
  *inChannel = false;
}

// Parse one rough-tier dump file.  Returns false if it has no usable rank
// header, or if the file ends with zero channels (truncation in practice —
// a post-init comm always dumps at least one channel).  *anomalies counts
// lines that looked like state but failed to parse; *noChannelData marks
// the zero-channel case, *truncatedMidFile a channel or peer record that
// stops before the lines the writer format promises (state lines per
// channel, step lines per connected leg) — both for the caller's skip message.
static bool parseCommDumpFile(FILE* f, offlineRank* out, uint64_t* commHash, int* anomalies,
                              bool* noChannelData, bool* truncatedMidFile) {
  char* line = nullptr;
  size_t cap = 0;
  bool sawRank = false;
  bool inChannel = false;
  bool sawStateLine = false; // channelId: line was followed by a channelKernelIdx: line
  int pendingSteps = 0;     // Step lines promised by connect flags but not yet seen
  // The writer always flushes Ring: and then writes Tree:, so a Ring without
  // its Tree can only mean a crash between the two writes (Ring is fflushed
  // immediately, Tree is still in the stdio buffer) — the typical crash cut.
  bool sawRing = false, sawTree = false;
  // Every channel's writer ends with a peers: section or an unavailability
  // marker; a state/topology section with neither can only be a truncation.
  bool sawPeersSection = false;
  offlineChannel cur;
  int curPeer = -1;

  while (getline(&line, &cap, f) != -1) {
    const char* p = skipSpace(line);
    if (*p == '\0' || *p == '\n' || *p == '\r') continue;

    if (!sawRank) {
      if (strncmp(p, "rank:", 5) == 0) sawRank = parseRankLine(p, out, commHash);
      continue; // "DUMP VERSION:" and anything before the rank line is metadata.
    }

    // Peer-connect lines (and kernel work lines) start with a digit; the
    // kernel work lines ("0: work nThreads:...") match nothing below.  A
    // numeric line that parses as neither invalidates curPeer so a later
    // stray step line cannot attach to the previous peer.
    if (*p >= '0' && *p <= '9') {
      int peerRank = -1;
      int nConnected = 0;
      if (parsePeerConnectLine(p, &peerRank, &nConnected)) {
        if (pendingSteps > 0) { *truncatedMidFile = true; break; }
        curPeer = peerRank;
        pendingSteps = nConnected;
      } else if (strstr(p, "connect(") != nullptr) {
        // A peer header naming connect( that fails to parse is a truncated
        // line, not an unrelated numeric line (kernel work lines never have it).
        *truncatedMidFile = true;
        break;
      } else {
        curPeer = -1;
      }
      continue;
    }

    // Channel start: "channelId:%d" on its own line, or fused with the
    // leading "totalKernelNum:%lu " of the first channel.
    if (strncmp(p, "channelId:", 10) == 0 || strncmp(p, "totalKernelNum:", 15) == 0) {
      const char* q = strstr(p, "channelId:");
      if (q != nullptr) {
        if (inChannel && (!sawStateLine || pendingSteps > 0 || (sawRing && !sawTree) ||
                          !sawPeersSection)) {
          *truncatedMidFile = true;
          break;
        }
        pushChannel(&out->channels, &cur, &inChannel);
        int channelId = -1;
        if (sscanf(q, "channelId:%d", &channelId) == 1 && channelId >= 0) {
          cur.channelId = channelId;
          inChannel = true;
        }
        sawStateLine = false;
        pendingSteps = 0;
        sawRing = sawTree = false;
        sawPeersSection = false;
        curPeer = -1;
      }
      continue;
    }

    if (strncmp(p, "channelKernelIdx:", 17) == 0) {
      if (inChannel) {
        if (!parseChannelStateLine(p, &cur)) { *truncatedMidFile = true; break; }
        sawStateLine = true;
      }
      continue;
    }
    if (strncmp(p, "Ring:", 5) == 0) {
      if (inChannel) {
        if (!parseRingLine(p, &cur)) { *truncatedMidFile = true; break; }
        sawRing = true;
      }
      continue;
    }
    if (strncmp(p, "peers:", 6) == 0) {
      sawPeersSection = true;
      continue;
    }
    if (strstr(p, "no valid peers head&tail") != nullptr) {
      out->peersUnavailable = true;
      sawPeersSection = true;
      continue;
    }
    if (strncmp(p, "Tree:", 5) == 0) {
      if (inChannel) {
        if (!parseTreeLine(p, &cur)) { *truncatedMidFile = true; break; }
        sawTree = true;
      }
      continue;
    }
    if (strncmp(p, "send step:", 10) == 0 || strncmp(p, "recv step:", 10) == 0) {
      if (!inChannel || curPeer < 0) { (*anomalies)++; continue; }
      offlinePeer pe{};
      pe.peerRank = curPeer;
      pe.direction = (p[0] == 's') ? 0 : 1;
      if (parseStepLine(p + 5, &pe)) { cur.peers.push_back(pe); if (pendingSteps > 0) pendingSteps--; }
      else (*anomalies)++;
      continue;
    }
  }
  free(line);
  if (*truncatedMidFile || ferror(f) ||
      (inChannel && (!sawStateLine || pendingSteps > 0 || (sawRing && !sawTree) ||
                     !sawPeersSection))) {
    *truncatedMidFile = true;
    return false;
  }
  pushChannel(&out->channels, &cur, &inChannel);
  if (sawRank && out->channels.empty()) {
    *noChannelData = true;
    return false;
  }
  return sawRank;
}

// The ring (and tree) is only reported when at least one coll leg of the channel
// is connected, matching what an online snapshot of the same state would carry.
static void fillDiagChannel(const offlineChannel& oc, struct rasDiagChannel* dc) {
  memset(dc, 0, sizeof(*dc));
  dc->channelId = oc.channelId;
  dc->workFifoTail = oc.workFifoTail;
  dc->devHangReportedKernelIdx = oc.hangReportedIdx;
  dc->devHangReportedKernelIdxValid = oc.hangReportedValid;
  dc->devCompletedKernelIdx = oc.completedIdx;
  dc->devCompletedKernelIdxValid = oc.completedValid;

  bool collEstablished = false;
  for (const offlinePeer& pe : oc.peers) {
    if (pe.connector != 0) continue;
    if (pe.direction == 0 && pe.peerRank == oc.ringNext) collEstablished = true;
    if (pe.direction == 1 && pe.peerRank == oc.ringPrev) collEstablished = true;
  }
  bool hasRing = collEstablished && (oc.ringPrev >= 0 || oc.ringNext >= 0);
  bool hasTree = hasRing;

  if (hasRing) {
    dc->ringPrev = oc.ringPrev;
    dc->ringNext = oc.ringNext;
    dc->ringSendConn = oc.ringSendConn;
    dc->ringRecvConn = oc.ringRecvConn;
  } else {
    dc->ringPrev = -1;
    dc->ringNext = -1;
    dc->ringSendConn = RAS_DIAG_CONN_NONE;
    dc->ringRecvConn = RAS_DIAG_CONN_NONE;
  }
  if (hasTree) {
    dc->treeUp = oc.treeUp;
    dc->treeUpConn = oc.treeUpConn;
    bool hasTreeDown = false;
    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
      dc->treeDown[i] = oc.treeDown[i];
      if (oc.treeDown[i] >= 0) hasTreeDown = true;
    }
    dc->treeDownConn = hasTreeDown ? oc.treeDownConn : RAS_DIAG_CONN_NONE;
  } else {
    dc->treeUp = -1;
    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) dc->treeDown[i] = -1;
    dc->treeUpConn = RAS_DIAG_CONN_NONE;
    dc->treeDownConn = RAS_DIAG_CONN_NONE;
  }
  dc->fcConnType = RAS_DIAG_CONN_NONE; // Not carried by the rough dump; NONE passes the conntype check.
}

// Must mirror buildBlobFromGroup's write caps exactly (channel and peer
// truncation); otherwise nDiagBytes over-reports the bytes actually written.
static size_t diagRankSize(const offlineRank& r) {
  size_t sz = sizeof(struct rasDiagRankHeader);
  size_t nChannels = std::min<size_t>(r.channels.size(), MAXCHANNELS);
  for (size_t c = 0; c < nChannels; c++) {
    const offlineChannel& oc = r.channels[c];
    sz += sizeof(struct rasDiagChannel);
    sz += std::min<size_t>(oc.peers.size(), 1000000) * sizeof(struct rasDiagPeer);
  }
  return sz;
}

static struct rasCollComms* buildBlobFromGroup(uint64_t commHash,
                                               const std::map<int, offlineRank>& byRank,
                                               int nRanks) {
  typedef struct rasCollComms::comm::rank blobRank_t;
  int nValid = (int)byRank.size();
  size_t diagBytes = 0;
  for (const auto& kv : byRank) diagBytes += diagRankSize(kv.second);
  size_t total = sizeof(struct rasCollComms)
               + sizeof(struct rasCollComms::comm)
               + (size_t)nValid * sizeof(blobRank_t)
               + diagBytes;
  if (diagBytes == 0 || total > (size_t)INT32_MAX) return nullptr;

  char* buf = (char*)calloc(1, total);
  if (buf == nullptr) return nullptr;

  auto* blob = (struct rasCollComms*)buf;
  blob->nComms = 1;
  auto* comm = (struct rasCollComms::comm*)(buf + sizeof(struct rasCollComms));
  comm->commId.commHash = commHash;
  comm->nRanks = nRanks > 0 ? nRanks : nValid;
  comm->nValidRanks = nValid;
  comm->nMissingRanks = 0;
  comm->nDiagBytes = (int)diagBytes;

  auto* ranksArr = (blobRank_t*)((char*)comm + sizeof(*comm));
  char* cursor = (char*)ranksArr + (size_t)nValid * sizeof(blobRank_t);
  int i = 0;
  for (const auto& kv : byRank) {
    const offlineRank& orank = kv.second;
    ranksArr[i].commRank = orank.rank;
    ranksArr[i].hggcDev = (char)orank.hggcDev;
    i++;

    auto* rh = (struct rasDiagRankHeader*)cursor;
    rh->wireVersion = RAS_DIAG_WIRE_VERSION;
    rh->commRank = orank.rank;
    rh->nChannels = (int)std::min<size_t>(orank.channels.size(), MAXCHANNELS);
    cursor += sizeof(*rh);
    for (size_t c = 0; c < (size_t)rh->nChannels; c++) {
      const offlineChannel& oc = orank.channels[c];
      auto* dc = (struct rasDiagChannel*)cursor;
      fillDiagChannel(oc, dc);
      cursor += sizeof(*dc);
      size_t nPeers = std::min<size_t>(oc.peers.size(), 1000000);
      for (size_t pIdx = 0; pIdx < nPeers; pIdx++) {
        const offlinePeer& pe = oc.peers[pIdx];
        auto* dp = (struct rasDiagPeer*)cursor;
        dp->peerRank = pe.peerRank;
        dp->connector = pe.connector;
        dp->direction = pe.direction;
        dp->connected = 1;
        dp->head = pe.head;
        dp->tail = pe.tail;
        dp->devStep = pe.devStep;
        cursor += sizeof(*dp);
      }
      dc->nPeerEntries = (int)nPeers;
    }
    rh->totalBytes = (uint32_t)(cursor - (char*)rh);
  }
  return blob;
}

}  // namespace

bool analyzerTimeArgValid(const char* s) {
  long long out = 0;
  return parseTimeArg(s, &out);
}

int analyzerFormatReportFromCommDumps(FILE* fp, const char* dir, int round, const char* timeStr,
                                    int* outFilesParsed) {
  if (fp == nullptr || dir == nullptr) return -1;
  if (outFilesParsed != nullptr) *outFilesParsed = 0;
  if (round < 1) round = 1;
  long long timeTarget = -1;
  if (timeStr != nullptr && !parseTimeArg(timeStr, &timeTarget)) {
    fprintf(stderr, "Invalid time: %s (expected YYYYMMDD-HHMMSS or a shorter prefix)\n", timeStr);
    return -1;
  }
  DIR* dp = opendir(dir);
  if (dp == nullptr) {
    fprintf(stderr, "Cannot open directory: %s (%s)\n", dir, strerror(errno));
    return -1;
  }

  std::vector<std::string> names;
  struct dirent* ent;
  while ((ent = readdir(dp)) != nullptr) {
    if (ent->d_name[0] == '.') continue;
    names.push_back(ent->d_name);
  }
  closedir(dp);
  std::sort(names.begin(), names.end()); // Deterministic; timestamped names sort oldest-first.

  // commHash is derived from the uniqueId alone (not nRanks), so runs of
  // different sizes sharing an ID would otherwise merge into one report.
  std::map<std::pair<uint64_t, int>, offlineCommGroup> groups;
  int nFiles = 0, nParsed = 0, nSkipped = 0, nAnomalies = 0, nVersionWarn = 0;

  for (const std::string& name : names) {
    std::string path = std::string(dir) + "/" + name;
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;

    FILE* f = fopen(path.c_str(), "r");
    if (f == nullptr) { nSkipped++; continue; }
    char first[64];
    if (fgets(first, sizeof(first), f) == nullptr || strncmp(first, "DUMP VERSION:", 13) != 0) {
      fclose(f);
      continue; // Not a rough-tier comm dump (bin blob, report, foreign file, ...).
    }
    nFiles++;

    unsigned ver = 0;
    if (sscanf(first, "DUMP VERSION: %x", &ver) != 1 || ver != OFFLINE_COMM_DUMP_VERSION_KNOWN) {
      fprintf(stderr, "WARN: %s has dump version 0x%x (expected 0x%x), skipped (unsupported version)\n",
              name.c_str(), ver, OFFLINE_COMM_DUMP_VERSION_KNOWN);
      fclose(f);
      nSkipped++;
      nVersionWarn++;
      continue;
    }

    offlineRank orank;
    uint64_t commHash = 0;
    int anomalies = 0;
    bool noChannelData = false;
    bool truncatedMidFile = false;
    if (parseCommDumpFile(f, &orank, &commHash, &anomalies, &noChannelData, &truncatedMidFile)) {
      offlineCommGroup& g = groups[{commHash, orank.nRanks}];
      g.nRanks = std::max(g.nRanks, orank.nRanks);
      g.byRank[orank.rank].push_back({parseStampFromName(name), std::move(orank)}); // Oldest-first.
      nAnomalies += anomalies;
      nParsed++;
    } else {
      nSkipped++;
      if (truncatedMidFile)
        fprintf(stderr, "WARN: %s has incomplete or malformed channel/peer data, skipped\n",
                name.c_str());
      else if (noChannelData)
        fprintf(stderr, "WARN: %s has a rank header but no channel data (truncated or empty dump), skipped\n",
                name.c_str());
      else
        fprintf(stderr, "WARN: %s has no usable rank header, skipped\n", name.c_str());
    }
    fclose(f);
  }

  if (outFilesParsed != nullptr) *outFilesParsed = nParsed;
  if (groups.empty()) return 0;

  fprintf(fp, "\nOffline comm-dump analysis: parsed %d/%d dump file(s) into %zu comm(s)"
              " (skipped %d, %d parse anomalies)\n",
          nParsed, nFiles, groups.size(), nSkipped, nAnomalies);
  if (nVersionWarn > 0)
    fprintf(fp, "  NOTE: %d file(s) skipped for an unsupported dump version\n", nVersionWarn);

  // Filename order is PID-dominated (the pid segment precedes the timestamp),
  // so sort each rank's files by the parsed stamp before applying the round
  // index; undated files first, ties keep the deterministic filename order.
  for (auto& kv : groups) {
    for (auto& rk : kv.second.byRank) {
      std::stable_sort(rk.second.begin(), rk.second.end(),
                       [](const offlineRankFile& a, const offlineRankFile& b) { return a.stamp < b.stamp; });
    }
  }

  int nAnalyzed = 0;
  for (const auto& kv : groups) {
    const offlineCommGroup& g = kv.second;
    int nRoundsMax = 1;
    for (const auto& rk : g.byRank) nRoundsMax = std::max(nRoundsMax, (int)rk.second.size());

    std::map<int, offlineRank> chosen;
    int nShort = 0;
    long long minStamp = -1, maxStamp = -1;
    for (const auto& rk : g.byRank) {
      const std::vector<offlineRankFile>& files = rk.second;
      const offlineRankFile* pick = nullptr;
      if (timeTarget >= 0) {
        for (const offlineRankFile& rf : files) {
          if (rf.stamp < 0) continue;
          long long dNew = llabs(rf.stamp - timeTarget);
          long long dOld = pick ? llabs(pick->stamp - timeTarget) : 0;
          if (pick == nullptr || dNew < dOld || (dNew == dOld && rf.stamp < pick->stamp)) pick = &rf;
        }
      } else {
        if ((int)files.size() < round) { nShort++; continue; }
        pick = &files[files.size() - round];
      }
      if (pick == nullptr) { nShort++; continue; }
      chosen[rk.first] = pick->data;
      if (pick->stamp >= 0) {
        if (minStamp < 0 || pick->stamp < minStamp) minStamp = pick->stamp;
        if (maxStamp < 0 || pick->stamp > maxStamp) maxStamp = pick->stamp;
      }
    }
    if (chosen.empty()) {
      if (timeTarget >= 0)
        fprintf(fp, "\n========== comm 0x%lx : no timestamped dump for time selection, skipped ==========\n",
                (unsigned long)kv.first.first);
      else
        fprintf(fp, "\n========== comm 0x%lx : no rank has a round-%d dump (max %d round(s)), skipped ==========\n",
                (unsigned long)kv.first.first, round, nRoundsMax);
      continue;
    }

    struct rasCollComms* blob = buildBlobFromGroup(kv.first.first, chosen, g.nRanks);
    if (blob == nullptr) {
      fprintf(stderr, "WARN: comm 0x%lx: failed to build diag blob\n", (unsigned long)kv.first.first);
      continue;
    }
    fprintf(fp, "\n========== comm 0x%lx : %d/%d rank dump(s)",
            (unsigned long)kv.first.first, (int)chosen.size(), g.nRanks);
    if (timeTarget >= 0) {
      char tbuf[32], lo[32], hi[32];
      formatStamp(timeTarget, tbuf, sizeof(tbuf));
      formatStamp(minStamp, lo, sizeof(lo));
      formatStamp(maxStamp, hi, sizeof(hi));
      fprintf(fp, " (%d dump round(s), time target %s, chosen %s..%s)", nRoundsMax, tbuf, lo, hi);
    } else if (nRoundsMax > 1) {
      fprintf(fp, " (%d dump round(s), using round %d%s)",
              nRoundsMax, round, round == 1 ? " = newest" : "");
    }
    if (nShort > 0) {
      if (timeTarget >= 0) fprintf(fp, " [WARN %d rank(s) had no timestamped dump, skipped]", nShort);
      else fprintf(fp, " [WARN %d rank(s) had fewer than %d round(s), skipped]", nShort, round);
    }
    fprintf(fp, " ==========\n");
    if (timeTarget >= 0 && minStamp >= 0 && (timeTarget < minStamp || timeTarget > maxStamp))
      fprintf(fp, "  [WARN] target time falls outside the chosen span; it may lie between dump rounds\n");
    int nPeersUnavailable = 0;
    for (const auto& rk : chosen) if (rk.second.peersUnavailable) nPeersUnavailable++;
    if (nPeersUnavailable > 0) {
      fprintf(fp, "  [WARN] %d/%d rank dump(s) have unavailable peer head&tail "
                  "(invalid hggc eit or uninitialized channel/peer): "
                  "ring/p2p edge checks are inconclusive for this comm\n",
              nPeersUnavailable, (int)chosen.size());
    }
    fprintf(fp, "  [NOTE] rough dumps do not carry the fc connection type; fc legs of the ConnType check are unverified\n");
    analyzerFormatReport(fp, blob, /*nLegTimeouts*/0, /*singleCategory*/-1,
                         "PCCL Offline Comm-Dump Analyzer Report");
    free(blob);
    nAnalyzed++;
  }
  if (nAnalyzed == 0 && !groups.empty()) {
    if (timeTarget >= 0) fprintf(stderr, "WARN: no comm has a timestamped dump in %s\n", dir);
    else fprintf(stderr, "WARN: no comm has a round-%d dump in %s\n", round, dir);
  }
  return nAnalyzed;
}
