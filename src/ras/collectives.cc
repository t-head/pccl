/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#define NDEBUG // Comment out during development only!
#include <cassert>
#include <mutex>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "checks.h"
#include "pccl_internal.h"
#include "transport.h"
#include "utils.h"
#include "ras_internal.h"

// The number of recent collectives to keep track of.  Completely arbitrary.
#define COLL_HISTORY_SIZE 64

// An entry in the rasCollHistory array keeping track of recently completed collectives (to make it possible to
// identify and drop duplicates arriving over different links).
struct rasCollHistoryEntry {
  union ncclSocketAddress rootAddr;
  uint64_t rootId;
};

// Array keeping track of recently completed collectives (to avoid infinite loops).  LRU-based replacement.
static struct rasCollHistoryEntry rasCollHistory[COLL_HISTORY_SIZE];
static int nRasCollHistory, rasCollHistNextIdx;

// Monotonically increased to ensure that each collective originating locally has a unique Id.
static uint64_t rasCollLastId;

// Keeping track of ongoing collective operations (apart from broadcasts, which have no response so require
// no such tracking).
struct rasCollective* rasCollectivesHead;
struct rasCollective* rasCollectivesTail;

static ncclResult_t getNewCollEntry(struct rasCollective** pColl);
static ncclResult_t rasLinkSendCollReq(struct rasLink* link, struct rasCollective* coll,
                                       const struct rasCollRequest* req, size_t reqLen, struct rasConnection* fromConn);
static ncclResult_t rasConnSendCollReq(struct rasConnection* conn, const struct rasCollRequest* req, size_t reqLen);
static ncclResult_t rasCollReadyResp(struct rasCollective* coll);
static ncclResult_t rasConnSendCollResp(struct rasConnection* conn,
                                        const union ncclSocketAddress* rootAddr, uint64_t rootId,
                                        const union ncclSocketAddress* peers, int nPeers,
                                        const char* data, int nData, int nLegTimeouts);

static ncclResult_t rasCollConnsInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData);
static ncclResult_t rasCollConnsMerge(struct rasCollective* coll, struct rasMsg* msg);

static ncclResult_t rasCollCommsInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData);
static ncclResult_t rasCollCommsMerge(struct rasCollective* coll, struct rasMsg* msg);
static bool rasCollCommsSkipMissing(const struct rasCollRequest* req, ncclComm_t comm);
static int ncclCommsCompare(const void* p1, const void* p2);
static int peersHashesCompare(const void* p1, const void* p2);
static int peersHashesSearch(const void* k, const void* e);
static int rasCommIdCompare(const void* p1, const void* p2);
static int rasCollCommsMissingRankSearch(const void* k, const void* e);


///////////////////////////////////////////////////////////////////////////////////////
// Functions related to the initialization of collectives and the message exchanges. //
///////////////////////////////////////////////////////////////////////////////////////

// Returns the index of the first available entry in the rasCollectives array, enlarging the array if necessary.
static ncclResult_t getNewCollEntry(struct rasCollective** pColl) {
  struct rasCollective* coll;
  int nRasConns;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECK(ncclCalloc(&coll, 1));

  coll->startTime = clockNano();
  coll->fromConn = nullptr;
  // We are unlikely to use the whole array, but at least we won't need to realloc.
  nRasConns = 0;
  for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next) nRasConns++;
  NCCLCHECKGOTO(ncclCalloc(&coll->fwdConns, nRasConns), ret, fail);

  if (rasCollectivesHead) {
    rasCollectivesTail->next = coll;
    coll->prev = rasCollectivesTail;
    rasCollectivesTail = coll;
  } else {
    rasCollectivesHead = rasCollectivesTail = coll;
  }

  *pColl = coll;
  return ncclSuccess;
fail:
  ncclFree(coll->fwdConns);
  ncclFree(coll);
  return ret;
}

// Initializes a collective request by giving it a unique ID.
void rasCollReqInit(struct rasCollRequest* req) {
  memcpy(&req->rootAddr, &rasNetListeningSocket.addr, sizeof(req->rootAddr));
  req->rootId = ++rasCollLastId;
}

// Sends a collective request message through all regular RAS network connections (effectively, broadcasts it).
// Also used for re-broadcasts (on peers receiving the request over the network).
// Checking for duplicates is the responsibility of the caller.
// For collectives other than broadcasts, initializes a rasCollective structure and fills it with local data,
// in preparation for collective response messages.
// pAllDone indicates on return if the collective operation is already finished, which is unusual, but possible
// in scenarios such as a total of two peers.
// pColl provides on return a pointer to the allocated rasCollective structure to track this collective (unless
// it's a broadcast, which require no such tracking).
ncclResult_t rasNetSendCollReq(const struct rasCollRequest* req, bool* pAllDone,
                               struct rasCollective** pColl, struct rasConnection* fromConn,
                               void (*onComplete)(struct rasCollective*)) {
  struct rasCollective* coll = nullptr;
  struct rasCollRequest* reqMod = (struct rasCollRequest*)req;
  size_t reqLen = 0;
  if (req->type >= RAS_COLL_CONNS) {
    // Keep track of this collective operation so that we can handle the responses appropriately.
    NCCLCHECK(getNewCollEntry(&coll));
    if (pColl)
      *pColl = coll;
    memcpy(&coll->rootAddr, &req->rootAddr, sizeof(coll->rootAddr));
    coll->rootId = req->rootId;
    coll->type = req->type;
    coll->timeout = req->timeout;
    coll->fromConn = fromConn;
    coll->onComplete = onComplete;
    if (ncclCalloc(&coll->peers, 1) == ncclSuccess) {
      memcpy(coll->peers, &rasNetListeningSocket.addr, sizeof(*coll->peers));
      coll->nPeers = 1;
    }

    // Collective-specific initialization of accumulated data (using local data for now).
    if (req->type == RAS_COLL_CONNS)
      (void)rasCollConnsInit(&reqMod, &reqLen, &coll->data, &coll->nData);
    else if (req->type == RAS_COLL_COMMS)
      (void)rasCollCommsInit(&reqMod, &reqLen, &coll->data, &coll->nData);
  } else { // req->type < RAS_COLL_CONNS
    // Add the info to the collective message history.
    nRasCollHistory = std::min(nRasCollHistory+1, COLL_HISTORY_SIZE);
    memcpy(&rasCollHistory[rasCollHistNextIdx].rootAddr, &req->rootAddr,
           sizeof(rasCollHistory[rasCollHistNextIdx].rootAddr));
    rasCollHistory[rasCollHistNextIdx].rootId = req->rootId;
    rasCollHistNextIdx = (rasCollHistNextIdx + 1) % COLL_HISTORY_SIZE;

    // Collective-specific message handling.
    bool done = false;
    if (req->type == RAS_BC_DEADPEER) {
      rasMsgHandleBCDeadPeer(&reqMod, &reqLen, &done);
    } else if (req->type == RAS_BC_DUMP) {
      rasMsgHandleBCDump(&reqMod, &reqLen, &done);
    }
    else if (req->type == RAS_BC_ABORT) {
      rasMsgHandleBCAbort(&reqMod, &reqLen, &done);
    }
    if (done) goto exit;
  } // req->type < RAS_COLL_CONNS

  for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next)
    conn->linkFlag = false;

  (void)rasLinkSendCollReq(&rasNextLink, coll, reqMod, reqLen, fromConn);
  (void)rasLinkSendCollReq(&rasPrevLink, coll, reqMod, reqLen, fromConn);

  if (coll && pAllDone)
    *pAllDone = (coll->nFwdSent == coll->nFwdRecv);
exit:
  if (reqMod != req)
    free(reqMod);
  return ncclSuccess;
}

// Sends the collective message through all connections associated with this link (with the exception of the one
// the message came from, if any).
static ncclResult_t rasLinkSendCollReq(struct rasLink* link, struct rasCollective* coll,
                                       const struct rasCollRequest* req, size_t reqLen,
                                       struct rasConnection* fromConn) {
  for (struct rasLinkConn* linkConn = link->conns; linkConn; linkConn = linkConn->next) {
    if (linkConn->conn && linkConn->conn != fromConn && !linkConn->conn->linkFlag) {
      // We send collective messages through fully established and operational connections only.
      if (linkConn->conn->sock && linkConn->conn->sock->status == RAS_SOCK_READY &&
          !linkConn->conn->experiencingDelays) {
        if (rasConnSendCollReq(linkConn->conn, req, reqLen) == ncclSuccess && coll != nullptr)
          coll->fwdConns[coll->nFwdSent++] = linkConn->conn;
      } // linkConn->conn is fully established and operational.
      linkConn->conn->linkFlag = true;
    } // if (linkConn->conn && linkConn->conn != fromConn && !linkConn->con->linkFlag)
  } // for (linkConn)

  return ncclSuccess;
}

// Sends a collective message down a particular connection.
static ncclResult_t rasConnSendCollReq(struct rasConnection* conn, const struct rasCollRequest* req, size_t reqLen) {
  struct rasMsg* msg = nullptr;
  int msgLen = rasMsgLength(RAS_MSG_COLLREQ) + reqLen;

  NCCLCHECK(rasMsgAlloc(&msg, msgLen));
  msg->type = RAS_MSG_COLLREQ;
  memcpy(&msg->collReq, req, reqLen);

  rasConnEnqueueMsg(conn, msg, msgLen);

  return ncclSuccess;
}

// Handles the RAS_MSG_COLLREQ collective message request on the receiver side.  Primarily deals with duplicates and
// re-broadcasts the message to local peers, though in case of a very limited RAS network it might be done right away,
// in which case it can immediately send the response.
ncclResult_t rasMsgHandleCollReq(struct rasMsg* msg, struct rasSocket* sock) {
  bool allDone = false;
  struct rasCollective* coll = nullptr;
  if (sock->conn == nullptr) {
    INFO(NCCL_RAS, "RAS socket lacks a connection: status %d -- internal error?", sock->status);
    return ncclInternalError;
  }

  // First check if we've already handled this request (through another connection).
  for (int i = 0; i < nRasCollHistory; i++) {
    // In principle we can use i to index the array but we convert it so that we check the most recent entries first.
    int collHistIdx = (rasCollHistNextIdx + COLL_HISTORY_SIZE - 1 - i) % COLL_HISTORY_SIZE;
    if (memcmp(&msg->collReq.rootAddr, &rasCollHistory[collHistIdx].rootAddr, sizeof(msg->collReq.rootAddr)) == 0 &&
        msg->collReq.rootId == rasCollHistory[collHistIdx].rootId) {
      if (msg->collReq.type >= RAS_COLL_CONNS) {
        // Send an empty response so that the sender can account for it.  The non-empty response has already been
        // sent through the connection that we received the request through first.
        NCCLCHECK(rasConnSendCollResp(sock->conn, &msg->collReq.rootAddr, msg->collReq.rootId,
                                      /*peers*/nullptr, /*nPeers*/0, /*data*/nullptr, /*nData*/0, /*nLegTimeouts*/0));
      }
      goto exit;
    }
  } // for (i)

  if (msg->collReq.type >= RAS_COLL_CONNS) {
    // Check if we're currently handling this collective request.
    for (coll = rasCollectivesHead; coll; coll = coll->next) {
      if (memcmp(&msg->collReq.rootAddr, &coll->rootAddr, sizeof(msg->collReq.rootAddr)) == 0 &&
          msg->collReq.rootId == coll->rootId) {
        if (msg->collReq.type == coll->type) {
          INFO(NCCL_RAS, "RAS found a duplicate ongoing collective");
          // Send an empty response so that the sender can account for it.  The non-empty response will be
          // sent through the connection that we received the request through first.
          NCCLCHECK(rasConnSendCollResp(sock->conn, &msg->collReq.rootAddr, msg->collReq.rootId,
                                        /*peers*/nullptr, /*nPeers*/0, /*data*/nullptr, /*nData*/0, /*nLegTimeouts*/0));
          goto exit;
        } else {
          // Should never happen.
          INFO(NCCL_RAS, "RAS collective type mismatch: request %d, ongoing %d -- internal error?", msg->collReq.type,
               coll->type);
        }
      } // if match
    } // for (coll)
  } // if (msg->collReq.type >= RAS_COLL_CONNS)

  // Re-broadcast the message to my peers (minus the one it came from) and handle it locally.
  NCCLCHECK(rasNetSendCollReq(&msg->collReq, &allDone, &coll, sock->conn));

  if (msg->collReq.type >= RAS_COLL_CONNS && allDone) {
    if (coll) {
      // We are a leaf process -- send the response right away.  This can probably trigger only for the case of a total
      // of two peers, and hence just one RAS connection, or during communication issues, because normally every peer
      // has more than one connection so there should always be _some_ other peer to forward the request to.
      NCCLCHECK(rasCollReadyResp(coll));
    } else {
      // Should never happen.
      INFO(NCCL_RAS, "RAS collective lacks a list entry -- internal error?");
    }
  }
exit:
  return ncclSuccess;
}

// Sends a collective response back to the process we received the collective request from.
// Invoked when we are finished waiting for the collective responses from other peers (i.e., either there weren't
// any peers (unlikely), the peers sent their responses (likely), or we timed out.
static ncclResult_t rasCollReadyResp(struct rasCollective* coll) {
  if (coll->fromConn) {
    // For remotely-initiated collectives, send the response back.
    NCCLCHECK(rasConnSendCollResp(coll->fromConn, &coll->rootAddr, coll->rootId,
                                  coll->peers, coll->nPeers, coll->data, coll->nData, coll->nLegTimeouts));

    // Add the identifying info to the collective message history.
    nRasCollHistory = std::min(nRasCollHistory+1, COLL_HISTORY_SIZE);
    memcpy(&rasCollHistory[rasCollHistNextIdx].rootAddr, &coll->rootAddr,
           sizeof(rasCollHistory[rasCollHistNextIdx].rootAddr));
    rasCollHistory[rasCollHistNextIdx].rootId = coll->rootId;
    rasCollHistNextIdx = (rasCollHistNextIdx + 1) % COLL_HISTORY_SIZE;

    rasCollFree(coll);
  } else if (coll->onComplete) {
    // RAS-thread-internal collective (e.g., hang analyzer triggered from the state monitor):
    // hand off to the registered callback, which owns the rasCollective and is responsible for
    // calling rasCollFree on it when done.
    coll->onComplete(coll);
  } else {
    // For locally-initiated collectives, invoke the client code again (which will release it, once finished).
    NCCLCHECK(rasClientResume(coll));
  }
  return ncclSuccess;
}

// Sends a collective response via the connection we originally received the request from.  The message should be
// a cumulative response from this process and all the processes that we forwarded the request to.
static ncclResult_t rasConnSendCollResp(struct rasConnection* conn,
                                        const union ncclSocketAddress* rootAddr, uint64_t rootId,
                                        const union ncclSocketAddress* peers, int nPeers,
                                        const char* data, int nData, int nLegTimeouts) {
  struct rasMsg* msg = nullptr;
  int msgLen = rasMsgLength(RAS_MSG_COLLRESP) + nPeers*sizeof(*peers);
  int dataOffset = 0;

  if (nData > 0) {
    ALIGN_SIZE(msgLen, alignof(int64_t));
    dataOffset = msgLen;
    msgLen += nData;
  }

  NCCLCHECK(rasMsgAlloc(&msg, msgLen));
  msg->type = RAS_MSG_COLLRESP;
  memcpy(&msg->collResp.rootAddr, rootAddr, sizeof(msg->collResp.rootAddr));
  msg->collResp.rootId = rootId;
  msg->collResp.nLegTimeouts = nLegTimeouts;
  msg->collResp.nPeers = nPeers;
  msg->collResp.nData = nData;
  if (nPeers)
    memcpy(msg->collResp.peers, peers, nPeers*sizeof(*msg->collResp.peers));
  if (nData)
    memcpy(((char*)msg)+dataOffset, data, nData);

  rasConnEnqueueMsg(conn, msg, msgLen);

  return ncclSuccess;
}

// Handles the collective response on the receiver side.  Finds the corresponding rasCollective structure, merges
// the data from the response into the accumulated data.  If all the responses have been accounted for, sends the
// accumulated response back.
ncclResult_t rasMsgHandleCollResp(struct rasMsg* msg, struct rasSocket* sock) {
  struct rasCollective* coll;
  char line[SOCKET_NAME_MAXLEN+1];

  for (coll = rasCollectivesHead; coll; coll = coll->next) {
    if (memcmp(&msg->collResp.rootAddr, &coll->rootAddr, sizeof(msg->collResp.rootAddr)) == 0 &&
        msg->collResp.rootId == coll->rootId)
      break;
  }
  if (coll == nullptr) {
    INFO(NCCL_RAS, "RAS failed to find a matching ongoing collective for response %s:%ld from %s!",
         ncclSocketToString(&msg->collResp.rootAddr, line), msg->collResp.rootId,
         ncclSocketToString(&sock->sock.addr, rasLine));
    goto exit;
  }

  coll->nLegTimeouts += msg->collResp.nLegTimeouts;
  if (sock->conn == nullptr) {
    INFO(NCCL_RAS, "RAS socket lacks a connection: status %d -- internal error?", sock->status);
    return ncclInternalError;
  }
  // Account for the received response in our collective operations tracking.
  for (int i = 0; i < coll->nFwdSent; i++) {
    if (coll->fwdConns[i] == sock->conn) {
      coll->fwdConns[i] = nullptr;
      break;
    }
  }
  coll->nFwdRecv++;
  if (msg->collResp.nData > 0) {
    // Collective-specific merging of the response into locally accumulated data.
    if (coll->type == RAS_COLL_CONNS)
      NCCLCHECK(rasCollConnsMerge(coll, msg));
    else if (coll->type == RAS_COLL_COMMS)
      NCCLCHECK(rasCollCommsMerge(coll, msg));
  }
  // We merge the peers after merging the data, so that the data merge function can rely on peers being unchanged.
  if (msg->collResp.nPeers > 0) {
    NCCLCHECK(ncclRealloc(&coll->peers, coll->nPeers, coll->nPeers + msg->collResp.nPeers));
    memcpy(coll->peers+coll->nPeers, msg->collResp.peers, msg->collResp.nPeers * sizeof(*coll->peers));
    coll->nPeers += msg->collResp.nPeers;
  }

  // If we received all the data we were waiting for, send our response back.
  if (coll->nFwdSent == coll->nFwdRecv)
    NCCLCHECK(rasCollReadyResp(coll));
exit:
  return ncclSuccess;
}

// Removes a connection from all ongoing collectives.  Called when a connection is experiencing a delay or is being
// terminated.
void rasCollsPurgeConn(struct rasConnection* conn) {
  for (struct rasCollective* coll = rasCollectivesHead; coll;) {
    struct rasCollective* collNext = coll->next;
    char line[SOCKET_NAME_MAXLEN+1];
    if (coll->fromConn == conn) {
      INFO(NCCL_RAS, "RAS purging collective %s:%ld because it comes from %s",
           ncclSocketToString(&coll->rootAddr, line), coll->rootId,
           ncclSocketToString(&conn->addr, rasLine));
      rasCollFree(coll);
    } else {
      for (int i = 0; i < coll->nFwdSent; i++) {
        if (coll->fwdConns[i] == conn) {
          coll->fwdConns[i] = nullptr;
          coll->nFwdRecv++;
          coll->nLegTimeouts++;
          INFO(NCCL_RAS, "RAS not waiting for response from %s to collective %s:%ld "
               "(nFwdSent %d, nFwdRecv %d, nLegTimeouts %d)",
               ncclSocketToString(&conn->addr, rasLine), ncclSocketToString(&coll->rootAddr, line), coll->rootId,
               coll->nFwdSent, coll->nFwdRecv, coll->nLegTimeouts);
          if (coll->nFwdSent == coll->nFwdRecv)
            (void)rasCollReadyResp(coll);
          break;
        }
      } // for (i)
    } // coll->fromConn != conn
    coll = collNext;
  } // for (coll)
}

// Frees a rasCollective entry and any memory associated with it.
void rasCollFree(struct rasCollective* coll) {
  if (coll == nullptr)
    return;

  free(coll->fwdConns);
  free(coll->peers);
  free(coll->data);

  if (coll == rasCollectivesHead)
    rasCollectivesHead = rasCollectivesHead->next;
  if (coll == rasCollectivesTail)
    rasCollectivesTail = rasCollectivesTail->prev;
  if (coll->prev)
    coll->prev->next = coll->next;
  if (coll->next)
    coll->next->prev = coll->prev;
  free(coll);
}

// Invoked from the main RAS thread loop to handle timeouts of the collectives.
// We obviously want to have a reasonable *total* timeout that the RAS client can rely on, but we don't have strict
// global coordination.  So we have, in effect, two timeouts: soft (5s) and hard (10s).  Soft equals the keep-alive
// timeout.
// When sending collective requests, we skip any connections that are experiencing delays.  After the 5s timeout, we
// check again the status of all outstanding connections and if any is now delayed, we give up on it.
// That works fine for directly observable delays, but if the problematic connection is further away from us, all
// we can do is trust that the other peers will "do the right thing soon".  However, if there is a cascade of
// problematic connections, they could still exceed the 5s total.  So after 10s we give up waiting no matter what
// and send back whatever we have.  Unfortunately, the peer that the RAS client is connected to will in all likelihood
// time out first, so at that point any delayed responses that eventually arrive are likely to be too late...
void rasCollsHandleTimeouts(int64_t now, int64_t* nextWakeup) {
  for (struct rasCollective* coll = rasCollectivesHead; coll;) {
    struct rasCollective* collNext = coll->next;
    if (coll->timeout > 0) {
      if (now - coll->startTime > coll->timeout) {
        // We've exceeded the leg timeout.  For all outstanding responses, check their connections.
        if (!coll->timeoutWarned) {
          INFO(NCCL_RAS, "RAS collective %s:%ld timeout warning (%lds) -- %d responses missing",
               ncclSocketToString(&coll->rootAddr, rasLine), coll->rootId,
               (now - coll->startTime) / CLOCK_UNITS_PER_SEC, coll->nFwdSent - coll->nFwdRecv);
          coll->timeoutWarned = true;
        }
        for (int i = 0; i < coll->nFwdSent; i++) {
          if (coll->fwdConns[i]) {
            struct rasConnection* conn = coll->fwdConns[i];
            char line[SOCKET_NAME_MAXLEN+1];
            if (!conn->experiencingDelays && conn->sock) {
              // Ensure that the connection is fully established and operational, and that the socket hasn't been
              // re-created during the handling of the collective (which would suggest that the request may have been
              // lost).
              if (conn->sock->status == RAS_SOCK_READY && conn->sock->createTime < coll->startTime)
                continue;
            }
            // In all other cases we declare a timeout so that we can (hopefully) recover.
            INFO(NCCL_RAS, "RAS not waiting for response from %s to collective %s:%ld "
                 "(nFwdSent %d, nFwdRecv %d, nLegTimeouts %d)",
                 ncclSocketToString(&conn->addr, rasLine), ncclSocketToString(&coll->rootAddr, line),
                 coll->rootId, coll->nFwdSent, coll->nFwdRecv, coll->nLegTimeouts);
            coll->fwdConns[i] = nullptr;
            coll->nFwdRecv++;
            coll->nLegTimeouts++;
          } // if (coll->fwdConns[i])
        } // for (i)
        if (coll->nFwdSent == coll->nFwdRecv) {
          (void)rasCollReadyResp(coll);
        } else {
          // At least some of the delays are *not* due to this process' connections experiencing delays, i.e., they
          // must be due to delays at other processes.  Presumably those processes will give up waiting soon and the
          // (incomplete) responses will arrive shortly, so we should wait a little longer.
          if (now - coll->startTime > coll->timeout + RAS_COLLECTIVE_EXTRA_TIMEOUT) {
            // We've exceeded even the longer timeout, which is unexpected.  Try to return whatever we have (though
            // the originator of the collective, if it's not us, may have timed out already anyway).
            INFO(NCCL_RAS, "RAS collective %s:%ld timeout error (%lds) -- giving up on %d missing responses",
                 ncclSocketToString(&coll->rootAddr, rasLine), coll->rootId,
                 (now - coll->startTime) / CLOCK_UNITS_PER_SEC, coll->nFwdSent - coll->nFwdRecv);
            coll->nLegTimeouts += coll->nFwdSent - coll->nFwdRecv;
            coll->nFwdRecv = coll->nFwdSent;
            (void)rasCollReadyResp(coll);
          } else {
            *nextWakeup = std::min(*nextWakeup, coll->startTime+coll->timeout+RAS_COLLECTIVE_EXTRA_TIMEOUT);
          }
        } // conn->nFwdRecv < conn->nFwdSent
      } else {
        *nextWakeup = std::min(*nextWakeup, coll->startTime+coll->timeout);
      }
    } // if (coll->timeout > 0)

    coll = collNext;
  } // for (coll)
}


/////////////////////////////////////////////////////////////////////////
// Functions related to the handling of the RAS_COLL_CONNS collective. //
/////////////////////////////////////////////////////////////////////////

// Initializes the accumulated data with just the local data for now.
// For this particular collective, we keep some reduced statistical data (min/max/avg travel time) as well
// as connection-specific info in case we observed a negative min travel time (which, ideally, shouldn't happen,
// but the system clocks may not be perfectly in sync).
static ncclResult_t rasCollConnsInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData) {
  struct rasCollConns connsData = {.travelTimeMin = INT64_MAX, .travelTimeMax = INT64_MIN};
  struct rasCollConns* pConnsData;

  *pReqLen = rasCollDataLength(RAS_COLL_CONNS);

  // Update the statistical data first and in the process also calculate how much connection-specific space we
  // will need.
  for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next) {
    if (conn->travelTimeCount > 0) {
      if (connsData.travelTimeMin > conn->travelTimeMin)
        connsData.travelTimeMin = conn->travelTimeMin;
      if (connsData.travelTimeMax < conn->travelTimeMax)
        connsData.travelTimeMax = conn->travelTimeMax;
      connsData.travelTimeSum += conn->travelTimeSum;
      connsData.travelTimeCount += conn->travelTimeCount;
      connsData.nConns++;
      if (conn->travelTimeMin < 0)
        connsData.nNegativeMins++;
    }
  }

  *pNData = sizeof(connsData) + connsData.nNegativeMins*sizeof(*connsData.negativeMins);
  NCCLCHECK(ncclCalloc(pData, *pNData));
  pConnsData = (struct rasCollConns*)*pData;
  memcpy(pConnsData, &connsData, sizeof(*pConnsData));
  if (connsData.nNegativeMins > 0) {
    int negMinsIdx = 0;
    for (struct rasConnection* conn = rasConnsHead; conn; conn = conn->next) {
      if (conn->travelTimeMin < 0) {
        struct rasCollConns::negativeMin* negativeMin = pConnsData->negativeMins+negMinsIdx;
        if (negMinsIdx >= connsData.nNegativeMins) {
          // Should never happen;
          INFO(NCCL_RAS, "RAS overflow of negativeMins: connsData.nNegativeMins %d, negMinsIdx %d -- internal error?",
               connsData.nNegativeMins, negMinsIdx);
          break;
        }
        memcpy(&negativeMin->source, &rasNetListeningSocket.addr, sizeof(negativeMin->source));
        memcpy(&negativeMin->dest, &conn->addr, sizeof(negativeMin->dest));
        negativeMin->travelTimeMin = conn->travelTimeMin;
        negMinsIdx++;
      }
    }
  }

  return ncclSuccess;
}

// Merges incoming collective RAS_COLL_CONNS response message into the local accumulated data.
static ncclResult_t rasCollConnsMerge(struct rasCollective* coll, struct rasMsg* msg) {
  struct rasCollConns* collData;
  struct rasCollConns* msgData;
  int dataOffset = rasMsgLength(RAS_MSG_COLLRESP) + msg->collResp.nPeers*sizeof(*msg->collResp.peers);
  ALIGN_SIZE(dataOffset, alignof(int64_t));

  msgData = (struct rasCollConns*)(((char*)msg) + dataOffset);
  collData = (struct rasCollConns*)coll->data;

  // Merge the stats.
  if (collData->travelTimeMin > msgData->travelTimeMin)
    collData->travelTimeMin = msgData->travelTimeMin;
  if (collData->travelTimeMax < msgData->travelTimeMax)
    collData->travelTimeMax = msgData->travelTimeMax;
  collData->travelTimeSum += msgData->travelTimeSum;
  collData->travelTimeCount += msgData->travelTimeCount;
  collData->nConns += msgData->nConns;

  // Append the info about negative minimums.
  if (msgData->nNegativeMins > 0) {
    int nData = sizeof(*collData) +
      (collData->nNegativeMins+msgData->nNegativeMins) * sizeof(*collData->negativeMins);
    NCCLCHECK(ncclRealloc(&coll->data, coll->nData, nData));
    collData = (struct rasCollConns*)coll->data;
    memcpy(coll->data+coll->nData, msgData->negativeMins,
           msgData->nNegativeMins * sizeof(*collData->negativeMins));
    coll->nData = nData;
    collData->nNegativeMins += msgData->nNegativeMins;
  }

  return ncclSuccess;
}


/////////////////////////////////////////////////////////////////////////
// Functions related to the handling of the RAS_COLL_COMMS collective. //
/////////////////////////////////////////////////////////////////////////

// Initializes the accumulated data with just the local data for now.
// For this particular collective, we keep for every communicator information about every rank, to help identify
// the missing ones and the discrepancies between the ones that did respond.
// For any new (previously unseen) communicator we also save the basic identification data about every rank that is
// "missing" (i.e., not part of this process).  During merging, this should be replaced by the actual data from
// those ranks, if they are responsive.  We want to provide this information to the user (so that we can say more
// than "rank xyz missing").
// Every "new" communicator is also recorded in the (updated) request, so that when that request is forwarded to our
// peers, those peers don't needlessly send us the same data.
static ncclResult_t rasCollCommsInit(struct rasCollRequest** pReq, size_t* pReqLen, char** pData, int* pNData) {
  ncclResult_t ret = ncclSuccess;
  struct rasCollComms* commsData;
  int nComms = 0, nRanks = 0, nMissingRanks = 0;
  bool skipMissing = false;
  std::lock_guard<std::recursive_mutex> lock(ncclCommsMutex);
  struct rasCollComms::comm* comm;
  struct rasCollRequest* req = nullptr;
  struct rasPeerInfo** peersReSorted = nullptr;
  int firstNewSkipMissingIdx = -1;
  bool includeDiag = (((*pReq)->comms.flags & RAS_COMMS_FLAG_INCLUDE_DIAG) != 0);
  uint64_t diagOnly = (*pReq)->comms.diagOnlyCommHash;
  size_t* diagBytesPerComm = nullptr; // Index is the collective comm index; only allocated when includeDiag.
  size_t totalDiagBytes = 0;
  // Item 3: when the synth dump runs, route its device reads through the diag
  // collector so the diag tier is filled as a byproduct (vs a second cold IO pass via
  // rasDiagSnapshotInto).  Stays null when no dump is being triggered, or when
  // creation fails (we just fall back to the cold path).
  struct pcclCommDiagCollector* diagCollector = nullptr;

  *pReqLen = rasCollDataLength(RAS_COLL_COMMS) +
    (*pReq)->comms.nSkipMissingRanksComms * sizeof(*(*pReq)->comms.skipMissingRanksComms);
  *pData = nullptr;

  if (includeDiag) {
    // Pessimistically allocate one slot per ncclComm; only the first nComms slots are used.
    NCCLCHECKGOTO(ncclCalloc(&diagBytesPerComm, nNcclComms + 1), ret, fail);
  }

  // Item 2: RAS_COMMS_FLAG_TRIGGER_LOCAL_DUMP folds what used to be a separate RAS_BC_DUMP
  // broadcast into this collective.  When set, every responder also runs pcclCommDump
  // (rough-by-default) for matching comms via the existing rasStateMonitorDump helper, which
  // shares the same lastTimeoutIndex/seq dedup as the legacy BC_DUMP path.  Honored only
  // with INCLUDE_DIAG; without diag the COMMS collective has no business firing dumps.
  if (includeDiag && ((*pReq)->comms.flags & RAS_COMMS_FLAG_TRIGGER_LOCAL_DUMP) != 0) {
    diagCollector = rasDiagCollectorCreate();
    struct rasCollRequest synthDumpReq;
    memset(&synthDumpReq, 0, sizeof(synthDumpReq));
    synthDumpReq.type = RAS_BC_DUMP;
    synthDumpReq.dump.dumpAllComms = (diagOnly == 0);
    synthDumpReq.dump.dumpLevel = (*pReq)->comms.dumpLevel;
    synthDumpReq.dump.commHash = diagOnly;
    synthDumpReq.dump.seq = (*pReq)->comms.diagSeq;
    bool dumpDone = false;
    rasStateMonitorDump(&synthDumpReq, &dumpDone, diagCollector);
  }

  // Start by counting the communicators so that we know how much space to allocate.
  // We also need to sort the comms array, to make the subsequent merging easier, both between the ranks (in case
  // of multiple GPUs per process) and between the peers.
  if (!ncclCommsSorted) {
    qsort(ncclComms, nNcclComms, sizeof(*ncclComms), &ncclCommsCompare);
    ncclCommsSorted = true;
  }
  for (int commIdx = 0; commIdx < nNcclComms; commIdx++) {
    if (ncclComms[commIdx].comm == nullptr) // nullptr's are always at the end after sorting.
      break;
    if (!__atomic_load_n(&ncclComms[commIdx].comm->peerInfoValid, __ATOMIC_ACQUIRE)) {
      // Critical data is not yet initialized -- ignore the communicator.
      continue;
    }
    // A process may manage multiple GPUs and thus have multiple communicators with the same commHash.
    // Comparing just the commHash is OK though within communicators that are part of the same process.
    if (commIdx == 0 || ncclComms[commIdx].comm->commHash != ncclComms[commIdx-1].comm->commHash) {
      skipMissing = rasCollCommsSkipMissing(*pReq, ncclComms[commIdx].comm);
      if (!skipMissing) {
        // Add this communicator to the request so that the processes we forward the request to know not to fill in
        // the missing rank info.
        struct rasCommId* skipComm;
        if (req == nullptr) {
          // We pessimistically allocate space for all the remaining communicators so that we don't need to reallocate.
          int newSize = *pReqLen + (nNcclComms-commIdx) * sizeof(*req->comms.skipMissingRanksComms);
          NCCLCHECKGOTO(ncclCalloc((char**)&req, newSize), ret, fail);
          memcpy(req, *pReq, *pReqLen);
          *pReq = req;
          firstNewSkipMissingIdx = req->comms.nSkipMissingRanksComms;
        }
        skipComm = req->comms.skipMissingRanksComms + req->comms.nSkipMissingRanksComms++;
        skipComm->commHash = ncclComms[commIdx].comm->commHash;
        skipComm->hostHash = ncclComms[commIdx].comm->peerInfo->hostHash;
        skipComm->pidHash = ncclComms[commIdx].comm->peerInfo->pidHash;

        nMissingRanks += ncclComms[commIdx].comm->nRanks;
      } // if (!skipMissing)
      nComms++;
    } // if encountered a new communicator
    nRanks++;
    if (!skipMissing)
      nMissingRanks--;

    if (includeDiag) {
      uint64_t h = ncclComms[commIdx].comm->commHash;
      if (diagOnly == 0 || h == diagOnly) {
        size_t sz = rasDiagTotalSnapshotDataSize(ncclComms[commIdx].comm);
        diagBytesPerComm[nComms - 1] += sz;
        totalDiagBytes += sz;
      }
    }
  } // for (commIdx)

  // rasCollComms has nested variable-length arrays, which makes the size calculation and subsequent
  // pointer manipulations somewhat unwieldy...
  // This is extra complicated because of the "hidden" array of struct rasCollCommsMissingRank following the
  // ranks array for each communicator.
  *pNData = sizeof(*commsData) + nComms * sizeof(*commsData->comms) + nRanks * sizeof(*commsData->comms[0].ranks) +
    nMissingRanks * sizeof(struct rasCollCommsMissingRank) + totalDiagBytes;
  NCCLCHECKGOTO(ncclCalloc(pData, *pNData), ret, fail);
  commsData = (struct rasCollComms*)*pData;
  commsData->nComms = nComms;

  // comm points at the space in the accumulated data where the info about the current communicator is to be stored.
  comm = commsData->comms;
  // collCommIdx counts rasCollComms::comm (comm); commIdx indexes ncclComms.
  for (int collCommIdx = 0, commIdx = 0; collCommIdx < nComms; collCommIdx++) {
    ncclComm_t ncclComm = ncclComms[commIdx].comm;
    if (!__atomic_load_n(&ncclComm->peerInfoValid, __ATOMIC_ACQUIRE))
      continue;

    if ((char*)(comm + 1) - (char*)commsData > *pNData) {
      // Should never happen.
      INFO(NCCL_RAS, "RAS overflow of commsData: collCommIdx %d, nComms %d, pNData %d, needs %td -- internal error?",
           collCommIdx, nComms, *pNData, (char*)(comm + 1) - (char*)commsData);
      break;
    }

    comm->commId.commHash = ncclComm->commHash;
    comm->commId.hostHash = ncclComm->peerInfo->hostHash;
    comm->commId.pidHash = ncclComm->peerInfo->pidHash;
    comm->nRanks = ncclComm->nRanks;
    comm->nValidRanks = comm->nMissingRanks = 0;
    comm->nDiagBytes = 0;
    int rankStartIdx = commIdx; // For walking the same ncclComms range again for the diag tier.

    // Fill in the comm->ranks array.
    for (; commIdx < nNcclComms && ncclComms[commIdx].comm && ncclComms[commIdx].comm->commHash == comm->commId.commHash;
         commIdx++) {
      ncclComm = ncclComms[commIdx].comm;
      struct rasCollComms::comm::rank* rank = comm->ranks+comm->nValidRanks;
      if ((char*)(rank + 1) - (char*)commsData > *pNData) {
        // Should never happen.
        INFO(NCCL_RAS,
             "RAS overflow of commsData: collCommIdx %d, nComms %d, rank %d, commIdx %d, pNData %d, "
             "needs %td -- internal error?",
             collCommIdx, nComms, comm->nValidRanks, commIdx, *pNData, (char*)(rank + 1) - (char*)commsData);
        break;
      }
      rank->commRank = ncclComm->rank;
      // rasNetSendCollReq initializes coll->peers[0] to our rasNetListeningSocket.addr, so peerIdx is initially
      // always 0.  It will increase after we send this response back to the peer we got the request from.
      rank->peerIdx = 0;
      rank->collOpCount = ncclComm->collOpCount;
      rank->status.initState = (uint32_t)ncclComm->initState;
      rank->status.asyncError = (uint32_t)__atomic_load_n(&ncclComm->asyncResult, __ATOMIC_ACQUIRE);
      // TODO: check proxyState asyncResult
      // if (rank->status.asyncError == (uint32_t)ncclSuccess && ncclComm->proxyState)
      //   rank->status.asyncError = (uint32_t)__atomic_load_n(&ncclComm->proxyState->asyncResult, __ATOMIC_ACQUIRE);
      rank->status.finalizeCalled = (ncclComm->finalizeCalled != 0);
      rank->status.destroyFlag = (ncclComm->destroyFlag != 0);
      rank->status.abortFlag = (__atomic_load_n(ncclComm->abortFlag, __ATOMIC_ACQUIRE) != 0);
      rank->hggcDev = ncclComm->hggcDev;
      rank->hgmlDev = ncclComm->hgmlDev;
      comm->nValidRanks++;
    } // for (commIdx)

    if (__atomic_load_n(&ncclComm->peerInfoValid, __ATOMIC_ACQUIRE) && firstNewSkipMissingIdx != -1 &&
        memcmp(req->comms.skipMissingRanksComms+firstNewSkipMissingIdx, &comm->commId, sizeof(comm->commId)) == 0) {
      // Fill in the missingRanks array that follows the comm->ranks.
      struct rasCollCommsMissingRank* missingRanks = (struct rasCollCommsMissingRank*)(comm->ranks+comm->nValidRanks);

      if (peersReSorted == nullptr) {
        // Create a lookup table to rasPeers that is sorted by hostHash and pidHash, to reduce the complexity of the
        // lookups in the missingRankIdx loop below.
        NCCLCHECKGOTO(ncclCalloc(&peersReSorted, nRasPeers), ret, fail);
        for (int peerIdx = 0; peerIdx < nRasPeers; peerIdx++)
          peersReSorted[peerIdx] = rasPeers+peerIdx;
        qsort(peersReSorted, nRasPeers, sizeof(*peersReSorted), peersHashesCompare);
      }

      comm->nMissingRanks = comm->nRanks - comm->nValidRanks;
      for (int missingRankIdx = 0, rankIdx = 0; missingRankIdx < comm->nMissingRanks; missingRankIdx++) {
        struct rasCollCommsMissingRank* missingRank;
        struct ncclPeerInfo* info;
        struct rasPeerInfo** peer;
        uint64_t key[2];
        // Look for the next "hole" in the ranks array.
        while (rankIdx < comm->nValidRanks && comm->ranks[rankIdx].commRank == rankIdx+missingRankIdx)
          rankIdx++;

        missingRank = missingRanks + missingRankIdx;
        if ((char*)(missingRank + 1) - (char*)commsData > *pNData) {
          // Should never happen.
          INFO(NCCL_RAS,
               "RAS overflow of commsData: collCommIdx %d, nComms %d, nValidRanks %d, missingRankIdx %d, "
               "nMissingRanks %d, pNData %d, needs %td -- internal error?",
               collCommIdx, nComms, comm->nValidRanks, missingRankIdx, comm->nMissingRanks, *pNData,
               (char*)(missingRank + 1) - (char*)commsData);
          comm->nMissingRanks = missingRankIdx; // Keep the count in sync with the data actually written.
          break;
        }
        missingRank->commRank = rankIdx + missingRankIdx;
        info = ncclComm->peerInfo + missingRank->commRank;
        key[0] = info->hostHash - ncclComm->commHash;
        key[1] = info->pidHash - ncclComm->commHash;
        peer = (struct rasPeerInfo**)bsearch(key, peersReSorted, nRasPeers, sizeof(*peersReSorted), peersHashesSearch);
        if (peer)
          memcpy(&missingRank->addr, &(*peer)->addr, sizeof(missingRank->addr));
        missingRank->hggcDev = info->hggcDev;
        missingRank->hgmlDev = info->hgmlDev;
      } // for (missingRankIdx)

      if (++firstNewSkipMissingIdx == req->comms.nSkipMissingRanksComms)
        firstNewSkipMissingIdx = -1;
    } // if need to fill in the missingRanks

    // Diag tier (per-rank snapshots) follows the missingRanks array.  We only emit it when the
    // request asked for it (RAS_COMMS_FLAG_INCLUDE_DIAG) and either this is the comm of interest
    // (diagOnly == its commHash) or all comms were requested (diagOnly == 0).  The byte budget
    // for this comm was computed in the first pass and stored in diagBytesPerComm.
    //
    // Item 3: when the synth dump above ran with a collector attached, the per-rank diag bytes
    // were already captured during the dump's device IO -- copy them in from the collector and
    // skip the cold rasDiagSnapshotInto pass.  Per-rank fallback: a rank that wasn't
    // visited by the dump (e.g. dedup'd via lastTimeoutIndex or aborted) gets the cold path.
    if (includeDiag && diagBytesPerComm != nullptr &&
        (diagOnly == 0 || comm->commId.commHash == diagOnly)) {
      char* diagBase = rasCollCommsDiagBytesMut(comm);
      size_t diagWritten = 0;
      size_t diagBudget = diagBytesPerComm[collCommIdx];
      // The overflow guards above keep every write inside commsData, but a comm/rank count drift between
      // the counting and filling passes can leave less room than the budget computed in the counting pass --
      // clamp it to what actually remains.
      size_t diagRoom = (diagBase < (char*)commsData + *pNData ?
                         (size_t)((char*)commsData + *pNData - diagBase) : 0);
      if (diagRoom < diagBudget) {
        INFO(NCCL_RAS, "RAS diag data would overflow commsData: collCommIdx %d, budget %zu, room %zu -- internal error?",
             collCommIdx, diagBudget, diagRoom);
        diagBudget = diagRoom;
      }
      for (int rIdx = rankStartIdx; rIdx < commIdx; rIdx++) {
        ncclComm_t rcomm = ncclComms[rIdx].comm;
        if (rcomm == nullptr) continue;
        if (!__atomic_load_n(&rcomm->peerInfoValid, __ATOMIC_ACQUIRE)) continue;
        size_t sz = 0;
        if (diagCollector != nullptr) {
          size_t got = 0;
          const char* bytes = rasDiagCollectorGet(diagCollector, rcomm, &got);
          if (bytes != nullptr && got > 0 && diagWritten + got <= diagBudget) {
            memcpy(diagBase + diagWritten, bytes, got);
            sz = got;
          }
        }
        if (sz == 0) {
          sz = rasDiagSnapshotInto(diagBase + diagWritten,
                                        diagBudget - diagWritten,
                                        rcomm);
        }
        diagWritten += sz;
      }
      comm->nDiagBytes = (int)diagWritten;
    }

    comm = rasCollCommsCommNext(comm);
  } // for (collCommIdx)

  if (req) {
    // Finish updating the request.
    *pReqLen = rasCollDataLength(RAS_COLL_COMMS) +
      req->comms.nSkipMissingRanksComms * sizeof(*req->comms.skipMissingRanksComms);
    qsort(req->comms.skipMissingRanksComms, req->comms.nSkipMissingRanksComms,
          sizeof(*req->comms.skipMissingRanksComms), rasCommIdCompare);
  }
ret:
  free(peersReSorted);
  free(diagBytesPerComm);
  if (diagCollector != nullptr) rasDiagCollectorDestroy(diagCollector);
  // rasDiagCollectorDestroy frees the underlying impl (recovered via ctx); diagCollector is not leaked.
  // coverity[leaked_storage]
  return ret;
fail:
  if (req) {
    free(req);
    *pReq = nullptr;
  }
  free(*pData);
  *pData = nullptr;
  goto ret;
}

// Merges incoming collective RAS_COLL_COMMS response message into the local accumulated data.
static ncclResult_t rasCollCommsMerge(struct rasCollective* coll, struct rasMsg* msg) {
  struct rasCollComms* collData; // Data previously stored (locally) by our process.
  struct rasCollComms* msgData; // Data just received from another process.
  int dataOffset = rasMsgLength(RAS_MSG_COLLRESP) + msg->collResp.nPeers*sizeof(*msg->collResp.peers);
  ALIGN_SIZE(dataOffset, alignof(int64_t));

  msgData = (struct rasCollComms*)(((char*)msg) + dataOffset);
  collData = (struct rasCollComms*)coll->data;

  if (msgData->nComms > 0) {
    struct rasCollComms* newData = nullptr; // Destination buffer for the merged data.

    // Allocate the new buffer pessimistically (sized as the sum of the two old ones).
    NCCLCHECK(ncclCalloc((char**)&newData, coll->nData + msg->collResp.nData));
    struct rasCollComms::comm* collComm = collData->comms;
    struct rasCollComms::comm* msgComm = msgData->comms;
    struct rasCollComms::comm* newComm = newData->comms;

    for (int collIdx = 0, msgIdx = 0; collIdx < collData->nComms || msgIdx < msgData->nComms; newData->nComms++) {
      int cmp;
      if (collIdx < collData->nComms && msgIdx < msgData->nComms)
        cmp = rasCommIdCompare(&collComm->commId, &msgComm->commId);
      else
        cmp = (collIdx < collData->nComms ? -1 : 1);

      if (cmp == 0 && collComm->nRanks != msgComm->nRanks) {
        INFO(NCCL_RAS, "RAS encountered inconsistent communicator data: size %d != %d -- "
             "possible hash collision (0x%lx, 0x%lx, 0x%lx)", collComm->nRanks, msgComm->nRanks,
             collComm->commId.commHash, collComm->commId.hostHash, collComm->commId.pidHash);
        cmp = (collComm->nRanks < msgComm->nRanks ? -1 : 1);
        // We try to preserve them both separately...
      }

      if (cmp == 0) {
        // Merge the comms.
        memcpy(&newComm->commId, &collComm->commId, sizeof(newComm->commId));
        newComm->nRanks = collComm->nRanks;
        if (collComm->nValidRanks + msgComm->nValidRanks > collComm->nRanks) {
          INFO(NCCL_RAS,
               "RAS encountered more ranks (%d) than the communicator size (%d) -- possible hash collision "
               "(0x%lx, 0x%lx, 0x%lx)", collComm->nValidRanks + msgComm->nValidRanks, newComm->nRanks,
               collComm->commId.commHash, collComm->commId.hostHash, collComm->commId.pidHash);
          newComm->nValidRanks = newComm->nRanks;
          // We'll skip the extras in the loop below.
        } else {
          newComm->nValidRanks = collComm->nValidRanks + msgComm->nValidRanks;
        }
        // Merge the ranks.
        for (int newRankIdx = 0, collRankIdx = 0, msgRankIdx = 0;
             collRankIdx < collComm->nValidRanks || msgRankIdx < msgComm->nValidRanks;
             newRankIdx++) {
          int cmpRank;
          if (newRankIdx == newComm->nRanks)
            break; // Short of failing, the best we can do is skip...
          if (collRankIdx < collComm->nValidRanks && msgRankIdx < msgComm->nValidRanks) {
            cmpRank = (collComm->ranks[collRankIdx].commRank < msgComm->ranks[msgRankIdx].commRank ? -1 :
                       (collComm->ranks[collRankIdx].commRank > msgComm->ranks[msgRankIdx].commRank ? 1 : 0));
          } else {
            cmpRank = (collRankIdx < collComm->nValidRanks ? -1 : 1);
          }

          // There shouldn't be any overlaps in ranks between different sources.
          if (cmpRank == 0) {
            INFO(NCCL_RAS, "RAS encountered duplicate data for rank %d -- possible hash collision "
                 "(0x%lx, 0x%lx, 0x%lx)", collComm->ranks[collRankIdx].commRank,
                 newComm->commId.commHash, newComm->commId.hostHash, newComm->commId.pidHash);
            msgRankIdx++; // Short of failing, the best we can do is skip...
          }
          memcpy(newComm->ranks+newRankIdx, (cmpRank <= 0 ? collComm->ranks+collRankIdx++ :
                                             msgComm->ranks+msgRankIdx++), sizeof(*newComm->ranks));
          if (cmpRank > 0) {
            // peerIdx values from msgComm need to shift after merge.
            newComm->ranks[newRankIdx].peerIdx += coll->nPeers;

            if (collComm->nMissingRanks > 0) {
              // Remove the corresponding entry from missingRanks.
              struct rasCollCommsMissingRank* missingRank;
              missingRank = (struct rasCollCommsMissingRank*)bsearch(&newComm->ranks[newRankIdx].commRank,
                                                                     collComm->ranks+collComm->nValidRanks,
                                                                     collComm->nMissingRanks,
                                                                     sizeof(struct rasCollCommsMissingRank),
                                                                     rasCollCommsMissingRankSearch);
              if (missingRank) {
                // Mark the entry as no longer needed.
                memset(&missingRank->addr, '\0', sizeof(missingRank->addr));
              } else {
                INFO(NCCL_RAS, "RAS failed to find missingRank data -- internal error?");
              }
            } // if (collComm->nMissingRanks > 0)
          } // if (cmpRank > 0)
        } // for (newRankIdx)
        if (collComm->nMissingRanks > 0) {
          // Copy the missingRanks to newComm, skipping over any no longer needed entries.
          union ncclSocketAddress emptyAddr;
          struct rasCollCommsMissingRank* collMissingRanks;
          struct rasCollCommsMissingRank* newMissingRanks;
          int newRankIdx;

          memset(&emptyAddr, '\0', sizeof(emptyAddr));
          collMissingRanks = (struct rasCollCommsMissingRank*)(collComm->ranks+collComm->nValidRanks);
          newMissingRanks = (struct rasCollCommsMissingRank*)(newComm->ranks+newComm->nValidRanks);
          newRankIdx = 0;
          for (int collRankIdx = 0; collRankIdx < collComm->nMissingRanks; collRankIdx++) {
            if (memcmp(&collMissingRanks[collRankIdx].addr, &emptyAddr, sizeof(emptyAddr))) {
              memcpy(newMissingRanks + newRankIdx++, collMissingRanks + collRankIdx, sizeof(*newMissingRanks));
            }
          }
          newComm->nMissingRanks = newRankIdx;
          // The best-effort handling above (duplicate rank data, failed missingRank lookups,
          // capped ranks) can leave the counts inconsistent.  This is diagnostics-only data:
          // warn and clamp instead of aborting the application.  The wire format only promises
          // nRanks >= nValidRanks + nMissingRanks, and every walker strides on the counts (see
          // rasCollCommsCommSize), so the merged record stays parseable either way.
          if (newComm->nValidRanks + newComm->nMissingRanks != newComm->nRanks) {
            int nMissing = std::min(newComm->nMissingRanks,
                                    std::max(0, newComm->nRanks - newComm->nValidRanks));
            WARN("RAS merged communicator (0x%lx, 0x%lx, 0x%lx) rank count inconsistency: "
                 "%d valid + %d missing != %d ranks -- possible hash collision or duplicate "
                 "rank data; reporting %d missing", newComm->commId.commHash,
                 newComm->commId.hostHash, newComm->commId.pidHash, newComm->nValidRanks,
                 newComm->nMissingRanks, newComm->nRanks, nMissing);
            newComm->nMissingRanks = nMissing;
          }
        }
        // Concatenate diag tiers (collComm's first, then msgComm's).  newData was sized
        // pessimistically as collData.nData+msgData.nData, so there is always room.
        if (collComm->nDiagBytes > 0 || msgComm->nDiagBytes > 0) {
          char* dst = rasCollCommsDiagBytesMut(newComm);
          if (collComm->nDiagBytes > 0) {
            memcpy(dst, rasCollCommsDiagBytes(collComm), collComm->nDiagBytes);
            dst += collComm->nDiagBytes;
          }
          if (msgComm->nDiagBytes > 0) {
            memcpy(dst, rasCollCommsDiagBytes(msgComm), msgComm->nDiagBytes);
          }
          newComm->nDiagBytes = collComm->nDiagBytes + msgComm->nDiagBytes;
        }
        newComm = rasCollCommsCommNext(newComm);
        collComm = rasCollCommsCommNext(collComm);
        collIdx++;
        msgComm = rasCollCommsCommNext(msgComm);
        msgIdx++;
      } else if (cmp < 0) {
        // Copy from collComm (including its diag tier, which the memcpy carries verbatim).
        size_t commSize = rasCollCommsCommSize(collComm);
        memcpy(newComm, collComm, commSize);
        newComm = (struct rasCollComms::comm*)(((char*)(newComm)) + commSize);
        collComm = (struct rasCollComms::comm*)(((char*)(collComm)) + commSize);
        collIdx++;
      } else { // cmp > 0
        // Copy from msgComm (including its diag tier).
        size_t commSize = rasCollCommsCommSize(msgComm);
        memcpy(newComm, msgComm, commSize);
        for (int i = 0; i < newComm->nValidRanks; i++) {
          // peerIdx values from msgComm need to shift after merge.
          newComm->ranks[i].peerIdx += coll->nPeers;
        }
        newComm = (struct rasCollComms::comm*)(((char*)(newComm)) + commSize);
        msgComm = (struct rasCollComms::comm*)(((char*)(msgComm)) + commSize);
        msgIdx++;
      } // cmp > 0
    } // for (collIdx and msgIdx)

    free(coll->data);
    coll->data = (char*)newData;
    // newComm points at the next element beyond the last one -- exactly what we need.
    coll->nData = ((char*)newComm) - (char*)newData;
  } // if (msgData->nComms > 0)

  return ncclSuccess;
}

// Checks if a given communicator is in the skipMissingRanksComms array of the request.
static bool rasCollCommsSkipMissing(const struct rasCollRequest* req, ncclComm_t comm) {
  struct rasCommId id;
  id.commHash = comm->commHash;
  id.hostHash = comm->peerInfo->hostHash;
  id.pidHash = comm->peerInfo->pidHash;
  return (bsearch(&id, req->comms.skipMissingRanksComms, req->comms.nSkipMissingRanksComms,
                  sizeof(*req->comms.skipMissingRanksComms), rasCommIdCompare) != nullptr);
}

// Sorting callback for the ncclComms array.
static int ncclCommsCompare(const void* p1, const void* p2) {
  const ncclComm_t comm1 = ((const rasNcclComm*)p1)->comm;
  const ncclComm_t comm2 = ((const rasNcclComm*)p2)->comm;

  // Put nullptr's at the end.
  if (comm1 == nullptr || comm2 == nullptr)
    return (comm1 != nullptr ? -1 : (comm2 != nullptr ? 1 : 0));

  if (comm1->commHash == comm2->commHash) {
    return (comm1->rank < comm2->rank ? -1 : (comm1->rank > comm2->rank ? 1 : 0));
  } else {
    return (comm1->commHash < comm2->commHash ? -1 : 1);
  }
}

// Sorting callback for a lookup table to rasPeers.  Sorts by the hostHash (primary) and pidHash (secondary).
static int peersHashesCompare(const void* p1, const void* p2) {
  const struct rasPeerInfo* pi1 = *(const struct rasPeerInfo**)p1;
  const struct rasPeerInfo* pi2 = *(const struct rasPeerInfo**)p2;

  if (pi1->hostHash == pi2->hostHash) {
    return (pi1->pidHash < pi2->pidHash ? -1 : (pi1->pidHash > pi2->pidHash ? 1 : 0));
  } else {
    return (pi1->hostHash < pi2->hostHash ? -1 : 1);
  }
}

// Search callback for a lookup table to rasPeers.  Searches by the hostHash and pidHash.  The key is an array
// containing the hostHash at index 0 and the pidHash at index 1.
static int peersHashesSearch(const void* k, const void* e) {
  const uint64_t* key = (const uint64_t*)k;
  const struct rasPeerInfo* elem = *(const struct rasPeerInfo**)e;

  if (key[0] == elem->hostHash) {
    return (key[1] < elem->pidHash ? -1 : (key[1] > elem->pidHash ? 1 : 0));
  } else {
    return (key[0] < elem->hostHash ? -1 : 1);
  }
}

// Sorting/searching callback for struct rasCommId.  Sorts by commHash, then hostHash, then pidHash.
static int rasCommIdCompare(const void* p1, const void* p2) {
  const struct rasCommId* i1 = (const struct rasCommId*)p1;
  const struct rasCommId* i2 = (const struct rasCommId*)p2;
  if (i1->commHash == i2->commHash) {
    if (i1->hostHash == i2->hostHash) {
      return (i1->pidHash < i2->pidHash ? -1 : (i1->pidHash > i2->pidHash ? 1 : 0));
    } else {
      return (i1->hostHash < i2->hostHash ? -1 : 1);
    }
  } else {
    return (i1->commHash < i2->commHash ? -1 : 1);
  }
}

// Search callback for rasCollComms::comm rasCollCommsMissingRank array.  The key is the commRank.
static int rasCollCommsMissingRankSearch(const void* k, const void* e) {
  int key = *(const int*)k;
  const struct rasCollCommsMissingRank* elem = (const struct rasCollCommsMissingRank*)e;

  return (key < elem->commRank ? -1 : (key > elem->commRank ? 1 : 0));
}

// Invoked during RAS termination to release all the allocated resources.
void rasCollectivesTerminate() {
  for (struct rasCollective* coll = rasCollectivesHead; coll;) {
    struct rasCollective* collNext = coll->next;
    rasCollFree(coll);
    coll = collNext;
  }

  // rasCollectivesHead and rasCollectivesTail are taken care of by rasCollFree().
}
