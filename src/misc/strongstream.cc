/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "strongstream.h"
#include "checks.h"
#include "param.h"
#include "drv_extension.h"
#include "utils.h"

// Tracks the chain of graph nodes for a given graph captured identified by
// its graph id. This state has to live for as long as captured work is being
// submitted. HGGC doesn't have mechanism to inform us when the user ends capture
// so the best we can do is get notified when the graph is destroyed.
struct ncclStrongStreamGraph {
  struct ncclStrongStreamGraph* next;
  // Atomically exchanged to false by both the main thread or the graph destructor
  // callback. The last to arrive deletes the node.
  bool alive;
  unsigned long long graphId;
  // For each graph we track the "tip" of the chain of graph nodes. A linear
  // chain would always have just one node at its tip, but since we have to merge
  // in chains from other streams (via ncclStrongStreamWaitStream) some spots
  // in the chain can be wider than a single node and thus need a list, so we
  // maintain a dynamically sized array of tip nodes.
  int tipCount, tipCapacity;
  hggcGraphNode_t* tipNodes;
};

static void ncclStrongStreamGraphDelete(struct ncclStrongStreamGraph* g) {
  free(g->tipNodes);
  free(g);
}

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclCudaGetCapturingGraph(
    struct ncclCudaGraph* graph, hggcStream_t stream
  ) {
  hggcStreamCaptureStatus status;
  unsigned long long gid;
#if HGGCRT_VERSION >= 13000
  HGGCCHECK(hggcStreamGetCaptureInfo(stream, &status, &gid, &graph->graph, nullptr, nullptr, nullptr));
#else
  HGGCCHECK(hggcStreamGetCaptureInfo_v2(stream, &status, &gid, &graph->graph, nullptr, nullptr));
#endif
  if (status != hggcStreamCaptureStatusActive) {
    graph->graph = nullptr;
    gid = ULLONG_MAX;
  }
  graph->graphId = gid;
  return ncclSuccess;
}

ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, hggcHostFn_t fn, void* arg) {
  hggcUserObject_t object;
  HGGCCHECK(hggcUserObjectCreate(
    &object, arg, fn, /*initialRefcount=*/1, hggcUserObjectNoDestructorSync
  ));
  // Hand over ownership to HGGC Graph
  HGGCCHECK(hggcGraphRetainUserObject(graph.graph, object, 1, hggcGraphUserObjectMove));
  return ncclSuccess;
}

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss) {
  HGGCCHECK(hggcEitInterface::getInstance().wrapHggcStreamCreateForCopyEngine(&ss->hggcStream));
  HGGCCHECK(hggcEventCreateWithFlags(&ss->serialEvent, hggcEventDisableTiming));
  ss->everCaptured = false;
  ss->serialEventNeedsRecord = false;
  ss->graphHead = nullptr;
  return ncclSuccess;
}

static void graphDestructor(void* arg) {
  struct ncclStrongStreamGraph* g = (struct ncclStrongStreamGraph*)arg;
  if (false == __atomic_exchange_n(&g->alive, false, __ATOMIC_ACQ_REL)) {
    // Last to arrive deletes list node.
    ncclStrongStreamGraphDelete(g);
  }
}

ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss) {
  HGGCCHECK(hggcStreamDestroy(ss->hggcStream));
  HGGCCHECK(hggcEventDestroy(ss->serialEvent));
  // Delete list of per-graph chains.
  struct ncclStrongStreamGraph* g = ss->graphHead;
  while (g != nullptr) {
    struct ncclStrongStreamGraph* next = g->next;
    if (false == __atomic_exchange_n(&g->alive, false, __ATOMIC_ACQ_REL)) {
      // Last to arrive deletes list node.
      ncclStrongStreamGraphDelete(g);
    }
    g = next;
  }
  return ncclSuccess;
}

static void ensureTips(struct ncclStrongStreamGraph* g, int n) {
  if (g->tipCapacity < n) {
    g->tipNodes = (hggcGraphNode_t*)realloc(g->tipNodes, n*sizeof(hggcGraphNode_t));
    g->tipCapacity = n;
  }
}

ncclResult_t ncclStrongStreamAcquire(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss
  ) {
  if (graph.graph == nullptr) {
    if (ss->everCaptured) {
      HGGCCHECK(hggcStreamWaitEvent(ss->hggcStream, ss->serialEvent, 0));
      ss->serialEventNeedsRecord = false;
    }
  } else {
    ss->everCaptured = true;
    // Find the current graph in our list of graphs if it exists.
    struct ncclStrongStreamGraph** pg = &ss->graphHead;
    struct ncclStrongStreamGraph* g;
    while (*pg != nullptr) {
      g = *pg;
      if (g->graphId == graph.graphId) {
        // Move to front of list so that operations after acquire don't have to search the list.
        *pg = g->next;
        g->next = ss->graphHead;
        ss->graphHead = g;
        return ncclSuccess;
      } else if (false == __atomic_load_n(&g->alive, __ATOMIC_ACQUIRE)) {
        // Unrelated graph that has been destroyed. Remove and delete.
        *pg = g->next;
        ncclStrongStreamGraphDelete(g);
      } else {
        pg = &g->next;
      }
    }

    // This is a new graph so add to the list.
    g = (struct ncclStrongStreamGraph*)malloc(sizeof(struct ncclStrongStreamGraph));
    g->graphId = graph.graphId;
    g->tipNodes = nullptr;
    g->tipCapacity = 0;
    g->tipCount = 0;
    g->next = ss->graphHead;
    ss->graphHead = g;
    g->alive = true;
    NCCLCHECK(ncclCudaGraphAddDestructor(graph, graphDestructor, (void*)g));

    if (ss->serialEventNeedsRecord) {
      // Can only be here if previous release was for uncaptured work that
      // elided updating the event because no capture had yet occurred.
      HGGCCHECK(hggcStreamWaitEvent(ss->hggcStream, ss->serialEvent, 0));
      HGGCCHECK(hggcEventRecord(ss->serialEvent, ss->hggcStream));
    }
    ss->serialEventNeedsRecord = false;

    // First node in the chain must be a wait on the serialEvent.
    ensureTips(g, 1);
    HGGCCHECK(hggcGraphAddEventWaitNode(&g->tipNodes[0], graph.graph, nullptr, 0, ss->serialEvent));
    g->tipCount = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamAcquireUncaptured(struct ncclStrongStream* ss) {
  if (ss->everCaptured) {
    HGGCCHECK(hggcStreamWaitEvent(ss->hggcStream, ss->serialEvent, 0));
  }
  ss->serialEventNeedsRecord = true; // Assume the caller is going to add work to stream.
  return ncclSuccess;
}

static ncclResult_t checkGraphId(struct ncclStrongStreamGraph* g, unsigned long long id) {
  if (g == nullptr || g->graphId != id) {
    ERROR("Expected graph id=%llu was not at head of strong stream's internal list.", id);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss) {
  if (ss->serialEventNeedsRecord) {
    if (graph.graph == nullptr) {
      if (ss->everCaptured) {
        HGGCCHECK(hggcEventRecord(ss->serialEvent, ss->hggcStream));
        ss->serialEventNeedsRecord = false;
      }
    } else {
      struct ncclStrongStreamGraph* g = ss->graphHead;
      NCCLCHECK(checkGraphId(g, graph.graphId));
      ensureTips(g, 1);
      HGGCCHECK(hggcGraphAddEventRecordNode(&g->tipNodes[0], graph.graph, g->tipNodes, g->tipCount, ss->serialEvent));
      g->tipCount = 1;
      ss->serialEventNeedsRecord = false;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamLaunchHost(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss, hggcHostFn_t fn, void* arg
  ) {
  if (graph.graph == nullptr) {
    HGGCCHECK(hggcLaunchHostFunc(ss->hggcStream, fn, arg));
  } else {
    hggcHostNodeParams p;
    p.fn = fn;
    p.userData = arg;
    struct ncclStrongStreamGraph* g = ss->graphHead;
    NCCLCHECK(checkGraphId(g, graph.graphId));
    ensureTips(g, 1);
    HGGCCHECK(hggcGraphAddHostNode(&g->tipNodes[0], graph.graph, g->tipNodes, g->tipCount, &p));
    g->tipCount = 1;
  }
  ss->serialEventNeedsRecord = true;
  return ncclSuccess;
}

// Merge node list `b` into list `a` but don't add duplicates.
static void mergeTips(struct ncclStrongStreamGraph* a, hggcGraphNode_t const* bNodes, int bn) {
  int an = a->tipCount;
  ensureTips(a, an + bn);
  for (int bi=0; bi < bn; bi++) {
    for (int ai=0; ai < an; ai++) {
      if (a->tipNodes[ai] == bNodes[bi]) goto next_b;
    }
    a->tipNodes[a->tipCount++] = bNodes[bi];
  next_b:;
  }
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, struct ncclStrongStream* b,
    bool b_subsumes_a
  ) {
  if (graph.graph == nullptr) {
    if (b->serialEventNeedsRecord) {
      b->serialEventNeedsRecord = false;
      HGGCCHECK(hggcEventRecord(b->serialEvent, b->hggcStream));
    }
    HGGCCHECK(hggcStreamWaitEvent(a->hggcStream, b->serialEvent, 0));
  } else {
    struct ncclStrongStreamGraph* ag = a->graphHead;
    NCCLCHECK(checkGraphId(ag, graph.graphId));
    struct ncclStrongStreamGraph* bg = b->graphHead;
    NCCLCHECK(checkGraphId(bg, graph.graphId));
    if (b_subsumes_a) ag->tipCount = 0;
    mergeTips(ag, bg->tipNodes, bg->tipCount);
  }
  a->serialEventNeedsRecord = true;
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, hggcStream_t b,
    bool b_subsumes_a
  ) {
  if (graph.graph == nullptr) {
    // It is ok to use a->serialEvent to record b since we'll be setting
    // a->serialEventNeedsRecord so the event won't be considered accurate
    // until re-recorded.
    HGGCCHECK(hggcEventRecord(a->serialEvent, b));
    HGGCCHECK(hggcStreamWaitEvent(a->hggcStream, a->serialEvent, 0));
  } else {
    hggcStreamCaptureStatus status;
    unsigned long long bGraphId;
    hggcGraphNode_t const* bNodes;
    size_t bCount = 0;
#if HGGCRT_VERSION >= 13000
    HGGCCHECK(hggcStreamGetCaptureInfo(b, &status, &bGraphId, nullptr, &bNodes, nullptr, &bCount));
#else
    HGGCCHECK(hggcStreamGetCaptureInfo_v2(b, &status, &bGraphId, nullptr, &bNodes, &bCount));
#endif
    if (status != hggcStreamCaptureStatusActive || graph.graphId != bGraphId) {
      ERROR("Stream is not being captured by the expected graph.");
      return ncclInvalidUsage;
    }
    struct ncclStrongStreamGraph* ag = a->graphHead;
    NCCLCHECK(checkGraphId(ag, graph.graphId));
    if (b_subsumes_a) ag->tipCount = 0;
    mergeTips(ag, bNodes, bCount);
  }
  a->serialEventNeedsRecord = true;

  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, hggcStream_t a, struct ncclStrongStream* b,
    bool b_subsumes_a
  ) {
  if (graph.graph == nullptr) {
    if (b->serialEventNeedsRecord) {
      b->serialEventNeedsRecord = false;
      HGGCCHECK(hggcEventRecord(b->serialEvent, b->hggcStream));
    }
    HGGCCHECK(hggcStreamWaitEvent(a, b->serialEvent, 0));
  } else {
    struct ncclStrongStreamGraph* bg = b->graphHead;
    NCCLCHECK(checkGraphId(bg, graph.graphId));
#if HGGCRT_VERSION >= 13000
    HGGCCHECK(hggcStreamUpdateCaptureDependencies(a, bg->tipNodes, nullptr, bg->tipCount,
      b_subsumes_a ? hggcStreamSetCaptureDependencies : hggcStreamAddCaptureDependencies
    ));
#else
    HGGCCHECK(hggcStreamUpdateCaptureDependencies(a, bg->tipNodes, bg->tipCount,
      b_subsumes_a ? hggcStreamSetCaptureDependencies : hggcStreamAddCaptureDependencies
    ));
#endif
  }
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss) {
  HGGCCHECK(hggcStreamWaitEvent(ss->hggcStream, ss->serialEvent, 0));
  ss->serialEventNeedsRecord = false;
  HGGCCHECK(hggcStreamSynchronize(ss->hggcStream));
  return ncclSuccess;
}
