/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2024-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_STRONGSTREAM_H_
#define NCCL_STRONGSTREAM_H_

#include "pccl.h"
#include "checks.h"

#include <stdint.h>

/* ncclCudaGraph: Wraps a hggcGraph_t so that we can support pre-graph CUDA runtimes
 * easily.
 */
struct ncclCudaGraph {
  hggcGraph_t graph;
  unsigned long long graphId;
};

inline struct ncclCudaGraph ncclCudaGraphNone() {
  struct ncclCudaGraph tmp;
  tmp.graph = nullptr;
  tmp.graphId = ULLONG_MAX;
  return tmp;
}

inline bool ncclCudaGraphValid(struct ncclCudaGraph graph) {
  return graph.graph != nullptr;
}

inline bool ncclCudaGraphSame(struct ncclCudaGraph a, struct ncclCudaGraph b) {
  return a.graphId == b.graphId;
}

ncclResult_t ncclCudaGetCapturingGraph(struct ncclCudaGraph* graph, hggcStream_t stream);
ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, hggcHostFn_t fn, void* arg);

/* ncclStrongStream: An abstraction over CUDA streams that do not lose their
 * identity while being captured. Regular streams have the deficiency that the
 * captured form of a stream in one graph launch has no relation to the
 * uncaptured stream or to the captured form in other graph launches. This makes
 * streams unfit for the use of serializing access to a persistent resource.
 * Strong streams have been introduced to address this need.
 *
 * - All updates to a strong stream must be enclosed by a Acquire/Release pair.
 *
 * - The Acquire, Release, and all updates take a ncclCudaGraph parameter
 *   indicating the currently capturing graph (or none). This parameter must be
 *   the same for the entire sequence of {Acquire; ...; Release}.
 *
 * - An {Acquire; ...; Release} sequence must not be concurrent with any
 *   other operations against the strong stream including graph launches which
 *   reference this stream.
 */
struct ncclStrongStream;

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss);
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss);

// Acquire-fence the strong stream.
ncclResult_t ncclStrongStreamAcquire(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss
);

// Acquire-fence the strong stream assuming no graph is capturing. This permits
// the caller to enqueue directly to the `ss->hggcStream` member using native CUDA
// calls. Strong stream still must be released via:
//   ncclStrongStreamRelease(ncclCudaGraphNone(), ss);
ncclResult_t ncclStrongStreamAcquireUncaptured(struct ncclStrongStream* ss);

// Release-fence of the strong stream.
ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss);

// Add a host launch to the stream.
ncclResult_t ncclStrongStreamLaunchHost(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss,
  hggcHostFn_t fn, void* arg
);

// Cause `a` to wait for the current state `b`. Both `a` and `b` must be acquired.
// `b_subsumes_a` indicates that all work in `a` is already present in `b`, thus
// we want to fast-forward `a` to be a clone of `b`. Knowing this permits the
// implementation to induce few graph dependencies.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, struct ncclStrongStream* a, struct ncclStrongStream* b, bool b_subsumes_a=false
);
// `b` must be capturing within `graph`.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, struct ncclStrongStream* a, hggcStream_t b, bool b_subsumes_a=false
);
// `a` must be capturing within `graph`.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, hggcStream_t a, struct ncclStrongStream* b, bool b_subsumes_a=false
);

// Synchrnoization does not need the strong stream to be acquired.
ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss);

////////////////////////////////////////////////////////////////////////////////

struct ncclStrongStreamGraph; // internal to ncclStrongStream

struct ncclStrongStream {
  // Used when not graph capturing.
  hggcStream_t hggcStream;
  // The event used to establish order between graphs and streams. During acquire
  // this event is waited on, during release it is recorded to.
  hggcEvent_t serialEvent;
  // This stream ever appeared in a graph capture.
  bool everCaptured;
  // Tracks whether serialEvent needs to be recorded to upon Release().
  bool serialEventNeedsRecord;
  struct ncclStrongStreamGraph* graphHead;
};

#endif
