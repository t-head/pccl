// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <pccl.h>
#include <mpi.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

static int rankId;
#define OK(call) do { int result = (call); if (result != 0) { \
  std::fprintf(stderr, "FAIL rank=%d line=%d call=%s rc=%d\n", rankId, __LINE__, #call, result); \
  MPI_Abort(MPI_COMM_WORLD, 91); } } while (0)

struct Message {
  int peer;
  int send;
  int recv;
};

static unsigned char payload(int rank, int msg, size_t i) {
  return (i * 29 + (i >> 13) + (i >> 20) * 17 + rank * 31 + msg * 43) % 251;
}

static void run(const char* label, const std::vector<Message>& msgs, pcclComm_t comm,
                hggcStream_t stream) {
  std::vector<size_t> offset;
  size_t size = 83;
  for (const auto& m : msgs) {
    offset.push_back(size);
    size += std::max(0, std::max(m.send, m.recv)) + 97;
  }
  std::vector<unsigned char> input(size), output(size, 253), expected(output);
  for (size_t j = 0; j < msgs.size(); j++) {
    for (int i = 0; i < msgs[j].send; i++) input[offset[j]+i] = payload(rankId, j, i);
    for (int i = 0; i < msgs[j].recv; i++) expected[offset[j]+i] = payload(msgs[j].peer, j, i);
  }
  unsigned char *src, *dst;
  OK(hggcMalloc((void**)&src, size)); OK(hggcMalloc((void**)&dst, size));
  OK(hggcMemcpy(src, input.data(), size, hggcMemcpyHostToDevice));
  OK(hggcMemcpy(dst, output.data(), size, hggcMemcpyHostToDevice));
  const bool capture = std::getenv("P2P_CHECK_GRAPH") && !std::strcmp(label, "duplex-asymmetric");
  hggcGraph_t graph;
  hggcGraphExec_t exec;
  if (capture) OK(hggcStreamBeginCapture(stream, hggcStreamCaptureModeThreadLocal));
  OK(pcclGroupStart());
  for (size_t j = 0; j < msgs.size(); j++) {
    const auto& m = msgs[j];
    if (m.send >= 0) OK(pcclSend(src + offset[j], m.send, pcclUint8, m.peer, comm, stream));
    if (m.recv >= 0) OK(pcclRecv(dst + offset[j], m.recv, pcclUint8, m.peer, comm, stream));
  }
  OK(pcclGroupEnd());
  if (capture) {
    OK(hggcStreamEndCapture(stream, &graph));
    OK(hggcGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    OK(hggcGraphLaunch(exec, stream)); OK(hggcStreamSynchronize(stream));
    OK(hggcMemcpy(output.data(), dst, size, hggcMemcpyDeviceToHost));
    if (output != expected) MPI_Abort(MPI_COMM_WORLD, 94);
    for (auto& value : input) value ^= 1;
    for (auto& value : expected) if (value != 253) value ^= 1;
    std::fill(output.begin(), output.end(), 253);
    OK(hggcMemcpy(src, input.data(), size, hggcMemcpyHostToDevice));
    OK(hggcMemcpy(dst, output.data(), size, hggcMemcpyHostToDevice));
    OK(hggcGraphLaunch(exec, stream));
  }
  OK(hggcStreamSynchronize(stream));
  OK(hggcMemcpy(output.data(), dst, size, hggcMemcpyDeviceToHost));
  for (size_t i = 0; i < size; i++) {
    if (output[i] != expected[i]) {
      std::fprintf(stderr, "MISMATCH rank=%d case=%s offset=%zu got=%u expected=%u\n",
                   rankId, label, i, output[i], expected[i]);
      MPI_Abort(MPI_COMM_WORLD, 92);
    }
  }
  if (capture) {
    OK(hggcGraphExecDestroy(exec)); OK(hggcGraphDestroy(graph));
    if (rankId == 0) std::puts("GRAPH_OK two replays, different payloads");
  }
  OK(hggcFree(dst)); OK(hggcFree(src));
  MPI_Barrier(MPI_COMM_WORLD);
  if (rankId == 0) std::printf("PASS %s guarded-bytes/rank=%zu\n", label, size);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  setvbuf(stdout, nullptr, _IOLBF, 0);
  const char* localRank = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (localRank == nullptr) MPI_Abort(MPI_COMM_WORLD, 95);
  int nranks, dev = std::atoi(localRank);
  MPI_Comm_rank(MPI_COMM_WORLD, &rankId); MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (nranks != 2 && nranks != 4) MPI_Abort(MPI_COMM_WORLD, 93);
  OK(hggcSetDevice(dev));
  char bus[64], host[256]; gethostname(host, sizeof(host));
  OK(hggcDeviceGetPCIBusId(bus, sizeof(bus), dev));
  std::printf("DEVICE rank=%d host=%s dev=%d pci=%s\n", rankId, host, dev, bus);
  pcclUniqueId id; pcclComm_t comm; hggcStream_t stream;
  if (rankId == 0) OK(pcclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  OK(pcclCommInitRank(&comm, nranks, id, rankId));
  OK(hggcStreamCreate(&stream));
  const int peer = rankId ^ (nranks / 2);
  const bool low = rankId < peer;
  const int large = (12 << 20) - 5;
  constexpr int small = (1 << 20) + 15;
  run("one-way", {{peer, low ? large : -1, low ? -1 : large}}, comm, stream);
  run("duplex-asymmetric", {{peer, low ? large : small, low ? small : large}}, comm, stream);
  run("group-same-peer", {{peer, large, large}, {peer, 13, 13}, {peer, 0, 0},
                          {peer, low ? 0 : small, low ? small : 0}}, comm, stream);
  run("self-tail", {{rankId, large, large}}, comm, stream);
  run("short-ll", {{peer, 16381, 16381}}, comm, stream);
  run("network-tail", {{peer, (8 << 20) + 1, (8 << 20) + 1}}, comm, stream);
  run("partition-tail", {{peer, large, large}}, comm, stream);
  OK(hggcStreamDestroy(stream)); OK(pcclCommDestroy(comm)); MPI_Finalize();
}
