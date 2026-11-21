//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <hggc_runtime.h>
#include <mpi.h>
#include <pccl.h>

#include <stdint.h>
#include <stdio.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "logger.h"

static uint8_t valueAt(int source, int destination, size_t offset, int iteration, int message) {
  return (source * 17u + destination * 31u + (offset % 251) * 13u + iteration * 7u + message * 11u) % 251;
}

static void fillBuffer(std::vector<uint8_t>& host, size_t bytes, int source, int destination,
                       int iteration, int message) {
  uint8_t pattern[251];
  for (size_t i = 0; i < sizeof(pattern); ++i) pattern[i] = valueAt(source, destination, i, iteration, message);
  for (size_t offset = 0; offset < bytes; offset += sizeof(pattern)) {
    std::memcpy(host.data() + offset, pattern, std::min(sizeof(pattern), bytes - offset));
  }
}

static int checkBuffer(const uint8_t* device, size_t bytes, size_t capacity, int source, int destination,
                       int iteration, int message, std::vector<uint8_t>& host) {
  HGGCCHECK(hggcMemcpy(host.data(), device, capacity, hggcMemcpyDeviceToHost));
  uint8_t pattern[251];
  for (size_t i = 0; i < sizeof(pattern); ++i) pattern[i] = valueAt(source, destination, i, iteration, message);
  for (size_t offset = 0; offset < bytes; offset += sizeof(pattern)) {
    if (std::memcmp(host.data() + offset, pattern, std::min(sizeof(pattern), bytes - offset)) != 0) {
      fprintf(stderr, "rank %d: source %d message %d: data mismatch at offset %zu\n",
              destination, source, message, offset);
      return 1;
    }
  }
  for (size_t offset = bytes; offset < capacity; ++offset) {
    if (host[offset] != 255) {
      fprintf(stderr, "rank %d: source %d message %d: unexpected write at offset %zu\n",
              destination, source, message, offset);
      return 1;
    }
  }
  return 0;
}

static bool hasMessage(const std::string& mode, int source, int destination, int nRanks, int message) {
  // The last rank is completely idle; the remaining ranks have unequal send/receive queues.
  return mode != "sparse" || (source != nRanks - 1 && destination != nRanks - 1 &&
                             (source * 7u + destination * 3u + message) % 4 != 0);
}

static size_t messageBytes(const std::string& mode, size_t peerBytes, int source, int destination, int message) {
  if (mode == "native" || mode == "sendrecv") return peerBytes;
  switch ((source * 17u + destination * 13u + message * 7u) % 6) {
    case 0: return 0;  // Enqueued zero-byte sync, distinct from an absent direction.
    case 1: return 1;
    case 2: return peerBytes / 2;
    case 3: return peerBytes == 0 ? 0 : peerBytes - 1;
    case 4: return peerBytes;
    default: return peerBytes + 1;
  }
}

int main(int argc, char** argv) {
  MPICHECK(MPI_Init(&argc, &argv));
  int rank, nRanks;
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  cxxopts::Options options("pcclAlltoAllGroupTest", "AlltoAll and variable P2P groups with rank-local extra tasks.");
  options.add_options()
      ("b,peer_bytes", "Reference bytes per peer", cxxopts::value<size_t>()->default_value("67108864"))
      ("i,iterations", "Repeats of each group layout", cxxopts::value<int>()->default_value("2"))
      ("m,messages", "Messages per peer in one group", cxxopts::value<int>()->default_value("2"))
      ("mode", "all, native, sendrecv, alltoallv, or sparse", cxxopts::value<std::string>()->default_value("all"))
      ("mixed_api", "Use native API on even ranks in sendrecv mode", cxxopts::value<bool>()->default_value("false"))
      ("graph", "Capture and replay each group twice", cxxopts::value<bool>()->default_value("false"))
      ("h,help", "Print usage");
  auto args = options.parse(argc, argv);
  if (args.count("help")) {
    if (rank == 0) printf("%s\n", options.help().c_str());
    MPICHECK(MPI_Finalize());
    return 0;
  }
  size_t peerBytes = args["peer_bytes"].as<size_t>();
  int iterations = args["iterations"].as<int>();
  int messages = args["messages"].as<int>();
  bool graphReplay = args["graph"].as<bool>();
  bool mixedApi = args["mixed_api"].as<bool>();
  std::string selectedMode = args["mode"].as<std::string>();
  std::vector<std::string> modes = {"native", "sendrecv", "alltoallv", "sparse"};
  if (nRanks < 3 || iterations < 1 || messages < 1 ||
      peerBytes == std::numeric_limits<size_t>::max() ||
      peerBytes + 1 > std::numeric_limits<size_t>::max() / nRanks / std::max(1, messages) ||
      (selectedMode != "all" && std::find(modes.begin(), modes.end(), selectedMode) == modes.end())) {
    if (rank == 0) {
      fprintf(stderr, "Require at least three ranks, positive iterations/messages, valid mode and size.\n");
    }
    MPICHECK(MPI_Finalize());
    return 1;
  }
  if (selectedMode != "all") modes = {selectedMode};

  MPI_Comm localComm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
  int localRank, localRanks, devices;
  MPICHECK(MPI_Comm_rank(localComm, &localRank));
  MPICHECK(MPI_Comm_size(localComm, &localRanks));
  HGGCCHECK(hggcGetDeviceCount(&devices));
  if (localRanks > devices) {
    fprintf(stderr, "rank %d: %d local ranks exceed %d visible devices\n", rank, localRanks, devices);
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  HGGCCHECK(hggcSetDevice(localRank));
  MPICHECK(MPI_Comm_free(&localComm));

  pcclUniqueId id;
  if (rank == 0) PCCLCHECK(pcclGetUniqueId(&id));
  MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  pcclComm_t comm;
  PCCLCHECK(pcclCommInitRank(&comm, nRanks, id, rank));
  hggcStream_t stream;
  HGGCCHECK(hggcStreamCreate(&stream));
  const size_t extraBytes = 1037;
  const size_t allocationBytes = (peerBytes + 1) * nRanks * messages;
  uint8_t *send, *recv, *extraSend, *extraRecv;
  HGGCCHECK(hggcMalloc(&send, allocationBytes));
  HGGCCHECK(hggcMalloc(&recv, allocationBytes));
  HGGCCHECK(hggcMalloc(&extraSend, extraBytes));
  HGGCCHECK(hggcMalloc(&extraRecv, extraBytes));
  std::vector<uint8_t> host(std::max(peerBytes + 1, extraBytes));
  int failed = 0;

  for (const std::string& mode : modes) {
    const bool native = mode == "native" || (mixedApi && mode == "sendrecv" && rank % 2 == 0);
    const bool variable = mode == "alltoallv" || mode == "sparse";
    const size_t slotBytes = std::max<size_t>(1, peerBytes + (variable ? 1 : 0));
    const size_t bufferBytes = slotBytes * nRanks * messages;
    for (int iteration = 0; iteration < iterations && !failed; ++iteration) {
      for (int message = 0; message < messages; ++message) {
        for (int peer = 0; peer < nRanks; ++peer) {
          size_t bytes = messageBytes(mode, peerBytes, rank, peer, message);
          fillBuffer(host, bytes, rank, peer, iteration, message);
          HGGCCHECK(hggcMemcpy(send + (size_t(message) * nRanks + peer) * slotBytes,
                              host.data(), bytes, hggcMemcpyHostToDevice));
        }
      }
      int source = iteration % 2, destination = 1 - source;
      fillBuffer(host, extraBytes, source, destination, iteration, messages);
      HGGCCHECK(hggcMemcpy(extraSend, host.data(), extraBytes, hggcMemcpyHostToDevice));

      for (int layout = 0; layout < 3 && !failed; ++layout) {
        auto enqueue = [&]() {
          PCCLCHECK(pcclGroupStart());
          for (int position = 0; position < 2; ++position) {
            if (layout == position + 1) {
              if (rank == source) PCCLCHECK(pcclSend(extraSend, extraBytes, pcclUint8, destination, comm, stream));
              if (rank == destination) PCCLCHECK(pcclRecv(extraRecv, extraBytes, pcclUint8, source, comm, stream));
            }
            if (position != 0) continue;
            for (int message = 0; message < messages; ++message) {
              size_t base = size_t(message) * nRanks * slotBytes;
              if (native) {
                PCCLCHECK(pcclAlltoAll(send + base, recv + base, peerBytes, pcclUint8, comm, stream));
              } else {
                // Vary peer order by rank while retaining FIFO order for each peer and direction.
                for (int p = 0; p < nRanks; ++p) {
                  int peer = (rank + (rank % 2 ? p : nRanks - p)) % nRanks;
                  if (hasMessage(mode, rank, peer, nRanks, message)) {
                    PCCLCHECK(pcclSend(send + base + peer * slotBytes,
                        messageBytes(mode, peerBytes, rank, peer, message), pcclUint8, peer, comm, stream));
                  }
                  if (hasMessage(mode, peer, rank, nRanks, message)) {
                    PCCLCHECK(pcclRecv(recv + base + peer * slotBytes,
                        messageBytes(mode, peerBytes, peer, rank, message), pcclUint8, peer, comm, stream));
                  }
                }
              }
            }
          }
          PCCLCHECK(pcclGroupEnd());
        };
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        hggcGraph_t graph;
        hggcGraphExec_t executable;
        if (graphReplay) {
          // Establish all peer connections before capture, including rank-local extra transfers.
          enqueue();
          HGGCCHECK(hggcStreamSynchronize(stream));
          HGGCCHECK(hggcStreamBeginCapture(stream, hggcStreamCaptureModeThreadLocal));
          enqueue();
          HGGCCHECK(hggcStreamEndCapture(stream, &graph));
          HGGCCHECK(hggcGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        }
        for (int replay = 0; replay < (graphReplay ? 2 : 1) && !failed; ++replay) {
          HGGCCHECK(hggcMemset(recv, 255, bufferBytes));
          HGGCCHECK(hggcMemset(extraRecv, 255, extraBytes));
          MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
          if (graphReplay) {
            HGGCCHECK(hggcGraphLaunch(executable, stream));
          } else {
            enqueue();
          }
          HGGCCHECK(hggcStreamSynchronize(stream));
          int errors = 0;
          for (int message = 0; message < messages; ++message) {
            for (int peer = 0; peer < nRanks; ++peer) {
              size_t bytes = hasMessage(mode, peer, rank, nRanks, message) ?
                  messageBytes(mode, peerBytes, peer, rank, message) : 0;
              errors += checkBuffer(recv + (size_t(message) * nRanks + peer) * slotBytes,
                                    bytes, slotBytes, peer, rank, iteration, message, host);
            }
          }
          if (layout != 0 && rank == destination) {
            errors += checkBuffer(extraRecv, extraBytes, extraBytes, source, rank, iteration, messages, host);
          }
          MPICHECK(MPI_Allreduce(&errors, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD));
          if (rank == 0) {
            printf("%s mode=%s ranks=%d peer_bytes=%zu messages=%d iteration=%d "
                   "layout=%s mixed_api=%d graph=%d replay=%d\n",
                   failed ? "FAIL" : "PASS", mode.c_str(), nRanks, peerBytes, messages, iteration,
                   layout == 0 ? "pure" : (layout == 1 ? "p2p-first" : "p2p-last"), mixedApi, graphReplay, replay);
            fflush(stdout);
          }
        }
        if (graphReplay) {
          HGGCCHECK(hggcGraphExecDestroy(executable));
          HGGCCHECK(hggcGraphDestroy(graph));
        }
      }
    }
  }

  PCCLCHECK(pcclCommDestroy(comm));
  HGGCCHECK(hggcStreamDestroy(stream));
  HGGCCHECK(hggcFree(extraRecv));
  HGGCCHECK(hggcFree(extraSend));
  HGGCCHECK(hggcFree(recv));
  HGGCCHECK(hggcFree(send));
  MPICHECK(MPI_Finalize());
  return failed ? 1 : 0;
}
