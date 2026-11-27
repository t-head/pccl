//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include "hggc_runtime.h"
#include "pccl.h"
#include "mpi.h"
#include "logger.h"
#include "cxxopts.hpp"

struct runArgs {
  int oldRanks;
  int root;
  int blocking;
  int splitShare;
  int iters;
  int timeout;
};

static void testTimeout(int) {
  const char message[] = "pcclCommGrowTest timed out\n";
  (void)write(STDERR_FILENO, message, sizeof(message) - 1);
  _exit(testInternalError);
}

static void waitAsync(pcclComm_t comm) {
  pcclResult_t state;
  do {
    PCCLCHECK(pcclCommGetAsyncError(comm, &state));
    if (state == pcclInProgress) usleep(1000);
  } while (state == pcclInProgress);
  PCCLCHECK(state);
}

static void checkComm(pcclComm_t comm, int myRank, int nRanks, int device) {
  int rank, count, dev;
  PCCLCHECK(pcclCommUserRank(comm, &rank));
  PCCLCHECK(pcclCommCount(comm, &count));
  PCCLCHECK(pcclCommCuDevice(comm, &dev));
  if (rank != myRank || count != nRanks || dev != device) {
    ERROR("Rank %d: communicator rank/count/device %d/%d/%d, expected %d/%d/%d",
          myRank, rank, count, dev, myRank, nRanks, device);
    exit(testInternalError);
  }
}

static void runAllReduce(pcclComm_t comm, hggcStream_t stream, int *sendbuff, int *recvbuff,
                         int myRank, int nRanks, int round) {
  const int count = 1024;
  int hostbuff[count];
  for (int i = 0; i < count; i++) hostbuff[i] = myRank + round + i % 7;
  HGGCCHECK(hggcMemcpy(sendbuff, hostbuff, sizeof(hostbuff), hggcMemcpyHostToDevice));
  HGGCCHECK(hggcMemset(recvbuff, 0, sizeof(hostbuff)));
  PCCLCHECK(pcclAllReduce(sendbuff, recvbuff, count, pcclInt32, pcclSum, comm, stream));
  waitAsync(comm);
  HGGCCHECK(hggcStreamSynchronize(stream));
  HGGCCHECK(hggcMemcpy(hostbuff, recvbuff, sizeof(hostbuff), hggcMemcpyDeviceToHost));
  for (int i = 0; i < count; i++) {
    int expected = nRanks * (nRanks - 1) / 2 + nRanks * (round + i % 7);
    if (hostbuff[i] != expected) {
      ERROR("Rank %d: AllReduce index %d got %d expected %d", myRank, i, hostbuff[i], expected);
      exit(testInternalError);
    }
  }
}

void parseAndSetupArguments(struct runArgs *args, int argc, char **argv) {
  cxxopts::Options options("pcclCommGrowTest", "Test PCCL communicator growth and unique ID generation.");
  options.add_options()
      ("n,old_ranks", "Number of ranks in the parent communicator.", cxxopts::value<int>()->default_value("1"))
      ("r,root", "Parent rank that calls pcclCommGetUniqueId.", cxxopts::value<int>()->default_value("0"))
      ("b,blocking", "Blocking (1) or nonblocking (0) communicators.", cxxopts::value<int>()->default_value("1"))
      ("s,split_share", "Parent communicator splitShare setting.", cxxopts::value<int>()->default_value("0"))
      ("i,iters", "Number of init/grow/destroy cycles.", cxxopts::value<int>()->default_value("1"))
      ("t,timeout", "Timeout for the whole test, in seconds.", cxxopts::value<int>()->default_value("120"))
      ("h,help", "Print usage");
  auto params = options.parse(argc, argv);
  if (params.count("help")) {
    std::cout << options.help() << std::endl;
    exit(testSuccess);
  }
  args->oldRanks = params["old_ranks"].as<int>();
  args->root = params["root"].as<int>();
  args->blocking = params["blocking"].as<int>();
  args->splitShare = params["split_share"].as<int>();
  args->iters = params["iters"].as<int>();
  args->timeout = params["timeout"].as<int>();
  if (args->oldRanks < 1 || args->root < 0 || args->root >= args->oldRanks ||
      (args->blocking != 0 && args->blocking != 1) || (args->splitShare != 0 && args->splitShare != 1) ||
      args->iters < 1 || args->timeout < 1) {
    ERROR("Require old_ranks > 0, 0 <= root < old_ranks, blocking/split_share in {0,1}, "
          "iters > 0, timeout > 0");
    exit(testInternalError);
  }
}

int main(int argc, char **argv) {
  struct runArgs args;
  parseAndSetupArguments(&args, argc, argv);
  if (signal(SIGALRM, testTimeout) == SIG_ERR) return testInternalError;
  alarm(args.timeout);

  int myRank, nRanks, localRank = 0;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  if (args.oldRanks >= nRanks) {
    ERROR("Require old_ranks < MPI size");
    exit(testInternalError);
  }

  std::vector<uint64_t> hostHashs(nRanks);
  char hostname[1024];
  getHostName(hostname, sizeof(hostname));
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs.data(),
                        sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p = 0; p < myRank; p++) {
    if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }
  HGGCCHECK(hggcSetDevice(localRank));
  int *sendbuff, *recvbuff;
  hggcStream_t stream;
  HGGCCHECK(hggcMalloc(&sendbuff, 1024 * sizeof(int)));
  HGGCCHECK(hggcMalloc(&recvbuff, 1024 * sizeof(int)));
  HGGCCHECK(hggcStreamCreate(&stream));

  pcclConfig_t config = PCCL_CONFIG_INITIALIZER;
  config.blocking = args.blocking;
  config.splitShare = args.splitShare;
  for (int iter = 0; iter < args.iters; iter++) {
    pcclUniqueId id;
    pcclComm_t comm = NULL, newcomm = NULL;
    if (myRank == 0) PCCLCHECK(pcclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    if (myRank < args.oldRanks) {
      PCCLCHECK(pcclCommInitRankConfig(&comm, args.oldRanks, id, myRank, &config));
      waitAsync(comm);
      runAllReduce(comm, stream, sendbuff, recvbuff, myRank, args.oldRanks, 2 * iter + 1);
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (myRank == args.root) PCCLCHECK(pcclCommGetUniqueId(comm, &id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, args.root, MPI_COMM_WORLD));
    if (myRank < args.oldRanks) {
      PCCLCHECK(pcclCommGrow(comm, nRanks, NULL, -1, &newcomm, &config));
    } else {
      PCCLCHECK(pcclCommGrow(NULL, nRanks, &id, myRank, &newcomm, &config));
    }
    // Grow completion is reported on the new communicator.
    waitAsync(newcomm);
    checkComm(newcomm, myRank, nRanks, localRank);
    if (comm != NULL) {
      PCCLCHECK(pcclCommFinalize(comm));
      waitAsync(comm);
      PCCLCHECK(pcclCommDestroy(comm));
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    runAllReduce(newcomm, stream, sendbuff, recvbuff, myRank, nRanks, 2 * iter + 2);
    PCCLCHECK(pcclCommFinalize(newcomm));
    waitAsync(newcomm);
    PCCLCHECK(pcclCommDestroy(newcomm));
    if (myRank == 0) {
      INFO("Cycle %d: Grow %d -> %d, root %d, blocking %d, splitShare %d: "
           "AllReduce verified after parent destruction",
           iter + 1, args.oldRanks, nRanks, args.root, args.blocking, args.splitShare);
    }
  }
  HGGCCHECK(hggcStreamDestroy(stream));
  HGGCCHECK(hggcFree(sendbuff));
  HGGCCHECK(hggcFree(recvbuff));
  MPICHECK(MPI_Finalize());
  alarm(0);
  INFO("[Rank %d] Success", myRank);
  return testSuccess;
}
