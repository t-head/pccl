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
  int excludeRank;
  int blocking;
  int shrinkShare;
  int inFlight;
  int iters;
  int timeout;
};

static void testTimeout(int) {
  const char message[] = "pcclCommRevokeTest timed out\n";
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
  cxxopts::Options options("pcclCommRevokeTest", "Test PCCL communicator revocation and recovery with shrink.");
  options.add_options()
      ("r,exclude_rank", "Rank that skips Revoke and Shrink (default: last rank).", cxxopts::value<int>())
      ("b,blocking", "Blocking (1) or nonblocking (0) communicators.", cxxopts::value<int>()->default_value("1"))
      ("s,shrink_share", "Parent communicator shrinkShare setting.", cxxopts::value<int>()->default_value("0"))
      ("f,in_flight", "Leave an AllReduce pending before Revoke (1) or revoke idle communicators (0).",
                      cxxopts::value<int>()->default_value("1"))
      ("i,iters", "Number of init/revoke/shrink/destroy cycles.", cxxopts::value<int>()->default_value("1"))
      ("t,timeout", "Timeout for the whole test, in seconds.", cxxopts::value<int>()->default_value("120"))
      ("h,help", "Print usage");
  auto params = options.parse(argc, argv);
  if (params.count("help")) {
    std::cout << options.help() << std::endl;
    exit(testSuccess);
  }
  args->excludeRank = params.count("exclude_rank") ? params["exclude_rank"].as<int>() : -1;
  args->blocking = params["blocking"].as<int>();
  args->shrinkShare = params["shrink_share"].as<int>();
  args->inFlight = params["in_flight"].as<int>();
  args->iters = params["iters"].as<int>();
  args->timeout = params["timeout"].as<int>();
  if ((params.count("exclude_rank") && args->excludeRank < 0) ||
      (args->blocking != 0 && args->blocking != 1) || (args->shrinkShare != 0 && args->shrinkShare != 1) ||
      (args->inFlight != 0 && args->inFlight != 1) || args->iters < 1 || args->timeout < 1) {
    ERROR("Require exclude_rank >= 0, blocking/shrink_share/in_flight in {0,1}, "
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
  if (args.excludeRank == -1) args.excludeRank = nRanks - 1;
  if (nRanks < 2 || args.excludeRank >= nRanks) {
    ERROR("Require MPI size >= 2 and exclude_rank < MPI size");
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
  config.shrinkShare = args.shrinkShare;
  for (int iter = 0; iter < args.iters; iter++) {
    pcclUniqueId id;
    pcclComm_t comm = NULL, newcomm = NULL;
    if (myRank == 0) PCCLCHECK(pcclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    PCCLCHECK(pcclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
    waitAsync(comm);
    // Establish connections before one rank stops participating in collectives.
    runAllReduce(comm, stream, sendbuff, recvbuff, myRank, nRanks, 2 * iter + 1);
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (args.inFlight && myRank != args.excludeRank) {
      PCCLCHECK(pcclAllReduce(sendbuff, recvbuff, 1024, pcclInt32, pcclSum, comm, stream));
      waitAsync(comm);
      hggcError_t status = hggcStreamQuery(stream);
      if (status != hggcErrorNotReady) {
        ERROR("Rank %d: expected pending AllReduce, stream status %s", myRank, hggcGetErrorString(status));
        exit(testInternalError);
      }
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (myRank != args.excludeRank) {
      INFO("Rank %d cycle %d: entering Revoke with %s communicator", myRank, iter + 1,
           args.inFlight ? "pending AllReduce on" : "idle");
      fflush(stdout);
      PCCLCHECK(pcclCommRevoke(comm, PCCL_REVOKE_DEFAULT));
      waitAsync(comm);
      HGGCCHECK(hggcStreamSynchronize(stream));
      if (pcclAllReduce(sendbuff, recvbuff, 1024, pcclInt32, pcclSum, comm, stream) != pcclInvalidUsage) {
        ERROR("Rank %d: AllReduce on revoked communicator was not rejected with pcclInvalidUsage", myRank);
        exit(testInternalError);
      }
      if (pcclCommRevoke(comm, PCCL_REVOKE_DEFAULT) != pcclInvalidArgument) {
        ERROR("Rank %d: duplicate Revoke was not rejected with pcclInvalidArgument", myRank);
        exit(testInternalError);
      }
      PCCLCHECK(pcclCommShrink(comm, &args.excludeRank, 1, &newcomm, &config, PCCL_SHRINK_DEFAULT));
      // Shrink completion is reported on the parent communicator.
      waitAsync(comm);
      checkComm(newcomm, myRank - (myRank > args.excludeRank), nRanks - 1, localRank);
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (myRank == args.excludeRank) {
      PCCLCHECK(pcclCommAbort(comm));
    } else {
      PCCLCHECK(pcclCommDestroy(comm));
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (myRank != args.excludeRank) {
      int newRank = myRank - (myRank > args.excludeRank);
      runAllReduce(newcomm, stream, sendbuff, recvbuff, newRank, nRanks - 1, 2 * iter + 2);
      PCCLCHECK(pcclCommFinalize(newcomm));
      waitAsync(newcomm);
      PCCLCHECK(pcclCommDestroy(newcomm));
    }
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (myRank == 0) {
      INFO("Cycle %d: Revoke -> Shrink %d -> %d, exclude %d, blocking %d, shrinkShare %d: AllReduce verified",
           iter + 1, nRanks, nRanks - 1, args.excludeRank, args.blocking, args.shrinkShare);
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
