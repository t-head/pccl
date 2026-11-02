//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <stdio.h>
#include <iostream>
#include "hggc_runtime.h"
#include "pccl.h"
#include "logger.h"
#include "cxxopts.hpp"
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

struct runArgs {
  int shrinkShare;
  int shrinkFlag;
  int blocking;
  int excludeCount;
  std::vector<int> excludeRanks;
};

void parseAndSetupArguments(struct runArgs *args, int argc, char **argv, int myRank, int nRanks) {
  cxxopts::Options options("pcclCommShrinkTest", "Test program for PCCL communicator shrink feature.");

  options.add_options()
      ("s,shrink_share", "Flag indicating if this communicator shares resources with parent or children. Possible values: 0 (No sharing), 1 (Sharing enabled).", cxxopts::value<int>()->default_value("0"))
      ("f,shrink_flag", "pcclCommShrink flags. Possible values: 0 (PCCL_SHRINK_DEFAULT), 1 (PCCL_SHRINK_ABORT).", cxxopts::value<int>()->default_value(std::to_string(PCCL_SHRINK_DEFAULT)))
      ("b,blocking", "blocking flags. Possible values: 0 (non blocking), 1 (blocking).", cxxopts::value<int>()->default_value("1"))
      ("r,exclude_ranks", "Specify a rank to exclude. This option can be used multiple times to specify a list of ranks. E.g., `-r 0 -r 2 -r 5`.",
                    cxxopts::value<std::vector<int>>())
      ("h,help", "Print usage");

  auto params = options.parse(argc, argv);

  // Print help message.
  if (params.count("help")) {
    std::cout << options.help() << std::endl;
    exit(testSuccess);
  }

  // Parse arguments.
  args->shrinkShare = params["shrink_share"].as<int>();
  args->shrinkFlag= params["shrink_flag"].as<int>();
  args->blocking= params["blocking"].as<int>();

  if (params.count("exclude_ranks")) {
    args->excludeRanks = params["exclude_ranks"].as<std::vector<int>>();
    args->excludeCount = args->excludeRanks.size();
    if (args->excludeCount > nRanks) {
      ERROR("Rank %d: Exclude ranks %d is more than total ranks %d", myRank, args->excludeCount, nRanks);
      exit(testInternalError);
    }
  } else {
    ERROR("Rank %d: --exclude_ranks option is required.", myRank);
    exit(testInternalError);
  }

  if (myRank == 0) {
    INFO("Starting pccl comm shrink test:\n");
    INFO("  - GPUs: %d\n", nRanks);
    INFO("  - Applying shrinkShare = %d\n", args->shrinkShare);
    INFO("  - Applying shrinkFlag = %s\n", (args->shrinkFlag & PCCL_SHRINK_ABORT) ? "PCCL_SHRINK_ABORT" : "PCCL_SHRINK_DEFAULT");
    INFO("  - Applying blocking = %d\n", args->blocking);
    INFO("  - Number of excluded ranks specified: %d", args->excludeCount);
  }
}

int main(int argc, char* argv[])
{
  testResult_t result = testSuccess;
  int size = 32*1024*1024;
  int myRank, nRanks, localRank = 0;

#ifdef MPI_SUPPORT
  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
#endif
  struct runArgs args;
  parseAndSetupArguments(&args, argc, argv, myRank, nRanks);
  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
#ifdef MPI_SUPPORT
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
#endif

  for (int p=0; p<nRanks; p++) {
    if (p == myRank) break;
    if (hostHashs[p] == hostHashs[myRank]) localRank++;
  }

  pcclUniqueId id;
  pcclComm_t comm;
  float *sendbuff, *recvbuff;
  hggcStream_t s;

  //get PCCL unique ID at myRank 0 and broadcast it to all others
  if (myRank == 0) pcclGetUniqueId(&id);
#ifdef MPI_SUPPORT
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
#endif
  //picking a GPU based on localRank, allocate device buffers
  HGGCCHECK(hggcSetDevice(localRank));
  HGGCCHECK(hggcMalloc(&sendbuff, size * sizeof(float)));
  HGGCCHECK(hggcMalloc(&recvbuff, size * sizeof(float)));
  HGGCCHECK(hggcStreamCreate(&s));

  pcclResult_t state = pcclSuccess;
  pcclConfig_t config = PCCL_CONFIG_INITIALIZER;
  config.blocking = args.blocking;

  //initializing PCCL
  PCCLCHECK(pcclCommInitRankConfig(&comm, nRanks, id, myRank, &config));
  if (args.blocking == 0) {
    do {
      PCCLCHECK(pcclCommGetAsyncError(comm, &state));
    } while(state == pcclInProgress);
  }
  // communicating using PCCL
  PCCLCHECK(pcclAllReduce((const void*)sendbuff, (void*)recvbuff, size, pcclFloat, pcclSum,
          comm, s));
  if (args.blocking == 0) {
    do {
      PCCLCHECK(pcclCommGetAsyncError(comm, &state));
    } while(state == pcclInProgress);
  }
  HGGCCHECK(hggcStreamSynchronize(s));

  // comm shrink test
  pcclComm_t newcomm;
  config.shrinkShare = args.shrinkShare;

  // Only ranks that will be in the new communicator should call pcclCommShrink
  auto it = std::find(args.excludeRanks.begin(), args.excludeRanks.end(), myRank);
  if (it == args.excludeRanks.end()) {
    PCCLCHECK(pcclCommShrink(comm, args.excludeRanks.data(), args.excludeCount, &newcomm, &config, args.shrinkFlag));
    if (args.blocking == 0) {
      do {
        PCCLCHECK(pcclCommGetAsyncError(comm, &state));
      } while(state == pcclInProgress);
    }
    // Use the new communicator for collective operations
    PCCLCHECK(pcclAllReduce((const void*)sendbuff, (void*)recvbuff, size, pcclFloat, pcclSum,
            newcomm, s));
    if (args.blocking == 0) {
      do {
        PCCLCHECK(pcclCommGetAsyncError(newcomm, &state));
      } while(state == pcclInProgress);
    }
    //completing PCCL operation by synchronizing on the HGGC stream
    HGGCCHECK(hggcStreamSynchronize(s));
    // When done, destroy the new communicator
    pcclCommDestroy(newcomm);
  }
#ifdef MPI_SUPPORT
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  //free device buffers
  HGGCCHECK(hggcFree(sendbuff));
  HGGCCHECK(hggcFree(recvbuff));

  // finalizing PCCL
  pcclCommDestroy(comm);

#ifdef MPI_SUPPORT
  //finalizing MPI
  MPICHECK(MPI_Finalize());
#endif
  INFO("[Rank %d] Success", myRank);
  return 0;
}