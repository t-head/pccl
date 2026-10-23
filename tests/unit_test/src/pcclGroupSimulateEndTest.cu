//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "pccl.h"
#include "hggc_runtime.h"
#include "logger.h"
#include "cxxopts.hpp"
#ifdef MPI_SUPPORT
#include "mpi.h"
#endif
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define MAX_N_DEVS 16
#define MAX_GROUP_OPS 32

/******************************************************************************/
/*                          supported collective ops                         */
/******************************************************************************/

enum CollType {
  COLL_ALLREDUCE = 0,
  COLL_ALLGATHER,
  COLL_REDUCESCATTER,
  COLL_BROADCAST,
  COLL_REDUCE,
};

struct OpEntry {
  const char *name;
  CollType type;
};

static const OpEntry kOpTable[] = {
  {"all_reduce",     COLL_ALLREDUCE},
  {"all_gather",     COLL_ALLGATHER},
  {"reduce_scatter", COLL_REDUCESCATTER},
  {"broadcast",      COLL_BROADCAST},
  {"reduce",         COLL_REDUCE},
};
static const int kOpTableSize = sizeof(kOpTable) / sizeof(kOpTable[0]);

struct runArgs {
  int nGpus;
  pcclDataType_t dataType;
  const char *typeName;
  size_t beginBytes;
  size_t endBytes;
  size_t stepFactor;
  int root;
  std::vector<CollType> ops;
};

/******************************************************************************/
/*                                  helpers                                   */
/******************************************************************************/

static bool parseDataType(const std::string &s, pcclDataType_t *type,
                          const char **name) {
  static const struct {
    const char *n;
    pcclDataType_t t;
  } tbl[] = {
    {"int8", pcclInt8},       {"uint8", pcclUint8},
    {"int32", pcclInt32},     {"uint32", pcclUint32},
    {"int64", pcclInt64},     {"uint64", pcclUint64},
    {"half", pcclFloat16},    {"float", pcclFloat32},
    {"double", pcclFloat64},  {"bfloat16", pcclBfloat16},
    {"fp8_e4m3", pcclFloat8e4m3}, {"fp8_e5m2", pcclFloat8e5m2},
  };
  for (size_t i = 0; i < sizeof(tbl) / sizeof(tbl[0]); i++) {
    if (s == tbl[i].n) {
      *type = tbl[i].t;
      *name = tbl[i].n;
      return true;
    }
  }
  return false;
}

static size_t wordSize(pcclDataType_t type) {
  switch (type) {
    case pcclInt8:
    case pcclUint8:
    case pcclFloat8e4m3:
    case pcclFloat8e5m2:
      return 1;
    case pcclFloat16:
    case pcclBfloat16:
      return 2;
    case pcclInt32:
    case pcclUint32:
    case pcclFloat32:
      return 4;
    case pcclInt64:
    case pcclUint64:
    case pcclFloat64:
      return 8;
    default:
      return 4;
  }
}

static bool parseSize(const std::string &s, size_t *bytes) {
  double value;
  char suffix;
  size_t units = 1;
  int matched = sscanf(s.c_str(), "%lf %1s", &value, &suffix);
  if (matched < 1 || value <= 0) {
    return false;
  }
  if (matched == 2) {
    switch (suffix) {
      case 'K':
      case 'k':
        units = (size_t)1 << 10;
        break;
      case 'M':
      case 'm':
        units = (size_t)1 << 20;
        break;
      case 'G':
      case 'g':
        units = (size_t)1 << 30;
        break;
      default:
        return false;
    }
  }
  double total = value * units;
  if (total < 1.0 || total > (double)SIZE_MAX) {
    return false;
  }
  *bytes = (size_t)total;
  return true;
}

static CollType parseOp(const std::string &name) {
  for (int i = 0; i < kOpTableSize; i++) {
    if (name == kOpTable[i].name) {
      return kOpTable[i].type;
    }
  }
  ERROR("Unsupported op '%s' in --op_list. Supported: "
        "all_reduce/all_gather/reduce_scatter/broadcast/reduce",
        name.c_str());
  exit(testInternalError);
  return COLL_ALLREDUCE;  // unreachable, silences -Wreturn-type
}

static void getCollByteCount(CollType type, size_t count, int nranks,
                             size_t *sendCount, size_t *recvCount) {
  switch (type) {
    case COLL_ALLGATHER:
      *sendCount = count;
      *recvCount = count * nranks;
      break;
    case COLL_REDUCESCATTER:
      *sendCount = count * nranks;
      *recvCount = count;
      break;
    case COLL_ALLREDUCE:
    case COLL_BROADCAST:
    case COLL_REDUCE:
    default:
      *sendCount = count;
      *recvCount = count;
      break;
  }
}

static testResult_t runOneColl(CollType type, void *sendbuff, void *recvbuff,
                               size_t count, pcclDataType_t dtype,
                               pcclRedOp_t op, int root, pcclComm_t comm,
                               hggcStream_t stream) {
  switch (type) {
    case COLL_ALLREDUCE:
      PCCLCHECK(pcclAllReduce(sendbuff, recvbuff, count, dtype, op, comm, stream));
      break;
    case COLL_ALLGATHER:
      PCCLCHECK(pcclAllGather(sendbuff, recvbuff, count, dtype, comm, stream));
      break;
    case COLL_REDUCESCATTER:
      PCCLCHECK(pcclReduceScatter(sendbuff, recvbuff, count, dtype, op, comm, stream));
      break;
    case COLL_BROADCAST:
      PCCLCHECK(pcclBroadcast(sendbuff, recvbuff, count, dtype, root, comm, stream));
      break;
    case COLL_REDUCE:
      PCCLCHECK(pcclReduce(sendbuff, recvbuff, count, dtype, op, root, comm, stream));
      break;
  }
  return testSuccess;
}

static void printHeader() {
  printf("#\n");
  printf("# %10s  %12s  %8s  %6s  %5s  %12s\n",
         "size", "count", "type", "redop", "root", "est_time");
  printf("# %10s  %12s  %8s  %6s  %5s  %12s\n",
         "(B)", "(elements)", "", "", "", "(us)");
}

/******************************************************************************/
/*                                pre process                                 */
/******************************************************************************/

void parseAndSetupArguments(int argc, char **argv, struct runArgs *args,
                            int myRank) {
  cxxopts::Options options(
      "pcclGroupSimulateEndTest",
      "PCCL group simulation test using pcclGroupSimulateEnd(). All ops in "
      "--op_list are placed into one pcclGroupStart()/pcclGroupSimulateEnd() "
      "scope and a single estimated time is returned.");
  options.add_options()
    ("g,num_gpus", "Number of local GPUs per process.",
            cxxopts::value<int>()->default_value("2"))
    ("d,data_type", "Data type: int8/uint8/int32/uint32/int64/uint64/half/"
            "float/double/bfloat16/fp8_e4m3/fp8_e5m2.",
            cxxopts::value<std::string>()->default_value("float"))
    ("b,begin_size", "Begin data size in bytes; K/M/G suffixes allowed.",
            cxxopts::value<std::string>()->default_value("32"))
    ("e,end_size", "End data size in bytes; K/M/G suffixes allowed.",
            cxxopts::value<std::string>()->default_value("32"))
    ("f,step_factor", "Multiplication factor between sizes.",
            cxxopts::value<size_t>()->default_value("2"))
    ("r,root", "Root rank for broadcast/reduce.",
            cxxopts::value<int>()->default_value("0"))
    ("op_list", "Comma separated collective ops placed into one "
            "pcclGroupStart()/pcclGroupSimulateEnd() scope, e.g. "
            "all_reduce,all_gather. Supported: "
            "all_reduce/all_gather/reduce_scatter/broadcast/reduce.",
            cxxopts::value<std::string>())
    ("h,help", "Print usage");

  auto params = options.parse(argc, argv);

  // Print help message.
  if (params.count("help")) {
    if (myRank == 0) {
      std::cout << options.help() << std::endl;
    }
    exit(testSuccess);
  }

  // Parse num_gpus.
  args->nGpus = params["num_gpus"].as<int>();
  int maxNDevs = 0;
  HGGCCHECK(hggcGetDeviceCount(&maxNDevs));
  if (args->nGpus <= 0) {
    ERROR("Invalid GPU number %d!", args->nGpus);
    exit(testInternalError);
  } else if (args->nGpus > maxNDevs) {
    ERROR("Exceed the maximum number of GPU, which is %d, but input is %d!",
          maxNDevs, args->nGpus);
    exit(testInternalError);
  } else if (args->nGpus > MAX_N_DEVS) {
    ERROR("Exceed the maximum number of GPU per process supported by this "
          "test, which is %d, but input is %d!", MAX_N_DEVS, args->nGpus);
    exit(testInternalError);
  }

  // Parse data_type.
  std::string dt = params["data_type"].as<std::string>();
  if (!parseDataType(dt, &args->dataType, &args->typeName)) {
    ERROR("Invalid data_type '%s'!", dt.c_str());
    exit(testInternalError);
  }

  // Parse data sizes.
  std::string beginStr = params["begin_size"].as<std::string>();
  std::string endStr = params["end_size"].as<std::string>();
  if (!parseSize(beginStr, &args->beginBytes)) {
    ERROR("Invalid begin_size '%s'!", beginStr.c_str());
    exit(testInternalError);
  }
  if (!parseSize(endStr, &args->endBytes)) {
    ERROR("Invalid end_size '%s'!", endStr.c_str());
    exit(testInternalError);
  }
  args->stepFactor = params["step_factor"].as<size_t>();
  if (args->stepFactor < 2) {
    args->stepFactor = 2;
  }
  if (args->endBytes < args->beginBytes) {
    args->endBytes = args->beginBytes;
  }

  args->root = params["root"].as<int>();

  // Parse op_list.
  if (!params.count("op_list")) {
    ERROR("--op_list is required, e.g. --op_list all_reduce,all_gather");
    exit(testInternalError);
  }
  std::istringstream ss(params["op_list"].as<std::string>());
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token.empty()) {
      continue;
    }
    if ((int)args->ops.size() == MAX_GROUP_OPS) {
      ERROR("--op_list holds more than %d ops", MAX_GROUP_OPS);
      exit(testInternalError);
    }
    args->ops.push_back(parseOp(token));
  }
  if (args->ops.empty()) {
    ERROR("--op_list holds no valid op");
    exit(testInternalError);
  }
}

/******************************************************************************/
/*                                    main                                    */
/******************************************************************************/

int main(int argc, char *argv[]) {
  int proc = 0, nProcs = 1;

#ifdef MPI_SUPPORT
  // initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &proc));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nProcs));
#endif

  struct runArgs args;
  parseAndSetupArguments(argc, argv, &args, proc);

  int nGpus = args.nGpus;
  int nRanks = nProcs * nGpus;
  pcclRedOp_t redOp = pcclSum;
  const char *redOpName = "sum";

  if (args.root < 0 || args.root >= nRanks) {
    if (proc == 0) {
      ERROR("Invalid root %d, must be in [0, %d).", args.root, nRanks);
    }
    exit(testInternalError);
  }

  pcclUniqueId id;
  if (proc == 0) {
    PCCLCHECK(pcclGetUniqueId(&id));
  }
#ifdef MPI_SUPPORT
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
#endif

  int localProc = 0;
  {
    std::vector<uint64_t> hostHashs(nProcs);
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[proc] = getHostHash(hostname);
#ifdef MPI_SUPPORT
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           hostHashs.data(), sizeof(uint64_t), MPI_BYTE,
                           MPI_COMM_WORLD));
#endif
    for (int p = 0; p < nProcs; p++) {
      if (p == proc) break;
      if (hostHashs[p] == hostHashs[proc]) localProc++;
    }
  }
  int devOffset = localProc * nGpus;
  int maxNDevs = 0;
  HGGCCHECK(hggcGetDeviceCount(&maxNDevs));
  if (devOffset + nGpus > maxNDevs) {
    ERROR("Process %d needs devices [%d, %d) but only %d are visible; reduce "
          "num_gpus or the number of processes per node.",
          proc, devOffset, devOffset + nGpus, maxNDevs);
    exit(testInternalError);
  }

  pcclComm_t comms[MAX_N_DEVS];
  hggcStream_t streams[MAX_N_DEVS];
  void *sendbuffs[MAX_N_DEVS];
  void *recvbuffs[MAX_N_DEVS];

  size_t word = wordSize(args.dataType);
  size_t maxSendCount = 0, maxRecvCount = 0;
  for (size_t s = args.beginBytes; s <= args.endBytes;) {
    size_t count = s / word;
    for (size_t i = 0; i < args.ops.size(); i++) {
      size_t sc, rc;
      getCollByteCount(args.ops[i], count, nRanks, &sc, &rc);
      maxSendCount = std::max(maxSendCount, sc);
      maxRecvCount = std::max(maxRecvCount, rc);
    }
    if (s > args.endBytes / args.stepFactor) {
      break;
    }
    s *= args.stepFactor;
  }
  size_t sendBytesMax = std::max((size_t)1, maxSendCount) * word;
  size_t recvBytesMax = std::max((size_t)1, maxRecvCount) * word;

  for (int i = 0; i < nGpus; i++) {
    HGGCCHECK(hggcSetDevice(devOffset + i));
    HGGCCHECK(hggcStreamCreateWithFlags(&streams[i], hggcStreamNonBlocking));
    HGGCCHECK(hggcMalloc(&sendbuffs[i], sendBytesMax));
    HGGCCHECK(hggcMalloc(&recvbuffs[i], recvBytesMax));
  }

  PCCLCHECK(pcclGroupStart());
  for (int i = 0; i < nGpus; i++) {
    HGGCCHECK(hggcSetDevice(devOffset + i));
    PCCLCHECK(pcclCommInitRank(&comms[i], nRanks, id, proc * nGpus + i));
  }
  PCCLCHECK(pcclGroupEnd());

  if (proc == 0) {
    printHeader();
  }

  testResult_t result = testSuccess;
  for (size_t size = args.beginBytes; size <= args.endBytes;) {
    size_t count = size / word;
    if (count > 0) {
      size_t sendBytes = 0, recvBytes = 0;
      for (size_t i = 0; i < args.ops.size(); i++) {
        size_t sc, rc;
        getCollByteCount(args.ops[i], count, nRanks, &sc, &rc);
        sendBytes = std::max(sendBytes, sc * word);
        recvBytes = std::max(recvBytes, rc * word);
      }

      pcclSimInfo_t sim = PCCL_SIM_INFO_INITIALIZER;
      PCCLCHECK(pcclGroupStart());
      for (int i = 0; i < nGpus; i++) {
        HGGCCHECK(hggcSetDevice(devOffset + i));
        for (size_t c = 0; c < args.ops.size(); c++) {
          TESTCHECK(runOneColl(args.ops[c], sendbuffs[i], recvbuffs[i], count,
                               args.dataType, redOp, args.root, comms[i],
                               streams[i]));
        }
      }
      PCCLCHECK(pcclGroupSimulateEnd(&sim));

      if (proc == 0) {
        printf("%12li  %12li  %8s  %6s  %5d  %12.2f (us)\n",
               (long)std::max(sendBytes, recvBytes), (long)count,
               args.typeName, redOpName, args.root, sim.estimatedTime);
      }
    }
    if (size > args.endBytes / args.stepFactor) {
      break;
    }
    size *= args.stepFactor;
  }

  for (int i = 0; i < nGpus; i++) {
    HGGCCHECK(hggcSetDevice(devOffset + i));
    PCCLCHECK(pcclCommDestroy(comms[i]));
    HGGCCHECK(hggcStreamDestroy(streams[i]));
    HGGCCHECK(hggcFree(sendbuffs[i]));
    HGGCCHECK(hggcFree(recvbuffs[i]));
  }

#ifdef MPI_SUPPORT
  MPICHECK(MPI_Finalize());
#endif

  return result;
}
