//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <stdio.h>
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
#include <sys/time.h>
#include <pthread.h>

enum COLL_TYPE {
  ALL_REDUCE,
  ALL_TO_ALL,
  NUM_COLL
};

enum ABORT_TYPE {
  NO_ABORT,
  ABORT_DURING_RUNNING,
  ABORT_ON_COMPLETION,
  NO_DESTROY_NO_ABORT,
  NUM_ABORT
};

struct runArgs {
  int nThreads;
  int nGpus;
  size_t bytes;
  int iters;
  ABORT_TYPE abortType;
  COLL_TYPE coll;
};

struct testThread {
  int threadId;
  pthread_t thread;
  testResult_t ret;
  int nGpus;
  int* gpus;
  void** sendbuffs;
  void** recvbuffs;
  size_t count;
  pcclComm_t* comms;
  hggcStream_t* streams;
  double secondsElapsed;
  int nIters;
  COLL_TYPE coll;
  ABORT_TYPE abortType;
  int* ranks;
};

pthread_barrier_t barrierIn;
pthread_barrier_t barrierOut;

static double parseSize(const char *value) {
    long long int units;
    double size;
    char size_lit;

    int count = sscanf(value, "%lf %1s", &size, &size_lit);

    switch (count) {
    case 2:
      switch (size_lit) {
      case 'G':
      case 'g':
        units = 1024*1024*1024;
        break;
      case 'M':
      case 'm':
        units = 1024*1024;
        break;
      case 'K':
      case 'k':
        units = 1024;
        break;
      default:
        return -1.0;
      };
      break;
    case 1:
      units = 1;
      break;
    default:
      return -1.0;
    }

    return size * units;
}

testResult_t parseArguments(struct runArgs *args, int argc, char **argv) {
  double parsed;
  int longindex;
  static struct option longopts[] = {
    {"nthreads", required_argument, 0, 't'},
    {"ngpus", required_argument, 0, 'g'},
    {"bytes", required_argument, 0, 'b'},
    {"iters", required_argument, 0, 'n'},
    {"abort_type", required_argument, 0, 'a'},
    {"coll", required_argument, 0, 'c'},
    {"help", no_argument, 0, 'h'},
    {}
  };
  while(1) {
    int c;
    opterr = 0;
    c = getopt_long(argc, argv, "t:g:b:n:a:c:h", longopts, &longindex);

    if (c == -1)
      break;

    switch(c) {
      case 't':
        args->nThreads = strtol(optarg, NULL, 0);
        break;
      case 'g':
        args->nGpus = strtol(optarg, NULL, 0);
        break;
      case 'b':
        parsed = parseSize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'bytes'\n");
          return testInternalError;
        }
        args->bytes = (size_t)parsed;
        break;
      case 'n':
        args->iters = (int)strtol(optarg, NULL, 0);
        break;
      case 'a':
        args->abortType = (ABORT_TYPE)strtol(optarg, NULL, 0);
        if (args->abortType < 0 || args->abortType >= NUM_ABORT) {
          fprintf(stderr, "invalid abortType, should be 0/1/2/3 \n");
          return testInternalError;
        }
        break;
      case 'c':
        args->coll = (COLL_TYPE)strtol(optarg, NULL, 0);
        if (args->coll < 0 || args->coll >= NUM_COLL) {
          fprintf(stderr, "invalid coll, should be 0 (allReduce) or 1 (alltoall)\n");
          return testInternalError;
        }
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
        printf("USAGE: %s \n\t"
          "[-t,--nthreads <num threads>] \n\t"
          "[-g,--ngpus <gpus per thread>] \n\t"
          "[-b,--bytes <size in bytes>] \n\t"
          "[-n,--iters <iteration count>] \n\t"
          "[-a,--abort_type <abort_type, 0 destroy, 1 running time abort, 2 finish work abort, 3 neither abort nor destroy>] \n\t"
          "[-c,--coll <coll, 0 allreduce, 1 alltoall>] \n\t"
          "[-h,--help]\n",
        basename(argv[0]));
        exit(testSuccess);
    }
  }
  return testSuccess;
}

testResult_t getMemInfo(int* gpus, size_t* mems, int nDevs) {
  size_t freeMemory, totalMemory;
  for (int i = 0; i < nDevs; i++) {
    HGGCCHECK(hggcSetDevice(gpus[i]));
    HGGCCHECK(hggcDeviceReset());
    HGGCCHECK(hggcMemGetInfo(&freeMemory, &totalMemory));
    mems[i] = freeMemory;
  }
  return testSuccess;
}

testResult_t checkMemInfo(size_t* initMems, size_t* finalMems, int nDevs) {
  for (int i = 0; i < nDevs; i++) {
    if (initMems[i] - finalMems[i] > 10*1024*1024) {
      printf("dev %d memleak! init freeBuffer is %.4f GB, final freeBuffer is %.4f GB, remaining %.2f MB\n", i, initMems[i]/(1024.0*1024*1024), finalMems[i]/(1024.0*1024*1024), (initMems[i]-finalMems[i])/(1024.0*1024));
    }
  }
  return testSuccess;
}

testResult_t testStreamSynchronize(int ngpus, hggcStream_t* streams, pcclComm_t* comms, int abort, int* ranks) {
  hggcError_t hggcErr;
  int remaining = ngpus;
  int* done = (int*)malloc(sizeof(int)*ngpus);
  memset(done, 0, sizeof(int)*ngpus);

  while (remaining) {
    int idle = 1;
    for (int i=0; i<ngpus; i++) {
      if (done[i]) continue;

      hggcErr = hggcStreamQuery(streams[i]);
      if (hggcErr == hggcSuccess) {
        done[i] = 1;
        remaining--;
        idle = 0;
        continue;
      }
      if (hggcErr != hggcErrorNotReady) HGGCCHECK(hggcErr);

      if (abort == ABORT_DURING_RUNNING) {
        for (int i=0; i<ngpus; i++) {
          PCCLCHECK(pcclCommAbort(comms[i]));
        }
        printf("Running time abort, first rank is %d remaining is %d, total is %d.\n", ranks[0], remaining, ngpus);
        free(done);
        return testSuccess;
      }
    }
    if (idle) sched_yield();
  }
  free(done);
  return testSuccess;
}

void* runThreadTest(void* thread_) {
  struct testThread* args = (struct testThread*)thread_;
  args->ret = testInternalError;
  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);
  PCCLCHECK(pcclGroupStart());
  for (int i=0; i<args->nGpus; i++) {
    HGGCCHECK(hggcSetDevice(args->gpus[i]));
    if (args->coll == 0) {
      PCCLCHECK(pcclAllReduce((const void*)args->sendbuffs[i], (void*)args->recvbuffs[i], args->count, pcclFloat, pcclSum, args->comms[i], args->streams[i]));
    } else if (args->coll == 1) {
      PCCLCHECK(pcclAlltoAll((const void*)args->sendbuffs[i], (void*)args->recvbuffs[i], args->count, pcclFloat, args->comms[i], args->streams[i]));
    }
  }
  PCCLCHECK(pcclGroupEnd());
  TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms, args->nIters > 1 ? 0 : args->abortType, args->ranks));

  pthread_barrier_wait(&barrierIn);
#ifdef MPI_SUPPORT
  // check all comms are ready
  if (args->threadId == 0) MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
#endif
  pthread_barrier_wait(&barrierOut);

  if (args->nIters > 1) {
    gettimeofday(&start_time, NULL);
    for (int iter = 0; iter < args->nIters-1; iter++) {
      PCCLCHECK(pcclGroupStart());
      for (int i=0; i<args->nGpus; i++) {
        HGGCCHECK(hggcSetDevice(args->gpus[i]));
        if (args->coll == ALL_REDUCE) {
          PCCLCHECK(pcclAllReduce((const void*)args->sendbuffs[i], (void*)args->recvbuffs[i], args->count, pcclFloat, pcclSum, args->comms[i], args->streams[i]));
        } else if (args->coll == ALL_TO_ALL) {
          PCCLCHECK(pcclAlltoAll((const void*)args->sendbuffs[i], (void*)args->recvbuffs[i], args->count, pcclFloat, args->comms[i], args->streams[i]));
        }
      }
      PCCLCHECK(pcclGroupEnd());
    }
    TESTCHECK(testStreamSynchronize(args->nGpus, args->streams, args->comms, args->abortType, args->ranks));
  }
  gettimeofday(&end_time, NULL);
  args->secondsElapsed = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
  int nIters = args->nIters > 1 ? (args->nIters - 1) : 1;
  args->secondsElapsed /= nIters;
  // printf("secondsElapsed:%2f\n", args->secondsElapsed);
  args->ret = testSuccess;
  return NULL;
}

int main(int argc, char* argv[]) {
  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);

  testResult_t result = testSuccess;
  int totalProcs = 1, myProc = 0;
  int localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);
#ifdef MPI_SUPPORT
  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myProc));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &totalProcs));

  char* needSleepStr = getenv("NCCL_TEST_SLEEP");
  int needSleep = needSleepStr ? atoi(needSleepStr) : 0;
  if (myProc == 0 && needSleep) {
    printf("[DEBUG] Proc 0 (PID: %d) is paused because NCCL_TEST_SLEEP is set.\n To continue in GDB:\n 1. Switch to the 'run()' frame (e.g., 'frame <N>').\n 2. Execute 'set var flag = 1'.\n", getpid());
  }
  volatile int flag = 0;
  // Proc 0 blocks on the flag, which must be manually changed in a debugger (e.g., GDB). All other procs will block in MPI_Allgather waiting for proc 0.
  while (myProc==0 && needSleep && flag==0) {
    sleep(5);
  }

  uint64_t hostHashs[totalProcs];
  hostHashs[myProc] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  for (int p=0; p<totalProcs; p++) {
    if (p == myProc) break;
    if (hostHashs[p] == hostHashs[myProc]) localRank++;
  }
#endif

  struct runArgs args;
  args.nGpus = 1;
  args.nThreads = 1;
  args.bytes = 1024*1024;
  args.iters = 1;
  args.abortType = NO_ABORT;
  args.coll = ALL_REDUCE;

  TESTCHECK(parseArguments(&args, argc, argv));
  int nGpus = args.nGpus;
  int nThreads = args.nThreads;
  int nRanks = totalProcs * nGpus * nThreads;
  if (myProc == 0) printf("nProcs: %d, nRanks:%d, parse args: nThreads: %d, nGpus: %d, bytes: %zu, iters: %d, abortType: %d, coll: %d\n", totalProcs, nRanks, args.nThreads, args.nGpus, args.bytes, args.iters, args.abortType, args.coll);

  pthread_barrier_init(&barrierIn, NULL, nThreads);
  pthread_barrier_init(&barrierOut, NULL, nThreads);

  pcclUniqueId pcclId;
  if (myProc == 0){
    PCCLCHECK(pcclGetUniqueId(&pcclId));
  }
#ifdef MPI_SUPPORT
  MPICHECK(MPI_Bcast((void *)&pcclId, sizeof(pcclId), MPI_BYTE, 0, MPI_COMM_WORLD));
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
#endif
  pcclComm_t* comms = (pcclComm_t*)malloc(sizeof(pcclComm_t)*nThreads*nGpus);
  int gpus[nGpus*nThreads];
  hggcStream_t streams[nGpus*nThreads];
  void* sendbuffs[nGpus*nThreads];
  void* recvbuffs[nGpus*nThreads];
  int ranks[nGpus*nThreads];
  size_t sendBytes = args.bytes;
  size_t recvBytes = args.bytes;
  size_t count = sendBytes / sizeof(float);
  size_t initFreeMemory[nGpus*nThreads];
  size_t finalFreeMemory[nGpus*nThreads];

  for (int i=0; i<nGpus*nThreads; i++) {
    gpus[i] = localRank*nThreads*nGpus + i;
  }

  TESTCHECK(getMemInfo(gpus, initFreeMemory, nGpus*nThreads));

  for (int i=0; i<nGpus*nThreads; i++) {
    HGGCCHECK(hggcSetDevice(gpus[i]));
    HGGCCHECK(hggcMalloc(sendbuffs+i, sendBytes));
    HGGCCHECK(hggcMalloc(recvbuffs+i, recvBytes));
    HGGCCHECK(hggcStreamCreateWithFlags(streams+i, hggcStreamNonBlocking));
  }
  PCCLCHECK(pcclGroupStart());
  for (int i=0; i<nGpus*nThreads; i++) {
    HGGCCHECK(hggcSetDevice(gpus[i]));
    PCCLCHECK(pcclCommInitRank(comms+i, totalProcs*nThreads*nGpus, pcclId, myProc*nThreads*nGpus+i));
    ranks[i] = myProc*nThreads*nGpus+i;
  }
  PCCLCHECK(pcclGroupEnd());

#ifdef MPI_SUPPORT
  // check all comms are ready
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
#endif
  if (myProc == 0) printf("init comms done\n");

  struct testThread threads[nThreads];
  memset(threads, 0, sizeof(struct testThread)*nThreads);

  for (int t = 0; t < nThreads; t++) {
    threads[t].threadId = t;
    threads[t].nGpus = nGpus;
    threads[t].gpus = gpus + t*nGpus;
    threads[t].sendbuffs = sendbuffs + t*nGpus;
    threads[t].recvbuffs = recvbuffs + t*nGpus;
    threads[t].comms = comms + t*nGpus;
    threads[t].streams = streams + t*nGpus;
    threads[t].nIters = args.iters;
    threads[t].count = args.coll == 0 ? sendBytes / sizeof(float) : sendBytes / sizeof(float) / nRanks;
    threads[t].coll = args.coll;
    threads[t].abortType = args.abortType;
    threads[t].ranks = ranks + t*nGpus;
    pthread_create(&threads[t].thread, NULL, runThreadTest, (void*)(threads+t));
  }

  double secondsElapsed = 0.0;
  for (int t = 0; t < nThreads; t++) {
    pthread_join(threads[t].thread, NULL);
    TESTCHECK(threads[t].ret);
    secondsElapsed += threads[t].secondsElapsed;
  }
  secondsElapsed /= nThreads;
#ifdef MPI_SUPPORT
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &secondsElapsed, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  secondsElapsed /= totalProcs;
#endif
  if (myProc==0) printf("run colls done, alg bw is %.2fGB/s\n", 1.0*sendBytes/(1024*1024*1024)/secondsElapsed);

  for (int i=0; i<nGpus*nThreads; i++) {
    if (args.abortType == NO_ABORT) {
      PCCLCHECK(pcclCommDestroy(comms[i]));
    } else if (args.abortType == ABORT_ON_COMPLETION) {
      PCCLCHECK(pcclCommAbort(comms[i]));
    }
  }
  if (comms != NULL) free(comms);
#ifdef MPI_SUPPORT
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  if (myProc==0) printf("destroy comms done\n");

  for (int i=0; i<nGpus*nThreads; i++) {
    if (sendbuffs[i]) HGGCCHECK(hggcFree(sendbuffs[i]));
    if (recvbuffs[i]) HGGCCHECK(hggcFree(recvbuffs[i]));
    if (streams[i]) HGGCCHECK(hggcStreamDestroy(streams[i]));
  }

  pthread_barrier_destroy(&barrierIn);
  pthread_barrier_destroy(&barrierOut);

  TESTCHECK(getMemInfo(gpus, finalFreeMemory, nGpus*nThreads));
  TESTCHECK(checkMemInfo(initFreeMemory,finalFreeMemory,nGpus*nThreads));

#ifdef MPI_SUPPORT
  //finalizing MPI
  MPICHECK(MPI_Finalize());
#endif
  return 0;
}