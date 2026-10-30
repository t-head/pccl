//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <pthread.h>
#include <cstdio>
#include <libgen.h>
#include <sys/resource.h>
#include <map>
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>

#include "nlohmann/json.hpp"

#include "hggc_runtime.h"
#include "pccl.h"
#include "cxxopts.hpp"
#include "logger.h"

#ifdef MPI_SUPPORT
#include "mpi.h"
#endif

using json = nlohmann::json;

// Command line parameter defaults
static int nThreads = 8;
static int nGpus = 1;  // gpus per thread
static int core_dump=0;
static int init_blocking_timeout = 600; // default 10m
int worldSize = 0;

#define MAX_COMM_NUM 8

void GetCommSplitInfo(int commIdx, int rank, int nRanks, int* pColor, int* pKey) {
  switch (commIdx) {
    case 1: {
    // same as parent comm
      *pColor = 0;
      *pKey = rank;
      break;
    }
    case 2: {
      if (nRanks >= 2) {
        *pColor = rank/(nRanks/2);
        *pKey = rank%(nRanks/2);
        break;
      } else {
        ERROR("nRanks %d, can not use comm %d split strategy\n", commIdx, nRanks);
      }
    }
    case 3: {
      if (nRanks >= 2) {
        *pColor = rank%(nRanks/2);
        *pKey = rank/(nRanks/2);
        break;
      } else {
        ERROR("nRanks %d, can not use comm %d split strategy\n", commIdx, nRanks);
      }
    }
    case 4: {
      if (nRanks >= 4) {
        *pColor = rank/(nRanks/4);
        *pKey = rank%(nRanks/4);
        break;
      } else {
        ERROR("nRanks %d, can not use comm %d split strategy\n", commIdx, nRanks);
      }
    }
    case 5: {
      if (nRanks >= 4) {
        *pColor = rank%(nRanks/4);
        *pKey = rank/(nRanks/4);
        break;
      } else {
        ERROR("nRanks %d, can not use comm %d split strategy\n", commIdx, nRanks);
      }
    }
    case 6: {
      if (nRanks >= 8) {
        *pColor = rank/(nRanks/8);
        *pKey = rank%(nRanks/8);
        break;
      } else {
        ERROR("nRanks %d, can not use comm %d split strategy\n", commIdx, nRanks);
      }
    }
    case 7: {
      if (nRanks >= 8) {
        *pColor = rank%(nRanks/8);
        *pKey = rank/(nRanks/8);
        break;
      } else {
        ERROR("nRanks %d, can not use comm %d split strategy\n", commIdx, nRanks);
      }
    }
    default: {
      ERROR("fall back to default comm split strategy\n");
      *pColor = 0;
      *pKey = rank;
    }
  }
}

struct threadArgs {
  int nProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int* gpus;
  int niters;
  pcclComm_t* comms;
  json *configs;

  int* errors;
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread {
  pthread_t thread;
  threadFunc_t func;
  struct threadArgs args;
  testResult_t ret;
};

template<typename T>
__global__ void incrementKernel(T* ptr, int n) {
  int offset = blockIdx.x*blockDim.x + threadIdx.x;
  while(offset < n) {
    ptr[offset] += (T)1;
    offset += gridDim.x*blockDim.x;
  }
}

template<typename T>
__global__ void initData(T* ptr, T value, int n) {
  int offset = blockIdx.x*blockDim.x + threadIdx.x;
  while(offset < n) {
    ptr[offset] = value;
    offset += gridDim.x*blockDim.x;
  }
}

template<typename T>
__global__ void verifyData(T* ptr1, T* ptr2 , int n, int* flag) {
  int offset = blockIdx.x*blockDim.x + threadIdx.x;
  while(offset < n) {
    if(ptr1[offset] != ptr2[offset]) {
      atomicAdd(flag, 1);
      break;
    }
    offset += gridDim.x*blockDim.x;
  }
}

std::string fullfillStrWithDev(std::string& str, int dev) {
  std::string res = str + "_gpu_" + std::to_string(dev);
  return res;
}

bool isGlobalStream(std::string& streamName) {
  return streamName.find("global") != std::string::npos;
}

bool isGlobalEvent(std::string& eventName) {
  return eventName.find("global") != std::string::npos;
}

static const std::map<std::string, int> sizeOfDataType = {
  {"int", 4},
  {"float", 4},
  {"double", 8},
  {"half", 2},
  {"int8", 1},
  {"uint8", 1},
  {"int32", 4},
  {"uint32", 4},
  {"int64", 8},
  {"uint64", 8},
  {"bf16", 2}
};

static const std::map<std::string, pcclDataType_t> pcclTypeMap = {
  {"int", pcclInt32},
  {"float", pcclFloat},
  {"double", pcclDouble},
  {"half", pcclHalf},
  {"int8", pcclInt8},
  {"uint8", pcclUint8},
  {"int32", pcclInt32},
  {"uint32", pcclUint32},
  {"int64", pcclInt64},
  {"uint64", pcclUint64},
  {"bf16", pcclBfloat16}
};

struct Task {
  bool usingGraph;   // if usingGraph is true, the first launch would be a capture;
  std::string name;
  int commIdx;
  std::map<std::string, hggcStream_t> streams;
  std::map<std::string, hggcEvent_t> events;
  std::map<std::string, int> buffsizes;
  std::map<std::string, void*> buffs;
  std::vector<struct BaseOp*> ops;
  std::vector<struct BaseOp*> initOps;
  hggcGraph_t graph = nullptr;
  hggcGraphExec_t exec = nullptr;
  int iters = INT_MAX;
  std::map<std::string, hggcStream_t> *globalStreams;  // shared by all tasks, use useGlobalStream to check use global stream or not.
  std::vector<std::string> globalStreamNames;
  bool useGlobalStream = false;
  std::map<std::string, hggcEvent_t> *globalEvents;
  std::vector<std::string> globalEventNames;
  bool useGlobalEvent = false;
  std::string graphStreamName;
  int devId;
};

// Query InProgress done
testResult_t waitReady(int nGpus, pcclComm_t* comms) {
  int commCnts = nGpus;
  testResult_t ret = testSuccess;
  pcclResult_t state = pcclSuccess;
  auto startTime = std::chrono::steady_clock::now();
  while (commCnts) {
    PCCLCHECK(pcclCommGetAsyncError(comms[commCnts-1], &state));
    if (state == pcclInProgress) {
      if (init_blocking_timeout) {
        auto curTime = std::chrono::steady_clock::now();
        auto elaspeTime = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime).count();
        if (elaspeTime > init_blocking_timeout) {
          ret = testInternalError;
          ERROR("waitReady timeout !");
          break;
        }
      }
      continue;
    }
    // already done
    --commCnts;
  }

  if (ret == testInternalError) {
    for (int i=0; i<nGpus; i++)
      PCCLCHECK(pcclCommAbort(comms[i]));
  }
  return ret;
}

class BaseOp {
 public:
  std::string op;
  std::string streamName;
  std::string eventName;
  std::vector<std::string> buffNames;
  std::vector<void*> buffs;
  hggcStream_t stream;
  hggcEvent_t event;
  std::string typeName;
  int typeSize;
  int count;
  int root;
  int myRank;
  bool isPccl;
  BaseOp() {}

  BaseOp(std::vector<std::string>& args) {  // args[0] always match pccl
    // we put buffs at the tail
    op = args[1];
    if (op == "send" || op == "recv") {
      count = std::atoi(args[2].c_str());
      typeName = args[3];
      myRank = std::atoi(args[4].c_str());
      root = std::atoi(args[5].c_str());
      streamName = args[6];
      for (auto index=7; index<args.size(); ++index){
        buffNames.push_back(args[index]);
      }
      const auto& it = sizeOfDataType.find(typeName);
      if (it != sizeOfDataType.end()) {
        typeSize = it->second;
      } else {
        typeSize = 1;   // prevent 0 buffsize
        ERROR("type size %s not found", typeName.c_str());
      }

    } else if (args.size() >= 6) { // compute or pccl
      count = std::atoi(args[2].c_str());
      typeName = args[3];
      streamName = args[4];
      for (auto index=5; index<args.size(); ++index){
        buffNames.push_back(args[index]);
      }
      const auto& it = sizeOfDataType.find(typeName);
      if (it != sizeOfDataType.end()) {
        typeSize = it->second;
      } else {
        typeSize = 1;   // prevent 0 buffsize
        ERROR("type size %s not found", typeName.c_str());
      }
    } else if (args.size() == 4) { // eventRecord or streamWaitEvent
      eventName = args[2];
      streamName = args[3];
    }
  }

  virtual ~BaseOp() {};

  virtual void launchOp(void *args, int userRank) = 0;

  void presetBuffsize(const std::string& testCaseName, struct Task& task) {
    int size = 0;
    size = count*typeSize;
    if (op == "allgather" || op == "reducescatter") size = size * worldSize;
    for (auto index=0; index<buffNames.size(); ++index) {
      auto it = task.buffsizes.find(buffNames[index]);
      if (it == task.buffsizes.end()) {
        task.buffsizes[buffNames[index]] = 0;
      }
      if (task.buffsizes[buffNames[index]] < size) {
        task.buffsizes[buffNames[index]] = size;
      }
    }
  }

  void setBuffAndStream(struct Task& task) {
    std::string streamNameUsed = streamName;
    if (streamName.empty()) {
      stream = 0;
    } else if (isGlobalStream(streamName)) {
      streamNameUsed = fullfillStrWithDev(streamName, task.devId);
    }

    if (task.streams.find(streamNameUsed) != task.streams.end()) {
      stream = task.streams[streamNameUsed];
    }

    std::string eventNameUsed = eventName;
    if (isGlobalEvent(eventName)) {
      eventNameUsed = fullfillStrWithDev(eventName, task.devId);
    }
    if (task.events.find(eventNameUsed) != task.events.end()) {
      event = task.events[eventNameUsed];
    }

    for (auto& b: buffNames) {
      if (task.buffs.find(b) != task.buffs.end()) {
        buffs.push_back((float*)task.buffs[b]);
      } else {
        ERROR("op:%s buff:%s not found!", op.c_str(), b.c_str());
      }
    }
  }
};

class PcclOp : public BaseOp {
 public:
  pcclDataType_t dtype;
  PcclOp() {}
  PcclOp(std::vector<std::string>& args) : BaseOp(args) {
    isPccl = true;
    if (args.size() > 3) {
      const auto& it = pcclTypeMap.find(typeName);
      if (it != pcclTypeMap.end()) {
        dtype = it->second;
      } else {
        WARN("type %s not found set to pcclInt8", typeName.c_str());
        dtype = pcclInt8;
      }
    }
  }

  void launchOp(void* args, int userRank) override {
    pcclComm_t comm = (pcclComm_t)args;
    if (op == "groupStart") {
      waitReady(1, &comm);
      PCCLCHECK(pcclGroupStart());
    } else if (op == "groupEnd") {
      PCCLCHECK(pcclGroupEnd());
    } else if (op == "allgather") {
      PCCLCHECK(pcclAllGather((const void*)buffs[0], (void*)buffs[1], count, dtype, comm, stream));
    } else if (op == "allreduce") {
      PCCLCHECK(pcclAllReduce((const void*)buffs[0], (void*)buffs[1], count, dtype, pcclSum, comm, stream));
    } else if (op == "reducescatter") {
      PCCLCHECK(pcclReduceScatter((const void*)buffs[0], (void*)buffs[1], count, dtype, pcclSum, comm, stream));
    } else if (op == "alltoall") {
      int nRanks;
      PCCLCHECK(pcclCommCount(comm, &nRanks));
      size_t countPerRank = count / nRanks;
      size_t rankOffset = countPerRank * typeSize;
      PCCLCHECK(pcclGroupStart());
      for (int r=0; r<nRanks; r++) {
        PCCLCHECK(pcclSend(((char*)buffs[0])+r*rankOffset, countPerRank, dtype, r, comm, stream));
        PCCLCHECK(pcclRecv(((char*)buffs[1])+r*rankOffset, countPerRank, dtype, r, comm, stream));
      }
      PCCLCHECK(pcclGroupEnd());
    } else if (op == "send") {
      if (myRank == userRank) PCCLCHECK(pcclSend((const void*)buffs[0], count, dtype, root, comm, stream));
    } else if (op == "recv") {
      // tempRank = pcclCommUserRank(comm, &tempRank);
      if (myRank == userRank) PCCLCHECK(pcclRecv((void*)buffs[0], count, dtype, root, comm, stream));
    }
  }
};

class MatOp : public BaseOp {
 public:
  float dtype;
  MatOp() {}
  MatOp(std::vector<std::string>& args) : BaseOp(args) {
    isPccl = false;
  }

  template<typename T>
  void launchOpImpl(void* args) {
    if (op == "inc") {
      incrementKernel<<<16, 256, 0, stream>>>((T*)buffs[0], count);
    } else if (op == "init") {
      int dev;
      hggcGetDevice(&dev);
      initData<<<16, 256, 0, stream>>>((T*)buffs[0], (T)(10.1+(float)dev), count);
    }
  }

  void launchOp(void* args, int userRank) override {
    if (typeName == "float") {
      launchOpImpl<float>(args);
    } else if (typeName == "int") {
      launchOpImpl<int>(args);
    } else if (typeName == "int8") {
      launchOpImpl<int8_t>(args);
    } else if (typeName == "double") {
      launchOpImpl<double>(args);
    } else if (typeName == "half") {
      launchOpImpl<half>(args);
    } else if (typeName == "bf16") {
      launchOpImpl<half>(args);
    } else if (typeName == "int64") {
      launchOpImpl<int64_t>(args);
    }
  }
};

class EventOp : public BaseOp {
 public:
  EventOp() {}
  EventOp(std::vector<std::string>& args) : BaseOp(args) {
    isPccl = false;
 }

  void launchOp(void* args, int userRank) override {
    pcclComm_t comm = (pcclComm_t)args;
    if (op == "eventRecord") {
      HGGCCHECK(hggcEventRecord(event, stream));
    } else if (op == "streamWaitEvent") {
      HGGCCHECK(hggcStreamWaitEvent(stream, event));
    }
  }
};

void createOp(BaseOp*& op, std::vector<std::string>& args) {
  if (args[0] == "pccl"){
    op = new PcclOp(args);
  } else if (args[0] == "compute") {
    op = new MatOp(args);
  } else if (args[0] == "event") {
    op = new EventOp(args);
  }
}

void createOps(struct Task& task, const json& configs, const std::string& testName, std::string kname, std::vector<struct BaseOp*>& ops) {
  for (auto& k: configs["testConfigs"][testName][kname]) {
    std::vector<std::string> opArgs;
    for (auto& s: k) {
      opArgs.push_back(s.template get<std::string>());
    }

    // for (auto& s: opArgs) std::cout << s << std::endl;
    BaseOp *op = nullptr;
    createOp(op, opArgs);
    op->presetBuffsize(testName, task);
    ops.push_back(op);
  }
}

void setOpsBuffAndStream(struct Task& task, const json& configs, std::string& testName, std::vector<struct BaseOp*>& ops) {
  for (auto& op: ops) {
    op->setBuffAndStream(task);
  }
}

void createStreams(struct Task& task, const json& configs, std::string& testName) {
  for (auto& s: configs["testConfigs"][testName]["streams"]) {
    std::string streamName = s.template get<std::string>();
    if (streamName == "null") {
      task.streams[streamName] = 0;
    } else {
      hggcStream_t stream;
      if (isGlobalStream(streamName)) {
        task.useGlobalStream = true;
        std::string globalStreamName = fullfillStrWithDev(streamName, task.devId);
        auto it = task.globalStreams->find(globalStreamName);
        if (it == task.globalStreams->end()) {
          HGGCCHECK(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking));
          task.globalStreams->emplace(globalStreamName, stream);
        } else {
          stream = it->second;
        }
        DEBUG("task %s add global stream %s %p", task.name.c_str(), globalStreamName.c_str(), stream);
        task.streams[globalStreamName] = stream;
      } else {
        HGGCCHECK(hggcStreamCreateWithFlags(&stream, hggcStreamNonBlocking));
        DEBUG("task %s add stream %s %p", task.name.c_str(), streamName.c_str(), stream);
        task.streams[streamName] = stream;
      }
    }
  }
}

void createEvents(struct Task& task, const json& configs, std::string& testName) {
  if (!configs["testConfigs"][testName].contains("events")) {
    return;
  }
  for (auto& s: configs["testConfigs"][testName]["events"]) {
    std::string eventName = s.template get<std::string>();
    hggcEvent_t event;
    if (isGlobalEvent(eventName)) {
      task.useGlobalEvent = true;
      std::string globalEventName = fullfillStrWithDev(eventName, task.devId);
      auto it = task.globalEvents->find(globalEventName);
      if (it == task.globalEvents->end()) {
        HGGCCHECK(hggcEventCreate(&event));
        task.globalEvents->emplace(globalEventName, event);
      } else {
        event = it->second;
      }
      DEBUG("task %s add global event %s %p", task.name.c_str(), globalEventName.c_str(), event);
      task.events[globalEventName] = event;
    } else {
      HGGCCHECK(hggcEventCreate(&event));
      DEBUG("task %s add event %s %p", task.name.c_str(), eventName.c_str(), event);
      task.events[eventName] = event;
    }
  }
}

void createBuffs(struct Task& task, const json& configs, std::string& testName) {
  for (auto& b: configs["testConfigs"][testName]["buffs"]) {
    std::string buffName = b.template get<std::string>();
    void* buff;
    task.buffs[buffName] = buff;
    task.buffsizes[buffName] = 0;
  }
}

void mallocBuffs(struct Task& task, const json& configs, std::string& testName) {
  for (auto& b: configs["testConfigs"][testName]["buffs"]) {
    std::string buffName = b.template get<std::string>();
    if (task.buffs.find(buffName) == task.buffs.end()) {
      ERROR("buff %s not found", buffName.c_str());
    }
    int buffsize = task.buffsizes[buffName];
    HGGCCHECK(hggcMalloc(&task.buffs[buffName], buffsize));
    HGGCCHECK(hggcMemset(task.buffs[buffName], 0, buffsize));
    DEBUG("task %s malloc buff %s:%p size %d", task.name.c_str(), buffName.c_str(), task.buffs[buffName], buffsize);
  }
}

void buildTask(struct Task& task, const json &configs, std::string& testName) {
  createStreams(task, configs, testName);
  createEvents(task, configs, testName);
  createBuffs(task, configs, testName);
  createOps(task, configs, testName, "inits", task.initOps);
  createOps(task, configs, testName, "kernels", task.ops);
  mallocBuffs(task, configs, testName);
  setOpsBuffAndStream(task, configs, testName, task.ops);
  setOpsBuffAndStream(task, configs, testName, task.initOps);
}

void synchronizeTask(struct Task& task) {
  for (auto& it: task.streams) {
    DEBUG("sync streams %p", it.second);
    HGGCCHECK(hggcStreamSynchronize(it.second));
  }
}

void destroyTask(struct Task& task) {
  // INFO("destroy task %s", task.name.c_str());
  if (task.usingGraph) {
    if (task.graph != nullptr) {
      HGGCCHECK(hggcGraphDestroy(task.graph));
      task.graph = nullptr;
      HGGCCHECK(hggcGraphExecDestroy(task.exec));
      task.exec = nullptr;
    }
  }

  for (auto& it: task.streams) {
    std::string streamName = it.first;
    if (!isGlobalStream(streamName) && streamName != "null") HGGCCHECK(hggcStreamDestroy(it.second));
  }

  for (auto& it: task.buffs) {
    HGGCCHECK(hggcFree(it.second));
  }
}

void launchTaskInit(struct Task& task, pcclComm_t comm) {
  for (auto& op: task.initOps) {
    op->launchOp(comm, -1);
  }
}

void launchTask(struct Task& task, pcclComm_t comm, int iter) {
  if (task.name.empty()) ERROR("Empty task");
  if (iter > task.iters) {
    return;
  }

  int userRank;
  pcclCommUserRank(comm, &userRank);
  // for graph task, first launch is capture.
  if (task.usingGraph && task.graph == nullptr) {
    // use the right stream name, however one graph task allow only one stream now
    auto it = task.streams.begin();
    if (it == task.streams.end()) {
      ERROR("no stream found for task %s's capture", task.name.c_str());
      return;
    }
    hggcStream_t stream = it->second;
    hggcGraph_t graph;
    HGGCCHECK(hggcStreamBeginCapture(stream, hggcStreamCaptureModeThreadLocal));
    for (auto& op: task.ops) {
      op->launchOp(comm, userRank);
    }
    waitReady(1, &comm);
    HGGCCHECK(hggcStreamEndCapture(stream, &graph));
    hggcGraphExec_t exec;
    HGGCCHECK(hggcGraphInstantiate(&exec, graph, NULL, NULL, 0));
    task.graph = graph;
    task.exec = exec;

    if (task.devId == 0) {
      char dotName[1024];
      snprintf(dotName, 1024, "pcclCudaGraphTest.task-%s.dot", task.name.c_str());
      HGGCCHECK(hggcGraphDebugDotPrint(graph, dotName, hggcGraphDebugDotFlagsVerbose));
    }

    if (task.iters == iter) {
      if (task.usingGraph && task.graph) {
        HGGCCHECK(hggcGraphDestroy(task.graph));
        task.graph = nullptr;
        HGGCCHECK(hggcGraphExecDestroy(task.exec));
        task.exec = nullptr;
      }
    }
    return;
  }

  if (task.usingGraph) {
    hggcStream_t stream;
    std::string streamName = task.graphStreamName;

    // if unset, use the first stream in streams, although it's a map.
    if (streamName == "") {
      // if streams first is global, it has been fullfilled when emplace.
      streamName = task.streams.begin()->first;
    } else {
      if (isGlobalStream(streamName)) {
        streamName = fullfillStrWithDev(streamName, task.devId);
      }
    }

    auto it = task.streams.find(streamName);
    if (it != task.streams.end()) {
      stream = it->second;
    } else {
      ERROR("task %s do not find stream %s %d", task.name.c_str(), streamName.c_str());
    }

    if (task.exec == nullptr) { ERROR("no exec");}
    int dev=0;
    hggcGetDevice(&dev);
    DEBUG("task %s stream %p %s %d", task.name.c_str(), stream, streamName.c_str(), dev);
    HGGCCHECK(hggcGraphLaunch(task.exec, stream));
  } else {
    for (auto& op: task.ops) {
      op->launchOp(comm, userRank);
    }
  }
}

testResult_t testEntry(struct threadArgs* args) {
  pcclComm_t* parentComms = args->comms;
  pcclComm_t* comms[MAX_COMM_NUM] = {};
  comms[0] = parentComms;

  json configs = *args->configs;
  std::vector<std::string> tests;
  std::map<std::string, struct Task> tasks;
  json testConfigs = configs["testConfigs"];
  std::vector<std::string> order;
  std::vector<std::string> outbuffs;
  std::map<std::string, hggcStream_t> globalStreams;
  std::vector<std::string> compareTasks;


  // parse execution order
  for (auto &s: configs["order"]) {
    order.push_back(s.template get<std::string>());
  }

  // parse outbuffs, we only compare buffers in outbuffs
  for (auto &s: configs["outbuffs"]) {
    outbuffs.push_back(s.template get<std::string>());
  }

  // parse compare tasks, we only compare tasks's outputbuff
  for (auto &s: configs["compareTests"]) {
    compareTasks.push_back(s.template get<std::string>());
  }

  // get all task names
  for (json::iterator it = testConfigs.begin(); it != testConfigs.end(); ++it) {
    std::string testName = it.key();
    tests.push_back(testName);
  }

  // construct tasks
  for (int g=0; g<args->nGpus; g++) {
    int gpuid = args->gpus[g];
    HGGCCHECK(hggcSetDevice(gpuid));
    for (json::iterator it = testConfigs.begin(); it != testConfigs.end(); ++it) {
      struct Task task;
      task.globalStreams = &globalStreams;
      task.useGlobalStream = false;
      task.devId = gpuid;
      std::string testName = it.key();
      task.name = testName;
      task.usingGraph = testConfigs[testName]["graph"].template get<bool>();
      task.commIdx = testConfigs[testName].value("comm", 0);
      if (task.commIdx >= MAX_COMM_NUM) {
        ERROR("commIdx should be smaller than %d!", MAX_COMM_NUM);
        return testInternalError;
      }
      task.graphStreamName = testConfigs[testName].value("graph_stream", "");
      buildTask(task, configs, testName);
      std::string devTestName = fullfillStrWithDev(testName, gpuid);
      task.iters = testConfigs[testName].value("iters", INT_MAX);
      tasks[devTestName] = std::move(task);
    }
  }

  // init child comms
  for (auto& testName : order) {
    int gpuid = args->gpus[0];
    std::string devTestName = fullfillStrWithDev(testName, gpuid);
    int commIdx = tasks[devTestName].commIdx;
    if (comms[commIdx] == nullptr) {
      comms[commIdx] = (pcclComm_t*)malloc(sizeof(pcclComm_t)*args->nGpus);
      waitReady(args->nGpus, parentComms);
      PCCLCHECK(pcclGroupStart());
      for (int g=0; g< args->nGpus; g++) {
        gpuid = args->gpus[g];
        HGGCCHECK(hggcSetDevice(gpuid));
        int rank = args->proc * args->nThreads * args->nGpus + args->thread * args->nGpus + g;
        int nRanks = args->nProcs * args->nThreads * args->nGpus;
        int color = 0xffffffff;
        int key = 0xffffffff;
        GetCommSplitInfo(commIdx, rank, nRanks, &color, &key);
        PCCLCHECK(pcclCommSplit(parentComms[g], color, key, comms[commIdx] + g, NULL));
      }
      PCCLCHECK(pcclGroupEnd());
    }
  }

  // init input buffs
  for (int g=0; g<args->nGpus; g++) {
    int gpuid = args->gpus[g];
    HGGCCHECK(hggcSetDevice(gpuid));
    for (auto& testName : order) {
      std::string devTestName = fullfillStrWithDev(testName, gpuid);
      launchTaskInit(tasks[devTestName], comms[tasks[devTestName].commIdx][g]);
    }
  }

  waitReady(args->nGpus, parentComms);
  // launch task in order
  for (int i=0;i<args->niters;i++) {
    for (int g=0; g<args->nGpus; g++) {
      int gpuid = args->gpus[g];
      HGGCCHECK(hggcSetDevice(gpuid));
      for (auto& testName : order) {
        std::string devTestName = fullfillStrWithDev(testName, gpuid);
        launchTask(tasks[devTestName], comms[tasks[devTestName].commIdx][g], i);
      }
    }
  }

  // to compare final result, run graph task once
  if (args->niters > 0) {
    for (int g=0; g<args->nGpus; g++) {
      int gpuid = args->gpus[g];
      HGGCCHECK(hggcSetDevice(gpuid));
      for (auto& testName : order) {
        std::string devTestName = fullfillStrWithDev(testName, gpuid);
        if (tasks[devTestName].usingGraph) {
          launchTask(tasks[devTestName], comms[tasks[devTestName].commIdx][g], args->niters);
        }
      }
    }
  }

  // wait all stream done
  std::map<std::string, struct Task>::iterator it;
  for (it = tasks.begin(); it != tasks.end(); ++it) {
    struct Task& task= it->second;
    synchronizeTask(task);
  }

  // compare outputs
  int flag = 0;
  for (int g=0; g<args->nGpus; g++) {
    int gpuid = args->gpus[g];
    HGGCCHECK(hggcSetDevice(gpuid));
    int* dev_flag;
    hggcHostAlloc(&dev_flag, sizeof(int), hggcHostAllocDefault);
    *dev_flag = 0;
    // for now, two task with the same op should have the same result
    // TODO: make every task have one verify task.
    if (compareTasks.size() > 1) {
      for (int i=0; i<compareTasks.size()-1; i++) {
        std::string devTestName1 = fullfillStrWithDev(compareTasks[i], gpuid);
        std::string devTestName2 = fullfillStrWithDev(compareTasks[i+1], gpuid);
        auto& task1 = tasks[devTestName1];
        auto& task2 = tasks[devTestName2];
        for (auto& buffName : outbuffs) {
          verifyData<<<16, 512, 0, 0>>>((float*)task1.buffs[buffName], (float*)(task2.buffs[buffName]), task1.buffsizes[buffName]/sizeof(float), dev_flag);
        }
      }
    }
    hggcStreamSynchronize(0);
    flag += *dev_flag;
    if (flag != 0) ERROR("gpu:%d data compare failed error number is %d", gpuid, flag);
    hggcFreeHost(dev_flag);
  }

  // release task resources
  for (it = tasks.begin(); it != tasks.end(); ++it) {
    struct Task& task = it->second;
    destroyTask(task);
  }

  // release global resources
  for(std::map<std::string, hggcStream_t>::iterator it=globalStreams.begin(); it!=globalStreams.end(); ++it) {
    HGGCCHECK(hggcStreamDestroy(it->second));
  }

  // destroy child comms
  for (int idx = 1; idx < MAX_COMM_NUM; idx++) {
    if (comms[idx] != nullptr) {
      for (int g=0; g< args->nGpus; g++) {
        PCCLCHECK(pcclCommDestroy(comms[idx][g]));
      }
      free(comms[idx]);
      comms[idx] = nullptr;
    }
  }

  if (flag != 0) {
    ERROR("data compare failed error number is %d", flag);
    return testInternalError;
  } else {
    return testSuccess;
  }
}

void* threadLauncher(void* thread_) {
  struct testThread* thread = (struct testThread*)thread_;
  thread->ret = thread->func(&thread->args);
  return NULL;
}

testResult_t threadLaunch(struct testThread* thread) {
  pthread_create(&thread->thread, NULL, threadLauncher, thread);
  return testSuccess;
}

int multiThreadTestLauncher(const json& configs, int niters ) {
  int totalProcs = 1, proc = 0;
  int localRank = 0;
  char hostname[1024];
  getHostName(hostname, 1024);

#ifdef MPI_SUPPORT
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  uint64_t hostHashs[totalProcs];
  hostHashs[proc] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<totalProcs; p++) {
    if (p == proc) break;
    if (hostHashs[p] == hostHashs[proc]) localRank++;
  }
#endif

  pcclUniqueId ncclId;
  if (proc == 0) {
    PCCLCHECK(pcclGetUniqueId(&ncclId));
  }
#ifdef MPI_SUPPORT
  MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); // Ensure Bcast is complete for HCOLL
#endif

  int gpus[nGpus*nThreads];
  for (int i=0; i<nGpus*nThreads; i++) {
    gpus[i] = localRank*nThreads*nGpus + i;
  }

  hggcStream_t streams[nGpus*nThreads];

  //if parallel init is not selected, use main thread to initialize PCCL
  pcclComm_t* comms = (pcclComm_t*)malloc(sizeof(pcclComm_t)*nThreads*nGpus);
  int gpuArray[nGpus*nThreads];
  for (int i=0; i<nGpus*nThreads; i++) gpuArray[i] = i;
  if (totalProcs == 1) {
    worldSize = nGpus*nThreads;
    PCCLCHECK(pcclCommInitAll(comms, worldSize, gpus));
  } else {
    worldSize = totalProcs*nGpus*nThreads;
    PCCLCHECK(pcclGroupStart());
    for (int i=0; i<nGpus*nThreads; i++) {
      HGGCCHECK(hggcSetDevice(gpus[i]));
      PCCLCHECK(pcclCommInitRank(comms+i, worldSize, ncclId, proc*nThreads*nGpus+i));
    }
    PCCLCHECK(pcclGroupEnd());
  }

  int errors[nThreads];
  struct testThread threads[nThreads];
  memset(threads, 0, sizeof(struct testThread)*nThreads);
  int ret = 0;

  for (int t=nThreads-1; t>=0; t--) {
    threads[t].args.nProcs = totalProcs;
    threads[t].args.proc = proc;
    threads[t].args.nThreads = nThreads;
    threads[t].args.thread = t;
    threads[t].args.nGpus = nGpus;
    threads[t].args.gpus = gpus + t*nGpus;
    threads[t].args.comms = comms + t*nGpus;
    threads[t].args.errors = errors + t;
    threads[t].args.configs = (json*) & configs;
    threads[t].args.niters = niters;
    threads[t].func = testEntry;

    if (t) {
      threadLaunch(threads+t);
    }
    else {
      ret = threads[t].func(&threads[t].args);
    }
  }

  // Wait for other threads and accumulate stats and errors
  int pthread_ret = 0;
  for (int t=nThreads-1; t>=0; t--) {
    if (t) pthread_ret = pthread_join(threads[t].thread, NULL);
    errors[t] = threads[t].ret;
    if (t) {
      if (errors[t] != 0) {
        INFO("thread %d failed, ret is %d", t, errors[t]);
        ret += 1;
      }
    }

    if (pthread_ret != 0) {
      INFO("pthread_ret is not 0(%d)", pthread_ret);
      ret += 1;
    }
  }

#ifdef MPI_SUPPORT
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  for(int i=0; i<nGpus*nThreads; ++i)
    PCCLCHECK(pcclCommDestroy(comms[i]));
  free(comms);

  for (int i=0; i<nGpus*nThreads; i++) {
    HGGCCHECK(hggcSetDevice(gpus[i]));
    HGGCCHECK(hggcDeviceReset());
  }

  return ret;
}

void parseTestSequence(std::string const& testConfigs, json &configs) {
  std::ifstream f(testConfigs.c_str());
  configs = json::parse(f);
}

int main(int argc, char **argv) {
  cxxopts::Options options("hggcGraphUnitTest", "Test PCCL operations with HGGC graphs.");
  options.add_options()
    ("t,num_threads", "num_threads, one ppu per thread", cxxopts::value<int>()->default_value("2"))
    ("g,num_gpus", "num_gpus, ppu number per thread", cxxopts::value<int>()->default_value("1"))
    ("u,coredump", "enable coredump", cxxopts::value<bool>()->default_value("false"))
    ("f,test_config_file", "test config file", cxxopts::value<std::string>()->default_value("./testConfig.json"))
    ("n,num_iterations", "config test loop size", cxxopts::value<int>()->default_value("5"))
    ("h,help", "Print usage");
  auto args = options.parse(argc, argv);

  // Make sure everyline is flushed so that we see the progress of the test
  setlinebuf(stdout);
  int totalGpus;
  HGGCCHECK(hggcGetDeviceCount(&totalGpus));

  json configs;
  parseTestSequence(args["test_config_file"].as<std::string>(), configs);
  // std::cout << configs << std::endl;

  nThreads = args["num_threads"].as<int>();
  nGpus = args["num_gpus"].as<int>();
  std::cout << "num_threads: " << nThreads << ", num_gpus: " << nGpus << std::endl;
  if (nThreads * nGpus > totalGpus) {
    ERROR("num_threads * num_gpus(%d) > totalGpus(%d)\n", nThreads * nGpus, totalGpus);
    exit(EXIT_FAILURE);
  }

  if(args["coredump"].as<bool>()) {
      // enable core dump
      struct rlimit core_limits;
      core_limits.rlim_cur = core_limits.rlim_max = RLIM_INFINITY;
      setrlimit(RLIMIT_CORE, &core_limits);
  }

#ifdef MPI_SUPPORT
  MPI_Init(&argc, &argv);
#endif

  int error = multiThreadTestLauncher(configs, args["num_iterations"].as<int>());

#ifdef MPI_SUPPORT
  MPI_Finalize();
#endif

  if (error != 0) {
    ERROR("%s FAIL !!!\n", args["test_config_file"].as<std::string>().c_str());
    exit(EXIT_FAILURE);
  } else {
    INFO("%s PASS !!!\n", args["test_config_file"].as<std::string>().c_str());
    exit(EXIT_SUCCESS);
  }
}
