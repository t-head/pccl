//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <cstddef>
#include <stdio.h>
#include <random>
#include <thread>
#include <bits/stdc++.h>
#include "hggc_runtime.h"
#include "hggc_fp16.h"
#include "pccl.h"

#include "logger.h"
#include "cxxopts.hpp"

#include <sys/resource.h>
#include <hggc_profiler_api.h> // for Asight Compute profile

#define ALLREDUCE 0
#define BROADCAST 1
#define ALLGATHER 2
#define REDUCE 3
#define REDUCESCATTER 4
#define ALLTOALL 5
#define ALLTOALLARRAY 6
#define COLL_OP_NUM 7

#define host_half half

#define DELTA 0.000001
#define NAN_MAX_DELTA 10000.f

#define DELTA_FLOAT 1e-5
#define DELTA_HALF 1e-2
#define DELTA_INT8 1e-200
#define DELTA_DOUBLE 1e-12

#define MAX_SCALE_RANGE 17

typedef enum {
  DTYPE_FLOAT = 0,
  DTYPE_HALF = 1,
  DTYPE_INT8 = 2,
  DTYPE_DOUBLE = 3,
  DTYPE_NUM = 4
} DType;

// global test parameters
// collective pattern
int coll_type = ALLREDUCE;
// data type, default is float
DType data_type = DTYPE_FLOAT;
// delta of data type
double max_delta = DELTA_FLOAT;
// size of data type, default is sizeof(float) = 4
size_t data_size = sizeof(float);
// payload size for collective test, number of float elements
size_t num_elems = 32;
// minimum payload size for random collective test, number of float elements
size_t min_num_elems = 0;
// maximum payload size for random collective test, number of float elements
size_t max_num_elems = 0;
size_t buffer_offset = 0;
// number of ppu to be used for collective test
int num_gpus = 2;
// number of iterations to do collective test
int num_iterations = 1;
// do input data initialization and final output results verifying on host side.
int host_init_and_verify = 0;
// ignore data value initialization or not (1: ignore)
int ignore_data_init = 0;
// ignore data value comparison or not (1: ignore)
int ignore_data_verify = 0;
int core_dump = 0;
// run collective array tests for send/recv tests currently
int array_test_mode = 0;
// reuse input/output buffers of same size across iterations
bool reuse_buffer = false;
// seed to be used in collective test
int coll_seed = 996;
// do memcpy stress test only.
int local_memcpy_stress = 0;
// ignore error check results and keep stress test.
int ignore_error_check_result = 0;
// using ncu profile
int ncu_profile = 0;
int abnormal_mode = 0;

int inplace = 0;
int dump_full = 0;
int show_continuous_mismatch = 10;
dim3 grid_init = { 32, 1, 1 };
dim3 block_init = { 128, 1, 1 };
// used to launch data verify kernel in one block for threads synchronizing
const int blocks_verify_num = 512;

size_t wordSize(pcclDataType_t type) {
  switch(type) {
    case pcclInt8:
    case pcclUint8:
      return 1;
    case pcclFloat16:
      return 2;
    case pcclInt32:
    case pcclUint32:
    case pcclFloat32:
      return 4;
    case pcclInt64:
    case pcclUint64:
    case pcclFloat64:
      return 8;
    default: return 0;
  }
}

// AlltoAllArray implementation
pcclResult_t pcclAlltoAllArray(const void** sendbuff, void** recvbuff, size_t* sendcount,
    size_t* recvcount, pcclDataType_t datatype, pcclComm_t comm, hggcStream_t stream) {
  int cuDev, nRanks;
  PCCLCHECK(pcclCommCuDevice(comm, &cuDev));
  PCCLCHECK(pcclCommCount(comm, &nRanks));
  HGGCCHECK(hggcSetDevice(cuDev));

  PCCLCHECK(pcclGroupStart());
  for (int peerRank=0; peerRank<nRanks; peerRank++) {
    PCCLCHECK(pcclSend(((char*)sendbuff[peerRank]), sendcount[peerRank], datatype, peerRank, comm, stream));
    PCCLCHECK(pcclRecv(((char*)recvbuff[peerRank]), recvcount[peerRank], datatype, peerRank, comm, stream));
  }
  PCCLCHECK(pcclGroupEnd());
  return pcclSuccess;
}

// AlltoAll implementation
pcclResult_t pcclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
    pcclDataType_t datatype, pcclComm_t comm, hggcStream_t stream) {
  int cuDev, nRanks, word_size;
  PCCLCHECK(pcclCommCuDevice(comm, &cuDev));
  PCCLCHECK(pcclCommCount(comm, &nRanks));
  HGGCCHECK(hggcSetDevice(cuDev));
  word_size = wordSize(datatype);
  if (word_size <= 0) {
    ERROR("datatype %d is invalid.", datatype);
    return pcclInvalidArgument;
  }
  size_t rankOffset = count * word_size;
  if (count == 0) return pcclSuccess;

  PCCLCHECK(pcclGroupStart());
  for (int peerRank=0; peerRank<nRanks; peerRank++) {
    PCCLCHECK(pcclSend(((char*)sendbuff)+peerRank*rankOffset, count, datatype, peerRank, comm, stream));
    PCCLCHECK(pcclRecv(((char*)recvbuff)+peerRank*rankOffset, count, datatype, peerRank, comm, stream));
  }
  PCCLCHECK(pcclGroupEnd());
  return pcclSuccess;
}

size_t data_type_size() {
  switch(data_type) {
  case DTYPE_FLOAT:
    return 4;
  case DTYPE_HALF:
    return 2;
  case DTYPE_INT8:
    return 1;
  case DTYPE_DOUBLE:
    return 8;
  default:
    return 0;
  }
}

#define MAX_PPU_NUM_PER_NODE 8
// generate elements number for each device send/recv pair
void generateElementNumArray(size_t* send_elems_array, size_t* recv_elems_array) {
  // generate elements number randomly
  // obtain a random number from hardware
  static const size_t send_elems_arrays0[MAX_PPU_NUM_PER_NODE*MAX_PPU_NUM_PER_NODE] = {
    1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
    1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
    1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
    1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
    1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
    1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
    1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152,
    1048576, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 2097152
  };

  static const size_t send_elems_arrays1[MAX_PPU_NUM_PER_NODE*MAX_PPU_NUM_PER_NODE] = {
    1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576, 1048576,
    4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
    4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
    4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
    4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
    4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
    4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304, 4194304,
    2097152, 2097152, 2097152, 2097152, 2097152, 2097152, 2097152, 2097152
  };

  static const size_t send_elems_arrays2[MAX_PPU_NUM_PER_NODE*MAX_PPU_NUM_PER_NODE] = {
    0, 1048576, 1048576, 0, 1048576, 1048576, 1048576, 1048576,
    4194304, 0, 4194304, 4194304, 0, 4194304, 4194304, 4194304,
    4194304, 4194304, 0, 4194304, 4194304, 4194304, 0, 4194304,
    4194304, 4194304, 4194304, 0, 4194304, 4194304, 4194304, 0,
    4194304, 0, 4194304, 4194304, 0, 4194304, 4194304, 4194304,
    4194304, 4194304, 0, 4194304, 4194304, 0, 4194304, 4194304,
    4194304, 4194304, 0, 4194304, 4194304, 4194304, 0, 4194304,
    2097152, 2097152, 2097152, 0, 2097152, 2097152, 2097152, 0
  };

  int random_seed = coll_seed;
  if (random_seed < 0) {
    std::random_device rd;
    random_seed = rd();
  }

  // seed the generator
  INFO("Random seed to generate element numbers array: %d", random_seed);
  std::mt19937 gen(random_seed);
  // define the range
  size_t _min_elems = min_num_elems / num_gpus;
  size_t _max_elems = max_num_elems / num_gpus;
  std::uniform_int_distribution<size_t> distr(_min_elems, _max_elems);
  for (int i = 0; i < num_iterations; ++i) {
    for (int s_rank = 0; s_rank < num_gpus; ++s_rank) {
      for (int d_rank = 0; d_rank < num_gpus; ++d_rank) {
        size_t n_elems = 0;
        if (_min_elems > 0 && _max_elems > 0 && _min_elems < _max_elems) {
          n_elems = distr(gen);
        } else {
          // get elements number from cmd argument
          if (i % 3 == 0) {
            n_elems = send_elems_arrays0[s_rank*MAX_PPU_NUM_PER_NODE+d_rank];
          } else if (i % 3 == 1) {
            n_elems = send_elems_arrays1[s_rank*MAX_PPU_NUM_PER_NODE+d_rank];
          } else{
            n_elems = send_elems_arrays2[s_rank*MAX_PPU_NUM_PER_NODE+d_rank];
          }
        }
        send_elems_array[i*num_gpus*num_gpus+s_rank*num_gpus+d_rank] = n_elems;
        recv_elems_array[i*num_gpus*num_gpus+d_rank*num_gpus+s_rank] = n_elems;
      }
    }
  }
}

// generate elements number in each iteration
void generateElementNum(size_t* elems_array) {
  // generate elements number randomly
  // obtain a random number from hardware
  int random_seed = coll_seed;
  if (random_seed < 0) {
    std::random_device rd;
    random_seed = rd();
  }
  // seed the generator
  INFO("Random seed to generate element numbers array: %ld", random_seed);
  std::mt19937 gen(random_seed);
  // define the range
  std::uniform_int_distribution<size_t> distr(min_num_elems, max_num_elems);
  for (int i = 0; i < num_iterations; ++i) {
    size_t n_elems = 0;
    if (min_num_elems > 0 && max_num_elems > 0 \
          && min_num_elems < max_num_elems) {
      n_elems = distr(gen);
    } else {
      // get elements number from cmd argument
      n_elems = num_elems;
    }
    if (coll_type == ALLTOALL || coll_type == ALLGATHER \
          || coll_type == REDUCESCATTER) {
      if (n_elems < num_gpus) {
        n_elems = num_gpus;
        WARN("num_elems is reset equal to number of gpus, which is %lu", n_elems);
      } else if (n_elems % num_gpus != 0) {
        n_elems = size_t(n_elems / num_gpus * num_gpus);
        WARN("num_elems is reset to a multiple of number of  gpus, which is %lu", n_elems);
      }
    }
    elems_array[i] = n_elems;
  }
}

template <typename T>
void parseIntegerArguments(const std::vector<std::pair<std::string, T*>>& params, const cxxopts::ParseResult& args) {
    for (const auto& pair : params) {
        const std::string& key = pair.first;
        T* var_ptr = pair.second;

        if (args.count(key)) {
          *var_ptr = args[key].as<T>();
        }
    }
}

void parseArguments(int argc, char **argv) {
  cxxopts::Options options("collPcclUnitTest", "This is PCCL collective test.");
  options.add_options()
    ("c,coll_type", "Specific collective kernel to be used in PCCL collective test.",
      cxxopts::value<std::string>()->default_value("AllReduce"))
    ("d,dtype", "Specific data type be used in PCCL collective test.",
      cxxopts::value<std::string>()->default_value("float"))
    ("e,num_elems", "Number of <dtype> elements to be used in collective test.",
      cxxopts::value<size_t>()->default_value(std::to_string(num_elems)))
    ("n,min_num_elems", "Number of Minimum <dtype> elements to be used in collective test.",
      cxxopts::value<size_t>()->default_value(std::to_string(min_num_elems)))
    ("x,max_num_elems", "Number of Maximum <dtype> elements to be used in collective test.",
      cxxopts::value<size_t>()->default_value(std::to_string(max_num_elems)))
    ("b,buffer_offset", "Offset from start address of send&recvbuffer.",
      cxxopts::value<size_t>()->default_value(std::to_string(buffer_offset)))
    ("i,num_iterations", "Number of iterations to be run in collective test.",
      cxxopts::value<int>()->default_value(std::to_string(num_iterations)))
    ("g,num_gpus", "Number of gpus.", cxxopts::value<int>()->default_value(std::to_string(num_gpus)))
    ("z,host_init_and_verify", "Configure if we do input data initializing and final results verifying on host side.",
      cxxopts::value<int>()->default_value(std::to_string(host_init_and_verify)))
    ("t,ignore_data_init", "Configure to ignore data initialization.",
      cxxopts::value<int>()->default_value(std::to_string(ignore_data_init)))
    ("v,ignore_data_verify", "Configure to ignore data comparison.",
      cxxopts::value<int>()->default_value(std::to_string(ignore_data_verify)))
    ("o,core_dump", "Configure if enable core dump.",
      cxxopts::value<int>()->default_value(std::to_string(core_dump)))
    ("s,coll_seed", "Seed to generate random size sequences, by defualt is 996.",
      cxxopts::value<long>()->default_value(std::to_string(coll_seed)))
    ("l,local_memcpy_stress", "Configure if we do stress memcpy between host and device.",
      cxxopts::value<int>()->default_value(std::to_string(local_memcpy_stress)))
    ("r,ignore_error_check_result", "Configure if we ignore error check results and keep stress tests running, while result mismatched.",
      cxxopts::value<int>()->default_value(std::to_string(ignore_error_check_result)))
    ("p,ncu_profile", "Configure if using ncu profile.",
      cxxopts::value<int>()->default_value(std::to_string(ncu_profile)))
    ("a,abnormal_mode", "Set abnormal mode, if not zero, trigger abnormal using different method. 1 for hang with different op.",
      cxxopts::value<int>()->default_value(std::to_string(abnormal_mode)))
    ("y,inplace", "Configure test inplace or output place.",
      cxxopts::value<int>()->default_value(std::to_string(inplace)))
    ("m,show_continuous_mismatch", "Configure test show number for continuous mismatch data.",
      cxxopts::value<int>()->default_value(std::to_string(show_continuous_mismatch)))
    ("f,dump_full", "Configure if we dump all recv buff data while errors happened.",
      cxxopts::value<int>()->default_value(std::to_string(dump_full)))
    ("h,help", "Print usage");

  auto args = options.parse(argc, argv);

  // Print help mesage.
  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    exit(testSuccess);
  }

  // Parse collective type.
  if (args.count("coll_type")) {
    std::string coll_type_str = args["coll_type"].as<std::string>();
    static const std::unordered_map<std::string, int> coll_map = {
      {"AllReduce", ALLREDUCE}, {"BroadCast", BROADCAST}, {"AllGather", ALLGATHER},
      {"Reduce", REDUCE}, {"ReduceScatter", REDUCESCATTER}, {"AlltoAll", ALLTOALL},
      {"AlltoAllArray", ALLTOALLARRAY}
    };
    auto it = coll_map.find(coll_type_str);
    if (it != coll_map.end()) {
      coll_type = it->second;
    } else {
      ERROR("Unknown collective kernel type %s. Valid kernel types: "
        "AllReduce, BroadCast, AllGather, Reduce, ReduceScatter, AlltoAll, AlltoAllArray.",
        coll_type_str.c_str());
      exit(testInternalError);
    }

    if (coll_type == ALLTOALLARRAY) {
      array_test_mode = 1;
    }
  }

  // Parse data type.
  if (args.count("dtype")) {
    std::string dtype_str = args["dtype"].as<std::string>();
    const static std::unordered_map<std::string, std::tuple<DType, size_t, double>> dtype_map = {
      {"float",  {DTYPE_FLOAT,  wordSize(pcclFloat32), DELTA_FLOAT}},
      {"half",   {DTYPE_HALF,   wordSize(pcclHalf),    DELTA_HALF}},
      {"int8",   {DTYPE_INT8,   wordSize(pcclInt8),    DELTA_INT8}},
      {"double", {DTYPE_DOUBLE, wordSize(pcclDouble),  DELTA_DOUBLE}}
    };
  
    auto it = dtype_map.find(dtype_str);
    if (it != dtype_map.end()) {
        data_type = std::get<0>(it->second);
        data_size = std::get<1>(it->second);
        max_delta = std::get<2>(it->second);
    } else {
        ERROR("Unsupported data type %s, input must be 'float', 'half', 'int8' or 'double'!", dtype_str.c_str());
        exit(EXIT_FAILURE);
    }
  }
  if (data_type == DTYPE_HALF && num_gpus > 16) {
    max_delta = max_delta*(1.0*num_gpus)/16;
  }

  // Parse different integer type configuration parameters.
  std::vector<std::pair<std::string, size_t*>> sizet_param_pairs = {
    {"num_elems", &num_elems}, {"min_num_elems", &min_num_elems}, {"max_num_elems", &max_num_elems},
    {"buffer_offset", &buffer_offset}
  };
  parseIntegerArguments<size_t>(sizet_param_pairs, args);

  std::vector<std::pair<std::string, int*>> int_param_pairs = {
    {"num_iterations", &num_iterations}, {"num_gpus", &num_gpus}, {"host_init_and_verify", &host_init_and_verify},
    {"ignore_data_init", &ignore_data_init}, {"ignore_data_verify", &ignore_data_verify}, {"core_dump", &core_dump},
    {"coll_seed", &coll_seed}, {"local_memcpy_stress", &local_memcpy_stress}, {"ignore_error_check_result", &ignore_error_check_result},
    {"ncu_profile", &ncu_profile}, {"abnormal_mode", &abnormal_mode}, {"inplace", &inplace},
    {"show_continuous_mismatch", &show_continuous_mismatch}, {"dump_full", &dump_full}
  };
  parseIntegerArguments<int>(int_param_pairs, args);

  // Check legality of num_gpus parameter.
  if (num_gpus <= 0) {
    ERROR("Invalid GPU number: %d", num_gpus);
    exit(testInternalError);
  }

  int total_num_gpus = 0;
  HGGCCHECK(hggcGetDeviceCount(&total_num_gpus));
  if (num_gpus > total_num_gpus) {
    ERROR("exceed the maximum number of GPU, which is %d, but your input is %d!", total_num_gpus, num_gpus);
    exit(testInternalError);
  }
}

// init data

template <typename T>
void _allocTestMem(size_t* send_elems_num, T** sendbuff_origin, T** sendbuff,
                   size_t* recv_elems_num, T** recvbuff_origin, T** recvbuff, T** expected,
                   T** src_host, T** recv_host, T** ref_host) {
  for (int i = 0; i < num_gpus; ++i) {
    HGGCCHECK(hggcSetDevice(i));
    for (int j = 0; j < num_gpus; ++j) {
      // create device memory buffers and temp host buffers
      HGGCCHECK(hggcMalloc(sendbuff_origin+i*num_gpus+j, (send_elems_num[i*num_gpus+j]+buffer_offset)*sizeof(T)));
      sendbuff[i*num_gpus+j] = sendbuff_origin[i*num_gpus+j]+buffer_offset;
      HGGCCHECK(hggcMalloc(recvbuff_origin+i*num_gpus+j, (recv_elems_num[i*num_gpus+j]+buffer_offset)*sizeof(T)));
      recvbuff[i*num_gpus+j] = recvbuff_origin[i*num_gpus+j]+buffer_offset;  // send & recv use the same offset
      DEBUG("dev %d, peer dev %d, sendbuff_origin addr %p, sendbuff addr %p, send elems num %d"
            ", recvbuff_origin addr %p, recvbuff addr %p, recv elems num %d", i, j,
            sendbuff_origin[i*num_gpus+j], sendbuff[i*num_gpus+j], send_elems_num[i*num_gpus+j],
            recvbuff_origin[i*num_gpus+j], recvbuff[i*num_gpus+j], recv_elems_num[i*num_gpus+j]);
      if (host_init_and_verify) {
        HGGCCHECK(hggcHostAlloc(src_host+i*num_gpus+j, send_elems_num[i*num_gpus+j]*sizeof(T), hggcHostAllocDefault));
        HGGCCHECK(hggcHostAlloc(recv_host+i*num_gpus+j, recv_elems_num[i*num_gpus+j]*sizeof(T), hggcHostAllocDefault));
        HGGCCHECK(hggcHostAlloc(ref_host+i*num_gpus+j, recv_elems_num[i*num_gpus+j]*sizeof(T), hggcHostAllocDefault));
        DEBUG("dev %d, peer dev %d, src_host addr %p, recv_host addr %p, recv_host addr %p"
              ", ref_host addr %p", i, j, src_host[i*num_gpus+j], recv_host[i*num_gpus+j],
             ref_host[i*num_gpus+j]);
      } else {
        HGGCCHECK(hggcMalloc(expected+i*num_gpus+j, recv_elems_num[i*num_gpus+j]*sizeof(T)));
        DEBUG("dev %d, peer dev %d, expected addr %p", i, j, expected[i*num_gpus+j]);
      }
    }
  }
}

template <typename T>
void _allocTestMem(size_t elems_num, T** sendbuff_origin, T** sendbuff,
                   T** recvbuff_origin, T** recvbuff, T** expected,
                   T** src_host, T** recv_host, T** ref_host) {
  for (int i = 0; i < num_gpus; ++i) {
    // create device memory buffers and temp host buffers
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcMalloc(sendbuff_origin + i, (elems_num + buffer_offset) * sizeof(T)));
    sendbuff[i] = sendbuff_origin[i] + buffer_offset;
    if (inplace == 0) HGGCCHECK(hggcMalloc(recvbuff_origin + i, (elems_num + buffer_offset) * sizeof(T)));
    else recvbuff_origin[i] = sendbuff_origin[i];
    recvbuff[i] = recvbuff_origin[i] + buffer_offset;  // send & recv use the same offset
    DEBUG("dev %d, sendbuff_origin addr %p, sendbuff addr %p, recvbuff_origin addr %p, recvbuff addr %p",
           i, sendbuff_origin[i], sendbuff[i],
           recvbuff_origin[i], recvbuff[i]);
    if (host_init_and_verify) {
      HGGCCHECK(hggcHostAlloc(src_host + i, elems_num * sizeof(T), hggcHostAllocDefault));
      HGGCCHECK(hggcHostAlloc(recv_host + i, elems_num * sizeof(T), hggcHostAllocDefault));
      HGGCCHECK(hggcHostAlloc(ref_host + i, elems_num * sizeof(T), hggcHostAllocDefault));
      DEBUG("dev %d, src_host addr %p, recv_host addr %p, ref_host addr %p",
            i, src_host[i], recv_host[i],
            ref_host[i]);
    } else {
      HGGCCHECK(hggcMalloc(expected + i, elems_num * sizeof(T)));
      DEBUG("dev %d, expected addr %p", i, expected[i]);
    }
  }
}

void allocBuffer(size_t* send_elems_num, void** sendbuff_origin, void** sendbuff,
                 size_t* recv_elems_num, void** recvbuff_origin, void** recvbuff, void** expected,
                 void** src_host, void** recv_host, void** ref_host, double** deltas_host,
                 double** deltas_device, double** max_delta_device) {
  if (reuse_buffer) {
    INFO("Reuse buffer of last iteration, ignore repetitive alloc and free.");
    return;
  }

  switch (data_type)
  {
  case DTYPE_INT8:
    _allocTestMem<int8_t>(send_elems_num, (int8_t**)sendbuff_origin, (int8_t**)sendbuff,
                          recv_elems_num, (int8_t**)recvbuff_origin, (int8_t**)recvbuff,
                          (int8_t**)expected, (int8_t**)src_host, (int8_t**)recv_host,
                          (int8_t**)ref_host);
    break;
  case DTYPE_HALF:
    _allocTestMem<host_half>(send_elems_num, (host_half**)sendbuff_origin, (host_half**)sendbuff,
                          recv_elems_num, (host_half**)recvbuff_origin, (host_half**)recvbuff,
                          (host_half**)expected, (host_half**)src_host, (host_half**)recv_host,
                          (host_half**)ref_host);
    break;
  case DTYPE_FLOAT:
    _allocTestMem<float>(send_elems_num, (float**)sendbuff_origin, (float**)sendbuff,
                          recv_elems_num, (float**)recvbuff_origin, (float**)recvbuff,
                          (float**)expected, (float**)src_host, (float**)recv_host,
                          (float**)ref_host);
    break;
  case DTYPE_DOUBLE:
    _allocTestMem<double>(send_elems_num, (double**)sendbuff_origin, (double**)sendbuff,
                          recv_elems_num, (double**)recvbuff_origin, (double**)recvbuff,
                          (double**)expected, (double**)src_host, (double**)recv_host,
                          (double**)ref_host);
    break;
  default:
    break;
  }

  for (int i = 0; i < num_gpus; i++) {
    if (!host_init_and_verify) {
      HGGCCHECK(hggcSetDevice(i));
    }
    for (int j = 0; j < num_gpus; j++) {
      // allocate intermediate buffer of deltas_host
      deltas_host[i*num_gpus+j] = (double*)malloc(recv_elems_num[i*num_gpus+j]*sizeof(double));
      DEBUG("dev %d, peer dev %d, deltas_host addr %p", i, j, deltas_host[i*num_gpus+j]);

      // allocate intermediate buffer of deltas_device and max_delta_device
      if (!host_init_and_verify) {
        HGGCCHECK(hggcMalloc(deltas_device+i*num_gpus+j, recv_elems_num[i*num_gpus+j]*sizeof(double)));
        DEBUG("dev %d, peer dev %d, deltas_device addr %p", i, j, deltas_device[i*num_gpus+j]);
        HGGCCHECK(hggcMalloc(max_delta_device+i*num_gpus+j, sizeof(double)));
        DEBUG("dev %d, peer dev %d, max_delta_device addr %p", i, j, max_delta_device[i*num_gpus+j]);
      }
    }
  }
}


void allocBuffer(size_t elems_num, void** sendbuff_origin, void** sendbuff,
                 void** recvbuff_origin, void** recvbuff, void** expected,
                 void** src_host, void* recv_host, void** ref_host, double** deltas_host,
                 double** deltas_device, double** max_delta_device) {
  if (reuse_buffer) {
    INFO("Reuse buffer of last iteration, ignore repetitive alloc and free.");
    return;
  }

  switch (data_type)
  {
  case DTYPE_INT8:
    _allocTestMem<int8_t>(elems_num, (int8_t**)sendbuff_origin,
                          (int8_t**)sendbuff, (int8_t**)recvbuff_origin,
                          (int8_t**)recvbuff, (int8_t**)expected,
                          (int8_t**)src_host, (int8_t**)recv_host,
                          (int8_t**)ref_host);
    break;
  case DTYPE_HALF:
    _allocTestMem<host_half>(elems_num, (host_half**)sendbuff_origin,
                          (host_half**)sendbuff, (host_half**)recvbuff_origin,
                          (host_half**)recvbuff, (host_half**)expected,
                          (host_half**)src_host, (host_half**)recv_host,
                          (host_half**)ref_host);
    break;
  case DTYPE_FLOAT:
    _allocTestMem<float>(elems_num, (float**)sendbuff_origin,
                          (float**)sendbuff, (float**)recvbuff_origin,
                          (float**)recvbuff, (float**)expected,
                          (float**)src_host, (float**)recv_host,
                          (float**)ref_host);
    break;
  case DTYPE_DOUBLE:
    _allocTestMem<double>(elems_num, (double**)sendbuff_origin,
                          (double**)sendbuff, (double**)recvbuff_origin,
                          (double**)recvbuff, (double**)expected,
                          (double**)src_host, (double**)recv_host,
                          (double**)ref_host);
    break;
  default:
    break;
  }

  for (int i = 0; i < num_gpus; i++) {
    // allocate intermediate buffer of deltas_host
    deltas_host[i] = (double*)malloc(elems_num*sizeof(double));
    DEBUG("dev: %d, deltas_host addr %p", i, deltas_host[i]);

    // allocate intermediate buffer of deltas_device and max_delta_device
    if (!host_init_and_verify) {
      HGGCCHECK(hggcSetDevice(i));
      HGGCCHECK(hggcMalloc(deltas_device + i, elems_num * sizeof(double)));
      DEBUG("dev: %d, deltas_device addr %p", i, deltas_device[i]);
      HGGCCHECK(hggcMalloc(max_delta_device + i, sizeof(double)));
      DEBUG("dev: %d, max_delta_device addr %p", i, max_delta_device[i]);
    }
  }
}


// init data by kernel

// [init data by kernel] generate source val by element index and rank id
template <typename T>
__device__ T testValDevice(const size_t idx, const int rank) {
  uint8_t val = (idx + rank) % MAX_SCALE_RANGE;
  return (T)val;
}

template<>
__device__ float testValDevice(const size_t idx, const int rank) {
  return 1.0/(1.0 + testValDevice<int8_t>(idx, rank));
}

template<>
__device__ half testValDevice(const size_t idx, const int rank) {
  return __float2half(testValDevice<float>(idx, rank));
}

// [init data by kernel] calculate reference val directly by source val
template <typename T, int OP>
__device__ void refValDevice(T* expected, const size_t idx, const int rank, \
                              const int nRanks, const size_t N) {
  size_t offset = N / nRanks;
  if (OP == 0) {
    // AllReduce
    expected[idx] = testValDevice<T>(idx, 0);
    for (int i = 1; i < nRanks; i++) {
      expected[idx] += testValDevice<T>(idx, i);
    }
  } else if (OP == 1) {
    // BroadCast
    expected[idx] = testValDevice<T>(idx, 0);
  } else if (OP == 2) {
    // AllGather
    expected[idx] = testValDevice<T>(idx % offset, idx / offset);
  } else if (OP == 3) {
    // Reduce
    expected[idx] = testValDevice<T>(idx, rank);
    if (rank == 0) {
      for (int i = 1; i < nRanks; i++) {
        expected[idx] += testValDevice<T>(idx, i);
      }
    }
  } else if (OP == 4) {
    // ReduceScatter
    if (idx < offset) {
      expected[idx] = 0;
      for (int i = 0; i < nRanks; i++) {
        expected[idx] += testValDevice<T>(rank * offset + idx, i);
      }
    } else {
      expected[idx] = 0;
    }
  } else if (OP == 5) {
    // AlltoAll
    int _rank = int(idx / offset);
    size_t _idx = rank*offset + idx%offset;
    expected[idx] = testValDevice<T>(_idx, _rank);
  } else if (OP == 6) {
    // AlltoAllArray
    expected[idx] = testValDevice<T>(idx, rank);
  }
}

// [init data by kernel] initialize source and reference
// and set recvbuff default to 0
template <typename T, int OP>
__global__ void initDataKernel(T *sendbuff, T *recvbuff, T *expected,
                               const size_t N, const int rank,
                               const int nRanks) {
  for (size_t o = blockIdx.x * blockDim.x + threadIdx.x; o < N; o += gridDim.x * blockDim.x) {
    recvbuff[o] = 0;
    sendbuff[o] = testValDevice<T>(o, rank);
    refValDevice<T, OP>(expected, o, rank, nRanks, N);
  }
}

#define KERN(type, op) (void*)initDataKernel<type, op>
#define OPS(type) KERN(type, 0), KERN(type, 1), KERN(type, 2), \
                  KERN(type, 3), KERN(type, 4), KERN(type, 5), \
                  KERN(type, 6)

#define TOTAL_INIT_KERNELS_NUM ((COLL_OP_NUM)*(DTYPE_NUM))
static void* const initDataKernels[TOTAL_INIT_KERNELS_NUM] = {
  OPS(float), OPS(half), OPS(int8_t), OPS(double)
};

// init data by kernel
template<typename T>
void initDatabyKernel(size_t* send_elems_num, T** sendbuff, size_t* recv_elems_num,
    T** recvbuff, T** expected, hggcStream_t* s) {
  // init recvbuff all 0
  // generate source data(sendbuff)
  // calculate reference data(expected)
  for (int i = 0; i < num_gpus; ++i) {
    HGGCCHECK(hggcSetDevice(i));
    for (int j = 0; j < num_gpus; ++j) {
      if (send_elems_num[i*num_gpus+j] != recv_elems_num[j*num_gpus+i]) {
        ERROR("%d should be equal to %d here.", send_elems_num[i*num_gpus+j],
            recv_elems_num[j*num_gpus+i]);
        exit(EXIT_FAILURE);
      }

      void *args[] = {(void *)&(sendbuff[i*num_gpus+j]),   (void *)&(recvbuff[j*num_gpus+i]),
                      (void *)&(expected[j*num_gpus+i]),   (void *)&(send_elems_num[i*num_gpus+j]),
                      (void *)&(i),             (void *)&(num_gpus)};
      HGGCCHECK(hggcLaunchKernel(initDataKernels[int(data_type * COLL_OP_NUM + coll_type)],
                                 grid_init, block_init, args, 0, s[i]));
    }
  }
}

// init data by kernel
template<typename T>
void initDatabyKernel(size_t elems_num, T** sendbuff, T** recvbuff, T** expected, hggcStream_t* s) {
  // init recvbuff all 0
  // generate source data(sendbuff)
  // calculate reference data(expected)
  for (int i = 0; i < num_gpus; ++i) {
    HGGCCHECK(hggcSetDevice(i));
    void *args[] = {(void *)&(sendbuff[i]),   (void *)&(recvbuff[i]),
                    (void *)&(expected[i]),   (void *)&(elems_num),
                    (void *)&(i),             (void *)&(num_gpus)};
    HGGCCHECK(hggcLaunchKernel(initDataKernels[int(data_type * COLL_OP_NUM + coll_type)],
                               grid_init, block_init, args, 0, s[i]));
  }
}

// init data by H2D
// [init data by H2D]  generate source val by element index and rank id
template <typename T>
T testValHost(const size_t idx, const int rank) {
  return (T)(1.f / (1.f + testValHost<int8_t>(idx, rank)));
}

template <>
int8_t testValHost(const size_t idx, const int rank) {
  return (int8_t)((idx + rank) % MAX_SCALE_RANGE);
}

template <>
double testValHost(const size_t idx, const int rank) {
  return __float2half(testValHost<float>(idx, rank));
}

// [init data by D2H] calculate reference val directly by source val
template <typename T>
T refValHost(const size_t idx, const int rank, const int nRanks, \
              const size_t N) {
  int offset = N / nRanks;
  T ref_val = (T)0;
  if (coll_type == ALLREDUCE) {
    // AllReduce
    ref_val = testValHost<T>(idx, 0);
    for (int i = 1; i < nRanks; i++) {
      ref_val += testValHost<T>(idx, i);
    }
  } else if (coll_type == BROADCAST) {
    // BroadCast
    ref_val = testValHost<T>(idx, 0);
  } else if (coll_type == ALLGATHER) {
    // AllGather
    ref_val = testValHost<T>(idx % offset, idx / offset);
  } else if (coll_type == REDUCE) {
    // Reduce
    ref_val = testValHost<T>(idx, rank);
    if (rank == 0) {
      for (int i = 1; i < nRanks; i++) {
        ref_val += testValHost<T>(idx, i);
      }
    }
  } else if (coll_type == REDUCESCATTER) {
    // ReduceScatter
    if (idx < offset) {
      for (int i = 0; i < nRanks; i++) {
        ref_val += testValHost<T>(int(rank * offset + idx), i);
      }
    }
  } else if (coll_type == ALLTOALL) {
    // AlltoAll
    int _rank = int(idx / offset);
    int _idx = int(rank*offset + idx%offset);
    ref_val = testValHost<T>(_idx, _rank);
  } else if (coll_type == ALLTOALLARRAY) {
    // ALLTOALLARRAY
    ref_val = testValHost<T>(idx, rank);
  }
  return ref_val;
}

template <typename T>
void threadHostInit(T* src_buff, T *ref_buff, size_t thread_start_idx,
    size_t thread_elems_num, size_t N, int rank, int nRanks) {
  for (size_t i = 0; i < thread_elems_num; i++) {
    size_t global_idx = thread_start_idx + i;
    src_buff[global_idx] = testValHost<T>(global_idx, rank);
    if (local_memcpy_stress) {
      ref_buff[global_idx] = src_buff[global_idx];
    } else {
      ref_buff[global_idx] = refValHost<T>(global_idx, rank, nRanks, N);
    }
  }
}

void getWorkThreadNum(size_t elems_num, int& worker_thread_num,
    size_t& per_thread_elems_num, size_t& left_elems_num) {
  // per thread deals with 2MB
  static const size_t per_thread_size = 2 * 1024 * 1024;
  // total max thread number is 32
  static const int max_thread_num = 32;
  per_thread_elems_num = per_thread_size / data_type_size();
  worker_thread_num = (elems_num + per_thread_elems_num - 1) / per_thread_elems_num;
  if (worker_thread_num > max_thread_num) worker_thread_num = max_thread_num;
  per_thread_elems_num = elems_num / worker_thread_num;
  left_elems_num = elems_num - per_thread_elems_num * worker_thread_num;
  DEBUG("worker_thread_num: %d", worker_thread_num);
  DEBUG("per_thread_elems_num: %d", per_thread_elems_num);
  DEBUG("left_elems_num: %d", left_elems_num);
}

template <typename T>
void initDataMultiThreads(T* src_buff, T* ref_buff, size_t elems_num, int rank, int nRanks) {
  int thread_num;
  size_t thread_elems_num, left_elems_num;
  getWorkThreadNum(elems_num, thread_num, thread_elems_num, left_elems_num);
  std::thread* buffer_init_threads = new std::thread[thread_num];
  int increment = 0;
  for (int i = 0; i < thread_num; i++) {
    size_t thread_start_idx = i * thread_elems_num;
    size_t thread_size = thread_elems_num;

    thread_start_idx += increment;
    if (i < left_elems_num) {
      ++thread_size;
      ++increment;
    }

    DEBUG("Init thread: %d, thread_start_idx: %lu, thread_size: %lu", i, thread_start_idx, thread_size);
    buffer_init_threads[i] = std::thread(threadHostInit<T>, src_buff,
        ref_buff, thread_start_idx, thread_size, elems_num, rank, nRanks);
  }

  for (int i = 0; i < thread_num; i++) {
    buffer_init_threads[i].join();
  }
  delete []buffer_init_threads;
}

// init data by H2D
template <typename T>
void initDatabyH2D(size_t* send_elems_num, T** sendbuff, size_t* recv_elems_num, T** recvbuff,
    T** src_host, T**  ref_host, hggcStream_t* s) {
  // malloc buffer and init data
  for (int i = 0; i < num_gpus; i++) {
    for (int j = 0; j < num_gpus; j++) {
      // init device data buffers
      if (send_elems_num[i*num_gpus+j] != recv_elems_num[j*num_gpus+i]) {
        ERROR("%d should be equal to %d here.", send_elems_num[i*num_gpus+j],
              recv_elems_num[j*num_gpus+i]);
        exit(EXIT_FAILURE);
      }

      initDataMultiThreads<T>(src_host[i*num_gpus+j], ref_host[j*num_gpus+i],
          send_elems_num[i*num_gpus+j], i, num_gpus);
    }
  }

  // copy data from host to device
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    for (int j = 0; j < num_gpus; j++) {
      HGGCCHECK(hggcMemcpyAsync(sendbuff[i*num_gpus+j], src_host[i*num_gpus+j], send_elems_num[i*num_gpus+j]*sizeof(T),
                                hggcMemcpyHostToDevice, s[i]));
      // use a non-zero value to do initialization to ease debug
      if (inplace == 0) {
        HGGCCHECK(hggcMemsetAsync(recvbuff[i*num_gpus+j], 0xc, recv_elems_num[i*num_gpus+j]*sizeof(T), s[i]));
      }
    }
  }
}

// init data by H2D
template <typename T>
void initDatabyH2D(size_t elems_num, T** sendbuff, T** recvbuff,
    T** src_host, T**  ref_host, hggcStream_t* s) {
  // malloc buffer and init data
  for (int i = 0; i < num_gpus; i++) {
    // init device data buffers
    initDataMultiThreads<T>(src_host[i], ref_host[i], elems_num, i, num_gpus);
  }

  // copy data from host to device
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcMemcpyAsync(sendbuff[i], src_host[i], elems_num * sizeof(T),
                              hggcMemcpyHostToDevice, s[i]));
    // use a non-zero value to do initialization to ease debug
    if (inplace == 0) {
      HGGCCHECK(hggcMemsetAsync(recvbuff[i], 0xc, elems_num * sizeof(T), s[i]));
    }
  }
}

void initData(size_t* send_elems_num, void** sendbuff, size_t* recv_elems_num, void** recvbuff,
    void** expected, void** src_host, void **ref_host, double** deltas_host, double** deltas_device,
    double** max_delta_device, hggcStream_t* s, hggcEvent_t* pEvent) {
  switch (data_type)
  {
  case DTYPE_INT8:
    if (host_init_and_verify) {
      initDatabyH2D<int8_t>(send_elems_num, (int8_t**)sendbuff, recv_elems_num, (int8_t**)recvbuff,
          (int8_t**)src_host, (int8_t**)ref_host, s);
    } else {
      initDatabyKernel<int8_t>(send_elems_num, (int8_t**)sendbuff, recv_elems_num, (int8_t**)recvbuff,
          (int8_t**)expected, s);
    }
    break;
  case DTYPE_HALF:
    if (host_init_and_verify) {
      initDatabyH2D<host_half>(send_elems_num, (host_half**)sendbuff, recv_elems_num, (host_half**)recvbuff,
          (host_half**)src_host, (host_half**)ref_host, s);
    } else {
      initDatabyKernel<half>(send_elems_num, (half**)sendbuff, recv_elems_num, (half**)recvbuff,
          (half**)expected, s);
    }
    break;
  case DTYPE_FLOAT:
    if (host_init_and_verify) {
      initDatabyH2D<float>(send_elems_num, (float**)sendbuff, recv_elems_num, (float**)recvbuff,
          (float**)src_host, (float**)ref_host, s);
    } else {
      initDatabyKernel<float>(send_elems_num, (float**)sendbuff, recv_elems_num, (float**)recvbuff,
          (float**)expected, s);
    }
    break;
  case DTYPE_DOUBLE:
    if (host_init_and_verify) {
      initDatabyH2D<double>(send_elems_num, (double**)sendbuff, recv_elems_num, (double**)recvbuff,
          (double**)src_host, (double**)ref_host, s);
    } else {
      initDatabyKernel<double>(send_elems_num, (double**)sendbuff, recv_elems_num, (double**)recvbuff,
          (double**)expected, s);
    }
    break;
  default:
    break;
  }

  for (int i = 0; i < num_gpus; ++i) {
    // init intermediate device buffers of deltas_device and max_delta_device
    HGGCCHECK(hggcSetDevice(i));
    for (int j = 0; j < num_gpus; ++j) {
      // init intermediate host buffer of deltas_host
      memset(deltas_host[i*num_gpus+j], 0.0, recv_elems_num[i*num_gpus+j]*sizeof(double));

      if (!host_init_and_verify) {
        HGGCCHECK(hggcMemsetAsync(deltas_device[i*num_gpus+j], 0.0, recv_elems_num[i*num_gpus+j]*sizeof(double), s[i]));
        HGGCCHECK(hggcMemsetAsync(max_delta_device[i*num_gpus+j], 0.0, sizeof(double), s[i]));
      }
    }

    // record event
    HGGCCHECK(hggcEventRecord(pEvent[i], s[i]));
  }

  // wait event
  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < num_gpus; ++j) {
      HGGCCHECK(hggcStreamWaitEvent(s[i], pEvent[j], 0));
    }
  }
}

void initData(size_t elems_num, void** sendbuff, void** recvbuff, void** expected,
    void** src_host, void **ref_host, double** deltas_host, double** deltas_device,
    double** max_delta_device, hggcStream_t* s, hggcEvent_t* pEvent) {
  switch (data_type)
  {
  case DTYPE_INT8:
    if (host_init_and_verify) {
      initDatabyH2D<int8_t>(elems_num, (int8_t**)sendbuff, (int8_t**)recvbuff,
          (int8_t**)src_host, (int8_t**)ref_host, s);
    } else {
      initDatabyKernel<int8_t>(elems_num, (int8_t**)sendbuff, (int8_t**)recvbuff,
          (int8_t**)expected, s);
    }
    break;
  case DTYPE_HALF:
    if (host_init_and_verify) {
      initDatabyH2D<host_half>(elems_num, (host_half**)sendbuff, (host_half**)recvbuff,
          (host_half**)src_host, (host_half**)ref_host, s);
    } else {
      initDatabyKernel<half>(elems_num, (half**)sendbuff, (half**)recvbuff,
          (half**)expected, s);
    }
    break;
  case DTYPE_FLOAT:
    if (host_init_and_verify) {
      initDatabyH2D<float>(elems_num, (float**)sendbuff, (float**)recvbuff,
          (float**)src_host, (float**)ref_host, s);
    } else {
      initDatabyKernel<float>(elems_num, (float**)sendbuff, (float**)recvbuff,
          (float**)expected, s);
    }
    break;
  case DTYPE_DOUBLE:
    if (host_init_and_verify) {
      initDatabyH2D<double>(elems_num, (double**)sendbuff, (double**)recvbuff,
          (double**)src_host, (double**)ref_host, s);
    } else {
      initDatabyKernel<double>(elems_num, (double**)sendbuff, (double**)recvbuff,
          (double**)expected, s);
    }
    break;
  default:
    break;
  }

  for (int i = 0; i < num_gpus; ++i) {
    // init intermediate host buffer of deltas_host
    memset(deltas_host[i], 0.0, elems_num * sizeof(double));

    // init intermediate device buffers of deltas_device and max_delta_device
    HGGCCHECK(hggcSetDevice(i));
    if (!host_init_and_verify) {
      HGGCCHECK(hggcMemsetAsync(deltas_device[i], 0.0, elems_num * sizeof(double), s[i]));
      HGGCCHECK(hggcMemsetAsync(max_delta_device[i], 0.0, sizeof(double), s[i]));
    }

    // record event
    HGGCCHECK(hggcEventRecord(pEvent[i], s[i]));
  }

  // wait event
  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < num_gpus; ++j) {
      HGGCCHECK(hggcStreamWaitEvent(s[i], pEvent[j], 0));
    }
  }
}


// compare result, count number of mismatched values


template <typename T, int BSIZE>
__global__ void compDataKernel(T *reference, T *result, const size_t N,
                               double *deltas, double *max_delta_val) {
  int tid = threadIdx.x;
  __shared__ double thread_max_deltas[BSIZE];
  double loc_max_delta = 0.0;
  for (size_t o = tid; o < N; o += blockDim.x) {
    deltas[o] = fabs(((double)reference[o] - (double)result[o]));
    if (deltas[o] > loc_max_delta) loc_max_delta = deltas[o];
    if (isnan(deltas[o])) loc_max_delta = NAN_MAX_DELTA;
  }
  thread_max_deltas[tid] = loc_max_delta;
  for(int stride = BSIZE/2; stride > 1; stride >>= 1) {
    __syncthreads();
    if( tid < stride )
      thread_max_deltas[tid] = \
        thread_max_deltas[tid] > thread_max_deltas[tid + stride] ? \
          thread_max_deltas[tid] : thread_max_deltas[tid + stride];
  }
  __syncthreads();
  if(tid == 0)
    *max_delta_val = thread_max_deltas[0] > thread_max_deltas[1] ? \
                    thread_max_deltas[0] : thread_max_deltas[1];
}

static void* const compDataKerns[DTYPE_NUM] = {
  (void*)compDataKernel<   float, blocks_verify_num>,
  (void*)compDataKernel<    half, blocks_verify_num>,
  (void*)compDataKernel<  int8_t, blocks_verify_num>,
  (void*)compDataKernel<  double, blocks_verify_num>,
};

bool verifyDatabyKernel(size_t* recv_elem_num, void** recvbuff, void** expected,
    double** deltas_host, double* max_deltas_host, double** deltas_device,
    double** max_delta_device, hggcStream_t* s) {
  bool res = true;
  // compare reference and result
  if (coll_type != ALLTOALLARRAY) {
    ERROR("coll type %d should be equal to %d here", coll_type, ALLTOALLARRAY);
    return false;
  }

  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    for (int j = 0; j < num_gpus; j++) {
      void *args[] = {
          (void *)&(expected[i*num_gpus+j]),      (void *)&(recvbuff[i*num_gpus+j]),
          (void *)(recv_elem_num+i*num_gpus+j),         (void *)&(deltas_device[i*num_gpus+j]),
          (void *)&(max_delta_device[i*num_gpus+j])};
      dim3 grid_verify = { 1, 1, 1 };
      dim3 block_verify = { blocks_verify_num, 1, 1 };
      HGGCCHECK(hggcLaunchKernel(compDataKerns[data_type], grid_verify,
                                 block_verify, args, 0, s[i]));

      // copy compare result from device to host
      HGGCCHECK(hggcMemcpyAsync(max_deltas_host+i*num_gpus+j, max_delta_device[i*num_gpus+j], sizeof(double),
            hggcMemcpyDeviceToHost, s[i]));
      HGGCCHECK(hggcStreamSynchronize(s[i]));
      if (max_deltas_host[i*num_gpus+j] > max_delta) {
        HGGCCHECK(hggcMemcpyAsync(deltas_host[i*num_gpus+j], deltas_device[i*num_gpus+j],
              recv_elem_num[i*num_gpus+j]*sizeof(double), hggcMemcpyDeviceToHost, s[i]));
        res = false;
      }
    }
  }

  return res;
}

bool verifyDatabyKernel(size_t elem_num, void** recvbuff, void** expected,
    double** deltas_host, double* max_deltas_host, double** deltas_device,
    double** max_delta_device, hggcStream_t* s) {
  bool res = true;
  // compare reference and result
  if (coll_type == REDUCESCATTER) elem_num = size_t(elem_num / num_gpus);
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    if (i > 0 && coll_type == REDUCE) continue;
    void *args[] = {
        (void *)&(expected[i]),      (void *)&(recvbuff[i]),
        (void *)&(elem_num),         (void *)&(deltas_device[i]),
        (void *)&(max_delta_device[i])};
    dim3 grid_verify = { 1, 1, 1 };
    dim3 block_verify = { blocks_verify_num, 1, 1 };
    HGGCCHECK(hggcLaunchKernel(compDataKerns[data_type], grid_verify,
                               block_verify, args, 0, s[i]));

    // copy compare result from device to host
    HGGCCHECK(hggcMemcpyAsync(max_deltas_host + i, max_delta_device[i], sizeof(double),
          hggcMemcpyDeviceToHost, s[i]));
    HGGCCHECK(hggcStreamSynchronize(s[i]));
    if (max_deltas_host[i] > max_delta) {
      HGGCCHECK(hggcMemcpyAsync(deltas_host[i], deltas_device[i],
            elem_num * sizeof(double), hggcMemcpyDeviceToHost, s[i]));
      res = false;
    }
  }

  return res;
}

template <typename T>
double diffBetweenVals(T* buff0, T* buff1, int idx) {
  return fabs(double(buff0[idx] - buff1[idx]));
}

template <>
double diffBetweenVals(half* buff0, half* buff1, int idx) {
  return fabs(double(buff0[idx]) - double(buff1[idx]));
}

template <typename T>
void threadHostDeltaComp(T* recv_buff, T* ref_buff, double* deltas_host,
    double* max_delta_val_ptr, int  thread_start_idx, int thread_elems_num) {
  *max_delta_val_ptr = 0.0;
  for (int i = 0; i < thread_elems_num; ++i) {
    int global_idx = thread_start_idx + i;
    deltas_host[global_idx] = diffBetweenVals<T>(ref_buff, recv_buff, global_idx);
    if (deltas_host[global_idx] > *max_delta_val_ptr) {
      *max_delta_val_ptr = deltas_host[global_idx];
    }
    if (std::isnan(deltas_host[global_idx])) *max_delta_val_ptr = NAN_MAX_DELTA;
  }
}

template <typename T>
void compDeltaData(T* recv_buff, T* ref_buff, double* deltas_host_buff,
    double* max_delta_val_ptr, size_t elems_num) {
  int thread_num;
  size_t thread_elems_num, left_elems_num;
  getWorkThreadNum(elems_num, thread_num, thread_elems_num, left_elems_num);
  std::thread* buff_cmp_threads = new std::thread[thread_num];
  double* per_thread_max_delta = new double[thread_num];
  int increment = 0;
  for (int i = 0; i < thread_num; i++) {
    int thread_start_idx = i * thread_elems_num;
    int thread_size = thread_elems_num;

    thread_start_idx += increment;
    if (i < left_elems_num) {
      ++thread_size;
      ++increment;
    }

    DEBUG("result verify thread: %d, thread_start_idx: %lu, thread_size: %lu", i, thread_start_idx, thread_size);
    buff_cmp_threads[i] = std::thread(threadHostDeltaComp<T>, recv_buff,
        ref_buff, deltas_host_buff, per_thread_max_delta + i,
        thread_start_idx, thread_size);
  }

  *max_delta_val_ptr = 0.0;
  for (int i = 0; i < thread_num; i++) {
    buff_cmp_threads[i].join();
    if (per_thread_max_delta[i] > *max_delta_val_ptr) {
      *max_delta_val_ptr = per_thread_max_delta[i];
    }
  }

  delete []per_thread_max_delta;
  delete []buff_cmp_threads;
}

void dump_data(const char* marker_str1, const char* marker_str2, int dev, int peer_dev,
    int8_t* buff1, int8_t* buff2, size_t elem_num) {
  INFO("%s and %s, dev: %d, peer dev: %d:", marker_str1, marker_str2, dev, peer_dev);
  for (size_t i = 0; i < elem_num; ++i) {
    INFO("idx %d, %d, %d", i, buff1[i], buff2[i]);
  }
}

bool verifyDatabyHost(size_t* recv_elems_num, void** recvbuff, void** recv_host,
                      void** ref_host, double** deltas_host,
                      double* max_deltas_host, hggcStream_t* s) {
  bool res = true;
  // compare reference and result
  if (coll_type != ALLTOALLARRAY) {
    ERROR("coll type: %d should be %d here", coll_type, ALLTOALLARRAY);
    return false;
  }
  for (int i = 0; i < num_gpus; i++) {
    HGGCCHECK(hggcSetDevice(i));
    for (int j = 0; j < num_gpus; j++) {
      // copy device results from device to host
      HGGCCHECK(hggcMemcpyAsync(recv_host[i*num_gpus+j], recvbuff[i*num_gpus+j], recv_elems_num[i*num_gpus+j]*data_type_size(),
                                hggcMemcpyDeviceToHost, s[i]));
      HGGCCHECK(hggcStreamSynchronize(s[i]));

      // comparison between output and golden ref buffer, obtain delta host and max deltas
      switch(data_type) {
      case DTYPE_INT8:
          compDeltaData<int8_t>((int8_t*)(recv_host[i*num_gpus+j]), (int8_t*)(ref_host[i*num_gpus+j]), deltas_host[i*num_gpus+j],
              max_deltas_host+i*num_gpus+j, recv_elems_num[i*num_gpus+j]);
        break;
      case DTYPE_HALF:
          compDeltaData<half>((half*)(recv_host[i*num_gpus+j]), (half*)(ref_host[i*num_gpus+j]), deltas_host[i*num_gpus+j],
              max_deltas_host+i*num_gpus+j, recv_elems_num[i*num_gpus+j]);
        break;
      case DTYPE_FLOAT:
          compDeltaData<float>((float*)(recv_host[i*num_gpus+j]), (float*)(ref_host[i*num_gpus+j]), deltas_host[i*num_gpus+j],
              max_deltas_host+i*num_gpus+j, recv_elems_num[i*num_gpus+j]);
        break;
      case DTYPE_DOUBLE:
          compDeltaData<double>((double*)(recv_host[i*num_gpus+j]), (double*)(ref_host[i*num_gpus+j]), deltas_host[i*num_gpus+j],
              max_deltas_host+i*num_gpus+j, recv_elems_num[i*num_gpus+j]);
        break;
      default:
          ERROR("unsupported data type %d, your input "
                        "must be 'float', 'half', 'int8' or 'double'.", data_type);
        exit(EXIT_FAILURE);
      }

      if (max_deltas_host[i*num_gpus+j] > max_delta) {
        res = false;
        if (data_type == DTYPE_INT8 && 1 == dump_full) {
          const char* str1 = "recv_host";
          const char* str2 = "ref_host";
          dump_data(str1, str2, i, j, (int8_t*)(recv_host[i*num_gpus+j]),
              (int8_t*)(ref_host[i*num_gpus+j]), recv_elems_num[i*num_gpus+j]);
        }
      }
    }
  }

  return res;
}

bool verifyDatabyHost(size_t elems_num, void** recvbuff, void** recv_host,
                      void** ref_host, double** deltas_host,
                      double* max_deltas_host, hggcStream_t* s) {
  bool res = true;
  // compare reference and result
  if (coll_type == REDUCESCATTER) elems_num = size_t(elems_num / num_gpus);
  for (int i = 0; i < num_gpus; i++) {
    if (i > 0 && coll_type == REDUCE) continue;

    // copy device results from device to host
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcMemcpyAsync(recv_host[i], recvbuff[i], elems_num * data_type_size(),
                              hggcMemcpyDeviceToHost, s[i]));
    HGGCCHECK(hggcStreamSynchronize(s[i]));

    // comparison between output and golden ref buffer, obtain delta host and max deltas
    switch(data_type) {
    case DTYPE_INT8:
        compDeltaData<int8_t>((int8_t*)(recv_host[i]), (int8_t*)(ref_host[i]), deltas_host[i],
            max_deltas_host + i, elems_num);
      break;
    case DTYPE_HALF:
        compDeltaData<half>((half*)(recv_host[i]), (half*)(ref_host[i]), deltas_host[i],
            max_deltas_host + i, elems_num);
      break;
    case DTYPE_FLOAT:
        compDeltaData<float>((float*)(recv_host[i]), (float*)(ref_host[i]), deltas_host[i],
            max_deltas_host + i, elems_num);
      break;
    case DTYPE_DOUBLE:
        compDeltaData<double>((double*)(recv_host[i]), (double*)(ref_host[i]), deltas_host[i],
            max_deltas_host + i, elems_num);
      break;
    default:
        ERROR("unsupported data type %d, input must be 'float', 'half', 'int8' or 'double'.");
        exit(EXIT_FAILURE);
    }

    if (max_deltas_host[i] > max_delta) {
      res = false;
    }
  }

  return res;
}

bool verifyData(size_t* recv_elems_num, void** expected, void** recvbuff,
    void** recv_host, void** ref_host, double** deltas_host, double* max_deltas_host,
    double** deltas_device, double** max_delta_device, hggcStream_t* s) {
  if (host_init_and_verify) {
    return verifyDatabyHost(recv_elems_num, recvbuff, recv_host, ref_host,
        deltas_host, max_deltas_host, s);
  } else {
    return verifyDatabyKernel(recv_elems_num, recvbuff, expected, deltas_host,
        max_deltas_host, deltas_device, max_delta_device, s);
  }
}

bool verifyData(size_t elem_num, void** expected, void** recvbuff, void** recv_host,
    void** ref_host, double** deltas_host, double* max_deltas_host,
    double** deltas_device, double** max_delta_device, hggcStream_t* s) {
  if (host_init_and_verify) {
    return verifyDatabyHost(elem_num, recvbuff, recv_host, ref_host,
        deltas_host, max_deltas_host, s);
  } else {
    return verifyDatabyKernel(elem_num, recvbuff, expected, deltas_host,
        max_deltas_host, deltas_device, max_delta_device, s);
  }
}


// print data

template <typename T>
std::string valToString(T val) {
  return std::to_string(float(val));
}

template<typename T>
void printDataElems(size_t* send_elems_num, T** sendbuff, T** expected,
    size_t* recv_elems_num, T** recvbuff, T** src_host, T** recv_host,
    T** ref_host, double** deltas_host, double* max_deltas_host, hggcStream_t* s) {

  const int array_size = num_gpus * num_gpus;
  T* act[array_size];
  T* ref[array_size];
  T* send[array_size];
  // malloc host buffer
  for (int i = 0; i < num_gpus; i++) {
    for (int j = 0; j < num_gpus; j++) {
      if (host_init_and_verify) {
        act[i*num_gpus+j] = recv_host[i*num_gpus+j];
        ref[i*num_gpus+j] = ref_host[i*num_gpus+j];
        send[i*num_gpus+j] = src_host[i*num_gpus+j];
      } else {
        act[i*num_gpus+j] = (T*)malloc(recv_elems_num[i*num_gpus+j]*sizeof(T));
        ref[i*num_gpus+j] = (T*)malloc(recv_elems_num[i*num_gpus+j]*sizeof(T));
        send[i*num_gpus+j] = (T*)malloc(send_elems_num[i*num_gpus+j]*sizeof(T));
        // copy data from device to host
        if (max_deltas_host[i*num_gpus+j] > max_delta) {
          HGGCCHECK(hggcSetDevice(i));
          HGGCCHECK(hggcMemcpyAsync(act[i*num_gpus+j], recvbuff[i*num_gpus+j], recv_elems_num[i*num_gpus+j]*sizeof(T),
                hggcMemcpyDeviceToHost, s[i]));
          HGGCCHECK(hggcMemcpyAsync(ref[i*num_gpus+j], expected[i*num_gpus+j], recv_elems_num[i*num_gpus+j]*sizeof(T),
                                    hggcMemcpyDeviceToHost, s[i]));
          HGGCCHECK(hggcMemcpyAsync(send[i*num_gpus+j], sendbuff[i*num_gpus+j], send_elems_num[i*num_gpus+j]*sizeof(T),
                hggcMemcpyDeviceToHost, s[i]));
        }
      }
    }
  }

  // print mismatched elements (index, receive value, reference value)
  std::string info = "";
  for (int i = 0; i < num_gpus; i++) {
    for (int j = 0; j < num_gpus; j++) {
      if (max_deltas_host[i*num_gpus+j] > max_delta) {
        info += "\n  device " + std::to_string(i) + ", peer device " + std::to_string(j) + "\n";
        size_t pre_mismatch_index = 0;
        size_t continuous_len = 0;
        for (size_t k = 0; k < recv_elems_num[i*num_gpus+j]; k++) {
          if (deltas_host[i*num_gpus+j][k] > max_delta) {
            if ((k == 0) || (k - pre_mismatch_index) == 1) {
              continuous_len += 1;
            } else  {
              continuous_len = 0;
            }
            pre_mismatch_index = k;
            if (continuous_len < show_continuous_mismatch) {
                info += "    index " + std::to_string(k) + \
                    ", send " + valToString<T>(send[j*num_gpus+i][k]) + \
                    ", receive " + valToString<T>(act[i*num_gpus+j][k]) + \
                    ", reference " + valToString<T>(ref[i*num_gpus+j][k]) + \
                    ", delta " + std::to_string(deltas_host[i*num_gpus+j][k]) + \
                    ", max_delta " + std::to_string(max_delta) + "\n";
            }
            if (continuous_len == show_continuous_mismatch) {
              info += "    ... mismatch data excced " + std::to_string(show_continuous_mismatch) + "\n";
              info += "    ... increase --show_continuous_mismatch to show more mismatch data \n";
            }
          }
        }
      }
    }
  }
  INFO("%s", info.c_str());

  for (int i = 0; i < num_gpus; i++) {
    for (int j = 0; j < num_gpus; j++) {
      if (!host_init_and_verify) {
        free(act[i*num_gpus+j]);
        free(ref[i*num_gpus+j]);
      }
    }
  }
}

template<typename T>
void printDataElems(size_t elems_num, T** sendbuff, T** expected, T** recvbuff, T** src_host,
    T** recv_host, T** ref_host, double** deltas_host, double* max_deltas_host, hggcStream_t* s) {
  T* act[num_gpus];
  T* ref[num_gpus];
  T* send[num_gpus];
  // malloc host buffer
  for (int i = 0; i < num_gpus; i++) {
    if (host_init_and_verify) {
      act[i] = recv_host[i];
      ref[i] = ref_host[i];
      send[i] = src_host[i];
    } else {
      act[i] = (T*)malloc(elems_num*sizeof(T));
      ref[i] = (T*)malloc(elems_num*sizeof(T));
      send[i] = (T*)malloc(elems_num*sizeof(T));
      // copy data from device to host
      if (max_deltas_host[i] > max_delta) {
        HGGCCHECK(hggcSetDevice(i));
        HGGCCHECK(hggcMemcpyAsync(act[i], recvbuff[i], elems_num * sizeof(T),
              hggcMemcpyDeviceToHost, s[i]));
        HGGCCHECK(hggcMemcpyAsync(ref[i], expected[i], elems_num * sizeof(T),
                                  hggcMemcpyDeviceToHost, s[i]));
        HGGCCHECK(hggcMemcpyAsync(send[i], sendbuff[i], elems_num * sizeof(T),
              hggcMemcpyDeviceToHost, s[i]));
      }
    }
  }

  // print mismatched elements (index, receive value, reference value)
  for (int i = 0; i < num_gpus; i++) {
    std::string info = "";
    if (max_deltas_host[i] > max_delta || std::isnan(max_deltas_host[i])) {
      info += "\n  device " + std::to_string(i) + "\n";
      size_t pre_mismatch_index = 0;
      size_t continuous_len = 0;
      for (size_t j = 0; j < elems_num; j++) {
        if (deltas_host[i][j] > max_delta || std::isnan(deltas_host[i][j])) {
          if ((j == 0) || (j - pre_mismatch_index) == 1) {
            continuous_len += 1;
          } else  {
            continuous_len = 0;
          }
          pre_mismatch_index = j;
          if (continuous_len < show_continuous_mismatch) {
              info += "    index " + std::to_string(j) + \
                  ", send " + valToString<T>(send[i][j]) + \
                  ", receive " + valToString<T>(act[i][j]) + \
                  ", reference " + valToString<T>(ref[i][j]) + \
                  ", delta " + std::to_string(deltas_host[i][j]) + \
                  ", max_delta " + std::to_string(max_delta) + "\n";
          }
          if (continuous_len == show_continuous_mismatch) {
            info += "    ... mismatch data excced " + std::to_string(show_continuous_mismatch) + "\n";
            info += "    ... increase --show_continuous_mismatch to show more mismatch data \n";
          }
        }
      }
    }
    INFO("%s", info.c_str());
  }

  for (int i = 0; i < num_gpus; i++) {
    if (!host_init_and_verify) {
      free(act[i]);
      free(ref[i]);
    }
  }
}

void printData(size_t* send_elems_num, void **sendbuff, void** expected,
    size_t* recv_elems_num, void** recvbuff, void**src_host, void**recv_host, void** ref_host,
    double** deltas_host, double* max_deltas_host, hggcStream_t* s) {
  switch (data_type)
  {
  case DTYPE_INT8:
    printDataElems<int8_t>(send_elems_num, (int8_t **)sendbuff, (int8_t**)expected,
        recv_elems_num, (int8_t**)recvbuff, (int8_t**)src_host, (int8_t**)recv_host,
        (int8_t**)ref_host, deltas_host, max_deltas_host, s);
    break;
  case DTYPE_HALF:
    printDataElems<host_half>(send_elems_num, (host_half**)sendbuff, (host_half**)expected,
        recv_elems_num, (host_half**)recvbuff, (host_half**)src_host, (host_half**)recv_host,
        (host_half**)ref_host, deltas_host, max_deltas_host, s);
    break;
  case DTYPE_FLOAT:
    printDataElems<float>(send_elems_num, (float**)sendbuff, (float**)expected,
        recv_elems_num, (float**)recvbuff, (float**)src_host, (float**)recv_host,
        (float**)ref_host, deltas_host, max_deltas_host, s);
    break;
  case DTYPE_DOUBLE:
    printDataElems<double>(send_elems_num, (double**)sendbuff, (double**)expected,
        recv_elems_num, (double**)recvbuff, (double**)src_host, (double**)recv_host,
        (double**)ref_host, deltas_host, max_deltas_host, s);
    break;
  default:
    break;
  }
}

void printData(size_t elems_num, void **sendbuff, void** expected, void** recvbuff, void**src_host,
    void**recv_host, void** ref_host, double** deltas_host, double* max_deltas_host,
    hggcStream_t* s) {
  switch (data_type)
  {
  case DTYPE_INT8:
    printDataElems<int8_t>(elems_num, (int8_t **)sendbuff, (int8_t**)expected, (int8_t**)recvbuff,
        (int8_t**)src_host, (int8_t**)recv_host, (int8_t**)ref_host, deltas_host, max_deltas_host, s);
    break;
  case DTYPE_HALF:
    printDataElems<host_half>(elems_num, (host_half**)sendbuff, (host_half**)expected,
        (host_half**)recvbuff, (host_half**)src_host, (host_half**)recv_host, (host_half**)ref_host,
        deltas_host, max_deltas_host, s);
    break;
  case DTYPE_FLOAT:
    printDataElems<float>(elems_num, (float**)sendbuff, (float**)expected, (float**)recvbuff,
        (float**)src_host, (float**)recv_host, (float**)ref_host, deltas_host, max_deltas_host, s);
    break;
  case DTYPE_DOUBLE:
    printDataElems<double>(elems_num, (double**)sendbuff, (double**)expected, (double**)recvbuff,
        (double**)src_host, (double**)recv_host, (double**)ref_host, deltas_host, max_deltas_host, s);
    break;
  default:
    break;
  }
}

// device memory free
void bufferFreeArray(void** sendbuff_origin, void** recvbuff_origin, void** expected, void** src_host,
    void** recv_host, void** ref_host, double** deltas_host, double** deltas_device,
    double** max_delta_device) {
  if (reuse_buffer) {
    return;
  }

  for (int i = 0; i < num_gpus; ++i) {
    //free device buffers and temp cpu buffers
    HGGCCHECK(hggcSetDevice(i));
    for (int j = 0; j < num_gpus; ++j) {
      HGGCCHECK(hggcFree(sendbuff_origin[i*num_gpus+j]));
      if (inplace == 0) HGGCCHECK(hggcFree(recvbuff_origin[i*num_gpus+j]));
      free(deltas_host[i*num_gpus+j]);
      DEBUG("dev %d, peer dev %d, free sendbuff_origin addr %p"
            "free recvbuff_origin addr %p, free deltas_host addr %p", i, j,
            sendbuff_origin[i*num_gpus+j], recvbuff_origin[i*num_gpus+j], deltas_host[i*num_gpus+j]);

      if (host_init_and_verify) {
        HGGCCHECK(hggcFreeHost(src_host[i*num_gpus+j]));
        HGGCCHECK(hggcFreeHost(recv_host[i*num_gpus+j]));
        HGGCCHECK(hggcFreeHost(ref_host[i*num_gpus+j]));
        DEBUG("dev %d, peer dev %d, free src_host addr %p, free ref_host addr  %p",
              i, j, src_host[i*num_gpus+j], ref_host[i*num_gpus+j]);
      } else {
        HGGCCHECK(hggcFree(expected[i*num_gpus+j]));
        HGGCCHECK(hggcFree(deltas_device[i*num_gpus+j]));
        HGGCCHECK(hggcFree(max_delta_device[i*num_gpus+j]));
        DEBUG("dev %d, peer dev %d, free expected addr %p, free deltas_device addr %p,"
              "free max_delta_device addr %p", i, j, expected[i*num_gpus+j],
              deltas_device[i*num_gpus+j], max_delta_device[i*num_gpus+j]);
      }
    }
  }
}

// device memory free
void bufferFree(void** sendbuff_origin, void** recvbuff_origin, void** expected, void** src_host,
    void** recv_host, void** ref_host, double** deltas_host, double** deltas_device,
    double** max_delta_device) {
  if (reuse_buffer) {
    return;
  }

  for (int i = 0; i < num_gpus; ++i) {
    //free device buffers and temp cpu buffers
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcFree(sendbuff_origin[i]));
    if (inplace == 0) HGGCCHECK(hggcFree(recvbuff_origin[i]));
    free(deltas_host[i]);
    DEBUG("dev %d, free sendbuff_origin addr %p, free recvbuff_origin addr %p"
          ", free deltas_host addr %p", i, sendbuff_origin[i],
          recvbuff_origin[i], deltas_host[i]);

    if (host_init_and_verify) {
      HGGCCHECK(hggcFreeHost(src_host[i]));
      HGGCCHECK(hggcFreeHost(recv_host[i]));
      HGGCCHECK(hggcFreeHost(ref_host[i]));
      DEBUG("dev %d, free src_host addr %p, free ref_host addr %p", i,
            src_host[i], ref_host[i]);
    } else {
      HGGCCHECK(hggcFree(expected[i]));
      HGGCCHECK(hggcFree(deltas_device[i]));
      HGGCCHECK(hggcFree(max_delta_device[i]));
      DEBUG("dev %d, free expected addr %p, free deltas_device addr %p"
            ", free max_delta_device addr %p", i, expected[i],
            deltas_device[i], max_delta_device[i]);
    }
  }
}

// destroy device streams and events
void destroyDevices(hggcStream_t* s, hggcEvent_t* pEvent) {
  //free device data structures
  for (int i = 0; i < num_gpus; ++i) {
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcStreamDestroy(s[i]));
    HGGCCHECK(hggcEventDestroy(pEvent[i]));
  }
}

// call pccl collective array api
void coll_array_test(pcclComm_t* comms, size_t* send_elems_num, size_t* recv_elems_num,
    pcclDataType_t pccl_dtype, void **sendbuff_array, void **recvbuff_array, hggcStream_t *s) {

   //calling PCCL communication API. Group API is required when using
   //multiple devices per thread
  PCCLCHECK(pcclGroupStart());
  for (int i = 0; i < num_gpus; ++i) {
    switch (coll_type) {
      case ALLTOALLARRAY:
        PCCLCHECK(pcclAlltoAllArray((const void**)(sendbuff_array+i*num_gpus), (void**)(recvbuff_array+i*num_gpus),
              send_elems_num+i*num_gpus, recv_elems_num+i*num_gpus, pccl_dtype, comms[i], s[i]));
        break;
      default:
        ERROR("Not Implement now, coll_type %d", coll_type);
        exit(0);
    }
  }

  PCCLCHECK(pcclGroupEnd());
}

// call pccl collective api
void coll_test(pcclComm_t* comms, size_t elems_num, pcclDataType_t pccl_dtype,
    void **sendbuff, void **recvbuff, hggcStream_t *s) {
  // exec_counts, used to raise exception in the middle of tests.
  static int exec_counts = 0;

   //calling PCCL communication API. Group API is required when using
   //multiple devices per thread
  PCCLCHECK(pcclGroupStart());
  for (int i = 0; i < num_gpus; ++i) {
    if (abnormal_mode == 1 && exec_counts == 2) {
      if (i == 0) INFO("Abnormal Mode:%d, trigger hang", abnormal_mode);
      if (i == 0) coll_type = ALLREDUCE;
      if (i == 1) coll_type = BROADCAST;
    }
    switch (coll_type) {
      case ALLREDUCE :
        PCCLCHECK(pcclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i],
              elems_num, pccl_dtype, pcclSum, comms[i], s[i]));
        break;
      case BROADCAST:
        PCCLCHECK(pcclBroadcast((const void*)sendbuff[i], (void*)recvbuff[i],
              elems_num, pccl_dtype, 0 , comms[i], s[i]));
        break;
      case ALLGATHER:
        PCCLCHECK(pcclAllGather((const void*)sendbuff[i], (void*)recvbuff[i],
              elems_num/num_gpus, pccl_dtype, comms[i], s[i]));
        break;
      case REDUCE:
        PCCLCHECK(pcclReduce((const void*)sendbuff[i], (void*)recvbuff[i],
              elems_num, pccl_dtype, pcclSum, 0, comms[i], s[i]));
        break;
      case REDUCESCATTER:
        PCCLCHECK(pcclReduceScatter((const void*)sendbuff[i], (void*)recvbuff[i],
              elems_num/num_gpus, pccl_dtype, pcclSum, comms[i], s[i]));
        break;
      case ALLTOALL:
        PCCLCHECK(pcclAlltoAll((const void*)sendbuff[i], (void*)recvbuff[i],
              elems_num/num_gpus, pccl_dtype, comms[i], s[i]));
        break;
      default:
        ERROR("Not Implement now, coll_type %d", coll_type);
        exit(0);
    }
  }
  exec_counts++ ;

  PCCLCHECK(pcclGroupEnd());
}

void initDevices(hggcStream_t* s, hggcEvent_t* pEvent) {
  for (int i = 0; i < num_gpus; ++i) {
    HGGCCHECK(hggcSetDevice(i));
    HGGCCHECK(hggcStreamCreateWithFlags(s+i, hggcStreamNonBlocking));
    INFO("dev id: %d, hggcStream_t[%d] = %p", i, i, s[i]);
    HGGCCHECK(hggcEventCreate(pEvent+i, hggcEventDisableTiming));
  }
}

template<typename T>
__global__ void localTestKernel(T*__restrict__ send, T*__restrict__ recv, const size_t N) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;
  for (size_t o = globalId; o < N; o += gridSize) {
    recv[o] = send[o];
  }
}

static void* const localTestKernels[4] = {
  (void*)localTestKernel<   float>,
  (void*)localTestKernel<    half>,
  (void*)localTestKernel<  int8_t>,
  (void*)localTestKernel<  double>,
};

template<typename T>
void local_test(size_t elems_num, T** sendbuff, T** recvbuff, hggcStream_t* s) {
  dim3 grid_local = { 14, 1, 1 };
  dim3 block_local = { 512, 1, 1 };
  for (int i = 0; i < num_gpus; ++i) {
    HGGCCHECK(hggcSetDevice(i));
    void* args[3] = {
      (void*)&(sendbuff[i]),
      (void*)&(recvbuff[i]),
      (void*)&(elems_num),
    };
    HGGCCHECK(hggcLaunchKernel(localTestKernels[int(data_type)],
          grid_local, block_local, args, 0, s[i]));
  }
}

void runLocalTest(size_t elems_num, void** sendbuff, void** recvbuff,
    hggcStream_t* s) {
  switch (data_type)
  {
  case DTYPE_FLOAT:
    local_test<float>(elems_num, (float**)sendbuff, (float**)recvbuff, s);
    break;
  case DTYPE_HALF:
    local_test<half>(elems_num, (half**)sendbuff, (half**)recvbuff, s);
    break;
  case DTYPE_INT8:
    local_test<int8_t>(elems_num, (int8_t**)sendbuff, (int8_t**)recvbuff, s);
    break;
  case DTYPE_DOUBLE:
    local_test<double>(elems_num, (double**)sendbuff, (double**)recvbuff, s);
    break;
  default:
    break;
  }
}

void runArrayTest(pcclComm_t* comms, size_t* send_elems_num, size_t* recv_elems_num, void** sendbuff_array,
    void** recvbuff_array, hggcStream_t* s) {
  switch (data_type)
  {
  case DTYPE_FLOAT:
    coll_array_test(comms, send_elems_num, recv_elems_num, pcclFloat, sendbuff_array, recvbuff_array, s);
    break;
  case DTYPE_HALF:
    coll_array_test(comms, send_elems_num, recv_elems_num, pcclHalf, sendbuff_array, recvbuff_array, s);
    break;
  case DTYPE_INT8:
    coll_array_test(comms, send_elems_num, recv_elems_num, pcclInt8, sendbuff_array, recvbuff_array, s);
    break;
  case DTYPE_DOUBLE:
    coll_array_test(comms, send_elems_num, recv_elems_num, pcclDouble, sendbuff_array, recvbuff_array, s);
    break;
  default:
    break;
  }
}

// coll array test
int runCollArrayTest(pcclComm_t* comms, hggcStream_t* s, hggcEvent_t* e) {
  int res = 0;

  // buffers and arrays
  const int array_size = num_gpus * num_gpus;
  void* sendbuff_origin[array_size];
  void* recvbuff_origin[array_size];
  void* sendbuff[array_size];
  void* recvbuff[array_size];
  void* expected[array_size];
  void* src_host[array_size];
  void* recv_host[array_size];
  void* ref_host[array_size];
  double* deltas_host[array_size];
  double max_deltas_host[array_size];
  double* max_delta_device[array_size];
  double* deltas_device[array_size];
  size_t send_elems_array[num_iterations*array_size];
  size_t recv_elems_array[num_iterations*array_size];
  generateElementNumArray(send_elems_array, recv_elems_array);

  for (int c = 0; c < num_iterations; ++c) {
    INFO("Iteration: %d", c);

    // allocating buffers
    // check if we can reuse memory buffer of last iteration
    allocBuffer(send_elems_array+c*array_size, sendbuff_origin, sendbuff,
                recv_elems_array+c*array_size, recvbuff_origin, recvbuff,
                expected, src_host, recv_host, ref_host, deltas_host,
                deltas_device, max_delta_device);

    // initializing device buffers
    if (ignore_data_init == 0) {
      INFO("init data by %s +++", (host_init_and_verify ? "H2D" : "Kernel"));
      initData(send_elems_array+c*array_size, sendbuff, recv_elems_array+c*array_size, recvbuff, expected, src_host,
               ref_host, deltas_host, deltas_device, max_delta_device, s, e);
      INFO("init data by %s ---", (host_init_and_verify ? "H2D" : "Kernel"));
    } else {
      INFO("ignore init-data phase without caring for output results.");
    }

    // running coll test
    INFO("Call coll kernel +++");
    if (ncu_profile) {
      hggcProfilerStart();
    }
    runArrayTest(comms, send_elems_array+c*array_size, recv_elems_array+c*array_size, (void**)(sendbuff), (void**)(recvbuff), s);
    if (ncu_profile) {
      hggcProfilerStop();
    }
    INFO("Call coll kernel ---");

    // verify test results
    if (ignore_data_verify == 0) {
      INFO("Verify result start +++");
      bool iter_res = verifyData(recv_elems_array+c*array_size, expected, recvbuff, recv_host,
          ref_host, deltas_host, max_deltas_host, deltas_device, max_delta_device, s);

      if (!iter_res) {
        res = -1;
        ERROR("Result verify fail, print mismatched data.");
        printData(send_elems_array+c*array_size, sendbuff, expected, recv_elems_array+c*array_size, recvbuff,
            src_host, recv_host, ref_host, deltas_host, max_deltas_host, s);
        if (ignore_error_check_result == 0) {
          bufferFreeArray((void**)sendbuff_origin, (void**)recvbuff_origin, (void**)expected, src_host,
              recv_host, ref_host, deltas_host, deltas_device, max_delta_device);
          break;
        }
      }
      INFO("Verify result end ---");
    } else {
      INFO("Ignore result verify phase without caring about output results.");
    }

    // free buffers.
    bufferFreeArray((void**)sendbuff_origin, (void**)recvbuff_origin, (void**)expected, src_host,
        recv_host, ref_host, deltas_host, deltas_device, max_delta_device);
  }

  return res;
}

void runTest(pcclComm_t* comms, size_t elems_num, void** sendbuff,
    void** recvbuff, hggcStream_t* s) {
  switch (data_type)
  {
  case DTYPE_FLOAT:
    coll_test(comms, elems_num, pcclFloat, sendbuff, recvbuff, s);
    break;
  case DTYPE_HALF:
    coll_test(comms, elems_num, pcclHalf, sendbuff, recvbuff, s);
    break;
  case DTYPE_INT8:
    coll_test(comms, elems_num, pcclInt8, sendbuff, recvbuff, s);
    break;
  case DTYPE_DOUBLE:
    coll_test(comms, elems_num, pcclDouble, sendbuff, recvbuff, s);
    break;
  default:
    break;
  }
}

int main(int argc, char* argv[])
{
  gethostname(hostName, HOST_NAME_LEN);
  int res = 0;

  INFO("Coll (Collective) PPU feature Test.");

  // process command line args
  parseArguments(argc, argv);

  if(core_dump) {
    // enable core dump
    struct rlimit core_limits;
    core_limits.rlim_cur = core_limits.rlim_max = RLIM_INFINITY;
    setrlimit(RLIMIT_CORE, &core_limits);
  }

  //managing devices
  int devs[num_gpus];
  for (int i = 0; i < num_gpus; i++) { devs[i] = i;}

  //enabling device pairs peer access
  if (local_memcpy_stress == 0) {
    for (int i = 0; i < num_gpus; i++) {
      int access = 0;
      for (int j = 0; j < num_gpus; j++) {
        if (j >= i) continue;
        HGGCCHECK(hggcSetDevice(i));
        HGGCCHECK(hggcDeviceCanAccessPeer(&access, i, j));
        if (access) {
          HGGCCHECK(hggcDeviceEnablePeerAccess(j, 0));
          HGGCCHECK(hggcSetDevice(j));
          HGGCCHECK(hggcDeviceEnablePeerAccess(i, 0));
        }
      }
    }
  }

  pcclComm_t* comms;
  if (local_memcpy_stress == 0) {
    //allocating pccl comms and initializing PCCL
    comms = (pcclComm_t*)malloc(num_gpus*sizeof(pcclComm_t));
    PCCLCHECK(pcclCommInitAll(comms, num_gpus, devs));
  }

  // init device streams and events
  hggcStream_t* s = (hggcStream_t*)malloc(sizeof(hggcStream_t)*num_gpus);
  hggcEvent_t* pEvent = (hggcEvent_t *)malloc(sizeof(hggcEvent_t)*num_gpus);
  initDevices(s, pEvent);

  if (array_test_mode == 1) {
    res = runCollArrayTest(comms, s, pEvent);
  } else {
    // buffers and arrays
    void* sendbuff_origin[num_gpus];
    void* recvbuff_origin[num_gpus];
    void* sendbuff[num_gpus];
    void* recvbuff[num_gpus];
    void* expected[num_gpus];
    void* src_host[num_gpus];
    void* recv_host[num_gpus];
    void* ref_host[num_gpus];
    double* deltas_host[num_gpus];
    double max_deltas_host[num_gpus];
    double* max_delta_device[num_gpus];
    double* deltas_device[num_gpus];
    size_t elems_array[num_iterations];
    generateElementNum(elems_array);

    for (int c = 0; c < num_iterations; ++c) {
      INFO("Iteration: %d, Elements number: %lu", c, elems_array[c]);

      // allocating buffers
      // check if we can reuse memory buffer of last iteration
      allocBuffer(elems_array[c], sendbuff_origin, sendbuff, recvbuff_origin,
                  recvbuff, expected, src_host, recv_host, ref_host, deltas_host,
                  deltas_device, max_delta_device);

      // initializing device buffers
      if (ignore_data_init == 0) {
        INFO("init data by %s +++", host_init_and_verify ? "H2D" : "Kernel");
        initData(elems_array[c], sendbuff, recvbuff, expected, src_host, ref_host,
                 deltas_host, deltas_device, max_delta_device, s, pEvent);
        INFO("init data by %s ---", host_init_and_verify ? "H2D" : "Kernel");
      } else {
        INFO("ignore init-data phase without caring for output results.");
      }

      if (local_memcpy_stress) {
        // running single device memcpy test
        INFO("Call local kernel +++");
        runLocalTest(elems_array[c], (void**)(sendbuff), (void**)(recvbuff), s);
        INFO("Call local kernel ---");
      } else {
        // running coll test
        INFO("Call coll kernel +++");
        if (ncu_profile) {
          hggcProfilerStart();
        }

        runTest(comms, elems_array[c], (void**)(sendbuff), (void**)(recvbuff), s);

        if (abnormal_mode != 0) {
          for (int i = 0; i < num_gpus; i++) {
            HGGCCHECK(hggcEventRecord(pEvent[i], s[i]));
          }
          bool timeout = true;
          // check if last pccl kernel finished in 10s, if not throw timeout ERROR.
          for (int i = 0; i < 10; i++ ) {
            hggcError_t err ;
            // only check device 0's event for now.
            err = hggcEventQuery(pEvent[0]);
            if (err == hggcSuccess) {
              timeout = false;
              break;
            } else {
              sleep(1);
            }
          }

          if (timeout) {
            ERROR("Timeout! ");
          }
        }

        if (ncu_profile) {
          hggcProfilerStop();
        }
        INFO("Call coll kernel ---");
      }

      // verify test results
      if (ignore_data_verify == 0) {
        INFO("Verify result start +++");
        bool iter_res = verifyData(elems_array[c], expected, recvbuff, recv_host,
            ref_host, deltas_host, max_deltas_host, deltas_device, max_delta_device, s);

        if (!iter_res) {
          res = -1;
          ERROR("Result verify fail, print mismatched data.");
          printData(elems_array[c], sendbuff, expected, recvbuff, src_host, recv_host,
                  ref_host, deltas_host, max_deltas_host, s);
          if (ignore_error_check_result == 0) {
            bufferFree((void**)sendbuff, (void**)recvbuff, (void**)expected, src_host,
                recv_host, ref_host, deltas_host, deltas_device, max_delta_device);
            break;
          }
        }
        INFO("Verify result end ---");
      } else {
        INFO("Ignore result verify phase without caring about output results.");
      }

      // free buffers.
      // don't free buffers while current buffer size equal to next iteration's buffer size and
      // current iteration is not the last iteration
      if (c < num_iterations - 1 && elems_array[c] == elems_array[c+1]) {
        reuse_buffer = true;
      } else {
        reuse_buffer = false;
      }

      bufferFree((void**)sendbuff_origin, (void**)recvbuff_origin, (void**)expected, src_host,
          recv_host, ref_host, deltas_host, deltas_device, max_delta_device);
    }
  }

  // finalizing PCCL
  if (local_memcpy_stress == 0) {
    for(int i = 0; i < num_gpus; ++i) {
      pcclCommDestroy(comms[i]);
    }
    free(comms);
  }
  destroyDevices(s, pEvent);
  free(s);
  free(pEvent);

  // check result pass or not
  if (ignore_data_verify == 1) {
    INFO("Pass without data comparison.");
  } else if (res == 0) {
    INFO("Data Compare Success.");
  } else {
    ERROR("Data Compare Fail.");
  }

  return res;
}
