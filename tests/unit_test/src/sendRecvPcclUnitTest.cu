//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <hggc_runtime.h>
#include <pccl.h>

#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <vector>
#include <unistd.h>
#include <sys/types.h>

#include <mutex>
#include <condition_variable>
#include <memory>
#include <thread>
#include <iostream>

#include <string.h>
#include <unordered_map>

#include "logger.h"
#include "cxxopts.hpp"

int num_devs = 2;
int start_dev_id = 0;
int num_elems = 32;
int min_num_elems = 0;
int max_num_elems = 0;
int num_iterations = 1;
std::string mode = "single";

class Barrier {
    public:
        explicit Barrier(std::size_t count) { _count = count; }
        void wait() {
            std::unique_lock<std::mutex> lk(_lock);
            if(--_count == 0) {
                _cv.notify_all();
            } else {
                _cv.wait(lk, [this](){ return _count == 0; });
            }
        }
    private:
        std::size_t _count;
        std::mutex _lock;
        std::condition_variable _cv;
};

void parseArguments(int argc, char **argv) {
  cxxopts::Options options("sendRecvPcclUnitTest", "This is PCCL send&recv test.");
  options.add_options()
    ("h,help", "Print usage")
    ("g,num_gpus", "Number of devices for symm_alltoall (at least two).",
            cxxopts::value<int>()->default_value("2"))
    ("s,start_dev_id", "Starting GPU device ID (default: 0).",
            cxxopts::value<int>()->default_value(std::to_string(start_dev_id)))
    ("e,num_elems", "Number of elements per peer (uint32 for symm_alltoall, float otherwise).",
            cxxopts::value<int>()->default_value(std::to_string(num_elems)))
    ("n,min_num_elems", "Minimum number of float elements for random size testing.",
            cxxopts::value<int>()->default_value(std::to_string(min_num_elems)))
    ("a,max_num_elems", "Maximum number of float elements for random size testing.",
            cxxopts::value<int>()->default_value(std::to_string(max_num_elems)))
    ("i,num_iterations", "Number of test iterations.",
            cxxopts::value<int>()->default_value(std::to_string(num_iterations)))
    ("m,mode", "Execution mode: single, thread, or symm_alltoall (grouped all-to-all with symmetric windows).",
       cxxopts::value<std::string>()->default_value(mode));
  auto args = options.parse(argc, argv);

  // Print help message.
  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    exit(testSuccess);
  }

  num_devs = args["num_gpus"].as<int>();
  start_dev_id = args["start_dev_id"].as<int>();
  // Parse num_elems, min_num_elems, max_num_elems to be used in random send&recv test.
  if (args.count("num_elems")) {
    num_elems = args["num_elems"].as<int>();
  }
  if (args.count("min_num_elems")) {
    min_num_elems = args["min_num_elems"].as<int>();
  }
  if (args.count("max_num_elems")) {
    max_num_elems = args["max_num_elems"].as<int>();
  }

  // Parse num_iters.
  if (args.count("num_iterations")) {
    num_iterations = args["num_iterations"].as<int>();
  }

  mode = args["mode"].as<std::string>();
  if (mode == "symm_alltoall") {
    if (num_elems <= 0 || num_iterations <= 0 || min_num_elems || max_num_elems) {
      ERROR("symm_alltoall requires a fixed positive num_elems and positive num_iterations.");
      exit(testInternalError);
    }
    setenv("PCCL_P2P_DMA_ENABLE", "1", 1);
    setenv("PCCL_CTA_POLICY", "2", 1); // PCCL_CTA_POLICY_ZERO
    setenv("PCCL_P2P_DMA_MIN_BYTES", "0", 1);
  } else if ((mode != "single" && mode != "thread") || num_devs != 2) {
    ERROR("Use single or thread with two devices, or symm_alltoall with two or more.");
    exit(testInternalError);
  }

  int available = 0;
  HGGCCHECK(hggcGetDeviceCount(&available));
  if (num_devs < 2 || start_dev_id < 0 || num_devs > available - start_dev_id) {
    ERROR("Invalid device range: start=%d count=%d available=%d", start_dev_id, num_devs, available);
    exit(testInternalError);
  }
}

int runTest(int* devs, hggcStream_t* s, pcclComm_t* comms, int num_devs, int num_elems) {
  INFO("Running mode on send/recv: %s", mode.c_str());

  // rank 0 for sending, and rank 1 for receiving.
  float *sendBuffer;
  float *recvBuffer;
  float golden_data[num_elems];
  for(int i = 0; i < num_devs; i++) {
    HGGCCHECK(hggcSetDevice(devs[i]));
    if(i == 0) {
      HGGCCHECK(hggcMalloc((void **)&sendBuffer, num_elems * sizeof(float)));
      for(int i = 0; i < num_elems; i++) {
        golden_data[i] = 2 * i;
      }
      HGGCCHECK(hggcMemcpy(sendBuffer, golden_data, num_elems * sizeof(float), hggcMemcpyHostToDevice));
    } else {
      HGGCCHECK(hggcMalloc((void **)&recvBuffer, num_elems * sizeof(float)));
    }
  }

  if (mode == "thread") {
    // jobs: did the real pccl communication work
    INFO("running send/recv under multi-threading mode.");
    std::shared_ptr<std::thread> send_thread;
    std::shared_ptr<std::thread> recv_thread;

    std::shared_ptr<Barrier> _barrier (new Barrier(2)); // for send and recv threads sync.
    send_thread.reset(new std::thread([&](int cur_rank) {
                _barrier->wait();
                HGGCCHECK(hggcSetDevice(devs[cur_rank])); // here we assume the cur_rank is the same as hggc device id.
                INFO("Process[%d: cur_rank = %d Sending ...", getpid(), cur_rank);
                PCCLCHECK(pcclSend((const void *)sendBuffer, num_elems, pcclFloat, 1 /* peer rank for receiver */, comms[cur_rank], s[cur_rank]));
                }, 0));

    recv_thread.reset(new std::thread([&](int cur_rank) {
                _barrier->wait();
                HGGCCHECK(hggcSetDevice(devs[cur_rank])); // here we assume the cur_rank is the same as hggc device id.
                INFO("Process[%d: cur_rank = %d Receiving ...", getpid(), cur_rank);
                PCCLCHECK(pcclRecv((void *)recvBuffer, num_elems, pcclFloat, 0 /* peer rank for sender */, comms[cur_rank], s[cur_rank]));
                }, 1));

    INFO("Waiting Send & Recv Threads +++ ");
    send_thread->join();
    recv_thread->join();
    INFO("Waiting Send & Recv Threads --- ");
  } else {
    //calling PCCL communication API. Group API is required when using
    //multiple devices per thread
    INFO("Running send/recv under single thread mode.");
    PCCLCHECK(pcclGroupStart());
    PCCLCHECK(pcclSend((const void*)sendBuffer, num_elems, pcclFloat, 1, comms[0], s[0]));
    PCCLCHECK(pcclRecv((void*)recvBuffer, num_elems, pcclFloat, 0, comms[1], s[1]));
    PCCLCHECK(pcclGroupEnd());
  }

  // synchronize on HGGC streams to wait for completion of PCCL operations
  for(int i = 0; i < num_devs; i++) {
      HGGCCHECK(hggcSetDevice(devs[i]));
      HGGCCHECK(hggcStreamSynchronize(s[i]));
  }

  // check if receive data is ok
  float *check = (float *)malloc(num_elems * sizeof(float));
  HGGCCHECK(hggcSetDevice(devs[1]));
  HGGCCHECK(hggcMemcpy(check, recvBuffer, num_elems * sizeof(float), hggcMemcpyDeviceToHost));

  // free device buffers
  for(int i = 0; i < num_devs; ++i) {
    HGGCCHECK(hggcSetDevice(devs[i]));
    if(i == 0)
      HGGCCHECK(hggcFree(sendBuffer));
    else
      HGGCCHECK(hggcFree(recvBuffer));
  }

  int res = 0;
  std::string info = "Received Data: ";
  for(int i = 0; i < num_elems; i++) {
    info += std::to_string(check[i]) + " ";
    if (golden_data[i] != check[i]) {
      res = -1;
    }
  }
  INFO("%s", info.c_str());
  free(check);

  return res;
}

// Each sender addresses blocks by destination. Each receiver lays them out by source.
static uint32_t payload(int src, int dst, size_t i) {
  return 0x9e3779b9u * (src + 1u) + 0x85ebca6bu * (dst + 1u) + static_cast<uint32_t>(i);
}

static int runSymmAlltoAll(const int* devs, const hggcStream_t* streams, const pcclComm_t* comms) {
  const size_t count = num_elems;
  const size_t rankBytes = num_devs * count * sizeof(uint32_t);
  std::vector<uint32_t*> send(num_devs), recv(num_devs);
  std::vector<pcclWindow_t> sendWin(num_devs), recvWin(num_devs);
  std::vector<uint32_t> host(num_devs * count);
  for (int r = 0; r < num_devs; ++r) {
    HGGCCHECK(hggcSetDevice(devs[r]));
    PCCLCHECK(pcclMemAlloc((void**)&send[r], rankBytes));
    PCCLCHECK(pcclMemAlloc((void**)&recv[r], rankBytes));
    for (int peer = 0; peer < num_devs; ++peer) {
      for (size_t i = 0; i < count; ++i) host[peer * count + i] = payload(r, peer, i);
    }
    HGGCCHECK(hggcMemcpy(send[r], host.data(), rankBytes, hggcMemcpyHostToDevice));
    HGGCCHECK(hggcDeviceSynchronize());
  }

  PCCLCHECK(pcclGroupStart());
  for (int r = 0; r < num_devs; ++r) {
    PCCLCHECK(pcclCommWindowRegister(comms[r], send[r], rankBytes, &sendWin[r], PCCL_WIN_COLL_SYMMETRIC));
    PCCLCHECK(pcclCommWindowRegister(comms[r], recv[r], rankBytes, &recvWin[r], PCCL_WIN_COLL_SYMMETRIC));
  }
  PCCLCHECK(pcclGroupEnd());
  for (int r = 0; r < num_devs; ++r) {
    if (!sendWin[r] || !recvWin[r]) {
      ERROR("Symmetric windows unavailable on rank %d", r);
      exit(testPcclError);
    }
  }

  auto exchange = [&]() {
    PCCLCHECK(pcclGroupStart());
    for (int r = 0; r < num_devs; ++r) {
      for (int peer = 0; peer < num_devs; ++peer) {
        PCCLCHECK(pcclSend(send[r] + peer * count, count, pcclUint32, peer, comms[r], streams[r]));
        PCCLCHECK(pcclRecv(recv[r] + peer * count, count, pcclUint32, peer, comms[r], streams[r]));
      }
    }
    PCCLCHECK(pcclGroupEnd());
  };

  exchange(); // Warm up the same windows, message size and set of peers.
  for (int r = 0; r < num_devs; ++r) {
    HGGCCHECK(hggcSetDevice(devs[r]));
    HGGCCHECK(hggcStreamSynchronize(streams[r]));
  }
  for (int r = 0; r < num_devs; ++r) {
    HGGCCHECK(hggcSetDevice(devs[r]));
    HGGCCHECK(hggcMemsetAsync(recv[r], 0, rankBytes, streams[r]));
    HGGCCHECK(hggcStreamSynchronize(streams[r]));
  }

  const auto start = std::chrono::steady_clock::now();
  for (int iter = 0; iter < num_iterations; ++iter) exchange();
  for (int r = 0; r < num_devs; ++r) {
    HGGCCHECK(hggcSetDevice(devs[r]));
    HGGCCHECK(hggcStreamSynchronize(streams[r]));
  }
  const double us = std::chrono::duration<double, std::micro>(
      std::chrono::steady_clock::now() - start).count() / num_iterations;

  int result = testSuccess;
  for (int r = 0; r < num_devs; ++r) {
    HGGCCHECK(hggcSetDevice(devs[r]));
    HGGCCHECK(hggcMemcpy(host.data(), recv[r], rankBytes, hggcMemcpyDeviceToHost));
    for (size_t offset = 0; offset < host.size(); ++offset) {
      int src = offset / count;
      uint32_t expected = payload(src, r, offset % count);
      if (host[offset] != expected) {
        ERROR("src=%d dst=%d element=%zu expected=%u got=%u", src, r, offset % count, expected, host[offset]);
        result = testInternalError;
        break;
      }
    }
    PCCLCHECK(pcclCommWindowDeregister(comms[r], recvWin[r]));
    PCCLCHECK(pcclCommWindowDeregister(comms[r], sendWin[r]));
    PCCLCHECK(pcclMemFree(recv[r]));
    PCCLCHECK(pcclMemFree(send[r]));
  }
  if (result == testSuccess) {
    INFO("Grouped all-to-all: ranks=%d bytes/peer=%zu iterations=%d us/op=%.3f remote GB/s/rank=%.3f",
         num_devs, count * sizeof(uint32_t), num_iterations, us,
         (num_devs - 1) * count * sizeof(uint32_t) / (us * 1000.0));
  }
  return result;
}

int main(int argc, char **argv) {
  gethostname(hostName, HOST_NAME_LEN);
  // parse input arguments
  parseArguments(argc, argv);

  // Use consecutive devices starting at start_dev_id.
  std::vector<int> devs(num_devs);
  std::vector<hggcStream_t> streams(num_devs);
  for (int i = 0; i < num_devs; ++i) {
    devs[i] = i + start_dev_id;
    HGGCCHECK(hggcSetDevice(devs[i]));
    HGGCCHECK(hggcStreamCreate(&streams[i]));
  }

  // initialize pccl comm
  INFO("Init all comms start.");
  std::vector<pcclComm_t> comms(num_devs);
  PCCLCHECK(pcclCommInitAll(comms.data(), num_devs, devs.data()));
  INFO("Init all comms done.");

  // start peer2peer pccl send/recv testing
  int res = 0;
  bool isRandomTest = min_num_elems > 0 && max_num_elems > 0 && min_num_elems < max_num_elems;
  if (mode == "symm_alltoall") {
    res = runSymmAlltoAll(devs.data(), streams.data(), comms.data());
  } else if (isRandomTest) {
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(min_num_elems, max_num_elems); // define the range
    INFO("Start random test!");
    for (int i = 0; i < num_iterations; ++i) {
      num_elems = distr(gen);
      INFO("Elements number: %d", num_elems);
      res = runTest(devs.data(), streams.data(), comms.data(), num_devs, num_elems);
      if (res != 0) {
        ERROR("Test fail, result not matched.");
        return res;
      }
    }
  } else {
    INFO("Elements number: %d", num_elems);
    for (int i = 0; i < num_iterations; ++i) {
      INFO("Iteration = %d", i);
      res = runTest(devs.data(), streams.data(), comms.data(), num_devs, num_elems);
      if (res != 0) {
        ERROR("Test fail, result not matched.");
        return res;
      }
    }
  }

  // finalize PCCL
  for(int i = 0; i < num_devs; i++) {
    HGGCCHECK(hggcSetDevice(devs[i]));
    HGGCCHECK(hggcStreamDestroy(streams[i]));
    PCCLCHECK(pcclCommDestroy(comms[i]));
  }

  if (res == 0) {
    INFO("Test pass, results matched.");
  } else {
    ERROR("Test fail, result not matched.");
  }
  return res;
}
