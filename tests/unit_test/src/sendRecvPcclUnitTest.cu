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

const int num_devs = 2;
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
    ("s,start_dev_id", "Starting GPU device ID (default: 0).",
            cxxopts::value<int>()->default_value(std::to_string(start_dev_id)))
    ("e,num_elems", "Number of float elements for send/recv operation.",
            cxxopts::value<int>()->default_value(std::to_string(num_elems)))
    ("n,min_num_elems", "Minimum number of float elements for random size testing.",
            cxxopts::value<int>()->default_value(std::to_string(min_num_elems)))
    ("a,max_num_elems", "Maximum number of float elements for random size testing.",
            cxxopts::value<int>()->default_value(std::to_string(max_num_elems)))
    ("i,num_iterations", "Number of test iterations.",
            cxxopts::value<int>()->default_value(std::to_string(num_iterations)))
    ("m,mode", "Execution mode: 'single' (Group API) or 'thread' (multi-threaded with barrier synchronization). Default: 'single'.",
       cxxopts::value<std::string>()->default_value(mode));
  auto args = options.parse(argc, argv);

  // Print help message.
  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    exit(testSuccess);
  }

  // Parse start_dev_id.
  if (args.count("start_dev_id")) {
    int tmp = args["start_dev_id"].as<int>();
    int num_gpus = 0;
    HGGCCHECK(hggcGetDeviceCount(&num_gpus));
    if (tmp + num_devs - 1 >= num_gpus) {
      ERROR("Exceed the maximum id of GPU, which is %d, but yours is %d", num_gpus - 1, tmp + num_devs - 1);
      exit(testInternalError);
    }
    start_dev_id = tmp;
  }
  
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

  // Parse mode.
  if (args.count("mode")) {
    mode = args["mode"].as<std::string>();
    if (mode != "single" && mode != "thread") {
      ERROR("%s mode is not supported, your input must be 'single' or 'thread'!", mode.c_str());
      exit(testInternalError);
    }
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

int main(int argc, char **argv) {
  gethostname(hostName, HOST_NAME_LEN);
  // parse input arguments
  parseArguments(argc, argv);

  // set devices pair to do send/recv
  int devs[num_devs];
  hggcStream_t *s = (hggcStream_t *)malloc(sizeof(hggcStream_t) * num_devs);
  for (int i = 0; i < num_devs; ++i) {
    devs[i] = i + start_dev_id;
    HGGCCHECK(hggcSetDevice(devs[i]));
    HGGCCHECK(hggcStreamCreate(s + i));
  }

  // initialize pccl comm
  INFO("Init all comms start.");
  pcclComm_t comms[num_devs];
  PCCLCHECK(pcclCommInitAll(comms, num_devs, devs));
  INFO("Init all comms done.");

  // start peer2peer pccl send/recv testing
  int res = 0;
  bool isRandomTest = min_num_elems > 0 && max_num_elems > 0 && min_num_elems < max_num_elems;
  if (isRandomTest) {
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(min_num_elems, max_num_elems); // define the range
    INFO("Start random test!");
    for (int i = 0; i < num_iterations; ++i) {
      num_elems = distr(gen);
      INFO("Elements number: %d", num_elems);
      res = runTest(devs, s, comms, num_devs, num_elems);
      if (res != 0) {
        ERROR("Test fail, result not matched.");
        return res;
      }
    }
  } else {
    INFO("Elements number: %d", num_elems);
    for (int i = 0; i < num_iterations; ++i) {
      INFO("Iteration = %d", i);
      res = runTest(devs, s, comms, num_devs, num_elems);
      if (res != 0) {
        ERROR("Test fail, result not matched.");
        return res;
      }
    }
  }

  // finalize PCCL
  for(int i = 0; i < num_devs; i++) {
    pcclCommDestroy(comms[i]);
  }

  // free resources
  free(s);

  if (res == 0) {
    INFO("Test pass, results matched.");
  } else {
    ERROR("Test fail, result not matched.");
  }
  return res;
}
