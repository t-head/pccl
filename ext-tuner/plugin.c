/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2023-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "tuner.h"
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <map>
#include <pthread.h>
#include <mutex>

#define __hidden __attribute__ ((visibility("hidden")))

struct kernelConfig {
  int size;
  int algo;
  int proto;
  int nChannels;
  int nThreads;
};

int tunerImplStep;
int tunerImplChannels;
std::map<int, std::vector<struct kernelConfig>> tunerImplConfigs;
static pthread_mutex_t initMutex = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static pcclResult_t initResult = pcclSuccess;

void initDefaultConfigs(PPU_TYPE ppuType) {
  if (ppuType == _PPU_ZW810) {
    tunerImplStep = 524288;
    tunerImplChannels = 32;
    tunerImplConfigs.clear();
    std::vector<kernelConfig> configs;
    configs.push_back({524288,  1, 1, 24, 320});
    configs.push_back({1048576, 1, 1, 24, 320});
    configs.push_back({1572864, 1, 1, 24, 320});
    configs.push_back({2097152, 1, 1, 32, 160});
    configs.push_back({2621440, 1, 1, 24, 320});
    configs.push_back({3145728, 1, 1, 24, 320});
    configs.push_back({3670016, 1, 1, 24, 320});
    configs.push_back({4194304, 1, 1, 32, 320});
    configs.push_back({4718592, 1, 1, 32, 320});
    configs.push_back({5242880, 1, 1, 24, 320});
    configs.push_back({5767168, 1, 1, 24, 320});
    configs.push_back({6291456, 1, 1, 24, 320});
    configs.push_back({6815744, 1, 1, 24, 320});
    configs.push_back({7340032, 1, 1, 24, 320});
    configs.push_back({7864320, 1, 1, 32, 320});
    configs.push_back({8388608, 1, 1, 32, 320});
    configs.push_back({8388609, 1, 1, 24, 320});
    tunerImplConfigs[8] = configs;
  } else if (ppuType == _PPU_ZW810E) {
    tunerImplStep = 524288;
    tunerImplChannels = 32;
    tunerImplConfigs.clear();
    std::vector<kernelConfig> configs;
    configs.push_back({524288,  1, 1, 16, 512});
    configs.push_back({1048576, 1, 1, 16, 512});
    configs.push_back({1572864, 1, 1, 16, 320});
    configs.push_back({2097152, 1, 1, 16, 320});
    configs.push_back({2621440, 1, 1, 24, 320});
    configs.push_back({3145728, 1, 1, 24, 320});
    configs.push_back({3670016, 1, 1, 24, 320});
    configs.push_back({4194304, 1, 1, 32, 320});
    configs.push_back({4718592, 1, 1, 32, 320});
    configs.push_back({5242880, 1, 1, 32, 320});
    configs.push_back({5767168, 1, 1, 32, 320});
    configs.push_back({6291456, 1, 1, 32, 320});
    configs.push_back({6815744, 1, 1, 32, 320});
    configs.push_back({7340032, 1, 1, 24, 320});
    configs.push_back({7864320, 1, 1, 24, 320});
    configs.push_back({8388608, 1, 1, 32, 320});
    configs.push_back({8388609, 1, 1, 32, 320});
    tunerImplConfigs[8] = configs;
  }
}

__hidden pcclResult_t pluginInit(PPU_TYPE ppuType, size_t nRanks, size_t nNodes, pcclDebugLogger_t logFunction) {
  pthread_mutex_lock(&initMutex);  // Acquire mutex
  if (!initialized) {
    char *configFile = getenv("PCCL_TUNER_CONFIG_FILE");
    FILE* file = fopen(configFile,"r");
    if (file == NULL) {
      if (ppuType == _PPU_ZW810 || ppuType == _PPU_ZW810E) {
        logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                    "[TUNER]: load ppuType:%d default config.", ppuType);
        initDefaultConfigs(ppuType);
        initResult = pcclSuccess;
      } else {
        logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                    "[TUNER]: Unknown ppuType: %d", ppuType);
        initResult = pcclInvalidUsage;
      }
    } else {
      char line[256];
      int size, algo, proto, nChannels, nThreads;
      int configRanks;
      fgets(line, sizeof(line), file);
      int match = sscanf(line, "step:%d, maxChannels:%d.", &tunerImplStep, &tunerImplChannels);
      if (match != 2 || tunerImplStep <= 0) {
        initResult = pcclInternalError;
      } else {
        while (fgets(line, sizeof(line), file)) {
          if (sscanf(line, "nRanks:%d, size:%d, algo:%d, proto:%d, nChannels:%d, nThreads:%d",
                    &configRanks, &size, &algo, &proto, &nChannels, &nThreads) == 6) {
            if (tunerImplConfigs.find(configRanks) == tunerImplConfigs.end()) {
              tunerImplConfigs[configRanks] = std::vector<struct kernelConfig>();
            }
            tunerImplConfigs[configRanks].push_back({size, algo, proto, nChannels, nThreads});
          } else {
            logFunction(NCCL_LOG_WARN, NCCL_TUNING, __FILE__, __LINE__,
                       "Failed to parse config file '%s'", configFile);
            initResult = pcclInternalError;
          }
        }
      }
      fclose(file);
    }
    #if 0
    // Optional debug prints
    for (auto& configs : tunerImplConfigs) {
        for (const auto& k : configs.second) {
            printf("size %d : algo %d\n", k.size, k.algo);
        }
    }
    #endif
    initialized = true;
  }
  pthread_mutex_unlock(&initMutex);  // Release mutex
  return initResult;
}

__hidden pcclResult_t pluginGetCollInfo(pcclFunc_t collType, int nRanks, size_t nBytes,
                              int collNetSupport, int nvlsSupport, int numPipeOps,
                              int *algorithm, int *protocol, int* nChannels, int* nThreads) {

  *algorithm = NCCL_ALGO_RING; *protocol = NCCL_PROTO_SIMPLE; *nChannels = 0;
  if (collType == pcclFuncAllReduce)  {
    if (tunerImplConfigs.find(nRanks) == tunerImplConfigs.end()) goto out;
    {
      std::vector<kernelConfig>& configs = tunerImplConfigs[nRanks];
      int baseSize = configs[0].size;
      int maxSize = configs.back().size;
      if (nBytes < baseSize || nBytes >= maxSize) {
        goto out;
      }
      {
        int offset = (nBytes - baseSize) / tunerImplStep;
        *algorithm = configs[offset].algo;
        *protocol = configs[offset].proto;
        *nChannels = configs[offset].nChannels;
        *nThreads = configs[offset].nThreads;
      }
    }
  }
 out:
  return pcclSuccess;
}

__hidden pcclResult_t pluginGetChannels(int* channels) {
  *channels = tunerImplChannels;
  return pcclSuccess;
}

__hidden pcclResult_t pluginDestroy() {
  for (std::map<int, std::vector<kernelConfig>>::iterator it = tunerImplConfigs.begin(); it != tunerImplConfigs.end(); ++it) {
    it->second.clear();
  }
  tunerImplConfigs.clear();

  return pcclSuccess;
}

#define PLUGIN_NAME "PCCLTunerPlugin"

#ifdef __cplusplus
extern "C" {
#endif
pcclTuner_v1_t pcclTunerPlugin_v1 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .getChannels = pluginGetChannels,
  .destroy = pluginDestroy
};
#ifdef __cplusplus
}
#endif
