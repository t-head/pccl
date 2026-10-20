/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "pccl_internal.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "coll_net.h"
#include "enqueue.h"
#include "ce_coll.h"
#include "p2p_dma.h"
#include "graph.h"
#include "argcheck.h"
#include "tuner.h"
#include "ras.h"
#include "mnnvl.h"
#include "u2mm.h"
#include "utils.h"
#include "gin.h"
#include "mem_manager.h"
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include "asyswrap.h"
#include <signal.h>
#include <atomic>
#include "check_items.h"
#include "hgtx.h"
#include "hgtx_payload_schemas.h"
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
#include "ce_coll_oki.h"
#endif

#define STR2(v) #v
#define STR(v) STR2(v)

#ifdef ENABLE_TRACE
std::chrono::high_resolution_clock::time_point ncclEpoch;
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS+1] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "SendRecv"};
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNet", "FC", "PAT"};
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple", "Signal", "SimpleBulk", "SignalBulk", "SimpleBulkAR" };
const bool ncclSendRecvProtoSupport[NCCL_NUM_PROTOCOLS] = {false, false, true, true, true, true, true};

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
PCCL_PARAM(DisableAbort, "DISABLE_ABORT", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", PCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(RuntimeConnect, "RUNTIME_CONNECT", 1);
NCCL_PARAM(CtaPolicy, "CTA_POLICY", PCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(WinEnable, "WIN_ENABLE", 1);
NCCL_PARAM(SetCpuStackSize, "SET_CPU_STACK_SIZE", 1);
NCCL_PARAM(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);

// GDRCOPY support: Off by default
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);
NCCL_PARAM(P2pUseCudaMemcpy, "P2P_USE_CUDA_MEMCPY", 0);

extern int ppuDiesNum;

struct VersionMap {
  unsigned int ncclVer;
  unsigned int pcclVer;
};

static const VersionMap NCCL_TO_PCCL_VERSION_LIST[] = {
  //       NCCL Version   | Corresponding PCCL Version
  {NCCL_VERSION(2, 8, 3),  PCCL_VERSION(0, 0, 1)},
  {NCCL_VERSION(2, 17, 1), PCCL_VERSION(1, 4, 0)},
  {NCCL_VERSION(2, 27, 3), PCCL_VERSION(2, 0, 0)}
};

pcclResult_t pcclConfigCheck(unsigned int internalVersion, unsigned int magic, unsigned int* pcclVersion) {
  // Use magic numbers to differentiate API source (PCCL vs. NCCL) for version compatibility
  if (magic == PCCL_API_MAGIC) {
    *pcclVersion = internalVersion;
    return ncclSuccess;
  } else if (magic == NCCL_API_MAGIC) {
    int numEntries = sizeof(NCCL_TO_PCCL_VERSION_LIST) / sizeof(VersionMap);
    for (int i = numEntries - 1; i >= 0; i--) {
      if (internalVersion >= NCCL_TO_PCCL_VERSION_LIST[i].ncclVer) {
        *pcclVersion = NCCL_TO_PCCL_VERSION_LIST[i].pcclVer;
        VERBOSE(NCCL_INIT, "external lib version %d map to PCCL version %d", internalVersion, *pcclVersion);
        return ncclSuccess;
      }
    }
    *pcclVersion = PCCL_VERSION_CODE;
    WARN("Unsupported external lib version %u, using default PCCL version %u. Please check if PCCL upgrade is required.",
          internalVersion, *pcclVersion);
    // Log as error for visibility, but return success to permit permissive cross-version usage
    return ncclSuccess;
  } else {
    ERROR("pcclConfig_t argument not initialized via PCCL_CONFIG_INITIALIZER (invalid magic number: 0x%x)", magic);
    return ncclInvalidArgument;
  }
}

PCCL_PARAM(P2pThroughSwitchOptDisable, "P2P_THROUGH_SWITCH_OPT_DISABLE", 0);
bool pcclEnableGlobalIcnP2pThroughSwitchOpt(ncclComm_t comm) {
  return (pcclParamP2pThroughSwitchOptDisable() == 0 &&
          comm->minCompCap >= 89 &&
          comm->icnRmtType == PCCL_ICNLINK_SWITCH &&
          comm->globalIcnP2pSupport);
}

// GDRCOPY support
gdr_t ncclGdrCopy = NULL;

ncclResult_t initGdrCopy() {
  if (ncclParamGdrCopyEnable() == 1) {
    ncclGdrCopy = ncclGdrInit();
  }
  return ncclSuccess;
}

// The default Linux stack size (8MB) is safe.
#define SAFE_STACK_SIZE (8192*1024)

// pthread_setattr_default_np need glibc version >= 2.18
extern "C" int pthread_setattr_default_np(const pthread_attr_t *attr) __attribute__((weak));

static ncclResult_t setCpuStackSize() {
  if (ncclParamSetCpuStackSize() != 0) {
    // Query the stack size used for newly launched threads.
    pthread_attr_t attr;
    size_t stackSize;
    PTHREADCHECK(pthread_attr_init(&attr), "pthread_attr_init");
    PTHREADCHECK(pthread_attr_getstacksize(&attr, &stackSize), "pthread_attr_getstacksize");

    if (stackSize < SAFE_STACK_SIZE) {
      if (pthread_setattr_default_np != nullptr) {
        // GNU libc normally uses RLIMIT_STACK as the default pthread stack size, unless it's set to "unlimited" --
        // in that case a fallback value of 2MB (!) is used.

        // Query the actual resource limit so that we can distinguish between the settings of 2MB and unlimited.
        struct rlimit stackLimit;
        char buf[30];
        SYSCHECK(getrlimit(RLIMIT_STACK, &stackLimit), "getrlimit");
        if (stackLimit.rlim_cur == RLIM_INFINITY)
          strcpy(buf, "unlimited");
        else
          snprintf(buf, sizeof(buf), "%ldKB", stackLimit.rlim_cur/1024);
        INFO(NCCL_INIT|NCCL_ENV, "Stack size limit (%s) is unsafe; will use %dKB for newly launched threads",
            buf, SAFE_STACK_SIZE/1024);

        // Change the default pthread stack size (via a nonportable API, which will become necessary if we switch
        // to C++ threads).
        PTHREADCHECK(pthread_attr_setstacksize(&attr, SAFE_STACK_SIZE), "pthread_attr_setstacksize");
        PTHREADCHECK(pthread_setattr_default_np(&attr), "pthread_setattr_default_np");
      } else {
        INFO(NCCL_INIT|NCCL_ENV, "glibc < 2.18 detected; pthread_setattr_default_np not available. New threads will use system default stack size.");
      }
    }

    PTHREADCHECK(pthread_attr_destroy(&attr), "pthread_attr_destroy");
  }

  return ncclSuccess;
}

#define MAX_COMMS_PER_PROCESS 64
static ncclComm_t commArray[MAX_COMMS_PER_PROCESS] = {NULL};
// commArrayLock protects register/unregister; the dump signal handler reads without the lock
// (signal-handler context cannot safely acquire a mutex), relying on naturally-atomic
// pointer-sized loads on supported architectures.
static pthread_mutex_t commArrayLock = PTHREAD_MUTEX_INITIALIZER;

void pcclDumpCommList(int signal) {
  INFO(NCCL_DUMP, "signal %d triggered, start dump pccl comm list!", signal);
  pcclCommDumpMaskType dumpMask;
  NCCLVERIFY(getGlobalCommDumpMask(&dumpMask));
  ncclResult_t res;
  for (int i = 0; i < MAX_COMMS_PER_PROCESS; i++) {
    ncclComm_t comm = commArray[i];
    if (comm == NULL) continue;
    res = pcclCommDump(comm, dumpMask);
    if (res != ncclSuccess) {
      WARN("Fail to dump comm %p!", comm);
    }
  }
}

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
static ncclResult_t pcclOkiDestroyResources(ncclComm_t comm) {
  if (!comm->okiSupport) return ncclSuccess;
  for (int i = 0; i < comm->okiQpCount; i++) {
    LT_OKI_CHECK(ltOkiDestroyQP(comm->okiDevice, comm->okiQps[i]));
    comm->okiQps[i] = nullptr;
  }
  HGGCCHECK(hggcFreeHost(comm->okiQps));
  NCCLCHECK(ncclHggcFree(comm->okiDependencyAddr, MEM_NORMAL, comm->memManager));
  LT_OKI_CHECK(ltOkiCloseDevice(comm->okiDevice));
  comm->okiQps = NULL;
  comm->okiDevice = NULL;
  return ncclSuccess;
}
#endif

ncclResult_t initExtKernelPlugin(ncclComm_t comm) {
  NCCLCHECK(pcclExtKernelPluginLoad(comm));
  if (comm->extKernel) {
    ssize_t extKernelDevWorkspaceSize = 0;

    // Fill topology array and init info for plugin
    int* topoArray = NULL;
    NCCLCHECK(ncclCalloc(&topoArray, comm->localRanks * comm->localRanks));
    NCCLCHECK(pcclFillPPUTopoArray(topoArray, comm));
    struct pcclExtKernelInitInfo initInfo;
    initInfo.rank = comm->rank;
    initInfo.nRanks = comm->nRanks;
    initInfo.localRanks = comm->localRanks;
    initInfo.compCap = comm->compCap;
    initInfo.ppuType = comm->ppuType;
    initInfo.icnRmtType = comm->icnRmtType;
    initInfo.defaultStrategy = comm->defaultStrategy;
    initInfo.topoArray = topoArray;

    ncclResult_t initRes = comm->extKernel->kernelPluginInit(&initInfo, &comm->extKernelPluginWorkspace.extKernelPluginParams, &extKernelDevWorkspaceSize, ncclDebugLogForPlugin);
    NCCLCHECK(ncclFree(topoArray));
    NCCLCHECK(initRes);
    comm->extKernelPluginWorkspace.size = extKernelDevWorkspaceSize;
    if (extKernelDevWorkspaceSize > 0) {  // for now extKernel must have non-zero workspace
      int refCount = 0;
      struct ncclPeerInfo* myInfo = comm->peerInfo + comm->rank;
      for (int peer = 0; peer < comm->nRanks; peer++) {
        struct ncclPeerInfo* peerInfo = comm->peerInfo + peer;
        // do not allow 2 rank assign to one hggcDev
        if (PCCL_SAME_PID(myInfo, peerInfo)) {
          if (peerInfo->hggcDev != myInfo->hggcDev) {
            refCount++;
          }
        }
      }
      NCCLCHECK(ncclP2pAllocateShareableBuffer(extKernelDevWorkspaceSize, refCount, &(comm->extKernelPluginWorkspace.ipcDesc), &comm->extKernelPluginWorkspace.devPtr, &(comm->devMemUsed), MEM_NORMAL, -1, comm->memManager, ncclMemPersist));
    } else {
      NCCLCHECK(pcclExtKernelPluginUnload(comm));
    }
  }
  return ncclSuccess;
}

ncclResult_t destroyExtKernelPlugin(ncclComm_t comm) {
  for (int i=0; i<comm->extKernelPluginWorkspace.peerIpcPtrNum; i++) {
    if (ncclCuMemEnable()) {
      // hgMem API support
      if (comm->extKernelPluginWorkspace.sameProc[i]) {
        NCCLCHECK(ncclHgMemFreeAddr(comm->extKernelPluginWorkspace.peerIpcPtrs[i]));
      } else {
        NCCLCHECK(ncclHggcFree(comm->extKernelPluginWorkspace.peerIpcPtrs[i], MEM_NORMAL, comm->memManager));
      }
    } else {
      HGGCCHECK(hggcIpcCloseMemHandle(comm->extKernelPluginWorkspace.peerIpcPtrs[i]));
    }
    comm->extKernelPluginWorkspace.peerIpcPtrs[i] = NULL;
  }
  comm->extKernelPluginWorkspace.peerIpcPtrNum = 0;
  NCCLCHECK(ncclHggcFree(comm->extKernelPluginWorkspace.devPtr, MEM_NORMAL, comm->memManager));
  comm->extKernel->kernelPluginDestroy(comm->extKernelPluginWorkspace.extKernelPluginParams);
  NCCLCHECK(pcclExtKernelPluginUnload(comm));

  return ncclSuccess;
}

ncclResult_t buildConnectionForExtKernelPlugin(ncclComm_t comm) {
  ncclResult_t res;
  struct extKernelPeerMemInfo{
    void* directPtr;
    ncclIpcDesc ipcDesc;
    size_t size;
  };
  int rank = comm->rank;
  void* buffers[NCCL_MAX_LOCAL_RANKS];
  struct ncclPeerInfo* myInfo = comm->peerInfo+comm->rank;

  struct extKernelPeerMemInfo* peerMemInfo = NULL;
  NCCLCHECKGOTO(ncclCalloc(&peerMemInfo, comm->nRanks), res, fail);

  peerMemInfo[rank].directPtr = comm->extKernelPluginWorkspace.devPtr;
  peerMemInfo[rank].ipcDesc = comm->extKernelPluginWorkspace.ipcDesc;
  peerMemInfo[rank].size = comm->extKernelPluginWorkspace.size;

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, peerMemInfo, sizeof(struct extKernelPeerMemInfo)), res, fail);

  for (int peer=0; peer<comm->nRanks; peer++) {
    struct ncclPeerInfo* peerInfo = comm->peerInfo+peer;
    // do not allow 2 rank assign to one hggcDev
    if (PCCL_SAME_PID(myInfo, peerInfo)) {
      if (peerInfo->hggcDev == myInfo->hggcDev) {
        buffers[peer] = comm->extKernelPluginWorkspace.devPtr;
      } else {
        hggcError_t err = hggcDeviceEnablePeerAccess(peerInfo->hggcDev, 0);
        if (err == hggcErrorPeerAccessAlreadyEnabled) {
          err = hggcGetLastError();  // should do nothing
        } else if (err != hggcSuccess) {
          WARN("failed to enable peer access between rank %d-%d with error %s. Ignore Ext Kernel.", rank, peer,hggcGetErrorString(err));
          goto fail;
        }
        // hgMem API support
        if (ncclCuMemEnable()) {
          // for intra-process ranks, we should map memHandle of the peers to increase refcount.
          // Otherwise, if peers abort and free the buffer, the rank can suffer invalid access.
          NCCLCHECKGOTO(ncclHgMemAllocAddr(&buffers[peer], &peerMemInfo[peer].ipcDesc.memHandle, peerMemInfo[peer].size, NULL), res, fail);
          HGCHECKGOTO(hgMemRelease(peerMemInfo[peer].ipcDesc.memHandle), res, fail);
          comm->extKernelPluginWorkspace.sameProc[comm->extKernelPluginWorkspace.peerIpcPtrNum] = true;
          comm->extKernelPluginWorkspace.peerIpcPtrs[comm->extKernelPluginWorkspace.peerIpcPtrNum++] = buffers[peer];
        } else {
          buffers[peer] = peerMemInfo[peer].directPtr;
        }
      }
    } else {
      void *ptr;
      NCCLCHECKGOTO(ncclP2pImportShareableBuffer(comm, peer, peerMemInfo[peer].size, &peerMemInfo[peer].ipcDesc, &ptr, MEM_NORMAL, peerMemInfo[peer].directPtr, ncclMemPersist), res, fail);
      buffers[peer] = ptr;
      comm->extKernelPluginWorkspace.sameProc[comm->extKernelPluginWorkspace.peerIpcPtrNum] = false;
      comm->extKernelPluginWorkspace.peerIpcPtrs[comm->extKernelPluginWorkspace.peerIpcPtrNum++] = ptr;
    }
  }

  // set it to kernelPlugin Params
  NCCLCHECKGOTO(comm->extKernel->kernelPluginConfigWorkspace(buffers, comm->nRanks, comm->extKernelPluginWorkspace.extKernelPluginParams), res, fail);

 exit:
  NCCLCHECK(ncclFree(peerMemInfo));
  return res;

 fail:
  NCCLCHECKGOTO(destroyExtKernelPlugin(comm), res, exit);
  goto exit;
}

NCCL_PARAM(CollNetEnable, "COLLNET_ENABLE", 0);
PCCL_PARAM(MNNVLFakeAdvancedFeature, "MNNVL_FAKE_ADVANCED_FEATURE", 0);
PCCL_PARAM(NoMemset, "NO_MEMSET", 0);
PCCL_PARAM(SingleHostCrossNodeTest, "SINGLE_HOST_CROSS_NODE_TEST", 0);
PCCL_PARAM(PerfmodelSpeedup, "PERFMODEL_SPEEDUP", 0);
PCCL_PARAM(ChannelShuffle, "CHANNEL_SHUFFLE", -2);
PCCL_PARAM(StateMonitorDisable, "STATE_MONITOR_DISABLE", 0);
PCCL_PARAM(StateMonitorLevel, "STATE_MONITOR_LEVEL", 1);
PCCL_PARAM(StateMonitorTimeoutMs, "STATE_MONITOR_TIMEOUT_MS", 300*1e3 /* 5mins */);

int gModelType = HW_PPU;
int ncclPerfmodelSpeedup = 0;
const size_t ncclPerfmodelTotalDeviceMemSize = 1*1024*1024*1024;
const size_t ncclPerfmodelDevMemAlignSize = 2*1024*1024;
void* ncclPerfmodelHostPtr[8];
void* ncclPerfmodelDevicePtr[8];
size_t ncclPerfmodelMemOffset[8];
int ncclPerfmodelNeedCopy[8];

int ncclParamNoMemsetFlag;
int ncclParamSingleHostCrossNodeTestFlag;
int ncclParamStateMonitorLevelFlag;
int ncclParamMnnvlFakeAdvancedFeatureFlag;

int64_t ncclParamStateMonitorTimeoutMs;

enum ncclLaunchMode ncclParamLaunchMode;
int useMemcpy;

static void initGlobalVars() {
  ncclParamNoMemsetFlag = pcclParamNoMemset();
  ncclParamSingleHostCrossNodeTestFlag = pcclParamSingleHostCrossNodeTest();
  ncclParamStateMonitorLevelFlag = pcclParamStateMonitorDisable() ? 0 : pcclParamStateMonitorLevel();
  ncclParamMnnvlFakeAdvancedFeatureFlag = pcclParamMNNVLFakeAdvancedFeature();
  ncclParamStateMonitorTimeoutMs = pcclParamStateMonitorTimeoutMs();

  // check hw model type env setting
  const char* str = ncclGetEnv("ALIPPU_HW_TYPE");
  if (str) {
    if (strcmp(str, "MODEL_BEHAVIOR") == 0) {
      gModelType = HW_BEHAVIOR;
    } else if (strcmp(str, "MODEL_BIT_ACCURATE") == 0) {
      gModelType = HW_BIT_ACCURATE;
    } else if (strcmp(str, "MODEL_PERFORMANCE") == 0) {
      gModelType = HW_PERF;
    } else {
      WARN("invalid ALIPPU_HW_TYPE env (%s) set, set to ppu default", str);
      gModelType = HW_PPU;
    }
    INFO(NCCL_ENV, "ALIPPU_HW_TYPE set by environment to %s", str);
  }

  if (pcclParamPerfmodelSpeedup() == 1) {
    ncclPerfmodelSpeedup = 1;
    INFO(NCCL_ENV, "PCCL PERFMODEL SPEEDUP %d", ncclPerfmodelSpeedup);
  }
}

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static ncclResult_t ncclInit() {
  if (initialized) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    setCpuStackSize();
    initEnv();
    initGdrCopy();
    initGlobalVars();
    NCCLCHECK(bootstrapNetInit());
    initialized = true;
    // hgtx not support Enum in 2.0
    // initHgtxRegisteredEnums();
  }

  if (wrapAsysSymbols() == ncclSuccess) {
    wrapPcclPcclInjection();
  }

  pthread_mutex_unlock(&initLock);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclGetVersion, int* version);
ncclResult_t pcclGetVersion(int* version) {
  INFO_CALL("PCCL API: pcclGetVersion(version=%p)", version);
  if (version == NULL) return ncclInvalidArgument;
  *version = PCCL_VERSION_CODE;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclGetUniqueId, ncclUniqueId* out);
ncclResult_t pcclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  struct ncclBootstrapHandle handle;
  NCCLCHECK(bootstrapGetUniqueId(&handle, NULL));
  memset(out, 0, sizeof(ncclUniqueId));
  memcpy(out, &handle, sizeof(handle));

  INFO_CALL("PCCL API: pcclGetUniqueId(out=0x%llx)", (unsigned long long)getHash(out->internal, NCCL_UNIQUE_ID_BYTES));
  return ncclSuccess;
}

// Prevent compiler from optimizing out these operations
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  comm->rank = comm->hggcDev = comm->busId = comm->nRanks = -1;
}

#undef NCCL_NO_OPTIMIZE

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  NCCLCHECK(ncclCeFinalize(comm));

  if (comm->symmetricSupport) {
    NCCLCHECK(pcclP2pDmaFinalize(comm));
    NCCLCHECK(ncclSymkFinalize(comm));
    NCCLCHECK(ncclDevrFinalize(comm));
  }
  NCCLCHECK(ncclRasCommFini(comm));

  /* in commReclaim, we have guaranteed only last rank which calls ncclCommDestroy() will
   * free all intra-process communicators; therefore, we only need to focus on local
   * resource cleanup in commFree(). */
  if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread) {
    PTHREADCHECK(pthread_join(comm->proxyState->thread, nullptr), "pthread_join");
    if (comm->proxyState->threadUDS) {
      // UDS support
      PTHREADCHECK(pthread_join(comm->proxyState->threadUDS, nullptr), "pthread_join");
    }
  }

  if (comm->memPool) HGGCCHECK(hggcMemPoolDestroy(comm->memPool));

  NCCLCHECK(ncclFree(comm->connectSend));
  NCCLCHECK(ncclFree(comm->connectRecv));

  NCCLCHECK(ncclFree(comm->peerInfo));
  if (comm->topo)
    ncclTopoFree(comm->topo);
  if (comm->nodeRanks) {
    for (int n=0; n<comm->nNodes; n++) NCCLCHECK(ncclFree(comm->nodeRanks[n].localRankToRank));
    NCCLCHECK(ncclFree(comm->nodeRanks));
  }
  NCCLCHECK(ncclFree(comm->rankToNode));
  NCCLCHECK(ncclFree(comm->rankToLocalRank));

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  NCCLCHECK(ncclHggcFreeLegacy(comm->hostDevComm.comm.channels, MEM_NORMAL, comm->memManager));
  NCCLCHECK(ncclHggcFreeLegacy(comm->hostDevComm.comm.rankToLocalRank, MEM_NORMAL, comm->memManager));
  NCCLCHECK(ncclHggcFreeLegacy(comm->devComm, MEM_NORMAL, comm->memManager));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks, comm));

  if (comm->sharedRes && comm->ginPluginIndex >= 0) NCCLCHECK(ncclGinHostFinalize(comm));

  // Resource creation and GIN plugin assignment can fail independently.
  bool hadSharedRes = comm->sharedRes != nullptr;
  int sharedResRefCount = 0;
  if (comm->sharedRes) {
    sharedResRefCount = ncclAtomicRefCountDecrement(&comm->sharedRes->refCount);
    if (sharedResRefCount == 0) {
      for (int c=0; c<MAXCHANNELS; c++) {
        if (comm->sharedRes->peers[c]) NCCLCHECK(ncclFree(comm->sharedRes->peers[c]));
        if (comm->sharedRes->devPeers[c]) NCCLCHECK(ncclHggcFree(comm->sharedRes->devPeers[c], MEM_NORMAL, comm->memManager));
      }
      NCCLCHECK(ncclFree(comm->sharedRes->tpRankToLocalRank));
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->hostStream));
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->deviceStream));
      HGGCCHECK(hggcEventDestroy(comm->sharedRes->scratchEvent));
      NCCLCHECK(ncclProxyDestroy(comm));
      NCCLCHECK(ncclFree(comm->sharedRes));
    }
  }

  if (comm->workFifoBufGdrHandle) {
    // GDRCOPY support
    NCCLCHECK(ncclGdrCudaFree(comm->workFifoBufGdrHandle, comm->memManager));
  } else {
    if(comm->workFifoBufDev != comm->workFifoBuf) {
      ERROR("comm->workFifoBufDev(%p) is not equal to comm->workFifoBuf(%p)", comm->workFifoBufDev, comm->workFifoBuf);
      return ncclInternalError;
    }
    NCCLCHECK(ncclHggcHostFree(comm->workFifoBuf));
  }

  ncclMemoryStackDestruct(&comm->memScoped);
  ncclMemoryStackDestruct(&comm->memPermanent);

  if (ncclAtomicRefCountDecrement(comm->abortFlagRefCount) == 0) {
    NCCLCHECK(ncclFree(comm->abortFlag));
    NCCLCHECK(ncclHggcHostFree(comm->abortFlagDev));
    NCCLCHECK(ncclFree(comm->abortFlagRefCount));
  }

  NCCLCHECK(ncclFree(comm->topParentRanks));
  NCCLCHECK(ncclFree(comm->topParentLocalRanks));
  NCCLCHECK(ncclFree(comm->gproxyConn));

  NCCLCHECK(ncclRegCleanup(comm));

  // Poison comm to try and catch a double free
  int dev = 0;
  if (ncclPerfmodelSpeedup == 1) {
    dev = comm->hggcDev;
  }

  NCCLCHECK(ncclFree(comm->hgtxMessage));

  NCCLCHECK(ncclFree(comm->kernelInfoMonitorPtr));

  if (comm->ceStreams != nullptr) {
    for (int i = 0; i < comm->ceStreamsNum; ++i) {
      HGGCCHECK(hggcStreamDestroy(comm->ceStreams[i]));
      HGGCCHECK(hggcEventDestroy(comm->ceEvents[i]));
    }
    HGGCCHECK(hggcEventDestroy(comm->ceSyncEvent));

    NCCLCHECK(ncclFree(comm->ceStreams));
    NCCLCHECK(ncclFree(comm->ceEvents));
    NCCLCHECK(ncclFree(comm->ceCollConfigs.agOneStageStrategy));
    NCCLCHECK(ncclFree(comm->ceCollConfigs.agTwoStageStrategy));
  }

  // Remove from process comm dump list before freeing to prevent use-after-free in pcclDumpCommList
  {
    pthread_mutex_lock(&commArrayLock);
    for (int i = 0; i < MAX_COMMS_PER_PROCESS; i++) {
      if (commArray[i] == comm) {
        commArray[i] = NULL;
        break;
      }
    }
    pthread_mutex_unlock(&commArrayLock);
  }

  commPoison(comm);
  NCCLCHECK(ncclProfilerPluginFinalize(comm));
  NCCLCHECK(ncclHggcHostFree(comm->profiler.workStarted));
  NCCLCHECK(ncclHggcHostFree(comm->profiler.workCompleted));

  NCCLCHECK(ncclFree(comm->funcStrategy));

  // Release the per-comm net plugin reference taken in commAlloc; skip comms
  // that failed before the reference was taken to keep load/unload paired.
  if (comm->netPluginRetained) NCCLCHECK(ncclNetFinalize(comm));
  free((void*)comm->config.netName);
  comm->config.netName = nullptr;
  if (hadSharedRes && sharedResRefCount == 0 && comm->ginPluginIndex >= 0) {
    NCCLCHECK(ncclGinFinalize(comm));
  }

  // Free any pending suspend/resume tasks
  while (!ncclIntruQueueEmpty(&comm->suspendTaskQueue)) {
    struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->suspendTaskQueue);
    ncclFree(task);
  }
  while (!ncclIntruQueueEmpty(&comm->resumeTaskQueue)) {
    struct ncclMemManagerTask* task = ncclIntruQueueDequeue(&comm->resumeTaskQueue);
    ncclFree(task);
  }

  NCCLCHECK(ncclMemManagerDestroy(comm));

  NCCLCHECK(ncclFree(comm));

  if (ncclPerfmodelSpeedup) {
    NCCLCHECK(ncclFree(ncclPerfmodelHostPtr[dev]));
    INFO(NCCL_INIT, "rank:%d, free hggc ptr:%p", dev, ncclPerfmodelDevicePtr[dev]);
    HGGCCHECK(hggcFree(ncclPerfmodelDevicePtr[dev]));
  }

  return ncclSuccess;
}

// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
#define NCCL_WORK_FIFO_BYTES_DEFAULT (1<<25)
NCCL_PARAM(WorkFifoBytes, "WORK_FIFO_BYTES", NCCL_WORK_FIFO_BYTES_DEFAULT);
NCCL_PARAM(WorkArgsBytes, "WORK_ARGS_BYTES", INT64_MAX);

ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  /* comm must be ready, or error will be reported */
  ncclResult_t ret = ncclSuccess;

  if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
    ncclGroupJobAbort(comm->groupJob);
  } else {
    NCCLCHECK(pcclCommGetAsyncError(comm, &ret));
    if (ret == ncclInProgress) {
      ERROR("Attempt to use communicator before the previous operation returned pcclSuccess");
      ret = ncclInvalidArgument;
      goto exit;
    }
    /* if ret is not ncclInProgress, we just keep it. */
  }

exit:
  return ret;
}

static ncclResult_t commAlloc(ncclComm_t comm, ncclComm_t parentComm, int ndev, int rank) {
  if (ndev < 1) {
    ERROR("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    ERROR("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  HGGCCHECK(hggcGetDevice(&comm->hggcDev));

  ncclMemoryStackConstruct(&comm->memPermanent);
  ncclMemoryStackConstruct(&comm->memScoped);

  comm->rank = comm->hostDevComm.comm.rank = rank;
  comm->nRanks = comm->hostDevComm.comm.nRanks = ndev;
  // Skip GIN cleanup until a plugin has been assigned to this communicator.
  comm->ginPluginIndex = -1;

  // Take a per-comm reference on the net plugin (reloads it if it was fully
  // unloaded by the last comm destroy); released in commFree via ncclNetFinalize.
  NCCLCHECK(ncclNetInit(comm));
  INFO(NCCL_INIT, "Using network %s", comm->ncclNet->name);
  if (parentComm && parentComm->shareResources && parentComm->ncclNet != comm->ncclNet) {
    WARN("Split shares resources, but parent comm netName %s is different from child comm netName %s",
         parentComm->ncclNet->name, comm->ncclNet->name);
    return ncclInvalidUsage;
  }

  // Safe to read comm->ncclNet->name here: the reference taken above pins the plugin.
  // nccl_u2mm_init() is self-gated and idempotent across comms.
  if (strcmp(comm->ncclNet->name, EIC_PLUGIN_NAME) == 0) {
    nccl_use_eic = true;
    NCCLCHECK(nccl_u2mm_init());
  }

  comm->devMemUsed = 0;
  comm->pinnedMemUsed = 0;
  comm->hostDevComm.comm.stateMonitorLevel = ncclParamStateMonitorLevelFlag;
  // Initialize memory manager
  if (parentComm && parentComm->shareResources && parentComm->memManager) {
    // Share parent's memory manager
    comm->memManager = parentComm->memManager;
    ncclAtomicRefCountIncrement(&comm->memManager->refCount);
    INFO(NCCL_INIT, "MemManager: Shared from parent, refCount=%d", comm->memManager->refCount);
  } else {
    // Create new memory manager
    NCCLCHECK(ncclMemManagerInit(comm));
  }
  NCCLCHECK(getBusId(comm->hggcDev, &comm->busId));
  bool obtainHgmlInfo = false;
  if (pcclHgmlEnsureInitialized() == ncclSuccess) {
    hgmlDevice_t hgmlDev = NULL;
    char busId[HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    NCCLCHECK(int64ToBusId(comm->busId, busId));
    if (pcclHgmlDeviceGetHandleByPciBusId(busId, &hgmlDev) == ncclSuccess) {
      if ((pcclHgmlDeviceGetIndex(hgmlDev, (unsigned int*)&comm->hgmlDev) == ncclSuccess) &&
          (pcclHgmlDeviceGetCount(&comm->hgmlDeviceCount) == ncclSuccess)) {
        obtainHgmlInfo = true;
      }
    }
  }
  if (!obtainHgmlInfo) {
    if (strcmp(comm->ncclNet->name, EIC_PLUGIN_NAME) == 0) {
      ERROR("Query hgml info failed on EIC platform !");
      return pcclUnhandledHggcError;
    }
    comm->hgmlDev = -1;
    comm->hgmlDeviceCount = 0;
  }

  comm->compCap = ncclHggcCompCap();
  comm->ppuType = _PPU_UNKNOWN;
  comm->icnRmtType = PCCL_ICNLINK_UNKNOWN;
  struct ppuDeviceInfo* ppuInfo = pcclGetPpuDevInfo();
  comm->ppuType = ppuInfo->ppuType;
  comm->icnRmtType = ppuInfo->icnRmtType;
  VERBOSE(NCCL_INIT,"comm %p rank %d nranks %d hggcDev %d busId %x compCap %d ppuType %s icnRmtType %s", comm, rank, ndev, comm->hggcDev, comm->busId, comm->compCap, ppuTypeStr[comm->ppuType], icnRmtTypeStr[comm->icnRmtType]);

  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;

  comm->hostDevComm.comm.abortFlag = comm->abortFlagDev;

  *comm->abortFlag = 0;
  *comm->abortFlagDev = 0;

  comm->maxConnType = CONN_NORMAL;

  comm->collNetSupport = 0;

  comm->localPersistentRefs = 0;

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));

  // Mark channels as non initialized.
  for (int c=0; c<MAXCHANNELS; c++) comm->channels[c].id = -1;
  ncclIntruQueueConstruct(&comm->suspendTaskQueue);
  ncclIntruQueueConstruct(&comm->resumeTaskQueue);

  if (parentComm == NULL || !parentComm->shareResources) {
    NCCLCHECK(ncclCalloc(&comm->sharedRes, 1));
    /* most of attributes are assigned later in initTransportsRank(). */
    comm->sharedRes->owner = comm;
    comm->sharedRes->tpNRanks = comm->nRanks;
    comm->sharedRes->devMemUsed = 0;
    comm->sharedRes->pinnedMemUsed = 0;
    NCCLCHECK(ncclCalloc(&comm->sharedRes->tpRankToLocalRank, comm->nRanks));
    NCCLCHECK(ncclStrongStreamConstruct(&comm->sharedRes->deviceStream));
    NCCLCHECK(ncclStrongStreamConstruct(&comm->sharedRes->hostStream));
    HGGCCHECK(hggcEventCreateWithFlags(&comm->sharedRes->scratchEvent, hggcEventDisableTiming));
    comm->sharedRes->refCount = 1;
    NCCLCHECK(ncclGinInit(comm));
  } else {
    comm->sharedRes = parentComm->sharedRes;
    ncclAtomicRefCountIncrement(&parentComm->sharedRes->refCount);
    NCCLCHECK(ncclGinInitFromParent(comm, parentComm));
  }

  if (comm->topParentRanks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));
    for (int i = 0; i < comm->nRanks; ++i)
      comm->topParentRanks[i] = i;
  }

  for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
    comm->groupNext[type] = reinterpret_cast<ncclComm_t>(0x1);
  }

  comm->preconnectNext = reinterpret_cast<ncclComm_t>(0x1);

  /**
   * ring and tree are enabled by default
   * fc is enabled by default only in certain cases, other cases could be enabled by env
   * collnet is disabled by default
   * **/
  int algoDefaultEnable[NCCL_NUM_ALGORITHMS] = {ncclTuningEnable, ncclTuningEnable, ncclTuningDisable, ncclTuningCustomEnable, ncclTuningEnable};
  for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      comm->algoEnable[f*NCCL_NUM_ALGORITHMS+a] = algoDefaultEnable[a];
    }
  }

  NCCLCHECK(ncclCalloc(&comm->hgtxMessage, MAX_HGTX_MSG_LENGTH + ONE_HGTX_MSG_LENGTH));
  comm->hgtxMessageIndex = 0;
  comm->extKernelPluginWorkspace.peerIpcPtrNum = 0;

  ncclIntruQueueMpscConstruct(&comm->callbackQueue);
  ncclIntruQueueConstruct(&comm->legacyRegCleanupQueue);

  comm->regCache.pageSize = sysconf(_SC_PAGESIZE);

  do {
    hggcMemPoolProps props = {};
    props.allocType = hggcMemAllocationTypePinned;
    props.handleTypes = hggcMemHandleTypeNone;
    props.location.type = hggcMemLocationTypeDevice;
    props.location.id = comm->hggcDev;
    HGGCCHECK(hggcMemPoolCreate(&comm->memPool, &props));
    uint64_t releaseThreshold = ~uint64_t(0);
    HGGCCHECK(hggcMemPoolSetAttribute(comm->memPool, hggcMemPoolAttrReleaseThreshold, &releaseThreshold));
  } while (0);

  NCCLCHECK(ncclCalloc(&comm->kernelInfoMonitorPtr, pcclDevKernelArgsFifoDepth()));
  comm->kernelNum = 0;

  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->deviceStream));
  // Duplicate the channels on the device
  int nChannels = std::max(comm->nChannels, comm->p2pnChannels);
  NCCLCHECK(ncclHggcCallocAsyncLegacy(&comm->hostDevComm.comm.channels, nChannels, &(comm->devMemUsed), comm->sharedRes->deviceStream.hggcStream, comm->memManager));
  NCCLCHECK(ncclHggcMemcpyAsync(comm->hostDevComm.comm.channels, comm->channels, nChannels, comm->sharedRes->deviceStream.hggcStream));

  // Alloc profiler counters for the kernel
  NCCLCHECK(ncclHggcHostCalloc(&comm->profiler.workStarted, MAXCHANNELS, &(comm->pinnedMemUsed)));
  NCCLCHECK(ncclHggcHostCalloc(&comm->profiler.workCompleted, MAXCHANNELS, &(comm->pinnedMemUsed)));
  comm->hostDevComm.comm.workStarted = comm->profiler.workStarted;
  comm->hostDevComm.comm.workCompleted = comm->profiler.workCompleted;
  comm->hostDevComm.comm.profilerGlobalEnabled = ncclProfilerPluginLoaded() ? 1 : 0;

  // Copy userRanks and peers
  for (int r=0; r<comm->nChannels; r++) {
    NCCLCHECK(ncclHggcMemcpyAsync(comm->channels[r].ring.devUserRanks, comm->channels[r].ring.userRanks, comm->nRanks, comm->sharedRes->deviceStream.hggcStream));
  }

  int* rankToLocalRank;
  NCCLCHECK(ncclHggcCallocAsyncLegacy(&rankToLocalRank, comm->nRanks, &(comm->devMemUsed), comm->sharedRes->deviceStream.hggcStream, comm->memManager));
  NCCLCHECK(ncclHggcMemcpyAsync(rankToLocalRank, comm->rankToLocalRank, comm->nRanks, comm->sharedRes->deviceStream.hggcStream));
  comm->hostDevComm.comm.rankToLocalRank = rankToLocalRank;

  comm->workArgsBytes = std::min<size_t>(ncclParamWorkArgsBytes(), ncclMaxKernelArgsSize());

  comm->workFifoBytes = ncclParamWorkFifoBytes();
  if (0 != (comm->workFifoBytes & (comm->workFifoBytes-1))) {
    WARN("PCCL_WORK_FIFO_BYTES=%d is being ignored because it is not a power of 2.", comm->workFifoBytes);
    comm->workFifoBytes = NCCL_WORK_FIFO_BYTES_DEFAULT;
  }
  comm->workFifoBytes = std::min(comm->workFifoBytes, 1u<<30);

  if (comm->rank == 0) {
    INFO(NCCL_INIT, "workFifoBytes %d", comm->workFifoBytes);
  }

  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // The workFifoBuf lives in GDR mapped CUDA memory.
    NCCLCHECK(ncclGdrCudaCalloc((char**)(&comm->workFifoBuf), (char**)(&comm->workFifoBufDev), comm->workFifoBytes, &comm->workFifoBufGdrHandle, &(comm->devMemUsed), comm->memManager));
  } else {
    // The workFifoBuf lives in hggcHost memory.
    comm->workFifoBufGdrHandle = nullptr;
    NCCLCHECK(ncclHggcHostCalloc((char**)(&comm->workFifoBuf), comm->workFifoBytes, &(comm->pinnedMemUsed)));
    comm->workFifoBufDev = comm->workFifoBuf;
  }

  comm->workFifoProduced = 0;
  comm->workFifoProducedLastRecorded = 0;
  comm->workFifoConsumed = 0;

  for (int c=0; c < MAXCHANNELS; c++) {
    comm->hostDevComm.channels[c].peers = comm->channels[c].devPeers;
    comm->hostDevComm.channels[c].ring = comm->channels[c].ring;
    comm->hostDevComm.channels[c].tree = comm->channels[c].tree;
    comm->hostDevComm.channels[c].collTree = comm->channels[c].collTree;
    comm->hostDevComm.channels[c].fc = comm->channels[c].fc;

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECK(ncclHggcMemcpyAsync(comm->hostDevComm.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, comm->nRanks, comm->sharedRes->deviceStream.hggcStream));
    }
  }

  // Duplicate the dev comm on the device
  NCCLCHECK(ncclHggcCallocAsyncLegacy(reinterpret_cast<ncclKernelCommAndChannels**>(&comm->devComm), 1, &(comm->devMemUsed), comm->sharedRes->deviceStream.hggcStream, comm->memManager));
  NCCLCHECK(ncclHggcMemcpyAsync(reinterpret_cast<ncclKernelCommAndChannels*>(comm->devComm), &comm->hostDevComm, 1, comm->sharedRes->deviceStream.hggcStream));

  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream));

  return ncclSuccess;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define PCCL_VERSION_STRING "PCCL version " STR(PCCL_MAJOR) "." STR(PCCL_MINOR) "." STR(PCCL_PATCH) PCCL_SUFFIX
static ncclResult_t showVersion() {
  static int shown = 0;

  int hggcRuntimeVersion = 0;
  HGGCCHECKWITHWARN(hggcRuntimeGetVersion(&hggcRuntimeVersion));
  int hggcRuntimeVersionMajor = hggcRuntimeVersion / 1000;
  int hggcRuntimeVersionMinor = hggcRuntimeVersion % 1000 / 10;

  int hggcDriverVersion = 0;
  HGGCCHECKWITHWARN(hggcDriverGetVersion(&hggcDriverVersion));
  int hggcDriverVersionMajor = hggcDriverVersion / 1000;
  int hggcDriverVersionMinor = hggcDriverVersion % 1000 / 10;

  if (shown == 0 && ncclDebugLevel >= PCCL_LOG_VERSION) {
    printf("%s, HGGC Runtime version %d.%d, HGGC Driver version %d.%d \n", PCCL_VERSION_STRING,
            hggcRuntimeVersionMajor, hggcRuntimeVersionMinor, hggcDriverVersionMajor, hggcDriverVersionMinor);
    fflush(stdout);
    if (ncclDebugFile != stdout)
      INFO(NCCL_ALL,"%s, HGGC Runtime version %d.%d, HGGC Driver version %d.%d \n", PCCL_VERSION_STRING,
            hggcRuntimeVersionMajor, hggcRuntimeVersionMinor, hggcDriverVersionMajor, hggcDriverVersionMinor); // Also log PCCL version in one of the files
    shown = 1;
  }

  return ncclSuccess;
}

NCCL_PARAM(MNNVLUUID, "MNNVL_UUID", -1);
NCCL_PARAM(MNNVLCliqueId, "MNNVL_CLIQUE_ID", -1);

static ncclResult_t fillInfo(ncclComm_t comm, struct ncclPeerInfo* info, uint64_t commHash) {
  hggcDeviceProp prop;
  info->rank = comm->rank;
  info->hggcDev = comm->hggcDev;
  info->hgmlDev = comm->hgmlDev;
  NCCLCHECK(pcclGetVersion(&info->version));
  info->hostHash=getHostHash()+commHash;
  info->pidHash=getPidHash()+commHash;
  info->cuMemSupport = ncclCuMemEnable();
  HGGCCHECK(hggcGetDeviceProperties(&prop, comm->hggcDev));
  info->totalGlobalMem = ROUNDUP(prop.totalGlobalMem, (1L << 32));
  info->strideScale = 1;

  HGGCCHECK(hggcEitInterface::getInstance().wrapHggcDeviceGetAttributePpuId((int*)&info->ppuid, info->hggcDev));
  if (ncclCuMemEnable()) {
    size_t granularityVir = 0;
    size_t granularityPhy = 0;
    HGmemAllocationProp allocProp = {};
    allocProp.type = HG_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
    allocProp.requestedHandleTypes = ncclHgMemHandleType;
    allocProp.location.id = 0;
    HGCHECK(hgMemGetAllocationGranularity(&granularityVir, &allocProp, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    HGCHECK(hggcEitInterface::getInstance().wrapHgMemGetAllocationGranularityEx(&granularityPhy, &allocProp, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED));
    info->strideScale = divUp(granularityVir, granularityPhy);
  }

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  struct stat statbuf;
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = statbuf.st_dev;

  info->busId = comm->busId;

  NCCLCHECK(ncclGpuGdrSupport(&info->gdrSupport));
  info->comm = comm;
  info->hggcCompCap = comm->minCompCap = comm->maxCompCap = comm->compCap;

  NCCLCHECK(ncclTopoCheckCrossNicSupport(&info->crossNicSupport));
  int cuMemGdrFlag = 0;
  HGCHECK(hgDeviceGetAttribute(&cuMemGdrFlag, HG_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_HGGC_VMM_SUPPORTED, comm->hggcDev));
  info->cuMemGdrSupport = (cuMemGdrFlag == 1);
  info->supportedGinType = comm->sharedRes->ginState.ginType;

  // MNNVL support
  if (pcclHgmlEnsureInitialized() == ncclSuccess) {
    ncclResult_t mnnvlRes = ncclSuccess;
    info->fabricInfo.state = HGML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    // MNNVL: Request the fabric UUID and partition info
    char busId[HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    hgmlDevice_t hgmlDev;
    if (mnnvlRes == ncclSuccess) mnnvlRes = int64ToBusId(info->busId, busId);
    if (mnnvlRes == ncclSuccess) mnnvlRes = pcclHgmlDeviceGetHandleByPciBusId(busId, &hgmlDev);
    if (mnnvlRes == ncclSuccess) mnnvlRes = pcclHgmlDeviceGetGpuFabricInfo(hgmlDev, &info->fabricInfo);
    if (ncclParamMnnvlFakeAdvancedFeatureFlag || (mnnvlRes == ncclSuccess && info->fabricInfo.state != HGML_GPU_FABRIC_STATE_NOT_SUPPORTED)) {
      if (ncclParamMNNVLUUID() != -1) {
        ((long*)&info->fabricInfo.clusterUuid)[0] = ncclParamMNNVLUUID();
        ((long*)&info->fabricInfo.clusterUuid)[1] = ncclParamMNNVLUUID();
      }
      if (ncclParamMNNVLCliqueId() != -1) info->fabricInfo.cliqueId = ncclParamMNNVLCliqueId();
      INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x state %d",
           info->busId,
           ((long *)&info->fabricInfo.clusterUuid)[0], ((long *)&info->fabricInfo.clusterUuid)[1],
           info->fabricInfo.cliqueId, info->fabricInfo.state);
    }
  }

  return ncclSuccess;
}

static ncclResult_t setupChannel(ncclComm_t comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Reorganize ranks to start with rank.
  int shift;
  for (shift = 0; shift<nranks; shift++) {
    if (ringRanks[shift] == rank) {
      break;
    }
  }
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+shift)%nranks];
  }
  return ncclSuccess;
}

void* waitForNonNullPtr(void* p) {
  volatile void** ptr = (volatile void**) p;
  while (*ptr == NULL) sched_yield();
  return (void*)*ptr;
}

// Allocate/Set Intra Process Structures and set CG options
ncclResult_t ncclCommSetIntraProc(ncclComm_t comm) {
  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    const char* str = pcclGetEnvWithDualPrefix("PCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "PCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  return ncclSuccess;
}

#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_BUFFSIZE_ARM (1 << 20) /* 1MiB */
#define DEFAULT_SIGNAL_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_SIGNAL_BUFFSIZE_ARM (1 << 20) /* 1MiB */
#define DEFAULT_SIMPLE_BULK_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_SIMPLE_BULK_BUFFSIZE_ARM (1 << 20) /* 1MiB */
#define DEFAULT_SIGNAL_BULK_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_SIGNAL_BULK_BUFFSIZE_ARM (1 << 20) /* 1MiB */
#define DEFAULT_SIMPLE_BULK_AR_BUFFSIZE (1 << 22) /* 4MiB */
#define DEFAULT_SIMPLE_BULK_AR_BUFFSIZE_ARM (1 << 20) /* 1MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

static ncclResult_t computeBuffSizes(ncclComm_t comm) {
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));

  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize(), ncclParamBuffSize(), ncclParamBuffSize(), ncclParamBuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE, DEFAULT_SIGNAL_BUFFSIZE, DEFAULT_SIMPLE_BULK_BUFFSIZE, DEFAULT_SIGNAL_BULK_BUFFSIZE, DEFAULT_SIMPLE_BULK_AR_BUFFSIZE };

  if (cpuArch == NCCL_TOPO_CPU_ARCH_ARM) {
    defaults[NCCL_PROTO_SIMPLE] = DEFAULT_BUFFSIZE_ARM;
    defaults[NCCL_PROTO_SIGNAL] = DEFAULT_SIGNAL_BUFFSIZE_ARM;
    defaults[NCCL_PROTO_SIMPLE_BULK] = DEFAULT_SIMPLE_BULK_BUFFSIZE_ARM;
    defaults[NCCL_PROTO_SIGNAL_BULK] = DEFAULT_SIGNAL_BULK_BUFFSIZE_ARM;
    defaults[NCCL_PROTO_SIMPLE_BULK_AR] = DEFAULT_SIMPLE_BULK_AR_BUFFSIZE_ARM;
  }

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = comm->hostDevComm.comm.buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }
  return ncclSuccess;
}

static ncclResult_t copyChannel(ncclComm_t comm, struct ncclTopoGraph* graph, int num) {
  if (num <= graph->nChannels) {
    int localRanks = comm->localRanks;
    memcpy(graph->intra+graph->nChannels*localRanks, graph->intra, num*localRanks*sizeof(int));
    memcpy(graph->inter+graph->nChannels*2, graph->inter, num*2*sizeof(int64_t));
    graph->nChannels += num;
  } else {
    copyChannel(comm, graph, graph->nChannels);
    copyChannel(comm, graph, num - graph->nChannels);
  }
  return ncclSuccess;
}

static ncclResult_t ncclAlignRingAndTreeChannels(ncclComm_t comm, struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* treeGraph, bool algoRingEnable, bool algoTreeEnable) {
  if (algoRingEnable && algoTreeEnable) {
    INFO(NCCL_INIT, "Ring & Tree are both enabled, not necessary to adjust channels");
  } else {
    struct ncclTopoGraph* objGraph = NULL;
    int nc = 0;
    if (algoRingEnable) {
      objGraph = treeGraph;
      nc = ringGraph->nChannels - objGraph->nChannels;
      INFO(NCCL_INIT, "Only Ring is enabled, Tree need increase %d channels", nc);
    } else if (algoTreeEnable) {
      objGraph = ringGraph;
      nc = treeGraph->nChannels - objGraph->nChannels;
      INFO(NCCL_INIT, "Only Tree is enabled, Ring need increase %d channels", nc);
    } else {
      INFO(NCCL_INIT, "Ring & Tree are both disabled, not necessary to adjust channels");
    }

    if (nc > 0) {
      NCCLCHECK(copyChannel(comm, objGraph, nc));
    }
  }

  comm->nChannels = treeGraph->nChannels = ringGraph->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  comm->nGraphChannels = treeGraph->nGraphChannels = ringGraph->nGraphChannels = std::min(treeGraph->nGraphChannels, ringGraph->nGraphChannels);
  return ncclSuccess;
}

extern struct ncclTransport collNetTransport;

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
PCCL_PARAM(GraphTreePattern, "GRAPH_TREE_PATTERN", 0);

static int ncclTopoGraphTreePattern(ncclComm_t comm) {
  int pattern = pcclParamGraphTreePattern();
  if (pattern > NCCL_TOPO_PATTERN_TREE || pattern < NCCL_TOPO_PATTERN_BALANCED_TREE) {
    pattern = comm->nNodes <= 2 ? NCCL_TOPO_PATTERN_TREE : NCCL_TOPO_PATTERN_BALANCED_TREE;

    // for 8+ CX6, go simple tree.for now
    int netCount = 0;
    ncclTopoGetNetCount(comm->topo, &netCount);
    pattern = netCount >= 8 ? NCCL_TOPO_PATTERN_TREE : NCCL_TOPO_PATTERN_BALANCED_TREE;
  }

  return pattern;
}

// MNNVL: Flag to indicate whether to enable Multi-Node NVLink
NCCL_PARAM(MNNVLEnable, "MNNVL_ENABLE", 2);

#define TIMER_INIT_TOTAL 0
#define TIMER_INIT_KERNELS 1
#define TIMER_INIT_BOOTSTRAP 2
#define TIMER_INIT_ALLGATHER 3
#define TIMER_INIT_TOPO 4
#define TIMER_INIT_GRAPHS 5
#define TIMER_INIT_CONNECT 6
#define TIMER_INIT_ALLOC 7
#define TIMERS_INIT_COUNT 8

NCCL_PARAM(GroupSize, "P2P_SCHEDULE_GROUP_SIZE", -1);

static ncclResult_t ncclP2pSchedule(ncclComm_t comm) {
  struct ncclNodeRanks* nodeRanks = comm->nodeRanks;
  // For MNNVL systems, we need to split the nodes into different groups to guarantee the proper PXN
  // aggregation factor.
  int groupSize;
  if (comm->nNodes > 1) {
    if (ncclParamGroupSize() > 0) {
      groupSize = ncclParamGroupSize();
    } else {
      float gpuPerNetBw;
      pcclTopoGetGpuPerNetBw(comm, &gpuPerNetBw);
      if (gpuPerNetBw <= 12.0) {
        groupSize = NCCL_MAX_DEV_WORK_P2P_PER_BATCH;
      } else {
        groupSize = 2;
        while (comm->nRanks > comm->p2pnChannels * groupSize && groupSize * 2 <= NCCL_MAX_DEV_WORK_P2P_PER_BATCH) { 
          groupSize *= 2;
        }
      }
    }
  } else {
    groupSize = comm->maxLocalRanks;
  }

  for (int node = 0; node < comm->nNodes; node++) {
    int localRanks = nodeRanks[node].localRanks;
    if (localRanks % groupSize != 0 || localRanks < groupSize) groupSize = gcd(groupSize, nodeRanks[node].localRanks);
  }
  comm->p2pSchedGroupSize = groupSize;

  int local = comm->localRank % groupSize; // local id inside my group
  int group = comm->localRank / groupSize; // id of my group, incremented when going over the previous nodes
  int nGroups = comm->nRanks / groupSize;
  int nGroupsPow2 = pow2Up(nGroups);

  int *groupToNode = nullptr, *groupToLocal = nullptr;
  NCCLCHECK(ncclCalloc(&groupToNode, nGroups));  // node hosting the group
  ncclResult_t groupToLocalRet = ncclCalloc(&groupToLocal, nGroups); // local offset of the group
  if (groupToLocalRet != ncclSuccess) { // avoid leaking groupToNode if the second alloc fails
    NCCLCHECK(ncclFree(groupToNode));
    return groupToLocalRet;
  }
  int groupCount = 0;
  for (int n = 0; n < comm->nNodes; ++n) {
    if (0 != comm->nodeRanks[n].localRanks % groupSize) {
      WARN("nLocals = %d should be a diviser of the number of ranks in node %d = %d", groupSize, n,
           comm->nodeRanks[n].localRanks);
      NCCLCHECK(ncclFree(groupToNode));
      NCCLCHECK(ncclFree(groupToLocal));
      return ncclInternalError;
    }
    int nGroupsInNode = comm->nodeRanks[n].localRanks / groupSize;
    for (int g = 0; g < nGroupsInNode; ++g) {
      groupToLocal[groupCount] = g * groupSize;
      groupToNode[groupCount] = n;
      groupCount++;
    }
    if (n < comm->node) group += nGroupsInNode;
  }
  if (groupCount != nGroups) {
    WARN("Group creation failed: %d vs %d", groupCount, nGroups);
    NCCLCHECK(ncclFree(groupToNode));
    NCCLCHECK(ncclFree(groupToLocal));
    return ncclInternalError;
  }
  INFO(NCCL_GRAPH, "%s: group size used is %d", __func__, groupSize);

  uint32_t groupRound = 0, groupDelta = 0;
  int round = 0;
  // When enumerating peer deltas we use the quadratic formula (x*x+x)/2 mod N.
  // Since that formula only produces valid permutations when N is a pow of 2,
  // we let N = pow2Up(n) and filter out results greater-eq to n.
  // Example sequence for 16 ranks: 0, 1, 3, 6, 10, 15, 5, 12, 4, 13, 7, 2, 14, 11, 9, 8
  do {
    if (groupDelta < nGroups) {
      // Filter nonsensical group deltas
      int sendGroup = (group + groupDelta) % nGroups;
      int recvGroup = (group - groupDelta + nGroups) % nGroups;
      int sendNode = groupToNode[sendGroup];
      int recvNode = groupToNode[recvGroup];
      for (int delta = 0; delta < groupSize; delta++) {
        int sendLocal = groupToLocal[sendGroup] + (local + delta) % groupSize;
        int recvLocal = groupToLocal[recvGroup] + (local - delta + groupSize) % groupSize;
        comm->p2pSchedule[round].sendRank = nodeRanks[sendNode].localRankToRank[sendLocal];
        comm->p2pSchedule[round].recvRank = nodeRanks[recvNode].localRankToRank[recvLocal];
        round += 1;
      }
    }
    groupRound += 1;
    groupDelta = (groupDelta + groupRound) & (nGroupsPow2 - 1); // Quadratic update
  } while (groupRound != nGroupsPow2);

  NCCLCHECK(ncclFree(groupToNode));
  NCCLCHECK(ncclFree(groupToLocal));

  if (round != comm->nRanks) {
    WARN("P2p schedule creation has bugs.");
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t initTransportsRank(ncclComm_t comm, ncclComm_t parent, uint64_t timers[TIMERS_INIT_COUNT]) {
  ncclResult_t res = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nNodes = 1;

  struct graphInfo {
    int pattern;
    int sameChannels;
    float speedIntra;
    float speedInter;
    int typeIntra;
    int typeInter;
  };

  struct allGatherInfo {
    int64_t netId;
    int hggcCompCap;
    int nChannels;
    int nGraphChannels;
    int p2pnChannelsPerPeer;
    struct graphInfo graphInfo[NCCL_ALGO_COLLNET + 1];
    struct ncclTopoRanks topoRanks;
  } ;

  cpu_set_t affinitySave;
  int nChannelsOrig;

  struct ncclTopoGraph *ringGraph = &comm->graphs[NCCL_ALGO_RING];
  struct ncclTopoGraph *treeGraph = &comm->graphs[NCCL_ALGO_TREE];
  struct ncclTopoGraph *collNetGraph = &comm->graphs[NCCL_ALGO_COLLNET];
  struct ncclTopoGraph *fcGraph = &comm->graphs[NCCL_ALGO_FC];
  struct ncclTopoGraph *graphs[NCCL_ALGO_COLLNET + 1] = { treeGraph, ringGraph, collNetGraph };

  ncclComm_t intraRank0Comm = NULL;
  struct allGatherInfo *allGather3Data = NULL;
  int *nodesFirstRank = NULL;
  int *nodesTreePatterns = NULL;
  struct ncclTopoRanks** allTopoRanks = NULL;
  int *rings = NULL;
  int* pxnPeers = NULL;
  int *topParentLocalRanks = NULL;
  int netCount = 0;
  int p2pLevel = -1;
  bool globalGinSupport = false;
  bool globalCrossNicSupport = true;
  bool globalCuMemGdrSupport = true;

  // respect NCCL_ALGO when init
  const char *algoStr = pcclGetEnvWithDualPrefix("PCCL_ALGO");
  int algoEnableNum = 0;
  bool algoTreeEnable = false;
  bool algoRingEnable = false;
  if (algoStr) {
    INFO(NCCL_ENV, "PCCL_ALGO set by environment to %s", algoStr);
    NCCLCHECK(ncclParseList(algoStr, ncclFuncStr, NCCL_NUM_FUNCTIONS, ncclAlgoStr, NCCL_NUM_ALGORITHMS, comm->algoEnable));
  }

  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    bool algoEnableFlag = false;
    for (int f=0; f<NCCL_NUM_FUNCTIONS; f++) {
      if (comm->algoEnable[f*NCCL_NUM_ALGORITHMS+a] >= 1) {
        if (a == NCCL_ALGO_TREE) algoTreeEnable = true;
        if (a == NCCL_ALGO_RING) algoRingEnable = true;
        algoEnableFlag = true;
      }
    }
    // Each algo counted only once in all functions
    if (algoEnableFlag) algoEnableNum += 1;
  }

  timers[TIMER_INIT_ALLGATHER] = clockNano();
  // AllGather1 - begin
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks+1), res, fail); // Extra rank to represent CollNet root
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo+rank, comm->commHash), res, fail);
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), res, fail);
  __atomic_store_n(&comm->peerInfoValid, true, __ATOMIC_RELEASE);

  globalGinSupport = comm->sharedRes->ginState.ginType != NCCL_GIN_TYPE_NONE;
  comm->cuMemSupport = 1;
  for (int i = 0; i < nranks; i++) {
    if (comm->peerInfo[i].version != comm->peerInfo[rank].version) {
      ERROR("Mismatched PCCL version detected : rank %d version %d rank %d version %d",
            i, comm->peerInfo[i].version, rank, comm->peerInfo[rank].version);
      res = ncclInvalidUsage;
      goto fail;
    }
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) nNodes++;
    if (!comm->peerInfo[i].cuMemSupport) comm->cuMemSupport = 0;
    globalGinSupport &= (comm->peerInfo[i].supportedGinType == comm->sharedRes->ginState.ginType);
    globalCrossNicSupport &= comm->peerInfo[i].crossNicSupport;
    globalCuMemGdrSupport &= comm->peerInfo[i].cuMemGdrSupport;
    if (i == rank) comm->peerInfo[comm->rank].hostid = nNodes - 1;
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      ERROR("Duplicate PPU detected : rank %d and rank %d both on HGGC device %lx", rank, i, comm->peerInfo[rank].busId);
      res = ncclInvalidUsage;
      goto fail;
    }
  }

  // Compute intra ranks and minimum HGGC Compute capabilities of intra-node GPUs and all GPUs
  int intraRank0, intraRank, intraRanks, minCompCap, maxCompCap;
  minCompCap = comm->peerInfo[rank].hggcCompCap;
  maxCompCap = comm->peerInfo[rank].hggcCompCap;
  intraRank0 = -1; intraRank = -1; intraRanks = 0;
  for (int i = 0; i < nranks; i++) {
    if (PCCL_SAME_PID(comm->peerInfo+i, comm->peerInfo+rank)) {
      if (intraRanks == 0) intraRank0 = i;
      if (i == rank) intraRank = intraRanks;
      intraRanks++;
      if (intraRank0 == rank && rank != i) {
        comm->peerInfo[i].comm->intraNext = comm->intraNext;
        comm->intraNext = comm->peerInfo[i].comm;
      }
    }
    minCompCap = std::min(comm->peerInfo[i].hggcCompCap, minCompCap);
    maxCompCap = std::max(comm->peerInfo[i].hggcCompCap, maxCompCap);
  }
  TRACE(NCCL_INIT,"hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
        rank, comm->peerInfo[rank].hostHash, intraRank, intraRanks, intraRank0);
  if (intraRank == -1 || intraRank0 == -1 || comm->peerInfo[intraRank0].comm == NULL) {
    ERROR("Failed to determine intra ranks hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
         rank, comm->peerInfo[rank].hostHash, intraRank, intraRanks, intraRank0);
    res = ncclInternalError;
    goto fail;
  }
  intraRank0Comm = comm->peerInfo[intraRank0].comm;
  comm->intraComm0 = intraRank0Comm;
  comm->intraBarrierPhase = 0;
  comm->intraBarrierCounter = 0;
  comm->intraBarrierGate = 0;
  comm->intraRank = intraRank;
  comm->intraRanks = intraRanks;
  comm->channelShuffle = false;

  // Check for MNNVL support
  NCCLCHECKGOTO(ncclGetUserP2pLevel(&p2pLevel), res, fail);
  if ((nNodes > 1 && ncclParamMNNVLEnable() != 0 && p2pLevel != 0) || ncclParamMNNVLEnable() == 1) {
    NCCLCHECKGOTO(ncclMnnvlCheck(comm), res, fail);
  }

  if (pcclParamChannelShuffle() >= 0) {
    comm->channelShuffle = pcclParamChannelShuffle();
  } else if (minCompCap == 89 && nNodes == 1) {
    comm->channelShuffle = true;
  }

  useMemcpy = ncclParamP2pUseCudaMemcpy();

  // AllGather1 - end
  timers[TIMER_INIT_ALLGATHER] = clockNano() - timers[TIMER_INIT_ALLGATHER];

  // init cu resources usage policy
  NCCLCHECK(initSmResourcesConfig(comm));
  timers[TIMER_INIT_TOPO] = clockNano();
  // Topo detection / System graph creation
  NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), res, fail);
  // Compute paths between GPUs and NICs
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), res, fail);
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECKGOTO(ncclTopoTrimSystem(comm->topo, comm), res, fail);
  // Recompute paths after trimming
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), res, fail);
  // respect the pcie order to adjust PHB path
  NCCLCHECKGOTO(ncclTopoAdjustSystem(comm->topo), res, fail);
  // Init search
  NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), res, fail);
  // Print final topology
  NCCLCHECKGOTO(ncclTopoPrint(comm, comm->topo), res, fail);
  timers[TIMER_INIT_TOPO] = clockNano() - timers[TIMER_INIT_TOPO];

  timers[TIMER_INIT_GRAPHS] = clockNano();
  // Get rings and trees
  ringGraph->id = 0;
  ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph->collNet = 0;
  ringGraph->minChannels = 1;
  ringGraph->maxChannels = MAXCHANNELS/2;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), res, fail);
  if (comm->channelShuffle && minCompCap == 89) NCCLCHECKGOTO(pcclShuffleChannel(comm, ringGraph, nranks), res, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, ringGraph), res, fail);

  treeGraph->id = 1;
  treeGraph->pattern = ncclTopoGraphTreePattern(comm);
  treeGraph->collNet = 0;
  // treeGraph->minChannels = 1;
  // treeGraph->maxChannels = (algoEnableNum == 1 && comm->algoEnable[NCCL_ALGO_TREE] == 1) ? MAXCHANNELS/2 : ringGraph->nChannels ;
  // from nccl 2.18.1-1
  treeGraph->minChannels = (algoEnableNum == 1 && algoTreeEnable) ? 1 : ringGraph->nChannels ;
  treeGraph->maxChannels = (algoEnableNum == 1 && algoTreeEnable) ? MAXCHANNELS/2 : ringGraph->nChannels ;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, treeGraph), res, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, treeGraph), res, fail);

  collNetGraph->id = 2;
  collNetGraph->pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph->collNet = 1;
  collNetGraph->minChannels = collNetGraph->maxChannels = ringGraph->nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetGraph), res, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetGraph), res, fail);

  fcGraph->id = 3;
  fcGraph->pattern = NCCL_TOPO_PATTERN_FC;
  fcGraph->collNet = 0;
  fcGraph->minChannels = 1;
  fcGraph->maxChannels = MAXCHANNELS/2;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, fcGraph), res, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, fcGraph), res, fail);

  timers[TIMER_INIT_GRAPHS] = clockNano() - timers[TIMER_INIT_GRAPHS];

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* graphs[3] = { ringGraph, treeGraph, collNetGraph };
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 3, graphs), res, fail);
  }

  // Compute nChannels per peer for p2p
  NCCLCHECK(ncclTopoComputeP2pChannelsPerPeer(comm));

  timers[TIMER_INIT_CONNECT] = clockNano();
  // AllGather3 - begin

  NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), res, fail);
  NCCLCHECKGOTO(ncclTopoGetLocalNet(comm->topo, rank,  &allGather3Data[rank].netId, NULL), res, fail);

  NCCLCHECK(ncclAlignRingAndTreeChannels(comm, ringGraph, treeGraph, algoRingEnable, algoTreeEnable));

  if (fcGraph->nChannels>0) {
    comm->fcnChannels = fcGraph->nChannels;
  }
  for (int a = 0; a <= NCCL_ALGO_COLLNET; a++) {
    allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
    allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
    allGather3Data[rank].graphInfo[a].speedIntra = graphs[a]->speedIntra;
    allGather3Data[rank].graphInfo[a].speedInter = graphs[a]->speedInter;
    allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
    allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
  }
  allGather3Data[rank].p2pnChannelsPerPeer = comm->p2pnChannelsPerPeer;

  NCCLCHECKGOTO(ncclTopoPreset(comm, treeGraph, ringGraph, collNetGraph, fcGraph, &allGather3Data[rank].topoRanks), res, fail);

  allGather3Data[rank].nChannels = comm->nChannels;
  allGather3Data[rank].nGraphChannels = comm->nGraphChannels;

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), res, fail);

  // Determine nNodes, firstRanks, ...
  NCCLCHECKGOTO(ncclCalloc(&nodesFirstRank, nranks), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&nodesTreePatterns, nranks), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToNode, comm->nRanks),res, fail);
  comm->nNodes = 0;
  for (int i=0; i<nranks; i++) {
    int node = -1;
    int firstRank = allGather3Data[i].topoRanks.ringRecv[0];
    for (int n=0; n<comm->nNodes; n++) {
      if (nodesFirstRank[n] == firstRank) node = n;
    }
    if (node == -1) {
      node = comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[i].graphInfo[NCCL_ALGO_TREE].pattern;
    }
    comm->rankToNode[i] = node;
  }
  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), res, fail);
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), res, fail);
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  TRACE(NCCL_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    ERROR("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    res = ncclInternalError;
    goto fail;
  }

  NCCLCHECKGOTO(ncclLoadTunerPlugin(&comm->tuner), res, fail);
  if (comm->tuner) {
    ncclResult_t res = comm->tuner->init(comm->ppuType, comm->nRanks, comm->nNodes, ncclDebugLogForPlugin);
    if (res != ncclSuccess) {
      ERROR("TUNER PLUGIN INIT FAIL");
      comm->tuner->destroy();
      NCCLCHECK(ncclCloseTunerPlugin(&comm->tuner));
    }
  }

  nChannelsOrig = comm->nChannels;
  NCCLCHECKGOTO(ncclCalloc(&allTopoRanks, comm->nRanks), res, fail);
  for (int i=0; i<nranks; i++) {
    comm->peerInfo[i].netId = allGather3Data[i].netId;
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    // Make sure we align all ranks so that the tuning is consistent across ranks
    treeGraph->nChannels = ringGraph->nChannels = comm->nChannels = std::min(allGather3Data[i].nChannels, comm->nChannels);
    treeGraph->nGraphChannels = ringGraph->nGraphChannels = comm->nGraphChannels = std::min(allGather3Data[i].nGraphChannels, comm->nGraphChannels);
    for (int a = 0; a <= NCCL_ALGO_COLLNET; a++) {
      graphs[a]->sameChannels = std::min(allGather3Data[i].graphInfo[a].sameChannels, graphs[a]->sameChannels);
      graphs[a]->speedIntra = std::min(allGather3Data[i].graphInfo[a].speedIntra, graphs[a]->speedIntra);
      graphs[a]->speedInter = std::min(allGather3Data[i].graphInfo[a].speedInter, graphs[a]->speedInter);
      graphs[a]->typeIntra = std::max(allGather3Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
      graphs[a]->typeInter = std::max(allGather3Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
    }
    comm->p2pnChannelsPerPeer = std::min(comm->p2pnChannelsPerPeer, allGather3Data[i].p2pnChannelsPerPeer);
  }

  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  NCCLCHECKGOTO(ncclCalloc(&rings, nranks*MAXCHANNELS), res, fail);

  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks,
        rings), res, fail);
  if (comm->rank == 0) {
    ncclTopoDumpRings(rings, comm->nChannels, comm->nRanks);
  }

  // ensure that the die detecting / channel computing is done
  // funcStrategy is indexed by [0, TOTAL_FUNCS_NUM) for standard collectives and
  // by [TOTAL_FUNCS_NUM, TOTAL_FUNCS_NUM + ncclSymkKernelId_Count) for symk
  // entries (one slot per ncclSymkKernelId enum value, see enqueue.cc::
  // ncclSetDispatchStrategy and symmetric_sched.cc::plan->symFuncIndex).
  NCCLCHECKGOTO(ncclCalloc(&comm->funcStrategy, TOTAL_FUNCS_NUM + (int)ncclSymkKernelId_Count), res, fail);
  NCCLCHECKGOTO(initDefaultDispatchStrategy(comm, ppuDiesNum), res, fail);

  if (comm->nNodes > 1 &&
      ncclParamCollNetEnable() == 1 &&
      collNetSupport(comm) && collNetGraph->nChannels) {
    NCCLCHECKGOTO(ncclTopoConnectCollNet(comm, collNetGraph, rank), res, fail);
  }

  // AllGather3 - end
  timers[TIMER_INIT_ALLGATHER] += clockNano() - timers[TIMER_INIT_CONNECT];

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[1024];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    if (strlen(line) > 1023) break;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  if (fcGraph->nChannels > 0) {
    char line[1024];
    line[0]='\0';
    for (int c=0; c<comm->fcnChannels; c++) {
      if (strlen(line) > 1023) break;
      struct ncclFC* fc = &comm->channels[c].fc;
      snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d/%d->%d",
          c, fc->peerRanks[0], fc->peerRanks[1], fc->peerRanks[2],
          fc->peerRanks[3], fc->rank);
    }
    line[1023] = '\0';
    INFO(NCCL_INIT, "FCs%s", line);
  }

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  NCCLCHECKGOTO(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), res, fail);
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }

  NCCLCHECKGOTO(computeBuffSizes(comm), res, fail);

  NCCLCHECKGOTO(ncclTransportCheckP2pType(comm, &comm->localP2pSupport, &comm->directMode, &comm->globalP2pSupport), res, fail);
  NCCLCHECKGOTO(pcclTopoCheckGlobalIcnP2p(comm, comm->topo, &comm->globalIcnP2pSupport), res, fail);

  // Compute nChannels per peer for p2p
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), res, fail);

  /* until now, all info of comm should be known. We can initialize shared resources and
   * map localRanks to top parent local ranks. NOTE: this shareRes init must be put before
   * all proxy operations. */
  if (comm->sharedRes->owner == comm) {
    comm->sharedRes->tpNLocalRanks = comm->localRanks;
    comm->sharedRes->magic = comm->magic;
    comm->sharedRes->tpNChannels = comm->nChannels;
    comm->sharedRes->tpP2pNChannels = comm->p2pnChannels;
    memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  }
  NCCLCHECKGOTO(ncclCalloc(&topParentLocalRanks, comm->localRanks), res, fail);
  for (int i = 0; i < comm->localRanks; ++i) {
    int tpRank = comm->topParentRanks[comm->localRankToRank[i]];
    topParentLocalRanks[i] = comm->sharedRes->tpRankToLocalRank[tpRank];
  }
  comm->topParentLocalRanks = topParentLocalRanks;

  // Profiler plugin context has to be initialized before proxy thread
  NCCLCHECK(ncclProfilerPluginInit(comm));

  // Launch proxy service thread
  ncclTopoGetNetCount(comm->topo, &netCount);
  // Gates LL buffer allocation on the P2P net transport (intra-node always allocates).
  // Must be set before ncclProxyCreate snapshots it into proxyState.
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;
  // Launch proxy service thread, after this, the proxy calls can be used.
  if (parent && parent->shareResources && !comm->isGrow) {
    comm->proxyState = parent->sharedRes->proxyState;
    ncclAtomicRefCountIncrement(&parent->sharedRes->proxyState->refCount);
  } else {
    NCCLCHECKGOTO(ncclProxyCreate(comm), res, fail);
  }
  NCCLCHECKGOTO(ncclCalloc(&comm->gproxyConn, comm->nRanks), res, fail);

  INFO(NCCL_INIT|NCCL_ALLOC, "Begin connect topo graphs");
  timers[TIMER_INIT_CONNECT] = clockNano();

  comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();
  if (strcmp(comm->ncclNet->name, EIC_PLUGIN_NAME) == 0 && comm->nNodes > 1 && nccl_use_u2mm_dev) {
    comm->runtimeConn = 0;
    if (comm->rank == 0) {
      INFO(NCCL_INIT, "u2mm does not support vmm api for now, so we cannot use runtime connection establishment");
    }
  }

  // Build p2p schedule
  comm->p2pSchedule = ncclMemoryStackAlloc<ncclComm::P2pSchedulePair>(&comm->memPermanent, comm->nRanks);
  NCCLCHECKGOTO(ncclP2pSchedule(comm), res, fail);

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), res, fail);
  }
  if (comm->nRanks > 1) {
    NCCLCHECKGOTO(ncclConnTypeCollect(comm, &comm->graphs[NCCL_ALGO_RING]), res, fail);
    NCCLCHECKGOTO(ncclConnTypeCollect(comm, &comm->graphs[NCCL_ALGO_TREE]), res, fail);
  }

  if (fcGraph->nChannels > 0) {
    for (int c=0; c<comm->fcnChannels; c++) {
      if (c >= comm->nChannels) {
        NCCLCHECKGOTO(initChannel(comm, c), res, fail);
      }
    }
    NCCLCHECKGOTO(ncclConnTypeCollect(comm, &comm->graphs[NCCL_ALGO_FC]), res, fail);
  }
  if (!comm->runtimeConn) {
    NCCLCHECKGOTO(ncclTransportRingConnect(comm), res, fail);

    // Connect Trees
    NCCLCHECKGOTO(ncclTransportTreeConnect(comm), res, fail);

    // Connect PAT only for communicators with 1 GPU per node
    if (comm->maxLocalRanks == 1) NCCLCHECKGOTO(ncclTransportPatConnect(comm), res, fail);

    // Check if we can setup CollNet
    if (comm->nNodes > 1 &&
        ncclParamCollNetEnable() == 1 &&
        collNetSupport(comm) && collNetGraph->nChannels) {
      NCCLCHECKGOTO(ncclTransportCollnetConnect(comm), res, fail);
    }

    // Connect fc graphs
    if (fcGraph->nChannels > 0) {
      NCCLCHECKGOTO(ncclTransportFcConnect(comm), res, fail);
    }

  }

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  NCCLCHECKGOTO(initExtKernelPlugin(comm), res, fail);
  if (comm->extKernel) {
    NCCLCHECK(buildConnectionForExtKernelPlugin(comm));
  }

  // Compute time models for algorithm and protocol combinations
  NCCLCHECKGOTO(ncclTopoTuneModel(comm, minCompCap, maxCompCap, treeGraph, ringGraph,
        collNetGraph, fcGraph), res, fail);

  comm->planner.peers = ncclMemoryStackAlloc<ncclKernelPlanner::Peer>(&comm->memPermanent, comm->nRanks);

  if (netCount > 0) {
    // Connect to local net proxy
    struct ncclProxyConnector proxyConn;
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn), res, fail);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), res, fail);

    // Then to remote ones when using PXN
    if (ncclPxnDisable(comm) == 0) {
      int nranks;
      NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), res, fail);
      for (int r=0; r<nranks; r++) {
        NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), res, fail);
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), res, fail);
      }
    }
  }

  NCCLCHECKGOTO(ncclCommSetIntraProc(comm), res, fail);

  comm->globalGinSupport = NCCL_GIN_CONNECTION_NONE;
  if (globalGinSupport && globalCuMemGdrSupport) {
    comm->globalGinSupport = globalCrossNicSupport ? NCCL_GIN_CONNECTION_FULL : NCCL_GIN_CONNECTION_RAIL;
  }
  comm->symmetricSupport = ncclParamWinEnable() && ncclCuMemEnable() &&
      (comm->globalGinSupport != NCCL_GIN_CONNECTION_NONE || (comm->localP2pSupport && ncclDevrIsOneLsaTeam(comm)));
  if (ncclCeSupported(comm)) {
    comm->ceReduceStep = pcclCeReduceMaxSect;
    int p2pRingNeededPerDie = std::min(comm->nRanks, HG_DMA_RING_PER_DIE);
    int num = p2pRingNeededPerDie * ppuDiesNum;
    comm->ceStreamsNum = num;
    NCCLCHECKGOTO(ncclCalloc(&comm->ceStreams, num), res, fail);
    NCCLCHECKGOTO(ncclCalloc(&comm->ceEvents, num), res, fail);
    for (int r = 0; r < p2pRingNeededPerDie; ++r) {
      for (int dieId = 0; dieId < ppuDiesNum; ++dieId) {
        int streamId = r * ppuDiesNum + dieId;
        HGGCCHECKGOTO(hggcEitInterface::getInstance().wrapHggcStreamCreateForCopyEngine(comm->ceStreams + streamId), res, fail);
        HGGCCHECKGOTO(hggcEitInterface::getInstance().wrapHggcStreamHintChannel(comm->ceStreams[streamId], HG_STREAM_CHANNEL_PEER, dieId * HG_DMA_RING_PER_DIE + r), res, fail);
        HGGCCHECKGOTO(hggcEitInterface::getInstance().wrapHggcStreamHintChannel(comm->ceStreams[streamId], HG_STREAM_CHANNEL_COPY, dieId * HG_DMA_RING_PER_DIE + r), res, fail);
        HGGCCHECKGOTO(hggcEventCreateWithFlags(comm->ceEvents + streamId, hggcEventDisableTiming), res, fail);
      }
    }
    HGGCCHECKGOTO(hggcEventCreateWithFlags(&comm->ceSyncEvent, hggcEventDisableTiming), res, fail);
    NCCLCHECKGOTO(ncclCalloc(&comm->ceCollConfigs.agOneStageStrategy, 1), res, fail);
    NCCLCHECKGOTO(pcclInitCeAllGatherStrategy(comm->ceCollConfigs.agOneStageStrategy, comm, /* twoStage */ 0), res, fail);

    if (comm->nRanks == 8) {
      NCCLCHECKGOTO(ncclCalloc(&comm->ceCollConfigs.agTwoStageStrategy, 1), res, fail);
      NCCLCHECKGOTO(pcclInitCeAllGatherStrategy(comm->ceCollConfigs.agTwoStageStrategy, comm, /* twoStage */ 1), res, fail);
    }
  }

  // Call devCommSetup before the last barrier, making sure we don't have a thread running in front and starting to
  // launch NCCL kernels before all hggc mem allocation is complete. That could cause a deadlock.
  NCCLCHECKGOTO(devCommSetup(comm), res, fail);
  timers[TIMER_INIT_CONNECT] = clockNano() -  timers[TIMER_INIT_CONNECT];

  /* Local intra-node barrier */
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), res, fail);

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);
exit:
  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  // Unlink proxy shm to make sure it will be properly cleaned up.
  if (comm->sharedRes->owner == comm && !comm->shareResources && res == ncclSuccess && !ncclCuMemEnable()) NCCLCHECK(ncclProxyShmUnlink(comm));
  NCCLCHECK(ncclFree(allGather3Data));
  NCCLCHECK(ncclFree(nodesFirstRank));
  NCCLCHECK(ncclFree(nodesTreePatterns));
  NCCLCHECK(ncclFree(allTopoRanks));
  NCCLCHECK(ncclFree(rings));
  NCCLCHECK(ncclFree(pxnPeers));
  return res;
fail:
  goto exit;
}

// Match config max/minCTAs
NCCL_PARAM(MaxCTAs, "MAX_CTAS", PCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(MinCTAs, "MIN_CTAS", PCCL_CONFIG_UNDEF_INT);

struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
  ncclComm_t* newcomm;
  int hggcDev;
  // For ncclCommInitRank
  int nranks, myrank, nId;
  ncclUniqueId* commId;
  // for ncclCommSplit
  ncclComm_t parent;
  int color, key;
  int childCount;
  // For Shrink
  int* excludeRanksList;
  int excludeRanksCount;
  // for pcclCommGrow
  bool isGrow;
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

struct ncclCommRevokeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

typedef struct{
  int key;
  int color;
} commSplitInfo;

NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 1);
NCCL_PARAM(CommShrinkShareResources, "COMM_SHRINK_SHARE_RESOURCES", PCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES", PCCL_CONFIG_UNDEF_INT);

static ncclResult_t commGetSplitInfo(ncclComm_t parent, int color, int key, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int nRanks = 0, myRank = 0;
  ncclResult_t ret = ncclSuccess;

  commSplitInfo* info = NULL;
  NCCLCHECKGOTO(ncclCalloc(&info, parent->nRanks), ret, fail);

  // Compute nRanks, my rank and the ranks (of the original comm) before and after me
  info[parent->rank].color = color;
  info[parent->rank].key = key;
  NCCLCHECKGOTO(bootstrapAllGather(parent->bootstrap, info, sizeof(commSplitInfo)), ret, fail);

  // Negative color does not create a new comm. Return now.
  if (color == PCCL_SPLIT_NOCOLOR) goto exit;

  memset(parentRanksRet, 0xff, sizeof(int) * parent->nRanks);
  for (int i = 0; i < parent->nRanks; i++) {
    if (info[i].color != color) continue;
    // Find where to insert this rank
    int insert = 0;
    while (insert < nRanks && info[parentRanksRet[insert]].key <= info[i].key) insert++;
    // Shift ranks by one after insert
    for (int r = nRanks; r > insert; r--) parentRanksRet[r] = parentRanksRet[r - 1];
    // Insert our rank
    parentRanksRet[insert] = i;
    nRanks++;
  }

  for (int i = 0; i < nRanks; i++) {
    if (parentRanksRet[i] == parent->rank) myRank = i;
  }

  *nRanksRet = nRanks;
  *myRankRet = myRank;

exit:
  NCCLCHECK(ncclFree(info));
  return ret;
fail:
  goto exit;
}

static ncclResult_t getParentRanks(int parentRanks, int parentRank, int* excludeRanksList, int excludeRanksCount, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int count = 0, j = 0;
  for (int i = 0; i < parentRanks; i++) {
    // we assume excludeRanksList is sorted
    if (j < excludeRanksCount && excludeRanksList[j] == i) {
      j++;
      continue;
    }
    if (i == parentRank) *myRankRet = count;
    parentRanksRet[count++] = i;
  }
  *nRanksRet = parentRanks - excludeRanksCount;
  return ncclSuccess;
}

PCCL_PARAM(CommDumpSignal, "COMM_DUMP_SIGNAL", SIGUSR2);
static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  size_t maxLocalSizeBytes = 0;
  int hggcDev = job->hggcDev;
  int* parentRanks = NULL;
  const int sig = pcclParamCommDumpSignal();
  uint64_t timers[TIMERS_INIT_COUNT] = {0};

  timers[TIMER_INIT_TOTAL] = clockNano();
  HGGCCHECKGOTO(hggcSetDevice(hggcDev), res, fail);

  timers[TIMER_INIT_KERNELS] = clockNano();

  if (ncclPerfmodelSpeedup) {
    ncclPerfmodelHostPtr[hggcDev] = malloc(ncclPerfmodelTotalDeviceMemSize);
    memset(ncclPerfmodelHostPtr[hggcDev], 0, ncclPerfmodelTotalDeviceMemSize);
    HGGCCHECK(hggcMalloc(&ncclPerfmodelDevicePtr[hggcDev], ncclPerfmodelTotalDeviceMemSize));
  }

  NCCLCHECK(ncclInitKernelsForDevice(&maxLocalSizeBytes));
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    size_t curStackSize = 0;
    HGGCCHECK(hggcDeviceGetLimit(&curStackSize, hggcLimitStackSize));
    if (curStackSize < maxLocalSizeBytes) {
      VERBOSE(NCCL_INIT, "Change hggcLimitStackSize from %zu to %zu", curStackSize, maxLocalSizeBytes);
      HGGCCHECK(hggcDeviceSetLimit(hggcLimitStackSize, maxLocalSizeBytes));
    }
  }

  // dummpy kernel time here
  timers[TIMER_INIT_KERNELS] = clockNano() - timers[TIMER_INIT_KERNELS];

  if (job->parent && !job->isGrow) {
    NCCLCHECKGOTO(ncclCalloc(&parentRanks, job->parent->nRanks), res, fail);
    if (job->excludeRanksCount) {
      NCCLCHECKGOTO(getParentRanks(job->parent->nRanks, job->parent->rank, job->excludeRanksList, job->excludeRanksCount, &job->nranks, &job->myrank, parentRanks), res, fail);
    } else {
      NCCLCHECKGOTO(commGetSplitInfo(job->parent, job->color, job->key, &job->nranks, &job->myrank, parentRanks), res, fail);
      // Negative color does not create a new comm object. We needed to take part in the allgather, but we're done now.
      if (job->color == PCCL_SPLIT_NOCOLOR) goto exit;
    }

    // child hash obtained from (parent hash, child count, color)
    uint64_t hacc[2] = {1, 1};
    eatHash(hacc, &job->parent->commHash);
    eatHash(hacc, &job->childCount);
    eatHash(hacc, &job->color);
    comm->commHash = digestHash(hacc);
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, job->parent, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    INFO(NCCL_INIT, "pcclCommSplit comm %p rank %d nranks %d hggcDev %d busId %lx parent %p childCount %d color %d key %d - Init START",
         comm, comm->rank, comm->nRanks, comm->hggcDev, comm->busId, job->parent, job->childCount, job->color, job->key);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
  } else {
    if (job->isGrow) {
      struct ncclBootstrapHandle* growHandle = (struct ncclBootstrapHandle*)job->commId;
      uint64_t baseMagic = growHandle ? growHandle->magic : hashCombine(job->parent->magic, job->parent->childCount);
      comm->commHash = hashCombine(baseMagic, job->nranks);
    } else {
      comm->commHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);
    }
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, NULL, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    comm->isGrow = job->isGrow;
    INFO(NCCL_INIT,"pcclCommInitRank comm %p rank %d nranks %d hggcDev %d busId %lx commHash %lx - Init START",
    comm, comm->rank, comm->nRanks, comm->hggcDev, comm->busId, comm->commHash);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapInit(job->nId, job->commId, comm, job->parent), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
  }

  NCCLCHECKGOTO(initTransportsRank(comm, job->parent, timers), res, fail);

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  {
    uint systemDriverVersion = 0;
    NCCLCHECKGOTO(ncclSystemDriverVersion(&systemDriverVersion), res, fail);
    if (ncclCeSupported(comm) && (ncclCeOkiRequested() || pcclParamP2pDmaEnable())) {
      // support user compile install kmd version check for 0xffffffff
      if (systemDriverVersion < PCCL_MIN_KMD_VERSION_FOR_OKI) {
        INFO(NCCL_INIT, "disable OKI as system driver version smaller than %x(%x).", PCCL_MIN_KMD_VERSION_FOR_OKI, systemDriverVersion);
      } else if (comm->nRanks < 2) {
        // OKI needs at least one remote QP group.
        INFO(NCCL_INIT, "disable OKI as nRanks %d is below the minimum 2.", comm->nRanks);
      } else if (comm->nRanks > PCCL_OKI_MAX_LOCAL_RANKS) {
        // The batch op arrays are passed to the kernels by value with this fixed capacity.
        INFO(NCCL_INIT, "disable OKI as nRanks %d exceeds max %d.", comm->nRanks, PCCL_OKI_MAX_LOCAL_RANKS);
      } else if (ppuDiesNum > PCCL_CE_MAX_DIE_NUM) {
        // Kernels and fence arrays support only one or two dies.
        INFO(NCCL_INIT, "disable OKI as %d dies exceeds max %d.", ppuDiesNum, PCCL_CE_MAX_DIE_NUM);
      } else {
        comm->okiSupport = true;
        INFO(NCCL_INIT, "enable OKI");
      }
    }
  }

  if (comm->okiSupport) {
    LT_OKI_CHECK(ltOkiOpenDevice(&comm->okiDevice, comm->hggcDev));

    // Local D2D queues use a separate pool from the capped remote ICN rings.
    int remotePairs = std::min(comm->nRanks - 1, LT_OKI_DMA_RING_PER_DIE);
    int maxQPNum = ppuDiesNum + remotePairs * ppuDiesNum;
    NCCLCHECKGOTO(ncclHggcHostCalloc(&comm->okiQps, maxQPNum, &(comm->pinnedMemUsed)), res, fail);

    // qps[slot * ppuDiesNum + die]: slot 0 uses local queueId=die; remote queueId=die*LT_OKI_DMA_RING_PER_DIE+slot-1.
    for (int i = 0; i < maxQPNum; i++) {
      int slot = i / ppuDiesNum;
      int dieId = i % ppuDiesNum;
      bool isRemote = (slot != 0);
      int queueId = isRemote ? dieId * LT_OKI_DMA_RING_PER_DIE + (slot - 1) : dieId;
      LT_OKI_CHECKGOTO(ltOkiCreateQP(&comm->okiQps[i], comm->okiDevice, isRemote, queueId, PCCL_LT_OKI_QP_DEPTH), res, fail);
      comm->okiQpCount += 1;
    }
    // Validate one-warp-per-group launch limits once at init.
    {
      int numQpGroups = comm->okiQpCount / ppuDiesNum;
      if (numQpGroups < 2) {
        ERROR("OKI: need at least 2 QP groups (local + remote), got okiQpCount %d ppuDiesNum %d",
              comm->okiQpCount, ppuDiesNum);
        res = ncclInternalError;
        goto fail;
      }
      if (WARP_SIZE * numQpGroups > PCCL_OKI_MAX_THREADS) {
        ERROR("OKI: %d QP groups need %d threads, exceeding the %d hggc limit",
              numQpGroups, WARP_SIZE * numQpGroups, PCCL_OKI_MAX_THREADS);
        res = ncclInternalError;
        goto fail;
      }
    }
    INFO(NCCL_INIT, "OKI QPs: %d total = (1 local + %d remote) x %d dies",
         maxQPNum, remotePairs, ppuDiesNum);
    // Allocate PCCL-managed dependency fence address, used as the cross-collective
    // trigger point (written by gpuBarrierWithTrigger, read by okiMemcpyBatchAsync).
    NCCLCHECKGOTO(ncclHggcCalloc(&comm->okiDependencyAddr, 1, &comm->devMemUsed, comm->memManager), res, fail);
  }
#endif

  // add to process comm list and register comm dump.
  {
    pthread_mutex_lock(&commArrayLock);
    bool added = false;
    for (int i = 0; i < MAX_COMMS_PER_PROCESS; i++) {
      if (commArray[i] == NULL) {
        commArray[i] = comm;
        added = true;
        break;
      }
    }
    pthread_mutex_unlock(&commArrayLock);
    if (!added) {
      VERBOSE(NCCL_INIT, "More than %d comms in one process! comm(%p) will not be added to comm dump list.", MAX_COMMS_PER_PROCESS, comm);
    }
  }

  if (sig != -1) signal(sig, pcclDumpCommList);

  timers[TIMER_INIT_TOTAL] = clockNano() - timers[TIMER_INIT_TOTAL];

  // Trace this call for replay tool
  if (job->parent) {
    /* unlink child abort flag. */
    __atomic_store_n(&job->parent->childAbortFlag, NULL, __ATOMIC_RELEASE);
    if (job->isGrow) {
      TRACE_CALL("pcclCommGrow(%p, %d, %p)", job->parent, comm->nRanks, comm);
      INFO(NCCL_INIT,"pcclCommGrow comm %p rank %d nranks %d hggcDev %d busId %lx parent %p commHash %lx maxConnType %d - Init COMPLETE",
      comm, comm->rank, comm->nRanks, comm->hggcDev, comm->busId, job->parent, comm->commHash, comm->maxConnType);
    } else {
      TRACE_CALL("pcclCommSplit(%p, %d, %d, %p, %d, %d)",
                  job->parent, job->color, job->key, comm, comm->rank, comm->nRanks);
      INFO(NCCL_INIT,"pcclCommSplit comm %p rank %d nranks %d hggcDev %d busId %lx parent %p color %d key %d commHash %lx maxConnType %d - Init COMPLETE",
      comm, comm->rank, comm->nRanks, comm->hggcDev, comm->busId, job->parent, job->color, job->key, comm->commHash, comm->maxConnType);
    }
  } else {
    TRACE_CALL("pcclCommInitRank(%p, %d, 0x%lx, %d, %d)",
                comm, comm->nRanks, comm->commHash, comm->rank, comm->hggcDev);
    INFO(NCCL_INIT,"pcclCommInitRank comm %p rank %d nranks %d hggcDev %d busId %lx commHash %lx maxConnType %d - Init COMPLETE",
    comm, comm->rank, comm->nRanks, comm->hggcDev, comm->busId, comm->commHash, comm->maxConnType);
  }

  VERBOSE(NCCL_INIT|NCCL_PROFILE,"Init timings: rank %d nranks %d total %.2f (kernels %.2f, alloc %.2f, bootstrap %.2f, allgathers %.2f, topo %.2f, graphs %.2f, connections %.2f, rest %.2f)", comm->rank, comm->nRanks,
    timers[TIMER_INIT_TOTAL]/1e9,
    timers[TIMER_INIT_KERNELS]/1e9,
    timers[TIMER_INIT_ALLOC]/1e9,
    timers[TIMER_INIT_BOOTSTRAP]/1e9,
    timers[TIMER_INIT_ALLGATHER]/1e9,
    timers[TIMER_INIT_TOPO]/1e9,
    timers[TIMER_INIT_GRAPHS]/1e9,
    timers[TIMER_INIT_CONNECT]/1e9,
    (timers[TIMER_INIT_TOTAL]-timers[TIMER_INIT_KERNELS]-timers[TIMER_INIT_ALLOC]-timers[TIMER_INIT_BOOTSTRAP]-timers[TIMER_INIT_ALLGATHER]-timers[TIMER_INIT_TOPO]-timers[TIMER_INIT_GRAPHS]-timers[TIMER_INIT_CONNECT])/1e9);

  INFO(NCCL_PROFILE,"Init timings: rank %d nranks %d total %.2f", comm->rank, comm->nRanks, timers[TIMER_INIT_TOTAL]/1e9);
  INFO(NCCL_ALLOC, "comm %p totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d", comm, comm->devMemUsed, comm->sharedRes->devMemUsed, comm->config.splitShare, comm->nRanks, comm->rank);
  INFO(NCCL_ALLOC, "comm %p totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed, comm->sharedRes->pinnedMemUsed, comm->config.splitShare, comm->nRanks, comm->rank);

  // update communicator state
  comm->initState = ncclSuccess;

exit:
  if (job->newcomm) {
    /* assign it to user pointer. */
    __atomic_store_n(job->newcomm, comm, __ATOMIC_RELEASE);
  }
  NCCLCHECK(ncclFree(parentRanks));
  return res;
fail:
#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  if (comm->okiSupport) NCCLCHECK(pcclOkiDestroyResources(comm));
#endif

  comm->initState = res;
  goto exit;
}

#define NCCL_CONFIG_DEFAULT(config, field, undef, defvalue, fieldStr, format) \
  if (config->field == undef) { \
    config->field = defvalue; \
  } else { \
    INFO(NCCL_ENV, "Comm config " fieldStr " set to " format, config->field); \
  }

static ncclResult_t envConfigOverride(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  const char* netName = comm->config.netName;
  const char* envNetName = pcclGetEnvWithDualPrefix("PCCL_NET");
  if (envNetName != nullptr) netName = envNetName;
  // Each communicator owns its copy, including configs inherited by split.
  comm->config.netName = nullptr;
  if (netName != nullptr) {
    char* ownedNetName = nullptr;
    size_t size = strlen(netName) + 1;
    NCCLCHECK(ncclCalloc(&ownedNetName, size));
    memcpy(ownedNetName, netName, size);
    comm->config.netName = ownedNetName;
  }
  int blockingEnv;
  int minCTAsEnv;
  int maxCTAsEnv;
  int splitShareEnv;
  int ctaPolicyEnv;
  int shrinkShareEnv;

  /* override configuration from env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv == 0 || blockingEnv == 1)
    comm->config.blocking = blockingEnv;

  minCTAsEnv = ncclParamMinCTAs();
  if (minCTAsEnv != PCCL_CONFIG_UNDEF_INT) {
    if (minCTAsEnv <= 0)
      INFO(NCCL_ENV, "PCCL_MIN_CTAS %d is too low, leaving it set at %d", minCTAsEnv, comm->config.minCTAs);
    else
      comm->config.minCTAs = minCTAsEnv;
  }

  maxCTAsEnv = ncclParamMaxCTAs();
  if (maxCTAsEnv != PCCL_CONFIG_UNDEF_INT) {
    if (maxCTAsEnv <= 0)
      INFO(NCCL_ENV, "PCCL_MAX_CTAS %d is too low, leaving it set at %d", maxCTAsEnv, comm->config.maxCTAs);
    else
      comm->config.maxCTAs = maxCTAsEnv;
  }

  splitShareEnv = ncclParamCommSplitShareResources();
  if (splitShareEnv != PCCL_CONFIG_UNDEF_INT) {
    comm->config.splitShare = splitShareEnv;
  }

  shrinkShareEnv = ncclParamCommShrinkShareResources();
  if (shrinkShareEnv != PCCL_CONFIG_UNDEF_INT) {
    if (comm->config.shrinkShare != PCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config shrinkShare reset to PCCL_COMM_SHRINK_SHARE_RESOURCES=%d", shrinkShareEnv);
    comm->config.shrinkShare = shrinkShareEnv;
  }

  ctaPolicyEnv = ncclParamCtaPolicy();
  if (ctaPolicyEnv != PCCL_CONFIG_UNDEF_INT) {
    comm->config.CTAPolicy = ctaPolicyEnv;
  }

  /* cap channels if needed */
  if (comm->config.minCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "minCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.minCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.minCTAs = MAXCHANNELS;
  }

  if (comm->config.maxCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "maxCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.maxCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.maxCTAs = MAXCHANNELS;
  }

  if (comm->config.minCTAs > comm->config.maxCTAs) {
    INFO(NCCL_ENV, "minCTAs %d is larger than maxCTAs %d, set both to %d", comm->config.minCTAs, comm->config.maxCTAs, comm->config.maxCTAs);
    comm->config.minCTAs = comm->config.maxCTAs;
  }

  if (comm->config.splitShare != 1 && comm->config.splitShare != 0) {
    WARN("splitShare %d is not a valid value 0/1, set it to 0", comm->config.splitShare);
    comm->config.splitShare = 0;
  }

  if (comm->config.CTAPolicy < PCCL_CTA_POLICY_DEFAULT || comm->config.CTAPolicy > PCCL_CTA_POLICY_ZERO) {
    INFO(NCCL_ENV, "CTAPolicy %d is not a valid value, set it to %d", comm->config.CTAPolicy, PCCL_CTA_POLICY_DEFAULT);
    comm->config.CTAPolicy = PCCL_CTA_POLICY_DEFAULT;
  }
  return ret;
}

static ncclResult_t copyCommConfig(ncclComm_t childComm, ncclComm_t parentComm) {
  memcpy(&childComm->config, &parentComm->config, sizeof(ncclConfig_t));
  NCCLCHECK(envConfigOverride(childComm));
  return ncclSuccess;
}

static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t defaultConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr;
  size_t realSize;

  internalConfig.magic = 0;
  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);

    unsigned int pcclVersion = 0;
    NCCLCHECKGOTO(pcclConfigCheck(internalConfigPtr->version, internalConfigPtr->magic, &pcclVersion), ret, fail);
    if (pcclVersion < PCCL_VERSION(1, 4, 0)) {
      internalConfigPtr->blocking = defaultConfig.blocking;
      internalConfigPtr->cgaClusterSize = defaultConfig.cgaClusterSize;
      internalConfigPtr->minCTAs = defaultConfig.minCTAs;
      internalConfigPtr->maxCTAs = defaultConfig.maxCTAs;
      internalConfigPtr->netName = defaultConfig.netName;
      internalConfigPtr->splitShare = defaultConfig.splitShare;
    }

    if (pcclVersion < PCCL_VERSION(2, 0, 0)) {
      internalConfigPtr->trafficClass = defaultConfig.trafficClass;
      internalConfigPtr->commName = defaultConfig.commName;
      internalConfigPtr->collnetEnable = defaultConfig.collnetEnable;
      internalConfigPtr->CTAPolicy = defaultConfig.CTAPolicy;
      internalConfigPtr->shrinkShare = defaultConfig.shrinkShare;
      internalConfigPtr->nvlsCTAs = defaultConfig.nvlsCTAs;
    }
  }

  /* check input config attributes, -1 means user-undefined and we should use default value from NCCL. */
  if (internalConfigPtr->blocking != PCCL_CONFIG_UNDEF_INT && internalConfigPtr->blocking != 0 && internalConfigPtr->blocking != 1) {
    ERROR("Invalid config blocking attribute value %d", internalConfigPtr->blocking);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->splitShare != PCCL_CONFIG_UNDEF_INT && internalConfigPtr->splitShare != 0 && internalConfigPtr->splitShare != 1) {
    ERROR("Invalid config splitShare attribute value %d", internalConfigPtr->splitShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->CTAPolicy != PCCL_CONFIG_UNDEF_INT && (internalConfigPtr->CTAPolicy < PCCL_CTA_POLICY_DEFAULT ||
    internalConfigPtr->CTAPolicy > PCCL_CTA_POLICY_ZERO)) {
    ERROR("Invalid config policy attribute value %d", internalConfigPtr->CTAPolicy);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->shrinkShare != PCCL_CONFIG_UNDEF_INT && internalConfigPtr->shrinkShare != 0 && internalConfigPtr->shrinkShare != 1) {
    ERROR("Invalid config shrinkShare attribute value %d", internalConfigPtr->shrinkShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* default config value can be tuned on different platform. */
  NCCL_CONFIG_DEFAULT(internalConfigPtr, blocking, PCCL_CONFIG_UNDEF_INT, 1, "Blocking", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, minCTAs, PCCL_CONFIG_UNDEF_INT, 1, "Min CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxCTAs, PCCL_CONFIG_UNDEF_INT, MAXCHANNELS, "Max CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, netName, PCCL_CONFIG_UNDEF_PTR, NULL, "Net name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, splitShare, PCCL_CONFIG_UNDEF_INT, 0, "Split share", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, commName, PCCL_CONFIG_UNDEF_PTR, NULL, "Comm name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, CTAPolicy, PCCL_CONFIG_UNDEF_INT, PCCL_CTA_POLICY_DEFAULT, "CTA policy flags", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, shrinkShare, PCCL_CONFIG_UNDEF_INT, 0, "shrinkShare", "%d");

  /* assign config to communicator */
  comm->config.blocking = internalConfigPtr->blocking;
  comm->config.splitShare = internalConfigPtr->splitShare;
  comm->config.commName = internalConfigPtr->commName;
  comm->config.CTAPolicy = internalConfigPtr->CTAPolicy;
  comm->config.minCTAs = internalConfigPtr->minCTAs;
  comm->config.maxCTAs = internalConfigPtr->maxCTAs;
  comm->config.netName = internalConfigPtr->netName;
  comm->config.shrinkShare = internalConfigPtr->shrinkShare;

  NCCLCHECK(envConfigOverride(comm));

end:
  return ret;
fail:
  goto end;
}

static void ncclCommInitJobFree(void* _job) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)_job;
  ncclFree(job->commId);
  ncclFree(_job);
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, int nId, ncclUniqueId* commId, int myrank, int hggcDev, ncclConfig_t *config) {
  ncclResult_t res = ncclSuccess;
  const char* commIdEnv = NULL;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob *job = NULL;
  bool launchedJob = false;

  NCCLCHECKGOTO(ncclInit(), res, fail);

  if (myrank == 0) {
    if(showVersion() != ncclSuccess) WARN("Get version error");
  }

  // Make sure the HGGC runtime is initialized.
  HGGCCHECKGOTO(hggcFree(NULL), res, fail);

  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, fail);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclHggcHostCalloc((uint32_t**)&comm->abortFlagDev, 1, &(comm->pinnedMemUsed)), res, fail);
  // set up comm state and abortFlag only
  comm->hostDevComm.comm.abortFlag = comm->abortFlagDev;
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlagRefCount, 1), res, fail);
  *comm->abortFlagRefCount = 1;
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);
  /* start with ncclInProgress and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInProgress;
  *newcomm = comm;

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->nId = nId;
  job->comm = comm;
  job->nranks = nranks;
  job->myrank = myrank;
  job->hggcDev = hggcDev;
  // need to copy the commIds to allow async commInit and to avoid alignement issues when casting from ncclUNiqueId and ncclBootstrapHandle
  // ncclUniqueIds and ncclBootstrapHandle don't have the same alignment requirements.
  // Therefore the array of Ids coming from the user might not be properly aligned to be cast into a ncclBootstrapHandle
  // copying into allocated memory guarantees that the memory is properly aligned for any objects, removing that issue
  NCCLCHECKGOTO(ncclCalloc(&job->commId, nId), res, fail);
  memcpy(job->commId, commId, nId * NCCL_UNIQUE_ID_BYTES);

  commIdEnv = pcclGetEnvWithDualPrefix("PCCL_COMM_ID");
  if (commIdEnv && myrank == 0) {
    INFO(NCCL_ENV, "PCCL_COMM_ID set by environment to %s", commIdEnv);
    if (nId > 1) {
      INFO(NCCL_INIT | NCCL_ENV, "PCCL_COMM_ID cannot be used with more than one pcclUniqueId");
      job->nId = 1;
    }
    NCCLCHECKGOTO(bootstrapCreateRoot((struct ncclBootstrapHandle*)&job->commId[0], true), res, fail);
  }
  launchedJob = true;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm), res, fail);

exit:
  if (ncclAsyncMode()) return ncclGroupErrCheck(res);
  else return res;
fail:
  if (job && !launchedJob) ncclCommInitJobFree(job);
  if (comm) {
    NCCLCHECK(ncclFree(comm->abortFlag));
    ncclHggcHostFree(comm->abortFlagDev);
    NCCLCHECK(ncclFree(comm->abortFlagRefCount));
    free((void*)comm->config.netName);
    comm->config.netName = nullptr;
    NCCLCHECK(ncclFree(comm));
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t pcclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  HGTX3_RANGE(PcclHgtxParamsCommInitRank);
  // Load the HGGC driver and dlsym hooks (can fail on old drivers)
  (void)ncclHggcLibraryInit();

  int hggcDev;
  HGGCCHECK(hggcGetDevice(&hggcDev));
  INFO_CALL("PCCL API: pcclCommInitRank(newcomm=%p, nranks=%d, commId=%p, myrank=%d)", newcomm, nranks, (void *)&commId, myrank);
  uint64_t commHash = getHash(commId.internal, NCCL_UNIQUE_ID_BYTES);
  ncclPerCommHashDebugInit(commHash);

  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, hggcDev, NULL));
  HGTX3_RANGE_ADD_PAYLOAD(CommInitRank, PcclHgtxParamsCommInitRankSchema,
    HGTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, hggcDev));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t pcclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  ncclResult_t ret = ncclSuccess;
  uint64_t commHash = 0UL;
  HGTX3_RANGE(PcclHgtxParamsCommInitAll);

  // Load the HGGC driver and dlsym hooks (can fail on old drivers)
  (void)ncclHggcLibraryInit();

  char *devlist_str = NULL;
  int offset = 0;
  NCCLCHECKGOTO(ncclCalloc(&devlist_str, 128), ret, fail);
  for (int i = 0; i < ndev; i++) {
    int n = snprintf(devlist_str + offset, 128 - offset, i > 0 ? " %d" : "%d", devlist ? devlist[i] : i);
    if (n < 0 || n >= 128 - offset) break;
    offset += n;
  }
  INFO_CALL("PCCL API: pcclCommInitAll(comms=%p, ndev=%d, devlist=[%s])", comms, ndev, devlist_str);

  NCCLCHECKGOTO(PtrCheck(comms, "CommInitAll", "comms"), ret, fail);
  if (ndev < 0) {
    ERROR("Invalid device count requested : %d", ndev);
    ret = ncclInvalidArgument;
    goto fail;
  }

  ncclUniqueId uniqueId;
  NCCLCHECKGOTO(pcclGetUniqueId(&uniqueId), ret, fail);

  commHash = getHash(uniqueId.internal, NCCL_UNIQUE_ID_BYTES);
  ncclPerCommHashDebugInit(commHash);

  NCCLCHECKGOTO(pcclGroupStart(), ret, fail);
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    ncclCommInitRankDev(comms+i, ndev, 1, &uniqueId, i, devlist ? devlist[i] : i, NULL);
  }
  NCCLCHECKGOTO(pcclGroupEnd(), ret, fail);
  HGTX3_RANGE_ADD_PAYLOAD(CommInitAll, PcclHgtxParamsCommInitAllSchema,
    HGTX3_PAYLOAD(comms[0]->commHash, ndev));
exit:
  NCCLCHECK(ncclFree(devlist_str));
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t pcclCommInitRankConfig(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  int hggcDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;
  uint64_t commHash = getHash(commId.internal, NCCL_UNIQUE_ID_BYTES);

  INFO_CALL("PCCL API: pcclCommInitRankConfig(newcomm=%p, nranks=%d, commId=%p, myrank=%d, config=%p)", newcomm, nranks, (void *)&commId, myrank, config);
  HGTX3_RANGE(PcclHgtxParamsCommInitRankConfig);
  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclHggcLibraryInit();

  HGGCCHECKGOTO(hggcGetDevice(&hggcDev), ret, exit);

  ncclPerCommHashDebugInit(commHash);

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, hggcDev, internalConfigPtr), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if(!(*newcomm)->config.blocking) {
      (void) pcclCommGetAsyncError(*newcomm, &ret);
    }
    HGTX3_RANGE_ADD_PAYLOAD(CommInitRankConfig, PcclHgtxParamsCommInitRankSchema,
      HGTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, hggcDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommInitRankScalable, ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config);
ncclResult_t pcclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config) {
  HGTX3_RANGE(PcclHgtxParamsCommInitRankScalable);
  int hggcDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;
  uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);

  INFO_CALL("PCCL API: pcclCommInitRankScalable(newcomm=%p, nranks=%d, nId=%d, commId=%p, myrank=%d, config=%p)", newcomm, nranks, nId, (void *)&commId, myrank, config);
  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclHggcLibraryInit();
  HGGCCHECKGOTO(hggcGetDevice(&hggcDev), ret, exit);

  ncclPerCommHashDebugInit(commHash);

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, nId, commId, myrank, hggcDev, internalConfigPtr), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void) pcclCommGetAsyncError(*newcomm, &ret);
    }
    HGTX3_RANGE_ADD_PAYLOAD(CommInitRankScalable, PcclHgtxParamsCommInitRankSchema,
      HGTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, hggcDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  int savedDevice = -1;
  int commDevice = comm->hggcDev;
  ncclResult_t ret = ncclSuccess;

  HGGCCHECKGOTO(hggcGetDevice(&savedDevice), ret, fail);
  if (savedDevice != commDevice) {
    HGGCCHECKGOTO(hggcSetDevice(commDevice), ret, fail);
  }

  VERBOSE(NCCL_DESTROY, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag, comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->hostStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync hostStream error %d", comm, comm->rank, ret);
    }
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync deviceStream error %d", comm, comm->rank, ret);
    }
  }

  NCCLCHECK(ncclCommPollCallbacks(comm, false));
  // And keep polling until all graphs referencing us die.
  while (comm->localPersistentRefs != 0) {
    NCCLCHECK(ncclCommPollCallbacks(comm, /*waitSome=*/true));
  }

  while (!ncclIntruQueueEmpty(&comm->legacyRegCleanupQueue)) {
    struct ncclCommCallback* cb = ncclIntruQueueDequeue(&comm->legacyRegCleanupQueue);
    if (cb->fn(comm, cb) != ncclSuccess) {
      WARN("Legacy IPC cleanup callback failed comm %p (rank = %d) cb %p", comm, comm->rank, cb);
    }
  }

  if (savedDevice != commDevice) {
    HGGCCHECKGOTO(hggcSetDevice(savedDevice), ret, fail);
  }

  if ((ret = ncclProxyStop(comm)) != ncclSuccess) {
    WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, ret);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t commCleanup(ncclComm_t comm) {
  int savedDevice = -1;
  int commDevice = comm->hggcDev;

  HGGCCHECK(hggcGetDevice(&savedDevice));
  if (savedDevice != commDevice) {
    HGGCCHECK(hggcSetDevice(commDevice));
  }

  if (comm->tuner) {
    NCCLCHECK(comm->tuner->destroy());
  }
  NCCLCHECK(ncclLoadTunerPlugin(&comm->tuner));

  if (comm->extKernel) {
    NCCLCHECK(destroyExtKernelPlugin(comm));
  }

#if PCCL_HGGC_VERSION >= PCCL_HGGC_DRIVER_VERSION(2, 2, 0)
  if (comm->okiSupport) NCCLCHECK(pcclOkiDestroyResources(comm));
#endif

  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice) {
    HGGCCHECK(hggcSetDevice(savedDevice));
  }

  return ncclSuccess;
}

static ncclResult_t commFinalize(ncclComm_t comm, bool userCalled) {
  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob *job = NULL;

  comm->finalizeCalled = true;
  /* launch async thread to finalize comm. */
  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;

  if (userCalled) {
    NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, commDestroySync, NULL, free, comm), ret, exit);
  } else {
    NCCLCHECKGOTO(commDestroySync(&job->base), ret, fail);
    NCCLCHECK(ncclFree(job));
  }

exit:
  return ncclGroupErrCheck(ret);
fail:
  ncclFree(job);
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommFinalize, ncclComm_t comm);
ncclResult_t pcclCommFinalize(ncclComm_t comm) {
  HGTX3_RANGE(PcclHgtxParamsCommFinalize);
  ncclResult_t ret = ncclSuccess;

  INFO_CALL("PCCL API: pcclCommFinalize(comm=%p)", comm);
  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* wait comm ready before finalize. */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* prevent double finalize. */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* finalize comm. */
  ret = commFinalize(comm, true);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
    if (comm) {
      if (!comm->config.blocking) {
        NCCLCHECK(pcclCommGetAsyncError(comm, &ret));
      }
      HGTX3_RANGE_ADD_PAYLOAD(CommFinalize, PcclHgtxParamsCommFinalizeSchema,
        HGTX3_PAYLOAD(comm->commHash));
  }
  return ret;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;
  ncclResult_t state;
  int curRank; /* Debug info */

  NCCLCHECKGOTO(pcclCommGetAsyncError(comm, &state), ret, fail);
  VERBOSE(NCCL_INIT, "commReclaim: reclaim comm %p rank %d state %d", comm, comm->rank, state);
  if (state == ncclSuccess && __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) == 0 && comm->finalizeCalled == false) {
    /* user does not call ncclCommFinalize and this is a normal comm destroy. ncclCommDestroy
     * should be nonblocking until last call of ncclCommDestroy. */
    NCCLCHECKGOTO(commFinalize(comm, false), ret, fail);
  }

  if (comm->initState != ncclSuccess) {
    /* if init errors happen, no finalize thread should have been launched. Main thread can reclaim
     * everything since no NCCL kernel was issued. */
    struct ncclCommFinalizeAsyncJob finalizeJob;

    finalizeJob.comm = comm;
    curRank = comm->rank;
    /* comm aborts, commDestroySync should not be blocked. */
    if ((ret = commDestroySync((struct ncclAsyncJob*) &finalizeJob)) != ncclSuccess) {
      ERROR("commReclaim: comm %p (rank = %d) in abort, error %d", comm, curRank, ret);
    }

    if ((ret = commCleanup(comm)) != ncclSuccess) {
      // We pass a freed pointer, but we don't dereference; we merely print its value, so it's OK.
      // coverity[pass_freed_arg]
      ERROR("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", comm, curRank, ret);
    }
  } else {
    int curRankCnt;
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    curRankCnt = __atomic_add_fetch(finalizeRankCnt, 1, __ATOMIC_ACQ_REL);
    if (curRankCnt == intraRanks) {
      ncclComm_t curIntraComm;
      ncclComm_t nextIntraComm = intracomm0;

      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if (curIntraComm->finalizeCalled == false) {
          struct ncclCommFinalizeAsyncJob finalizeJob;
          finalizeJob.comm = curIntraComm;
          /* every comm aborts, commDestroySync should not be blocked. */
          if ((ret = commDestroySync((struct ncclAsyncJob*) &finalizeJob)) != ncclSuccess)
            ERROR("commReclaim: comm %p (rank = %d) in abort, error %d", curIntraComm, curRank, ret);
        }
      }

      /* free local resources. */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          // We pass a freed pointer, but we don't dereference; we merely print its value, so it's OK.
          // coverity[pass_freed_arg]
          ERROR("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommDestroy, ncclComm_t comm);
ncclResult_t pcclCommDestroy(ncclComm_t comm) {
  INFO_CALL("PCCL API: pcclCommDestroy(comm=%p)", comm);
  if (comm == NULL) {
    PCCL_HGTX3_FUNC_RANGE;
    return ncclSuccess;
  }
  int rank = comm->rank, nranks = comm->nRanks, hggcDev = comm->hggcDev;
  int64_t busId = comm->busId;
  uintptr_t commAddress = reinterpret_cast<uintptr_t>(comm);
  struct ncclCommFinalizeAsyncJob* job = NULL;
  ncclResult_t res = ncclSuccess;

  HGTX3_FUNC_WITH_PARAMS(CommDestroy, PcclHgtxParamsCommInitRank,
    HGTX3_PAYLOAD(comm->commHash, nranks, rank, hggcDev));
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks == -1 || comm->hggcDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  ncclPerCommHashDebugClose();

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, commReclaim, NULL, free, comm), res, fail);

exit:
  INFO(NCCL_DESTROY,"comm 0x%lx rank %d nranks %d hggcDev %d busId %lx - Destroy COMPLETE", (unsigned long)commAddress, rank, nranks, hggcDev, busId);
  // coverity[leaked_storage]
  return res;
fail:
  goto exit;
}

static ncclResult_t setCommAbortFlags(ncclComm_t comm, int value) {
  // Set abort flags
  // Ask anything that might still be running on the device to quit
  if (comm->childAbortFlag != NULL) {
    __atomic_store_n(comm->childAbortFlag, value, __ATOMIC_RELEASE);
    __atomic_store_n(comm->childAbortFlagDev, value, __ATOMIC_RELEASE);
  }
  __atomic_store_n(comm->abortFlag, value, __ATOMIC_RELEASE);
  __atomic_store_n(comm->abortFlagDev, value, __ATOMIC_RELEASE);
  return ncclSuccess;
}

static ncclResult_t commRevokeAsync(struct ncclAsyncJob* job_) {
  struct ncclCommRevokeAsyncJob* job = (struct ncclCommRevokeAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;
  NCCLCHECKGOTO(PtrCheck(comm, "CommRevokeAsync", "comm"), ret, exit);
  INFO(NCCL_DESTROY, "CommRevokeAsync START comm %p rank %d nranks %d nNodes %d localRank %d hggcDev %d", comm,
       comm->rank, comm->nRanks, comm->nNodes, comm->localRank, comm->hggcDev);
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), ret, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->hostStream), ret, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), ret, exit);
  NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/true), ret, exit);
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), ret, exit);
  {
    ncclResult_t tmpRet = ncclSuccess;
    if ((tmpRet = ncclProxyStop(comm)) != ncclSuccess) {
      WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, tmpRet);
    }
    if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread) {
      PTHREADCHECK(pthread_join(comm->proxyState->thread, nullptr), "pthread_join");
      comm->proxyState->thread = 0;
      if (comm->proxyState->threadUDS) {
        PTHREADCHECK(pthread_join(comm->proxyState->threadUDS, nullptr), "pthread_join");
        comm->proxyState->threadUDS = 0;
      }
    }
  }
  NCCLCHECKGOTO(setCommAbortFlags(comm, 0), ret, exit);
exit:
  (void)ncclCommSetAsyncError(comm, ret);
  INFO(NCCL_DESTROY, "CommRevokeAsync END comm %p result %d", comm, ret);
  return ret;
}

NCCL_API(ncclResult_t, pcclCommRevoke, ncclComm_t comm, int revokeFlags);
ncclResult_t pcclCommRevoke(ncclComm_t comm, int revokeFlags) {
  HGTX3_RANGE(PcclHgtxParamsCommRevoke);

  INFO_CALL("PCCL API: pcclCommRevoke(comm=%p, revokeFlags=%d)", comm, revokeFlags);
  if (comm == NULL) {
    return ncclSuccess;
  }
  if (revokeFlags != PCCL_REVOKE_DEFAULT) {
    return ncclInvalidArgument;
  }
  if (comm->destroyFlag || comm->finalizeCalled) {
    return ncclInvalidArgument;
  }
  if (comm->revokedFlag) {
    return ncclInvalidArgument;
  }

  INFO(NCCL_DESTROY, "comm %p rank %d nranks %d hggcDev %d busId %lx commHash %lx - Revoke START", comm, comm->rank, comm->nRanks, comm->hggcDev, comm->busId, comm->commHash);

  NCCLCHECK(ncclGroupStartInternal());
  (void)setCommAbortFlags(comm, 1);
  comm->revokedFlag = 1;
  (void)ncclCommEnsureReady(comm);
  comm->finalizeCalled = true;

  int rank = comm->rank, nranks = comm->nRanks, hggcDev = comm->hggcDev;
  struct ncclCommRevokeAsyncJob* job = NULL;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, commRevokeAsync, NULL, free, comm), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(pcclCommGetAsyncError(comm, &ret));
    }
    HGTX3_RANGE_ADD_PAYLOAD(CommRevoke, PcclHgtxParamsCommInitRankSchema,
      HGTX3_PAYLOAD(comm->commHash, nranks, rank, hggcDev));
  }
  INFO(NCCL_DESTROY, "comm %p rank %d nranks %d hggcDev %d busId %lx commHash %lx - Revoke COMPLETE, result %d", comm, rank, nranks, hggcDev, comm->busId, comm->commHash, ret);
  return ret;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommAbort, ncclComm_t comm);
ncclResult_t pcclCommAbort(ncclComm_t comm) {
  HGTX3_RANGE(PcclHgtxParamsCommAbort);
  INFO_CALL("PCCL API: pcclCommAbort(comm=%p)", comm);
  INFO(NCCL_DESTROY, "pcclCommAbort!");
  if (comm == NULL)
    return ncclSuccess;

  struct ncclCommFinalizeAsyncJob* job = NULL;
  ncclResult_t res = ncclSuccess;

  const char* str = pcclGetEnvWithDualPrefix("PCCL_ENABLE_ABORT_DUMP");
  if (str) {
    // do not add NCCLCHECK, otherwise it may corrupt current state.
    pcclCommDumpMaskType dumpMask;
    NCCLVERIFY(getGlobalCommDumpMask(&dumpMask));
    ncclResult_t dumpRes = pcclCommDump(comm, dumpMask);
    if (dumpRes != ncclSuccess) {
      WARN("Failed to dump comm:%p!", comm);
    }
  }

  // Ask anything that might still be running on the device to quit
  if (pcclParamDisableAbort() == 0) {
    (void)setCommAbortFlags(comm, 1);
  }
  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  (void)ncclCommEnsureReady(comm);

  // once the comm is ready, we can access ranks etc
  int rank = comm->rank, nranks = comm->nRanks, hggcDev = comm->hggcDev;
  int64_t busId = comm->busId;
  uintptr_t commAddress = reinterpret_cast<uintptr_t>(comm);
  HGTX3_RANGE_ADD_PAYLOAD(CommAbort, PcclHgtxParamsCommInitRankSchema,
    HGTX3_PAYLOAD(comm->commHash, nranks, rank, hggcDev));

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, commReclaim, NULL, free, comm), res, fail);

exit:
  INFO(NCCL_DESTROY,"comm 0x%lx rank %d nranks %d hggcDev %d busId %lx - Abort COMPLETE", (unsigned long)commAddress, rank, nranks, hggcDev, busId);
  /* abort should never report an error to the user. */
  // coverity[leaked_storage]
  return ncclSuccess;
fail:
  goto exit;
}

static void childCommCleanupJob(void* job) {
  struct ncclCommInitRankAsyncJob* initJob = (struct ncclCommInitRankAsyncJob*)job;
  if (initJob->excludeRanksList) free(initJob->excludeRanksList);
  free(job);
}

// initializing a child communicator (for both split and shrink)
static ncclResult_t ncclCommInitChildComm(ncclComm_t comm, ncclComm_t* newcomm, bool isShrink, int flags, int color, int key, int* excludeRanksList, int excludeRanksCount,
                                          ncclConfig_t* config, const char* caller) {
  struct ncclCommInitRankAsyncJob *job = NULL;
  ncclComm_t childComm = PCCL_COMM_NULL;
  ncclResult_t res = ncclSuccess;

  int oldDev;
  HGGCCHECK(hggcGetDevice(&oldDev));
  NCCLCHECKGOTO(CommCheck(comm, caller, "comm"), res, exit);
  NCCLCHECKGOTO(PtrCheck(newcomm, caller, "newcomm"), res, exit);
  if (isShrink) {
    NCCLCHECKGOTO(PtrCheck(excludeRanksList, caller, "excludeRanksList"), res, exit);
    NCCLCHECKGOTO(excludeRanksCount > 0 ? ncclSuccess : ncclInvalidArgument, res, exit);
    // excludeRanksList may not be sorted, need to sort it
    qsort(excludeRanksList, excludeRanksCount, sizeof(int), compareInts);
    // ranks in excludeRanksList should not call into this function
    NCCLCHECKGOTO(bsearch(&comm->rank, excludeRanksList, excludeRanksCount, sizeof(int), compareInts) ? ncclInvalidArgument : ncclSuccess, res, exit);
  }
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, exit);
  HGGCCHECKGOTO(hggcSetDevice(comm->hggcDev), res, exit);

  /* *newcomm should be PCCL_COMM_NULL until comm split fully complete. */
  *newcomm = PCCL_COMM_NULL;
  if (!isShrink && color == PCCL_SPLIT_NOCOLOR) {
    INFO(NCCL_INIT, "Rank %d has color with PCCL_SPLIT_NOCOLOR, not creating a new communicator", comm->rank);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&childComm, 1), res, fail);

    // Set the shareResource field, this is used throughout the init and must be reset every time.
    // Never share resources if the parent communicator has been revoked.
    // If we shrink, we only reuse resources in default mode.
    comm->shareResources = !comm->revokedFlag && (isShrink ? (!(flags & PCCL_SHRINK_ABORT) && comm->config.shrinkShare) : comm->config.splitShare);
    if (comm->shareResources) {
      childComm->abortFlag = comm->abortFlag;
      childComm->abortFlagDev = comm->abortFlagDev;
      childComm->abortFlagRefCount = comm->abortFlagRefCount;
      comm->childAbortFlag = NULL;
      ncclAtomicRefCountIncrement(comm->abortFlagRefCount);
    } else {
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlag, 1), res, fail);
      NCCLCHECKGOTO(ncclHggcHostCalloc((uint32_t**)&childComm->abortFlagDev, 1, &(childComm->pinnedMemUsed)), res, fail);
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlagRefCount, 1), res, fail);
      /* temporarily used to abort everything during child comm init. */
      comm->childAbortFlag = childComm->abortFlag;
      comm->childAbortFlagDev = childComm->abortFlagDev;
      *childComm->abortFlagRefCount = 1;
    }
    if (config == NULL) {
      NCCLCHECKGOTO(copyCommConfig(childComm, comm), res, fail);
    } else {
      NCCLCHECKGOTO(parseCommConfig(childComm, config), res, fail);
    }

    /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
    childComm->initState = ncclInternalError;
  }
  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = childComm;
  job->newcomm = newcomm;
  job->parent = comm;
  job->color = color;
  job->key = key;
  if (excludeRanksList) {
    // need to copy the list of ranks to exclude because the job is async
    job->excludeRanksCount = excludeRanksCount;
    NCCLCHECKGOTO(ncclCalloc(&job->excludeRanksList, excludeRanksCount), res, fail);
    memcpy(job->excludeRanksList, excludeRanksList, excludeRanksCount * sizeof(int));
  } else {
    // each split has to lead to a unique comm, so increment the child count
    job->childCount = ++comm->childCount;
    job->excludeRanksList = NULL;
  }
  job->hggcDev = comm->hggcDev;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, /*undo=*/NULL, /*destructor=*/childCommCleanupJob, comm), res, fail);

exit:
  (void)hggcSetDevice(oldDev);
  return res;
fail:
  if (childComm) {
    if (!comm->shareResources) {
      NCCLCHECK(ncclFree(childComm->abortFlag));
      if (childComm->abortFlagDev) ncclHggcHostFree(childComm->abortFlagDev);
      if (childComm->abortFlagRefCount) NCCLCHECK(ncclFree(childComm->abortFlagRefCount));
    }
    free((void*)childComm->config.netName);
    childComm->config.netName = nullptr;
    NCCLCHECK(ncclFree(childComm));
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommShrink, ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t  pcclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm, ncclConfig_t* config, int shrinkFlags) {
  HGTX3_RANGE(PcclHgtxParamsCommShrink)
  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  // Handle error mode by setting abort flags and waiting for kernels to complete and unset the flags to avoid bootstrap issues
  if (shrinkFlags & PCCL_SHRINK_ABORT) {
    NCCLCHECKGOTO(setCommAbortFlags(comm, 1), res, exit);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
    NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
  }
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/true, shrinkFlags, /*color=*/0, /*key=*/comm->rank, excludeRanksList, excludeRanksCount, config, __func__), res, exit);

  if (*newcomm) HGTX3_RANGE_ADD_PAYLOAD(CommShrink, PcclHgtxParamsCommShrinkSchema, HGTX3_PAYLOAD(comm->commHash, comm->nRanks, comm->rank, comm->hggcDev, excludeRanksCount));

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
}

NCCL_API(ncclResult_t, pcclCommGetUniqueId, ncclComm_t comm, ncclUniqueId* uniqueId);
ncclResult_t pcclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  NCCLCHECK(PtrCheck(uniqueId, "CommGetUniqueId", "uniqueId"));

  struct ncclBootstrapHandle growHandle;
  NCCLCHECK(bootstrapGetUniqueId(&growHandle, comm));
  NCCLCHECK(bcastGrowHandle(&growHandle, comm, /*isRoot=*/true));
  memset(uniqueId, 0, sizeof(ncclUniqueId));
  memcpy(uniqueId, &growHandle, sizeof(growHandle));

  INFO_CALL("PCCL API: pcclCommGetUniqueId(comm=%p, out=0x%llx)", comm, (unsigned long long)getHash(uniqueId->internal, NCCL_UNIQUE_ID_BYTES));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclCommGrow, ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config);
ncclResult_t pcclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config) {
  HGTX3_RANGE(PcclHgtxParamsCommGrow)
  ncclResult_t res = ncclSuccess;
  bool isExistingRank = (comm != NULL);
  struct ncclCommInitRankAsyncJob* job = NULL;
  ncclComm_t newComm = NULL;
  struct ncclBootstrapHandle recvHandle;
  const void* growHandle = uniqueId;
  bool launchedJob = false;

  INFO_CALL("PCCL API: pcclCommGrow(comm=%p, nRanks=%d, uniqueId=%p, rank=%d, newcomm=%p, config=%p)", comm, nRanks, (void*)uniqueId, rank, (void*)newcomm, (void*)config);
  if (newcomm == NULL) return ncclInvalidArgument;
  if (nRanks <= 0) {
    ERROR("pcclCommGrow: total ranks must be positive, got %d", nRanks);
    return ncclInvalidArgument;
  }
  if (comm && nRanks <= comm->nRanks) {
    ERROR("pcclCommGrow: total ranks %d is less than current ranks %d", nRanks, comm->nRanks);
    return ncclInvalidArgument;
  }

  *newcomm = NULL;

  NCCLCHECK(ncclGroupStartInternal());

  if (isExistingRank) {
    NCCLCHECKGOTO(CommCheck(comm, __func__, "comm"), res, exit);
    NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, exit);
    if (rank != -1) {
      ERROR("pcclCommGrow: existing ranks must pass rank=-1, got %d", rank);
      res = ncclInvalidArgument;
      goto exit;
    }

    // each grow has to lead to a unique comm, so increment the child count
    ++comm->childCount;

    // Only boundary ranks (0 and N-1) receive grow handle from coordinator
    if (comm->rank == 0 || comm->rank == comm->nRanks - 1) {
      NCCLCHECKGOTO(bcastGrowHandle(&recvHandle, comm, /*isRoot=*/false), res, exit);
      if (recvHandle.magic != hashCombine(comm->magic, comm->childCount)) {
        ERROR("pcclCommGrow: magic mismatch computed by the root, got %lx expected %lx", recvHandle.magic,
              hashCombine(comm->magic, comm->childCount));
        res = ncclInvalidArgument;
        goto exit;
      }
      growHandle = &recvHandle;
    }
  } else {
    if (rank < 0) {
      ERROR("pcclCommGrow: new ranks must pass valid rank >= 0, got %d", rank);
      res = ncclInvalidArgument;
      goto exit;
    }
    if (uniqueId == NULL) {
      ERROR("pcclCommGrow: new ranks must pass non-NULL uniqueId");
      res = ncclInvalidArgument;
      goto exit;
    }
    if (rank >= nRanks) {
      ERROR("pcclCommGrow: new rank %d exceeds total ranks %d", rank, nRanks);
      res = ncclInvalidArgument;
      goto exit;
    }

    memcpy(&recvHandle, uniqueId, sizeof(recvHandle));
    if (recvHandle.nRanks <= 0 || recvHandle.nRanks >= nRanks || rank < recvHandle.nRanks) {
      ERROR("pcclCommGrow: invalid existing rank count %d for rank %d of %d", recvHandle.nRanks, rank, nRanks);
      res = ncclInvalidArgument;
      goto exit;
    }

    (void)ncclHggcLibraryInit();
    NCCLCHECKGOTO(ncclInit(), res, exit);
    if (showVersion() != ncclSuccess) WARN("Get version error");
    HGGCCHECKGOTO(hggcFree(NULL), res, exit);
  }

  NCCLCHECKGOTO(ncclCalloc(&newComm, 1), res, fail);

  NCCLCHECKGOTO(ncclCalloc(&newComm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclHggcHostCalloc((uint32_t**)&newComm->abortFlagDev, 1, &(newComm->pinnedMemUsed)), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&newComm->abortFlagRefCount, 1), res, fail);
  *newComm->abortFlagRefCount = 1;

  if (isExistingRank && config == NULL) {
    NCCLCHECKGOTO(copyCommConfig(newComm, comm), res, fail);
  } else {
    NCCLCHECKGOTO(parseCommConfig(newComm, config), res, fail);
  }

  newComm->initState = ncclInProgress;
  *newcomm = newComm;

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->nId = (growHandle != NULL) ? 1 : 0;
  if (job->nId == 1) {
    NCCLCHECKGOTO(ncclCalloc(&job->commId, 1), res, fail);
  }

  if (isExistingRank) {
    job->parent = comm;
    job->hggcDev = comm->hggcDev;
    job->myrank = comm->rank;
  } else {
    int device;
    HGGCCHECKGOTO(hggcGetDevice(&device), res, fail);
    job->parent = NULL;
    job->hggcDev = device;
    job->myrank = rank;
  }

  job->comm = newComm;

  if (job->nId == 1) memcpy(job->commId, growHandle, sizeof(struct ncclBootstrapHandle));
  job->nranks = nRanks;
  job->newcomm = newcomm;
  job->isGrow = true;
  job->color = 0;
  job->key = job->myrank;
  launchedJob = true;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, newComm), res, fail);

exit:
  if (*newcomm) {
    uint64_t parentHash = isExistingRank ? comm->commHash : 0;
    HGTX3_RANGE_ADD_PAYLOAD(CommGrow, PcclHgtxParamsCommGrowSchema,
      HGTX3_PAYLOAD((*newcomm)->commHash, parentHash, nRanks, (*newcomm)->rank, (*newcomm)->hggcDev));
  }
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;

fail:
  if (job && !launchedJob) ncclCommInitJobFree(job);
  if (newComm) {
    NCCLCHECK(ncclFree(newComm->abortFlag));
    if (newComm->abortFlagDev) ncclHggcHostFree(newComm->abortFlagDev);
    NCCLCHECK(ncclFree(newComm->abortFlagRefCount));
    free((void*)newComm->config.netName);
    newComm->config.netName = nullptr;
    NCCLCHECK(ncclFree(newComm));
    if (newcomm) *newcomm = NULL;
  }
  goto exit;
}

NCCL_API(ncclResult_t, pcclCommSplit, ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config);
ncclResult_t pcclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  HGTX3_RANGE(PcclHgtxParamsCommSplit)

  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/false, /*shrink mode=*/PCCL_SHRINK_DEFAULT, color, key, NULL, 0, config, __func__), res, exit);

  if (*newcomm)
    HGTX3_RANGE_ADD_PAYLOAD(CommSplit, PcclHgtxParamsCommSplitSchema, HGTX3_PAYLOAD((*newcomm)->commHash, comm->commHash, comm->nRanks, comm->rank, comm->hggcDev, color, key));

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
}

NCCL_API(const char*, pcclGetErrorString, ncclResult_t code);
const char* pcclGetErrorString(ncclResult_t code) {
  INFO_CALL("PCCL API: pcclGetErrorString(code=%d)", code);
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledHggcError     : return "unhandled hggc error";
    case ncclSystemError            : return "unhandled system error";
    case ncclInternalError          : return "internal error";
    case ncclInvalidArgument        : return "invalid argument";
    case ncclInvalidUsage           : return "invalid usage";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "PCCL operation in progress";
    default                         : return "unknown result code";
  }
}

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
NCCL_API(const char*, pcclGetLastError, const ncclComm_t comm);
const char* pcclGetLastError(ncclComm_t comm) {
  INFO_CALL("PCCL API: pcclGetLastError(comm=%p)", comm);
  return ncclLastError;
}

NCCL_API(void, pcclResetDebugInit);
void pcclResetDebugInit() {
  ERROR("%s not implemented yet!", __func__);
}

NCCL_API(ncclResult_t, pcclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t pcclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(PtrCheck(comm, "pcclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "pcclGetAsyncError", "asyncError"));

  *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
  if (*asyncError == ncclSuccess && comm->proxyState)
    *asyncError = __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);

  /* Check gin status */
  if (*asyncError == ncclSuccess && comm->sharedRes && comm->sharedRes->ginState.connected) {
    struct pcclGinState* ginState = &comm->sharedRes->ginState;
    // Gin progress thread status
    if (ginState->needsProxyProgress) *asyncError = __atomic_load_n(&ginState->asyncResult, __ATOMIC_ACQUIRE);
    // Gin side errors, also works when we have no GIN progress thread.
    if (*asyncError == ncclSuccess) {
      bool ginError;
      NCCLCHECK(ncclGinQueryLastError(ginState, &ginError));
      if (ginError) {
        WARN("GIN Error detected");
        *asyncError = ncclRemoteError;
      }
    }
  }

  /* if there is linked group job, we should complete it. */
  if (*asyncError == ncclSuccess && comm->groupJob) {
    NCCLCHECK(ncclGroupJobComplete(comm->groupJob));
    comm->groupJob = NULL;
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclCommCount, const ncclComm_t comm, int* count);
ncclResult_t pcclCommCount(const ncclComm_t comm, int* count) {
  PCCL_HGTX3_FUNC_RANGE;
  TRACE_CALL("PCCL API: pcclCommCount(comm=%p, count=%p)", comm, count);
  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* init thread must be joined before we access the attributes of comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t pcclCommCuDevice(const ncclComm_t comm, int* devid) {
  PCCL_HGTX3_FUNC_RANGE;
  INFO_CALL("PCCL API: pcclCommCuDevice(comm=%p, devid=%p)", comm, devid);
  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  *devid = comm->hggcDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, pcclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t pcclCommUserRank(const ncclComm_t comm, int* rank) {
  PCCL_HGTX3_FUNC_RANGE;
  TRACE_CALL("PCCL API: pcclCommUserRank(comm=%p, rank=%p)", comm, rank);
  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  *rank = comm->rank;
  return ncclSuccess;
}
