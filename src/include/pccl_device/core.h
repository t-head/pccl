/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _PCCL_DEVICE_CORE_H_
#define _PCCL_DEVICE_CORE_H_
#include "pccl.h"
#include "coop.h"
#include "utility.h"

struct pcclDevComm;
typedef struct pcclDevComm pcclDevComm_t;

struct pcclTeam;
typedef struct pcclTeam pcclTeam_t;

// typedef struct pcclWindow_vidmem* pcclWindow_t; // in pccl_internal.h
typedef struct pcclWindow_vidmem pcclWindow_vidmem_t;

struct pcclMultimemHandle;
typedef struct pcclMultimemHandle pcclMultimemHandle_t;

typedef uint32_t pcclDevResourceHandle;
typedef pcclDevResourceHandle pcclDevResourceHandle_t;

typedef uint32_t pcclGinSignal_t;
typedef uint32_t pcclGinCounter_t;

typedef struct alignas(uint64_t) {
  char opaque[16];
} pcclGinRequest_t;

struct pcclLsaBarrierHandle;
typedef struct pcclLsaBarrierHandle pcclLsaBarrierHandle_t;

struct pcclGinBarrierHandle;
typedef struct pcclGinBarrierHandle pcclGinBarrierHandle_t;

struct pcclLLA2AHandle;
typedef struct pcclLLA2AHandle pcclLLA2AHandle_t;

struct pcclTeam {
  int nRanks, rank, stride;
};

#if __cplusplus
template<typename T> struct pcclSymPtr;
#endif

#if __cplusplus
struct pcclTeamTagWorld {};
struct pcclTeamTagLsa {};
struct pcclTeamTagRail {};
#endif

struct pcclDevCommRequirements;
typedef struct pcclDevCommRequirements pcclDevCommRequirements_t;

struct pcclDevResourceRequirements;
typedef struct pcclDevResourceRequirements pcclDevResourceRequirements_t;

struct pcclTeamRequirements;
typedef struct pcclTeamRequirements pcclTeamRequirements_t;

struct pcclCommProperties;
typedef struct pcclCommProperties pcclCommProperties_t;

typedef enum {
  PCCL_GIN_CONNECTION_NONE,
  PCCL_GIN_CONNECTION_FULL,
  PCCL_GIN_CONNECTION_RAIL,
} pcclGinConnectionType_t;

struct pcclDevCommRequirements {
  /* attributes that users should never touch. */
  size_t size;
  unsigned int magic;
  unsigned int version;

  /* attributes that users are able to customize. */
  pcclDevResourceRequirements_t* resourceRequirementsList;
  pcclTeamRequirements_t* teamRequirementsList;

  bool lsaMultimem; // Enable multimem on lsa team

  int barrierCount;
  int lsaBarrierCount;
  int railGinBarrierCount;

  int lsaLLA2ABlockCount, lsaLLA2ASlotCount;

  bool ginForceEnable;

  int ginContextCount; // This is a hint, the actual context count in the devcomm may not match.
  int ginSignalCount; // Guaranteed to start at id=0
  int ginCounterCount; // Guaranteed to start at id=0
  pcclGinConnectionType_t ginConnectionType;
  bool ginExclusiveContexts;
  int ginQueueDepth;
  int ginTrafficClass;

  int worldGinBarrierCount;
};

#define PCCL_DEV_COMM_REQUIREMENTS_INITIALIZER {                               \
    sizeof(pcclDevCommRequirements_t),           /* size */                    \
    PCCL_API_MAGIC,                              /* magic */                   \
    PCCL_VERSION_CODE,                           /* version */                 \
    nullptr,                                     /* resourceRequirementsList*/ \
    nullptr,                                     /* teamRequirementsList */    \
    false,                                       /* lsaMultimem */             \
    0,                                           /* barrierCount */            \
    0,                                           /* lsaBarrierCount */         \
    0,                                           /* railGinBarrierCount */     \
    0,                                           /* lsaLLA2ABlockCount */      \
    0,                                           /* lsaLLA2ASlotCount */       \
    false,                                       /* ginForceEnable */          \
    4,                                           /* ginContextCount */         \
    0,                                           /* ginSignalCount */          \
    0,                                           /* ginCounterCount */         \
    PCCL_GIN_CONNECTION_NONE,                    /* ginConnectionType */       \
    false,                                       /* ginExclusiveContexts */    \
    0,                                           /* ginQueueDepth */           \
    PCCL_CONFIG_UNDEF_INT,                       /* ginTrafficClass */         \
    0,                                           /* worldGinBarrierCount */    \
}

struct pcclDevResourceRequirements {
  pcclDevResourceRequirements_t* next;
  size_t bufferSize, bufferAlign;
  pcclDevResourceHandle_t* outBufferHandle; // If non-null, target assigned during pcclDevCommCreate.
  int ginSignalCount;
  int ginCounterCount;
  pcclGinSignal_t* outGinSignalStart;
  pcclGinCounter_t* outGinCounterStart;
};

struct pcclTeamRequirements {
  pcclTeamRequirements_t* next;
  pcclTeam_t team;
  bool multimem;
  pcclMultimemHandle_t* outMultimemHandle; // If non-null, target assigned during pcclDevCommCreate.
};

#define PCCL_COMM_PROPERTIES_INITIALIZER {                           \
  sizeof(pcclCommProperties_t),                  /* size */          \
  PCCL_API_MAGIC,                                /* magic */         \
  PCCL_VERSION_CODE,                             /* version */       \
}

typedef enum {
  PCCL_GIN_TYPE_NONE = 0,
  PCCL_GIN_TYPE_PROXY = 2, // intentially not 1. Must match PCCL_NET_DEVICE_GIN_PROXY for backward compatibility
  PCCL_GIN_TYPE_GDAKI = 3, // intentially not 2. Must match PCCL_NET_DEVICE_GIN_GDAKI for backward compatibility
} pcclGinType_t;

struct pcclCommProperties {
  /* internal use only */
  size_t size;
  unsigned int magic;
  unsigned int version;

  /* attributes for users. */
  int rank;
  int nRanks;
  int hggcDev;
  int hgmlDev;
  bool deviceApiSupport;
  bool multimemSupport;
  pcclGinType_t ginType;
  int nLsaTeams;
  bool hostRmaSupport;
  pcclGinType_t railedGinType;
};

PCCL_EXTERN_C __host__ pcclResult_t pcclCommQueryProperties(pcclComm_t comm, pcclCommProperties_t* props);
PCCL_EXTERN_C __host__ pcclResult_t pcclDevCommCreate(pcclComm_t comm, pcclDevCommRequirements_t const* reqs, pcclDevComm_t* outDevComm);
PCCL_EXTERN_C __host__ pcclResult_t pcclDevCommDestroy(pcclComm_t comm, pcclDevComm_t const* devComm);

PCCL_EXTERN_C __host__ pcclResult_t pcclGetLsaMultimemDevicePointer(pcclWindow_t window, size_t offset, void** outPtr);
PCCL_EXTERN_C __host__ pcclResult_t pcclGetMultimemDevicePointer(pcclWindow_t window, size_t offset, pcclMultimemHandle_t multimem, void** outPtr);
PCCL_EXTERN_C __host__ pcclResult_t pcclGetLsaDevicePointer(pcclWindow_t window, size_t offset, int lsaRank, void** outPtr);
PCCL_EXTERN_C __host__ pcclResult_t pcclGetPeerDevicePointer(pcclWindow_t window, size_t offset, int peer, void** outPtr);

////////////////////////////////////////////////////////////////////////////////
// Team API:
#if __cplusplus
PCCL_IR_EXTERN_C PCCL_HOST_DEVICE_INLINE pcclTeam pcclTeamWorld(pcclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
PCCL_EXTERN_C __host__ pcclTeam_t pcclTeamWorld(pcclComm_t);
#endif

#if __cplusplus
PCCL_IR_EXTERN_C PCCL_HOST_DEVICE_INLINE pcclTeam pcclTeamLsa(pcclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
PCCL_EXTERN_C __host__ pcclTeam_t pcclTeamLsa(pcclComm_t);
#endif

PCCL_EXTERN_C PCCL_HOST_DEVICE_INLINE bool pcclTeamRankIsMember(pcclTeam_t a, pcclTeam_t b, int bPeer);
PCCL_EXTERN_C PCCL_HOST_DEVICE_INLINE int pcclTeamRankToTeam(pcclTeam_t a, pcclTeam_t b, int bPeer);

#if __cplusplus
PCCL_IR_EXTERN_C PCCL_HOST_DEVICE_INLINE int pcclTeamRankToWorld(pcclDevComm const&, pcclTeam, int rank);
#endif
#ifndef __clang_llvm_bitcode_lib__
PCCL_EXTERN_C __host__ int pcclTeamRankToWorld(pcclComm_t, pcclTeam_t, int rank);
#endif

#if __cplusplus
PCCL_IR_EXTERN_C PCCL_HOST_DEVICE_INLINE int pcclTeamRankToLsa(pcclDevComm const&, pcclTeam, int rank);
#endif
#ifndef __clang_llvm_bitcode_lib__
PCCL_EXTERN_C __host__ int pcclTeamRankToLsa(pcclComm_t, pcclTeam_t, int rank);
#endif

PCCL_EXTERN_C PCCL_HOST_DEVICE_INLINE pcclTeam_t pcclTeamInnerFactor(pcclTeam_t parent, int innerSize);
PCCL_EXTERN_C PCCL_HOST_DEVICE_INLINE pcclTeam_t pcclTeamOuterFactor(pcclTeam_t parent, int innerSize);

// Interpret each team as a set of ranks. This function assumes that `subset`
// is a subset of `parent`. Thus the number of ranks in the set difference of
// `parent` minus `subset` is `super.nRanks - subset.nRanks`. Given `index` this
// function returns the index'th element of `parent` minus `subset`.
PCCL_EXTERN_C PCCL_HOST_DEVICE_INLINE int pcclTeamRankInDifference(pcclTeam_t parent, pcclTeam_t subset, int index);

// Equivalent to pcclTeamOuterFactor of lsa team.
#if __cplusplus
PCCL_IR_EXTERN_C PCCL_HOST_DEVICE_INLINE pcclTeam pcclTeamRail(pcclDevComm const&);
#endif
#ifndef __clang_llvm_bitcode_lib__
PCCL_EXTERN_C __host__ pcclTeam_t pcclTeamRail(pcclComm_t);
#endif

// Get offset of resource buffer within `comm.resourceWindow`.
PCCL_EXTERN_C PCCL_HOST_DEVICE_INLINE size_t pcclGetResourceBufferOffset(pcclDevResourceHandle_t);

PCCL_DEVICE_INLINE pcclSymPtr<char> pcclGetResourceBuffer(pcclDevComm const&, pcclDevResourceHandle);

////////////////////////////////////////////////////////////////////////////////
// Window API:

PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetLocalPointer(pcclWindow_t w, size_t offset);
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetLsaPointer(pcclWindow_t w, size_t offset, int peer);
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetPeerPointer(pcclWindow_t w, size_t offset, int peer);
PCCL_DEVICE_INLINE void* pcclGetPeerPointer(pcclWindow_t w, size_t offset, pcclTeam tm, int peer);
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetMultimemPointer(pcclWindow_t w, size_t offset, pcclMultimemHandle mmHandle);
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetLsaMultimemPointer(pcclWindow_t w, size_t offset, pcclDevComm const&);

// Convenience for combining pcclGet***Pointer() with resource handle.
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetResourceBufferLocalPointer(pcclDevComm const&, pcclDevResourceHandle);
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetResourceBufferLsaPointer(pcclDevComm const&, pcclDevResourceHandle, int peer);
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetResourceBufferPeerPointer(pcclDevComm const&, pcclDevResourceHandle, pcclTeam, int peer);
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetResourceBufferMultimemPointer(pcclDevComm const&, pcclDevResourceHandle, pcclMultimemHandle);
PCCL_IR_EXTERN_C PCCL_DEVICE_INLINE void* pcclGetResourceBufferLsaMultimemPointer(pcclDevComm const&, pcclDevResourceHandle);

#endif // _PCCL_DEVICE_CORE_H_
