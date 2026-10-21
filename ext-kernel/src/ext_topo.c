//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <utility>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include "pccl/ext_kernel.h"
#include "pccl/ext_kernel_utils.h"
#include "ext_topo.h"

const char*  topoStr[] = {
  "TWO_RANKS_810_TOPO",
  "TWO_RANKS_810E_TOPO",
  "TWO_RANKS_M890P_TOPO",
  "TWO_RANKS_M890P_SHM_TOPO",
  "FOUR_RANKS_810_FC_TOPO",
  "FOUR_RANKS_810_SQUARE_TOPO",
  "FOUR_RANKS_810E_FC_TOPO",
  "FOUR_RANKS_810E_SQUARE1_TOPO",
  "FOUR_RANKS_810E_SQUARE2_TOPO",
  "FOUR_RANKS_M890P_FC_TOPO",
  "FOUR_RANKS_M890P_SHM_TOPO",
  "EIGHT_RANKS_810_TOPO",
  "EIGHT_RANKS_810E_CROSS_REAR_TOPO",
  "EIGHT_RANKS_810E_ONE_REAR_TOPO",
  "EIGHT_RANKS_M890P_TOPO",
  "EIGHT_RANKS_M890P_SHM_TOPO",
  "SIXTEEN_RANKS_810E",
  "SIXTEEN_RANKS_M890P_SHM_TOPO",
  "THIRTY_TWO_RANKS_M890P_SHM_TOPO"
};

const int TWO_RANKS_1LINK_TOPO_IMPL[] = {
  0, 1,
  1, 0
};

const int TWO_RANKS_2LINKS_TOPO_IMPL[] = {
  0, 2,
  2, 0
};

// 810:   0,1,2,3;  4,5,6,7.
const int FOUR_RANKS_810_FC_TOPO_IMPL[] = {
  0, 1, 1, 1,
  1, 0, 1, 1,
  1, 1, 0, 1,
  1, 1, 1, 0
};

// 810E:  0,4,9,13; 2,3,10,11; 6,7,14,15; 1,5,8,12.
const int FOUR_RANKS_810_SQUARE_TOPO_IMPL[] = {
  0, 1, 1, 0,
  1, 0, 0, 1,
  1, 0, 0, 1,
  0, 1, 1, 0
};

// eg. 810E: 0,1,2,3
const int FOUR_RANKS_810E_FC_TOPO_IMPL[] = {
  0, 2, 1, 2,
  2, 0, 2, 1,
  1, 2, 0, 1,
  2, 1, 1, 0
};

// eg. 810E: 0,1,8,9
const int FOUR_RANKS_810E_SQUARE1_TOPO_IMPL[] = {
  0, 2, 1, 0,
  2, 0, 0, 1,
  1, 0, 0, 2,
  0, 1, 2, 0
};

// eg. 810E: 0,3,7,4
const int FOUR_RANKS_810E_SQUARE2_TOPO_IMPL[] = {
  0, 2, 1, 0,
  2, 0, 0, 2,
  1, 0, 0, 2,
  0, 2, 2, 0
};

// eg. 890M: 0,1,2,3
const int FOUR_RANKS_890M_FC_TOPO_IMPL[] = {
  0, 2, 2, 2,
  2, 0, 2, 2,
  2, 2, 0, 2,
  2, 2, 2, 0
};

const int EIGHT_RANKS_810_TOPO_IMPL[] = {
  0, 1, 1, 1, 0, 0, 1, 0,
  1, 0, 1, 1, 0, 0, 0, 1,
  1, 1, 0, 1, 1, 0, 0, 0,
  1, 1, 1, 0, 0, 1, 0, 0,
  0, 0, 1, 0, 0, 1, 1, 1,
  0, 0, 0, 1, 1, 0, 1, 1,
  1, 0, 0, 0, 1, 1, 0, 1,
  0, 1, 0, 0, 1, 1, 1, 0
};

// eg. 810E: 0,1,2,3,4,5,6,7
const int EIGHT_RANKS_810E_ONE_REAR_TOPO_IMPL[] = {
  0, 2, 1, 2, 1, 0, 0, 0,
  2, 0, 2, 1, 0, 1, 0, 0,
  1, 2, 0, 1, 0, 0, 2, 0,
  2, 1, 1, 0, 0, 0, 0, 2,
  1, 0, 0, 0, 0, 2, 1, 2,
  0, 1, 0, 0, 2, 0, 2, 1,
  0, 0, 2, 0, 1, 2, 0, 1,
  0, 0, 0, 2, 2, 1, 1, 0
};

// eg. 810E: 0,1,2,3,8,9,10,11
const int EIGHT_RANKS_810E_CROSS_REAR_TOPO_IMPL[] = {
  0, 2, 1, 2, 0, 1, 0, 0,
  2, 0, 2, 1, 1, 0, 0, 0,
  1, 2, 0, 1, 0, 0, 0, 1,
  2, 1, 1, 0, 0, 0, 1, 0,
  0, 1, 0, 0, 0, 2, 1, 2,
  1, 0, 0, 0, 2, 0, 2, 1,
  0, 0, 0, 1, 1, 2, 0, 1,
  0, 0, 1, 0, 2, 1, 1, 0
};

const int EIGHT_RANKS_M890P_TOPO_IMPL[] = {
  0, 2, 2, 2, 0, 0, 2, 0,
  2, 0, 2, 2, 0, 0, 0, 2,
  2, 2, 0, 2, 2, 0, 0, 0,
  2, 2, 2, 0, 0, 2, 0, 0,
  0, 0, 2, 0, 0, 2, 2, 2,
  0, 0, 0, 2, 2, 0, 2, 2,
  2, 0, 0, 0, 2, 2, 0, 2,
  0, 2, 0, 0, 2, 2, 2, 0
};

// PpuFamily groups PPU types that share the same ext-kernel tuning.
// Adding a chip equivalent to an existing family only needs a new case
// in getPpuFamily; the tuner tables below stay untouched.
typedef enum {
  PPU_FAMILY_810,
  PPU_FAMILY_810E,
  PPU_FAMILY_M890,
  PPU_FAMILY_UNKNOWN,
} PpuFamily;

static PpuFamily getPpuFamily(PPU_TYPE ppuType) {
  switch (ppuType) {
    case _PPU_ZW810:  return PPU_FAMILY_810;
    case _PPU_ZW810E: return PPU_FAMILY_810E;
    case _ZW_M890P:
    case _ZW_M890L:
    case _ZW_M530:
    case _ZW_M530P:   return PPU_FAMILY_M890;
    default:          return PPU_FAMILY_UNKNOWN;
  }
}

typedef struct {
  PpuFamily family;
  int nRanks;
  union {
    const int* topoArray;  // non-SHM: adjacency matrix for compareTopo
    int linkCount;         // SHM: sub-link count per peer
  } topoInfo;
  extTopo topo;
} extKernelTuner;

// Non-SHM tuner table: matched by (family, nRanks, compareTopo)
extKernelTuner topoTuners[] = {
  // 2 ranks
  {.family = PPU_FAMILY_810  , .nRanks = 2 , .topoInfo = {.topoArray = TWO_RANKS_1LINK_TOPO_IMPL     }, .topo = TWO_RANKS_810_TOPO       } ,
  {.family = PPU_FAMILY_810E , .nRanks = 2 , .topoInfo = {.topoArray = TWO_RANKS_1LINK_TOPO_IMPL     }, .topo = TWO_RANKS_810_TOPO       } ,
  {.family = PPU_FAMILY_810E , .nRanks = 2 , .topoInfo = {.topoArray = TWO_RANKS_2LINKS_TOPO_IMPL    }, .topo = TWO_RANKS_810E_TOPO      } ,
  {.family = PPU_FAMILY_M890 , .nRanks = 2 , .topoInfo = {.topoArray = TWO_RANKS_2LINKS_TOPO_IMPL    }, .topo = TWO_RANKS_M890P_TOPO     } ,
  // 4 ranks
  {.family = PPU_FAMILY_810  , .nRanks = 4 , .topoInfo = {.topoArray = FOUR_RANKS_810_FC_TOPO_IMPL      }, .topo = FOUR_RANKS_810_FC_TOPO       } ,
  {.family = PPU_FAMILY_810  , .nRanks = 4 , .topoInfo = {.topoArray = FOUR_RANKS_810_SQUARE_TOPO_IMPL  }, .topo = FOUR_RANKS_810_SQUARE_TOPO   } ,
  {.family = PPU_FAMILY_810E , .nRanks = 4 , .topoInfo = {.topoArray = FOUR_RANKS_810E_FC_TOPO_IMPL     }, .topo = FOUR_RANKS_810E_FC_TOPO      } ,
  {.family = PPU_FAMILY_810E , .nRanks = 4 , .topoInfo = {.topoArray = FOUR_RANKS_810E_SQUARE1_TOPO_IMPL}, .topo = FOUR_RANKS_810E_SQUARE1_TOPO } ,
  {.family = PPU_FAMILY_810E , .nRanks = 4 , .topoInfo = {.topoArray = FOUR_RANKS_810E_SQUARE2_TOPO_IMPL}, .topo = FOUR_RANKS_810E_SQUARE2_TOPO } ,
  {.family = PPU_FAMILY_M890 , .nRanks = 4 , .topoInfo = {.topoArray = FOUR_RANKS_890M_FC_TOPO_IMPL     }, .topo = FOUR_RANKS_M890P_FC_TOPO     } ,
  // 8 ranks
  {.family = PPU_FAMILY_810  , .nRanks = 8 , .topoInfo = {.topoArray = EIGHT_RANKS_810_TOPO_IMPL            }, .topo = EIGHT_RANKS_810_TOPO             } ,
  {.family = PPU_FAMILY_810  , .nRanks = 8 , .topoInfo = {.topoArray = EIGHT_RANKS_810E_CROSS_REAR_TOPO_IMPL}, .topo = EIGHT_RANKS_810E_CROSS_REAR_TOPO } ,
  {.family = PPU_FAMILY_810E , .nRanks = 8 , .topoInfo = {.topoArray = EIGHT_RANKS_810E_ONE_REAR_TOPO_IMPL  }, .topo = EIGHT_RANKS_810E_ONE_REAR_TOPO   } ,
  {.family = PPU_FAMILY_M890 , .nRanks = 8 , .topoInfo = {.topoArray = EIGHT_RANKS_M890P_TOPO_IMPL          }, .topo = EIGHT_RANKS_M890P_TOPO           } ,
};

// SHM tuner table: matched by (family, nRanks, linkCount) when icnRmtType==SWITCH.
// linkCount is the sub-link count per peer, read from topoArray[0][1].
// To add family-specific SHM tuning, append a row here.
extKernelTuner shmTuners[] = {
  {.family = PPU_FAMILY_M890 , .nRanks = 2 , .topoInfo = {.linkCount = 8} , .topo = TWO_RANKS_M890P_SHM_TOPO   } ,
  {.family = PPU_FAMILY_M890 , .nRanks = 4 , .topoInfo = {.linkCount = 8} , .topo = FOUR_RANKS_M890P_SHM_TOPO  } ,
  {.family = PPU_FAMILY_M890 , .nRanks = 8 , .topoInfo = {.linkCount = 8} , .topo = EIGHT_RANKS_M890P_SHM_TOPO } ,
  {.family = PPU_FAMILY_M890 , .nRanks = 16, .topoInfo = {.linkCount = 8} , .topo = SIXTEEN_RANKS_M890P_SHM_TOPO } ,
  {.family = PPU_FAMILY_M890 , .nRanks = 32, .topoInfo = {.linkCount = 8} , .topo = THIRTY_TWO_RANKS_M890P_SHM_TOPO } ,
};


const extTopoConfig topoConfigs[] = {
//{ topoId                        , boundLow , boundHigh , oneShotLLBoundHigh , twoShotSwitchPoint , maxBlockNum , agBoundHigh , rsBoundHigh}
{TWO_RANKS_810_TOPO               , 0        , 32_MB     , 16_KB              , 190_KB             , 32          , 32_MB       , 32_MB}       ,
{TWO_RANKS_810E_TOPO              , 0        , 32_MB     , 16_KB              , 190_KB             , 32          , 32_MB       , 32_MB}       ,
{TWO_RANKS_M890P_TOPO             , 0        , 32_MB     , 16_KB              , 190_KB             , 32          , 32_MB       , 8_MB}        ,
{TWO_RANKS_M890P_SHM_TOPO         , 0        , 8_MB      , 16_KB              , 190_KB             , 32          , 16_MB       , 4_MB}        ,
{FOUR_RANKS_810_FC_TOPO           , 0        , 32_MB     , 8_KB               , 248_KB             , 32          , 4_MB        , 4_MB }       ,
{FOUR_RANKS_810_SQUARE_TOPO       , 0        , 4_MB      , 8_KB               , 152_KB             , 32          , 4_MB        , 4_MB }       ,
{FOUR_RANKS_810E_FC_TOPO          , 0        , 16_MB     , 8_KB               , 312_KB             , 32          , 4_MB        , 4_MB }       ,
{FOUR_RANKS_810E_SQUARE1_TOPO     , 0        , 4_MB      , 8_KB               , 184_KB             , 32          , 4_MB        , 4_MB }       ,
{FOUR_RANKS_810E_SQUARE2_TOPO     , 0        , 8_MB      , 8_KB               , 212_KB             , 32          , 4_MB        , 4_MB }       ,
{FOUR_RANKS_M890P_FC_TOPO         , 0        , 32_MB     , 16_KB              , 312_KB             , 32          , 32_MB       , 32_MB}       ,
{FOUR_RANKS_M890P_SHM_TOPO        , 0        , 8_MB      , 16_KB              , 312_KB             , 32          , 16_MB       , 8_MB}        ,
{EIGHT_RANKS_810_TOPO             , 0        , 2_MB      , 8_KB               , 64_KB              , 32          , 2_MB        , 2_MB }       ,
{EIGHT_RANKS_810E_CROSS_REAR_TOPO , 0        , 2_MB      , 8_KB               , 64_KB              , 32          , 2_MB        , 2_MB }       ,
{EIGHT_RANKS_810E_ONE_REAR_TOPO   , 0        , 2_MB      , 8_KB               , 64_KB              , 32          , 2_MB        , 2_MB }       ,
{EIGHT_RANKS_M890P_TOPO           , 0        , 8_MB      , 16_KB              , 64_KB              , 32          , 4_MB        , 4_MB }       ,
{EIGHT_RANKS_M890P_SHM_TOPO       , 0        , 16_MB     , 16_KB              , 312_KB             , 32          , 16_MB       , 16_MB}       ,
{SIXTEEN_RANKS_810E               , 0        , 2_MB      , 8_KB               , 96_KB              , 16          , 2_MB        , 2_MB }       ,
{SIXTEEN_RANKS_M890P_SHM_TOPO     , 0        , 32_MB     , 32_KB              , 128_KB             , 32          , 32_MB       , 16_MB}        ,
{THIRTY_TWO_RANKS_M890P_SHM_TOPO  , 0        , 32_MB     , 4_KB               , 4_KB               , 32          , 32_MB       , 16_MB}        ,
};

void swapRowsAndCols(int* matrix, const int N, int i, int j){
  if (i == j) return ;
  // Swap columns i and j
  for (int k = 0; k < N; ++k) {
    std::swap(matrix[k * N + i], matrix[k * N + j]);
  }
  // Swap rows i and j
  for (int k = 0; k < N; ++k) {
    std::swap(matrix[i * N + k], matrix[j * N + k]);
  }
}

// Function to compare two matrices,
// support 16 ranks for now, for more ranks, move the stack mem to heap.
template<int N>
bool compareTopo(int* A, const int* B) {
  int copyA[N*N];
  int copyB[N];
  memcpy(copyB, B, N*sizeof(int)); // just need to compare first line
  std::sort(copyB, copyB+N);
  for (int ii = 0; ii < N; ++ii) {
    memcpy(copyA, A, N * N * sizeof(int));
    // Step 1: swap A's column i and column j, then row i and row j
    swapRowsAndCols(copyA, N, 0, ii);

    // Step 2: compare copyA and B's first row, all topo Array should be skew symmetric matrix
    std::sort(copyA, copyA + N);
    bool firstDevCheck = true;
    for (int k = 0; k < N; ++k) {
      if (copyA[k] != copyB[k]) {
        firstDevCheck = false;
        break;
      }
    }
    if (!firstDevCheck) continue;

    // Step 3: create new matrices
    int subA[(N - 1) * (N - 1)];
    int subB[(N - 1) * (N - 1)];
    for (int i = 0; i < N-1; ++i) {
      for (int j = 0; j < N-1; ++j) {
        subA[i * (N - 1) + j] = copyA[(i + 1) * N + j + 1];
        subB[i * (N - 1) + j] = B[(i + 1) * N + j + 1];
      }
    }

    // Step 4: compare newA and newB recursively using CMP
    bool res = compareTopo<N-1>(subA, subB);
    if (res) return true;
  }
  return false;
}

template<>
bool compareTopo<1>(int* A, const int *B) {
  return A[0] == B[0];
}

static bool matchTopo(int* matrix, const int* ref, int N) {
  switch (N) {
    case 2:  return compareTopo<2>(matrix, ref);
    case 4:  return compareTopo<4>(matrix, ref);
    case 8:  return compareTopo<8>(matrix, ref);
    case 16: return compareTopo<16>(matrix, ref);
    default: return false;
  }
}

ncclResult_t parseTopo(PPU_TYPE ppuType, pcclICNRemoteDeviceType_t icnRmtType, int localRanks, const int* topoArray, extTopo* topo) {
  if (topo == NULL) {
    ERROR("EXT: topo pointer should not be null");
    return ncclInternalError;
  }
  if (localRanks < 2) {
    return ncclSuccess;
  }
  *topo = UNSUPPORTED_TOPO;

  // topoArray is pre-filled by pccl core, work with a mutable copy for compareTopo
  int* mutableTopo;
  NCCLCHECK(extCalloc(&mutableTopo, localRanks * localRanks));
  memcpy(mutableTopo, topoArray, localRanks * localRanks * sizeof(int));

#if 0 // for debug
  {
    char topoDbgStr[512];
    size_t offset = 0;
    for (int i = 0; i < localRanks; ++i) {
      for (int j = 0; j < localRanks; ++j) {
        if (offset >= sizeof(topoDbgStr) - 1) break;
        offset += snprintf(topoDbgStr + offset, sizeof(topoDbgStr) - offset,
                           "%d, ", mutableTopo[i * localRanks + j]);
      }
      if (offset >= sizeof(topoDbgStr) - 1) {
        topoDbgStr[offset] = '\0';
        break;
      }
      topoDbgStr[offset++] = '\n';
    }
    topoDbgStr[sizeof(topoDbgStr) - 1] = '\0';
    INFO(NCCL_EXT, "topo: \n%s", topoDbgStr);
  }
#endif
  PpuFamily family = getPpuFamily(ppuType);
  if (icnRmtType == PCCL_ICNLINK_SWITCH) {
    // SHM path: match by (family, nRanks, linkCount), skip compareTopo
    int linkCount = topoArray[1]; // topoArray[0][1]: sub-link count (uniform in SHM)
    for (int i = 0; i < sizeof(shmTuners) / sizeof(extKernelTuner); ++i) {
      extKernelTuner* tuner = shmTuners + i;
      if (tuner->nRanks == localRanks && tuner->topoInfo.linkCount == linkCount && tuner->family == family) {
        *topo = tuner->topo;
        break;
      }
    }
  } else {
    // Non-SHM path: match by (family, nRanks, compareTopo)
    for (int i = 0; i < sizeof(topoTuners) / sizeof(extKernelTuner); ++i) {
      extKernelTuner* tuner = topoTuners + i;
      // nRanks must match first to ensure topoArray dimensions align with compareTopo<N>
      if (tuner->nRanks != localRanks || tuner->family != family) continue;
      if (matchTopo(mutableTopo, tuner->topoInfo.topoArray, localRanks)) {
        *topo = tuner->topo;
        break;
      }
    }
    if (*topo == UNSUPPORTED_TOPO && localRanks == 16) {
      *topo = SIXTEEN_RANKS_810E;
    }
  }
  if (*topo != UNSUPPORTED_TOPO) INFO(NCCL_EXT, "EXT: topo type %s", topoStr[*topo]);
  NCCLCHECK(extFree(mutableTopo));
  return ncclSuccess;
}
