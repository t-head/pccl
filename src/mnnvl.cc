/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "mnnvl.h"
#include "transport.h"
#include <hggc.h>
#include "hggc_runtime.h"

extern int ncclParamMnnvlFakeAdvancedFeatureFlag;

// Determine if MNNVL support is available
ncclResult_t ncclMnnvlCheck(ncclComm_t comm) {
  // MNNVL requires cuMem to be enabled
  if (!ncclCuMemEnable()) {
    INFO(NCCL_INIT, "MNNVL cuMem is required but not enabled");
    return ncclSuccess;
  }

  // MNNVL also requires FABRIC handle support
  if (!ncclParamMnnvlFakeAdvancedFeatureFlag) {
    int hggcDev;
    int flag = 0;
    HGdevice hgrrentDev;
    HGGCCHECK(hggcGetDevice(&hggcDev));
    HGCHECK(hgDeviceGet(&hgrrentDev, hggcDev));
    // Ignore error if HG_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is not supported
    hgDeviceGetAttribute(&flag, HG_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, hgrrentDev);
    if (!flag) {
      INFO(NCCL_INIT, "MNNVL HG_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is not supported");
      return ncclSuccess;
    }
    // Check that all ranks have initialized the fabric fully
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].fabricInfo.state != HGML_GPU_FABRIC_STATE_COMPLETED) {
        INFO(NCCL_INIT, "MNNVL rank %d has not initialized fabric fully");
        return ncclSuccess;
      }
    }
  }

  // Determine our MNNVL domain/clique and NVL domain size
  NCCLCHECK(ncclCalloc(&comm->clique.ranks, comm->nRanks));
  comm->clique.id = comm->peerInfo[comm->rank].fabricInfo.cliqueId;
  comm->nvlDomainSize = 0;
  for (int i = 0; i < comm->nRanks; i++) {
    hgmlGpuFabricInfoV_t *fabricInfo1 = &comm->peerInfo[comm->rank].fabricInfo;
    hgmlGpuFabricInfoV_t *fabricInfo2 = &comm->peerInfo[i].fabricInfo;
    // Check if the cluster UUID and cliqueId match
    // A zero UUID means we don't have MNNVL fabric info - disable MNNVL
    if ((((long *)&fabricInfo2->clusterUuid)[0]|((long *)fabricInfo2->clusterUuid)[1]) == 0) {
      INFO(NCCL_INIT, "MNNVL fabric cluster uuid is 0");
      return ncclSuccess;
    }
    // Check if same NVL domain (clusterUuid match)
    if (memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, HGML_GPU_FABRIC_UUID_LEN) == 0) {
      comm->nvlDomainSize++;
      // Also check if same clique (cliqueId match)
      if (fabricInfo1->cliqueId == fabricInfo2->cliqueId) {
        if (i == comm->rank) {
          comm->cliqueRank = comm->clique.size;
        }
        comm->clique.ranks[comm->clique.size++] = i;
      }
    }
  }

  // No MNNVL clique found
  if (comm->clique.size <= 1 && comm->nvlDomainSize <= 1) {
    INFO(NCCL_INIT, "MNNVL no clique found");
    return ncclSuccess;
  }

  // Check that FABRIC handles can be exported & imported by IMEX
  {
    void *ptr = NULL;
    HGmemGenericAllocationHandle handle;
    ncclHgDesc hgDesc;
    HGresult err;

    // Allocate FABRIC handle compatible memory
    ncclResult_t ret = ncclHgMemAlloc(&ptr, &handle, HG_MEM_HANDLE_TYPE_FABRIC, HGGC_IPC_MIN, NULL, NULL);
    if (ret != ncclSuccess) {
      // Return an error if this is a MNNVL capable system but FABRIC handles are not supported
      ERROR("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX channel configuration (/dev/alixpu-caps-imex-channels). Set PCCL_MNNVL_ENABLE=0 to ignore this issue.",
            comm->clique.size);
      return ncclSystemError;
    }
    err = hgMemExportToShareableHandle(&hgDesc, handle, HG_MEM_HANDLE_TYPE_FABRIC, 0);
    if (err != HGGC_SUCCESS ||
        (err = hgMemImportFromShareableHandle(&handle, (void*)&hgDesc, HG_MEM_HANDLE_TYPE_FABRIC)) != HGGC_SUCCESS) {
      const char *errStr;
      hgGetErrorString(err, &errStr);
      NCCLCHECK(ncclHgMemFree(ptr));
      // Return an error if this is a MNNVL capable system but it's not working
      ERROR("MNNVL (cliqueSize %d) is available but not working on this system. Check the IMEX configuration (ptg-imex-ctl -N). Set PCCL_MNNVL_ENABLE=0 to ignore this issue.",
            comm->clique.size);
      return ncclSystemError;
    }
    NCCLCHECK(ncclHgMemFree(ptr));

    // Force the CUMEM handle type to be FABRIC for MNNVL
    ncclHgMemHandleType = HG_MEM_HANDLE_TYPE_FABRIC;
  }

  comm->MNNVL = 1;
  INFO(NCCL_INIT, "MNNVL %d cliqueId %x cliqueSize %d cliqueRank %d nvlDomainSize %d", comm->MNNVL, comm->clique.id, comm->clique.size, comm->cliqueRank, comm->nvlDomainSize);

  return ncclSuccess;
}
