/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*************************************************************************
 * Minimal EIT (Export Interface Table) helper for ext-kernel plugin.
 * Provides access to hgMemDump without depending on pccl internal
 * drv_extension.h.
 ************************************************************************/

#ifndef EXT_KERNEL_PCCL_EIT_H_
#define EXT_KERNEL_PCCL_EIT_H_

#include <hggc.h>
#include "ext_kernel_utils.h"

#define HG_UUID_CHAR(x) (char)((unsigned char)((x) & 0xff))

#define HG_DEFINE_UUID(var, timeLow, timeMid, timeHigh, varSeq, node) \
  static const HGuuid var =                                           \
  {                                                                   \
    {                                                                 \
      HG_UUID_CHAR((timeLow)  >> 24), HG_UUID_CHAR((timeLow)  >> 16), \
      HG_UUID_CHAR((timeLow)  >> 8 ), HG_UUID_CHAR((timeLow)  >> 0 ), \
      HG_UUID_CHAR((timeMid)  >> 8 ), HG_UUID_CHAR((timeMid)  >> 0 ), \
      HG_UUID_CHAR((timeHigh) >> 8 ), HG_UUID_CHAR((timeHigh) >> 0 ), \
      HG_UUID_CHAR((varSeq)   >> 8 ), HG_UUID_CHAR((varSeq)   >> 0 ), \
      HG_UUID_CHAR((node)     >> 40), HG_UUID_CHAR((node)     >> 32), \
      HG_UUID_CHAR((node)     >> 24), HG_UUID_CHAR((node)     >> 16), \
      HG_UUID_CHAR((node)     >> 8 ), HG_UUID_CHAR((node)     >> 0 ), \
    }                                                                 \
  };

HG_DEFINE_UUID(EXT_KERNEL_EIT_ID_PCCL, 0xc1ad3908, 0xed14, 0x4e3d, 0xb658, 0x783ca4bc01ae)

/* Minimal EIT table struct — only the hgMemDump entry we need */
typedef struct {
  uint32_t struct_size;
  HGresult (HGGCAPI *hgMemDump)(void* pDstCpu, HGdeviceptr dptr, size_t size);
  /* remaining fields not accessed */
} ExtKernelEitPccl;

class ExtKernelEit {
public:
  static ExtKernelEit& getInstance() {
    static ExtKernelEit instance;
    return instance;
  }

  hggcError_t memDump(void* cpuPtr, HGdeviceptr devPtr, ssize_t size) {
    if (pFuncTable != NULL && pFuncTable->hgMemDump != NULL) {
      return (hggcError_t)pFuncTable->hgMemDump(cpuPtr, devPtr, size);
    } else {
      WARN("EXT EIT: hgMemDump not available");
      return hggcSuccess;
    }
  }

private:
  ExtKernelEit() {
    pFuncTable = NULL;
    hggcError_t res = hggcGetExportTable(
        reinterpret_cast<const void**>(&pFuncTable),
        reinterpret_cast<const hggcUUID_t*>(&EXT_KERNEL_EIT_ID_PCCL));
    if (res != hggcSuccess || pFuncTable == NULL) {
      WARN("EXT EIT: failed to get pccl eit table");
      pFuncTable = NULL;
    }
  }

  ExtKernelEit(ExtKernelEit const&) = delete;
  void operator=(ExtKernelEit const&) = delete;

  const ExtKernelEitPccl* pFuncTable;
};

#endif /* EXT_KERNEL_PCCL_EIT_H_ */
