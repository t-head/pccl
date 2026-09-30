/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef __HGGC_EIT_PCCL__
#define __HGGC_EIT_PCCL__

#include "drv_uuid.h"
#include "drv_eit_common_types.h"
#include "hggc_pti.h"

HG_DEFINE_UUID(HG_EIT_ID_PCCL, 0xc1ad3908, 0xed14, 0x4e3d, 0xb658, 0x783ca4bc01ae)

enum HGpointerIpc_attribute
{
  HG_POINTER_IPC_ATTRIBUTE_INVALID    = 0, /**< Memory can not be used in any IPC apis */
  HG_POINTER_IPC_ATTRIBUTE_EXPORTABLE = 1, /**< Memory can be exported by hgMemGetIpcHandle */
  HG_POINTER_IPC_ATTRIBUTE_IMPORTED   = 2, /**< Memory was imported by hgMemOpenIpcHandle */
};

typedef enum HGmemCreateExFlags_enum
{
  HG_MEM_CREATE_MINI_GRANULARITY = 0x1,
  HG_MEM_CREATE_COPY_PRIMITIVE   = 0x2,
} HGmemCreateExFlags;

typedef enum HGstreamChannelType_enum
{
  HG_STREAM_CHANNEL_COMPUTE = 0, /**< Workload on the stream performed by kernel */
  HG_STREAM_CHANNEL_COPY    = 1, /**< Workload on the stream performed by dma */
  HG_STREAM_CHANNEL_PEER    = 2, /**< Workload on the stream performed by peer dma */
} HGstreamChannelType;

typedef enum HGeitStreamAttrID_enum
{
  HG_EIT_STREAM_ATTRIBUTE_MEMCPY_PRIMITIVE = 0x1,
} HGeitStreamAttrID;

typedef union HGeitStreamAttrValue_union
{
  HGdeviceptr cpdptr;
} HGeitStreamAttrValue;

typedef struct HGeitPccl_st
{
  // This export table supports versioning by adding to the end without changing the EIT ID.
  // The structSize field will always be set to the size in bytes of the entire export table structure.
  uint32_t struct_size;

  HGresult (HGGCAPI *hgMemDump)(void* pDstCpu, HGdeviceptr dptr, size_t size);

  HGresult (HGGCAPI *hgPointerGetIpcAttribute)(HGdeviceptr dptr, HGpointerIpc_attribute* attr);

  /// @brief Lock the specific device pointer to get a cpu-accessible address, the function may fail
  ///        once the device pointer was allocated on pci invisible heap (No indirect call supported yet).
  HGresult (HGGCAPI *hgMemLock)(HGdeviceptr dptr, void** ppCpuAddr);

  /// @brief Unlock the specific device pointer to release resources kept by the previous lock.
  /// @note User must ensure the pointer has been locked successfully.
  HGresult (HGGCAPI *hgMemUnlock)(HGdeviceptr dptr);

  /// @brief Allows caller to generated a coredump with specified attribute values.
  HGresult (HGGCAPI *hgCoredumpGenerate)(HGdevice device, const char* fileName, uint32_t flags);

  /// @brief Creates a stream and returns a handle in phStream. The Flags argument determines behaviors of
  //         the stream, see @HGeitStreamFlags.
  HGresult (HGGCAPI *hgStreamCreate)(HGstream* phStream, unsigned int Flags);

  /// @brief Read active warps' mask at specified sm.
  /// @note Callers MUST use hgDeviceGetAttribute to obtain attributes first before use the interface:
  //        sm id max = HG_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT - 1
  HGresult (HGGCAPI *hgReadActiveWarps)(uint32_t sm, uint64_t* pWarpsMask);

  /// @brief Read PC register on specified warp.
  /// @note Callers MUST use hgDeviceGetAttribute to obtain attributes first before use the interface:
  //        sm id max = HG_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT - 1
  //        warps id max = HG_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR / HG_DEVICE_ATTRIBUTE_WARP_SIZE - 1
  HGresult (HGGCAPI *hgReadPC)(uint32_t sm, uint32_t warp, uint64_t* pPc);

  /// @brief Get function name by PC.
  HGresult (HGGCAPI *hgGetFunctionPerPC)(uint64_t pc, const char** ppFuncName);

  HGresult(HGGCAPI *hgEitFuncSetAttribute)(HGfunction hFunc, HGeitFuncAttribute attr, void* pData, uint32_t size);

  /// @brief Set stream channel attributes per engine type to hint workload submission.
  /// @note Only user created stream is permitted.
  HGresult (HGGCAPI *hgStreamHintChannel)(HGstream hStream, HGstreamChannelType type, uint32_t ringId);

  /// @brief Query extension information of the device.
  HGresult (HGGCAPI *hgGetExtDeviceInfo)(HGdevice dev, HGextDeviceInfo_v2* pInfo);

  HGresult (HGGCAPI *hgMemGetAllocationGranularityEx)(size_t *granularity, const HGmemAllocationProp *prop, HGmemAllocationGranularity_flags option);

  HGresult (HGGCAPI *hgMemCreateEx)(HGmemGenericAllocationHandle *handle, size_t size, const HGmemAllocationProp *prop, unsigned long long flags);

  HGresult (HGGCAPI *hgMemAllocateCopyPrimitive)(HGdeviceptr cpdptr, HGdeviceptr* dptr);

  HGresult (HGGCAPI *hgMemReleaseCopyPrimitive)(HGdeviceptr cpdptr);

  HGresult (HGGCAPI *hgStreamSetAttribute)(HGstream hStream, HGeitStreamAttrID attr, HGeitStreamAttrValue* pValue);

  /// @brief Version numbers of the HGGC libraries. Note that this is distinct from the public interface
  ///        ::hgDriverGetVersion, the interface returns detailed interface version of libraries.
  HGresult (HGGCAPI *hgDriverGetVersion)(HGdriverVersionInfo* driverVersion);

} HGeitPccl;

#endif // __HGGC_EIT_PCCL__
