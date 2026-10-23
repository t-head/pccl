/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_HGMLWRAP_H_
#define NCCL_HGMLWRAP_H_

#include "pccl_internal.h"

//#define PCCL_HGML_DIRECT 1
#ifndef PCCL_HGML_DIRECT
#define PCCL_HGML_DIRECT 0
#endif

#if PCCL_HGML_DIRECT
#include "hgml.h"
#else
// Dynamically handle dependencies on HGML

/* Extracted from hgml.h */

#define HGML_API_VERSION            12

typedef struct hgmlDevice_st* hgmlDevice_t;
#define HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE   16
#define HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE_V2   32

/**
 * Maximum number of ICNLink links supported
 */
#define HGML_ICNLINK_MAX_LINKS 8

typedef enum hgmlEnableState_enum
{
    HGML_FEATURE_DISABLED    = 0,     //!< Feature disabled
    HGML_FEATURE_ENABLED     = 1      //!< Feature enabled
} hgmlEnableState_t;

typedef enum hgmlIcnLinkCapability_enum
{
    HGML_ICNLINK_CAP_P2P_SUPPORTED = 0,     // P2P over NVLink is supported
    HGML_ICNLINK_CAP_SYSMEM_ACCESS = 1,     // Access to system memory is supported
    HGML_ICNLINK_CAP_P2P_ATOMICS   = 2,     // P2P atomics are supported
    HGML_ICNLINK_CAP_SYSMEM_ATOMICS= 3,     // System memory atomics are supported
    HGML_ICNLINK_CAP_SLI_BRIDGE    = 4,     // SLI is supported over this link
    HGML_ICNLINK_CAP_VALID         = 5,     // Link is supported on this device
    // should be last
    HGML_ICNLINK_CAP_COUNT
} hgmlIcnLinkCapability_t;

typedef enum hgmlReturn_enum
{
    // cppcheck-suppress *
    HGML_SUCCESS = 0,                          //!< The operation was successful
    HGML_ERROR_UNINITIALIZED = 1,              //!< HGML was not first initialized with hgmlInit()
    HGML_ERROR_INVALID_ARGUMENT = 2,           //!< A supplied argument is invalid
    HGML_ERROR_NOT_SUPPORTED = 3,              //!< The requested operation is not available on target device
    HGML_ERROR_NO_PERMISSION = 4,              //!< The current user does not have permission for operation
    HGML_ERROR_ALREADY_INITIALIZED = 5,        //!< Deprecated: Multiple initializations are now allowed through ref counting
    HGML_ERROR_NOT_FOUND = 6,                  //!< A query to find an object was unsuccessful
    HGML_ERROR_INSUFFICIENT_SIZE = 7,          //!< An input argument is not large enough
    HGML_ERROR_INSUFFICIENT_POWER = 8,         //!< A device's external power cables are not properly attached
    HGML_ERROR_DRIVER_NOT_LOADED = 9,          //!< Driver is not loaded
    HGML_ERROR_TIMEOUT = 10,                   //!< User provided timeout passed
    HGML_ERROR_IRQ_ISSUE = 11,                 //!< Kernel detected an interrupt issue with a GPU
    HGML_ERROR_LIBRARY_NOT_FOUND = 12,         //!< HGML Shared Library couldn't be found or loaded
    HGML_ERROR_FUNCTION_NOT_FOUND = 13,        //!< Local version of HGML doesn't implement this function
    HGML_ERROR_CORRUPTED_INFOROM = 14,         //!< infoROM is corrupted
    HGML_ERROR_GPU_IS_LOST = 15,               //!< The GPU has fallen off the bus or has otherwise become inaccessible
    HGML_ERROR_RESET_REQUIRED = 16,            //!< The GPU requires a reset before it can be used again
    HGML_ERROR_OPERATING_SYSTEM = 17,          //!< The GPU control device has been blocked by the operating system/cgroups
    HGML_ERROR_LIB_RM_VERSION_MISMATCH = 18,   //!< RM detects a driver/library version mismatch
    HGML_ERROR_IN_USE = 19,                    //!< An operation cannot be performed because the GPU is currently in use
    HGML_ERROR_MEMORY = 20,                    //!< Insufficient memory
    HGML_ERROR_NO_DATA = 21,                   //!< No data
    HGML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,    //!< The requested vgpu operation is not available on target device, because ECC is enabled
    HGML_ERROR_INSUFFICIENT_RESOURCES = 23,    //!< Ran out of critical resources, other than memory
    HGML_ERROR_FREQ_NOT_SUPPORTED = 24,        //!< Ran out of critical resources, other than memory
    HGML_ERROR_ARGUMENT_VERSION_MISMATCH = 25, //!< The provided version is invalid/unsupported
    HGML_ERROR_DEPRECATED  = 26,               //!< The requested functionality has been deprecated
    HGML_ERROR_NOT_READY = 27,                 //!< The system is not ready for the request
    HGML_ERROR_GPU_NOT_FOUND = 28,             //!< No GPUs were found
    HGML_ERROR_UNKNOWN = 999                   //!< An internal driver error occurred
} hgmlReturn_t;

typedef struct hgmlPciInfo_st
{
    char busId[HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;             //!< The PCI domain on which the device's bus resides, 0 to 0xffff
    unsigned int bus;                //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;             //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;        //!< The combined 16-bit device id and 16-bit vendor id

    // Added in HGML 2.285 API
    unsigned int pciSubSystemId;     //!< The 32-bit Sub System Device ID

    // NVIDIA reserved for internal use only
    union {
      char busIdV2[HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE_V2];
      struct {
        unsigned int reserved0;
        unsigned int reserved1;
        unsigned int reserved2;
        unsigned int reserved3;
        unsigned int reserved4;
        unsigned int reserved5;
        unsigned int reserved6;
        int ppuId;
      } ppuInfo;
    };
} hgmlPciInfo_t;

/**
 * Enum to represent ICNLink's remote device type
 */
typedef enum hgmlIntIcnLinkDeviceType_enum
{
    HGML_ICNLINK_DEVICE_TYPE_PPU     = 0x00,
    HGML_ICNLINK_DEVICE_TYPE_SWITCH  = 0x02,
    HGML_ICNLINK_DEVICE_TYPE_UNKNOWN = 0xFF
} hgmlIntIcnLinkDeviceType_t;

/**
 * Represents the type for sample value returned
 */
typedef enum hgmlValueType_enum
{
    HGML_VALUE_TYPE_DOUBLE = 0,
    HGML_VALUE_TYPE_UNSIGNED_INT = 1,
    HGML_VALUE_TYPE_UNSIGNED_LONG = 2,
    HGML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3,
    HGML_VALUE_TYPE_SIGNED_LONG_LONG = 4,
    HGML_VALUE_TYPE_SIGNED_INT = 5,

    // Keep this last
    HGML_VALUE_TYPE_COUNT
} hgmlValueType_t;

/**
 * Union to represent different types of Value
 */
typedef union hgmlValue_st
{
    double dVal;                    //!< If the value is double
    int siVal;                      //!< If the value is signed int
    unsigned int uiVal;             //!< If the value is unsigned int
    unsigned long ulVal;            //!< If the value is unsigned long
    unsigned long long ullVal;      //!< If the value is unsigned long long
    signed long long sllVal;        //!< If the value is signed long long
} hgmlValue_t;

typedef struct hgmlFieldValue_st
{
    unsigned int fieldId;       //!< ID of the HGML field to retrieve. This must be set before any call that uses this struct. See the constants starting with HGML_FI_ above.
    unsigned int scopeId;       //!< Scope ID can represent data used by HGML depending on fieldId's context. For example, for ICNLink throughput counter data, scopeId can represent linkId.
    long long timestamp;        //!< CPU Timestamp of this value in microseconds since 1970
    long long latencyUsec;      //!< How long this field value took to update (in usec) within HGML. This may be averaged across several fields that are serviced by the same driver call.
    hgmlValueType_t valueType;  //!< Type of the value stored in value
    hgmlReturn_t hgmlReturn;    //!< Return code for retrieving this value. This must be checked before looking at value, as value is undefined if hgmlReturn != HGML_SUCCESS
    hgmlValue_t value;          //!< Value for this field. This is only valid if hgmlReturn == HGML_SUCCESS
} hgmlFieldValue_t;

/* ================================================ P2p Status ================================================ */
/* P2P Capability Index Status*/
typedef enum hgmlGpuP2PStatus_enum
{
    HGML_P2P_STATUS_OK     = 0,
    HGML_P2P_STATUS_CHIPSET_NOT_SUPPORTED,
    HGML_P2P_STATUS_GPU_NOT_SUPPORTED,
    HGML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED,
    HGML_P2P_STATUS_DISABLED_BY_REGKEY,
    HGML_P2P_STATUS_NOT_SUPPORTED,
    HGML_P2P_STATUS_UNKNOWN
} hgmlGpuP2PStatus_t;

/* P2P Capability Index*/
typedef enum hgmlGpuP2PCapsIndex_enum
{
    HGML_P2P_CAPS_INDEX_READ = 0,
    HGML_P2P_CAPS_INDEX_WRITE,
    HGML_P2P_CAPS_INDEX_ICNLINK,
    HGML_P2P_CAPS_INDEX_ATOMICS,
    HGML_P2P_CAPS_INDEX_PROP,
    HGML_P2P_CAPS_INDEX_UNKNOWN
}hgmlGpuP2PCapsIndex_t;

/* ================================================ FieldId ================================================ */
/**
 * Field Identifiers.
 *
 * All Identifiers pertain to a device. Each ID is only used once and is guaranteed never to change.
 */

/* ICNLink Speed */
#define HGML_FI_DEV_ICNLINK_SPEED_MBPS_L0     84  //!< ICNLink Speed in MBps for Link 0
#define HGML_FI_DEV_ICNLINK_SPEED_MBPS_L1     85  //!< ICNLink Speed in MBps for Link 1
#define HGML_FI_DEV_ICNLINK_SPEED_MBPS_L2     86  //!< ICNLink Speed in MBps for Link 2
#define HGML_FI_DEV_ICNLINK_SPEED_MBPS_L3     87  //!< ICNLink Speed in MBps for Link 3
#define HGML_FI_DEV_ICNLINK_SPEED_MBPS_L4     88  //!< ICNLink Speed in MBps for Link 4
#define HGML_FI_DEV_ICNLINK_SPEED_MBPS_L5     89  //!< ICNLink Speed in MBps for Link 5
#define HGML_FI_DEV_ICNLINK_SPEED_MBPS_COMMON 90  //!< Common NVLink Speed in MBps for active links
#define HGML_FI_DEV_ICNLINK_LINK_COUNT        91  //!< Number of NVLinks present on the device
#define HGML_FI_DEV_ICNLINK_SPEED_MBPS_L6     132  //!< ICNLink Speed in MBps for Link 6
#define HGML_FI_DEV_ICNLINK_SPEED_MBPS_L7     133  //!< ICNLink Speed in MBps for Link 7

#define HGML_FI_DEV_REAR_ID                           516 //!< Device rear id

/**
 * Remote device NVLink ID
 *
 * Link ID needs to be specified in the scopeId field in hgmlFieldValue_t.
 */
#define HGML_FI_DEV_ICNLINK_REMOTE_ICNLINK_ID     146 //!< Remote device NVLink ID

/**
 * NVSwitch: connected NVLink count
 */
#define HGML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT   147  //!< Number of NVLinks connected to NVSwitch

#define HGML_FI_DEV_ICNLINK_GET_SPEED                  164 //!< ICNLink Speed in MBps
#define HGML_FI_DEV_ICNLINK_GET_STATE                  165 //!< ICNLink State - Active,Inactive
#define HGML_FI_DEV_ICNLINK_GET_VERSION                166 //!< ICNLink Version

#define HGML_FI_DEV_C2C_LINK_COUNT                    170 //!< Number of C2C Links present on the device
#define HGML_FI_DEV_C2C_LINK_GET_STATUS               171 //!< C2C Link Status 0=INACTIVE 1=ACTIVE
#define HGML_FI_DEV_C2C_LINK_GET_MAX_BW               172 //!< C2C Link Speed in MBps for active links

#define HGML_FI_MAX                                   524 //!< One greater than the largest field ID defined above

/* ================================================ fabric manager ================================================ */
#define HGML_GPU_FABRIC_UUID_LEN 16

#define HGML_GPU_FABRIC_STATE_NOT_SUPPORTED 0
#define HGML_GPU_FABRIC_STATE_NOT_STARTED   1
#define HGML_GPU_FABRIC_STATE_IN_PROGRESS   2
#define HGML_GPU_FABRIC_STATE_COMPLETED     3

typedef unsigned char hgmlGpuFabricState_t;

typedef struct {
    unsigned char        clusterUuid[HGML_GPU_FABRIC_UUID_LEN]; //!< Uuid of the cluster to which this GPU belongs
    hgmlReturn_t         status;                                //!< Error status, if any. Must be checked only if state returns "complete".
    unsigned int         cliqueId;                              //!< ID of the fabric clique to which this GPU belongs
    hgmlGpuFabricState_t state;                                 //!< Current state of GPU registration process
} hgmlGpuFabricInfo_t;

/**
* GPU Fabric information (v2).
*
* Version 2 adds the \ref hgmlGpuFabricInfo_v2_t.version field
* to the start of the structure, and the \ref hgmlGpuFabricInfo_v2_t.healthMask
* field to the end. This structure is not backwards-compatible with
* \ref hgmlGpuFabricInfo_t.
*/
typedef struct {
    unsigned int         version;                               //!< Structure version identifier (set to \ref hgmlGpuFabricInfo_v2)
    unsigned char        clusterUuid[HGML_GPU_FABRIC_UUID_LEN]; //!< Uuid of the cluster to which this GPU belongs
    hgmlReturn_t         status;                                //!< Error status, if any. Must be checked only if state returns "complete".
    unsigned int         cliqueId;                              //!< ID of the fabric clique to which this GPU belongs
    hgmlGpuFabricState_t state;                                 //!< Current state of GPU registration process
    unsigned int         healthMask;                            //!< GPU Fabric health Status Mask
} hgmlGpuFabricInfo_v2_t;

typedef hgmlGpuFabricInfo_t hgmlGpuFabricInfoV_t;

/* End of hgml.h */
#endif // PCCL_HGML_DIRECT

constexpr int pcclHgmlMaxDevices = 32;
struct pcclHgmlDeviceInfo {
  hgmlDevice_t handle;
  int computeCapabilityMajor, computeCapabilityMinor;
};

extern int pcclHgmlDeviceCount;
extern pcclHgmlDeviceInfo pcclHgmlDevices[pcclHgmlMaxDevices];

struct pcclHgmlDevicePairInfo {
  hgmlGpuP2PStatus_t p2pStatusRead, p2pStatusWrite;
};
extern pcclHgmlDevicePairInfo pcclHgmlDevicePairs[pcclHgmlMaxDevices][pcclHgmlMaxDevices];

// All pcclHgmlFoo() functions call pcclHgmlEnsureInitialized() implicitly.
// Outsiders need only call it if they want to inspect the pcclHgml global
// tables above.
ncclResult_t pcclHgmlEnsureInitialized();

ncclResult_t pcclHgmlShutdown(void);
ncclResult_t pcclHgmlDeviceGetHandleByPciBusId(const char* pciBusId, hgmlDevice_t* device);
ncclResult_t pcclHgmlDeviceGetIndex(hgmlDevice_t device, unsigned* index);
ncclResult_t pcclHgmlDeviceGetHandleByIndex(unsigned int index, hgmlDevice_t *device);
ncclResult_t pcclHgmlDeviceGetIcnLinkState(hgmlDevice_t device, unsigned int link, hgmlEnableState_t *isActive);
ncclResult_t pcclHgmlDeviceGetPciInfo(hgmlDevice_t device, hgmlPciInfo_t *pci);
ncclResult_t pcclHgmlDeviceGetIcnLinkRemotePciInfo(hgmlDevice_t device, unsigned int link, hgmlPciInfo_t *pci);
ncclResult_t pcclHgmlDeviceGetIcnLinkRemoteDeviceType(hgmlDevice_t device, unsigned int link, hgmlIntIcnLinkDeviceType_t *pIcnLinkDeviceType);
ncclResult_t pcclHgmlDeviceGetIcnLinkCapability(hgmlDevice_t device, unsigned int link, hgmlIcnLinkCapability_t capability, unsigned int *capResult);
ncclResult_t pcclHgmlDeviceGetHggcComputeCapability(hgmlDevice_t device, int* major, int* minor);
ncclResult_t pcclHgmlDeviceGetCount(int*);
ncclResult_t pcclHgmlDeviceGetFieldValues(hgmlDevice_t device, int valuesCount, hgmlFieldValue_t *values);
ncclResult_t pcclHgmlDeviceGetName(hgmlDevice_t device, char *name, unsigned int length);
ncclResult_t pcclHgmlDeviceGetP2PStatus(hgmlDevice_t device1, hgmlDevice_t device2, hgmlGpuP2PCapsIndex_t p2pIndex, hgmlGpuP2PStatus_t* p2pStatus);
ncclResult_t pcclHgmlSystemGetDriverVersion(char *version, unsigned int length);
ncclResult_t pcclHgmlDeviceGetGpuFabricInfo(hgmlDevice_t device, hgmlGpuFabricInfoV_t *gpuFabricInfo);
ncclResult_t pcclHgmlDeviceGetGpuFabricInfoV(hgmlDevice_t device, hgmlGpuFabricInfoV_t *gpuFabricInfo);

#endif // End include guard
