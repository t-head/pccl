/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_NET_H_
#define PCCL_NET_H_

#include "pccl.h"
#include "pccl_common.h"
#include <stdint.h>

#define PCCL_NET_HANDLE_MAXSIZE 64

#define PCCL_PTR_HOST 0x1
#define PCCL_PTR_HGGC 0x2

// Maximum number of requests per comm object
#define PCCL_NET_MAX_REQUESTS 8

typedef struct {
  char* name;     // Used mostly for logging.
  char* pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // [PCCL_PTR_HOST|PCCL_PTR_HGGC|PCCL_PTR_DMABUF]
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  float latency;  // Network latency
  int maxComms;   // Maximum number of comms we can create
  int maxRecvs;   // Maximum number of grouped receives.
} pcclNetProperties_v6_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  pcclResult_t (*init)(pcclDebugLogger_t logFunction);
  // Return the number of adapters.
  pcclResult_t (*devices)(int* ndev);
  // Get various device properties.
  pcclResult_t (*getProperties)(int dev, pcclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to PCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  pcclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  pcclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  pcclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either PCCL_PTR_HOST or PCCL_PTR_HGGC.
  pcclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  pcclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  pcclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  pcclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  pcclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with PCCL_PTR_HGGC is
  // visible to the GPU
  pcclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  pcclResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  pcclResult_t (*closeSend)(void* sendComm);
  pcclResult_t (*closeRecv)(void* recvComm);
  pcclResult_t (*closeListen)(void* listenComm);
} pcclNet_v6_t;

typedef pcclNet_v6_t pcclNet_t;

#define PCCL_PLUGIN_SYMBOL pcclNetPlugin_v6

typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  pcclResult_t (*init)(pcclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  pcclResult_t (*devices)(int* ndev);
  // Get various device properties.
  pcclResult_t (*getProperties)(int dev, pcclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to PCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  pcclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  pcclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  pcclResult_t (*reduceSupport)(pcclDataType_t dataType, pcclRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either PCCL_PTR_HOST or PCCL_PTR_HGGC.
  pcclResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  pcclResult_t (*regMrDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  pcclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  pcclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      pcclDataType_t dataType, pcclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with PCCL_PTR_HGGC is
  // visible to the GPU
  pcclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  pcclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  pcclResult_t (*closeColl)(void* collComm);
  pcclResult_t (*closeListen)(void* listenComm);
} pcclCollNet_v6_t;

typedef pcclCollNet_v6_t pcclCollNet_t;

#define PCCL_COLLNET_PLUGIN_SYMBOL pcclCollNetPlugin_v6

// v5 struct for backwards compatibility
typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  pcclResult_t (*init)(pcclDebugLogger_t logFunction);
  // Return the number of adapters.
  pcclResult_t (*devices)(int* ndev);
  // Get various device properties.
  pcclResult_t (*getProperties)(int dev, pcclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to PCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  pcclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  pcclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  pcclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either PCCL_PTR_HOST or PCCL_PTR_HGGC.
  pcclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  pcclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  pcclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  pcclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with PCCL_PTR_HGGC is
  // visible to the GPU
  pcclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  pcclResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  pcclResult_t (*closeSend)(void* sendComm);
  pcclResult_t (*closeRecv)(void* recvComm);
  pcclResult_t (*closeListen)(void* listenComm);
} pcclNet_v5_t;

// v5 struct for backwards compatibility
typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  pcclResult_t (*init)(pcclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  pcclResult_t (*devices)(int* ndev);
  // Get various device properties.
  pcclResult_t (*getProperties)(int dev, pcclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to PCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  pcclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  pcclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  pcclResult_t (*reduceSupport)(pcclDataType_t dataType, pcclRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either PCCL_PTR_HOST or PCCL_PTR_HGGC.
  pcclResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  pcclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  pcclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      pcclDataType_t dataType, pcclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with PCCL_PTR_HGGC is
  // visible to the GPU
  pcclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  pcclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  pcclResult_t (*closeColl)(void* collComm);
  pcclResult_t (*closeListen)(void* listenComm);
} pcclCollNet_v5_t;

// v4 struct for backwards compatibility
typedef struct {
  char* name;     // Used mostly for logging.
  char* pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // PCCL_PTR_HOST or PCCL_PTR_HOST|PCCL_PTR_HGGC
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  int maxComms;   // Maximum number of comms we can create
} pcclNetProperties_v4_t;

// v4 struct for backwards compatibility
typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  pcclResult_t (*init)(pcclDebugLogger_t logFunction);
  // Return the number of adapters.
  pcclResult_t (*devices)(int* ndev);
  // Get various device properties.
  pcclResult_t (*getProperties)(int dev, pcclNetProperties_v4_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to PCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  pcclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  pcclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connectHandle
  pcclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either PCCL_PTR_HOST or PCCL_PTR_HGGC.
  pcclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  pcclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  pcclResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  pcclResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle, void** request);
  // Perform a flush/fence to make sure all data received with PCCL_PTR_HGGC is
  // visible to the GPU
  pcclResult_t (*iflush)(void* recvComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  pcclResult_t (*test)(void* request, int* done, int* size);
  // Close and free send/recv comm objects
  pcclResult_t (*closeSend)(void* sendComm);
  pcclResult_t (*closeRecv)(void* recvComm);
  pcclResult_t (*closeListen)(void* listenComm);
} pcclNet_v4_t;

// v4 struct for backwards compatibility
typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  pcclResult_t (*init)(pcclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  pcclResult_t (*devices)(int* ndev);
  // Get various device properties.
  pcclResult_t (*getProperties)(int dev, pcclNetProperties_v4_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to PCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  pcclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  pcclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  pcclResult_t (*reduceSupport)(pcclDataType_t dataType, pcclRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either PCCL_PTR_HOST or PCCL_PTR_HGGC.
  pcclResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  pcclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  pcclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      pcclDataType_t dataType, pcclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with PCCL_PTR_HGGC is
  // visible to the GPU
  pcclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  pcclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  pcclResult_t (*closeColl)(void* collComm);
  pcclResult_t (*closeListen)(void* listenComm);
} pcclCollNet_v4_t;

#endif // end include guard
