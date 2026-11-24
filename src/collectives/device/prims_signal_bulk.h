/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PRIMS_SIGNAL_BULK_H_
#define NCCL_PRIMS_SIGNAL_BULK_H_

// Implementation of primitive types
template <typename T, typename Fan, typename RedOp, int SlicePerChunk, int StepPerSlice,
         int Unroll, int Direct, ncclConn_t SENDCONNTYPE, ncclConn_t RECVCONNTYPE, int P2p, bool isNetOffload>
class Primitives<T, RedOp, Fan, Direct, ProtoSignalBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>, P2p, isNetOffload> :
    public PrimitivesSimpleBase<T, Fan, RedOp, SlicePerChunk, StepPerSlice, Unroll, Direct,
      ProtoSignalBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>,
      Primitives<T, RedOp, Fan, Direct, ProtoSignalBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>, P2p, isNetOffload>, P2p> {
 private:
  static_assert(!isNetOffload, "isNetOffload is not implemented on PPU");

  uint32_t minPathNum = 0;
  volatile uint32_t* signalMarkerPtr = NULL;
  uint32_t signalMarkerCache; // Cache last seen value

  __device__ __forceinline__ void loadRecvConnExtra(ncclDevChannelPeer *devChannelPeer, int connIndex, T* directBuff) {
    // Caller already gates with (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)).
    conn = &devChannelPeer->recv[connIndex];
    minPathNum = conn->minPathNum;
    if (role & ROLE_WAIT_RECV) {
      signalMarkerPtr = conn->signalMarker;
      signalMarkerCache = signalMarkerPtr[step % NCCL_STEPS];
    }
  }

  __device__ __forceinline__ void loadSendConnExtra(ncclDevChannelPeer *devChannelPeer, int connIndex) {
    // Caller already gates with (role & (ROLE_WAIT_SEND|ROLE_POST_SEND)).
    conn = &devChannelPeer->send[connIndex];
    if (role & ROLE_POST_SEND) {
      signalMarkerPtr = conn->signalMarker;
    }
  }

  using Base = PrimitivesSimpleBase<T, Fan, RedOp, SlicePerChunk, StepPerSlice, Unroll, Direct,
                 ProtoSignalBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>,
                 Primitives<T, RedOp, Fan, Direct, ProtoSignalBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>, P2p, isNetOffload>, P2p>;
  using Base::conn;
  using Base::connOffsFifoPtr;
  using Base::connTailPtr;
  using Base::connTailCache; // Cache last seen value

  using Base::index; // Peer index I'm responsible for
  using Base::peer;
  using Base::role;
  using Base::step;

  using Base::srcs;
  using Base::buff;

  using Base::spins;
  using Base::hangSpins;
  using Base::internalCheckAbort;

 public:
  __device__ __forceinline__ Primitives(
      const int tid, const int nthreads, int* recvPeers, int* sendPeers,
      T* directBuff, struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs,
      int group, uint8_t connIndexRecv=0, uint8_t connIndexSend=0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork=nullptr, int stepSize_=0, const uint64_t ncclSteps=NCCL_STEPS, bool noRed=false
      ): PrimitivesSimpleBase<T, Fan, RedOp, SlicePerChunk, StepPerSlice, Unroll, Direct,
        ProtoSignalBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>,
        Primitives<T, RedOp, Fan, Direct, ProtoSignalBulk<SlicePerChunk, StepPerSlice, Unroll, SENDCONNTYPE, RECVCONNTYPE>, P2p, isNetOffload>, P2p>
        (tid, nthreads, recvPeers, sendPeers, directBuff, comm, redOpArg, ptrs, group, connIndexRecv, connIndexSend, collWork, p2pWork, stepSize_, ncclSteps, noRed) {
    if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV))
      loadRecvConnExtra(getShmemPtr(comm)->channel.peers[peer], connIndexRecv, directBuff);
    if (role & (ROLE_WAIT_SEND|ROLE_POST_SEND))
      loadSendConnExtra(getShmemPtr(comm)->channel.peers[peer], connIndexSend);
  }

  template <int Src, int DirectRecv>
  inline __device__ void waitRecv(ssize_t directOffset) {
    hangSpins = spins = 0;
    do {
      signalMarkerCache = signalMarkerPtr[step%NCCL_STEPS];
      if (internalCheckAbort(spins, hangSpins)) break;
    } while (signalMarkerCache != minPathNum); // check if marker ready for all possible route paths
    signalMarkerPtr[step%NCCL_STEPS] = 0;
    if (connOffsFifoPtr) srcs[Src+index] = buff + loadInt(connOffsFifoPtr + (step%NCCL_STEPS))/sizeof(T);
    else srcs[Src+index] = this->template directPtr<DirectRecv>(directOffset);
    step += StepPerSlice;
  }

  template<int Send>
  __device__ __forceinline__ void memoryFence(int size) {
    if (Send && (role & ROLE_POST_SEND)) {
      #pragma unroll
      for (int i = 0; i < Fan::MaxSend; i++) {
        if (index == i) __ppu_signal_peer(&signalMarkerPtr[step%NCCL_STEPS]);
      }
    }
  }
};
#endif
