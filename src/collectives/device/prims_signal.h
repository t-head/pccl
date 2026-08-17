/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PRIMS_SIGNAL_H_
#define NCCL_PRIMS_SIGNAL_H_

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC, int P2P>
class Primitives<T, FUNC, ProtoSignal<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P> :
    public PrimitivesSimpleBase<T, NRECV, NSEND, FUNC, UNROLL, SLICESPERCHUNK, SLICESTEPS, DIRECT,
      ProtoSignal<UNROLL, SLICESPERCHUNK, SLICESTEPS>,
      Primitives<T, FUNC, ProtoSignal<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P>, P2P> {
 private:
  uint32_t minPathNum = 0;
  volatile uint32_t* signalMarkerPtr = NULL;
  uint32_t signalMarkerCache; // Cache last seen value

  __device__ __forceinline__ void loadRecvConnExtra(ncclDevChannelPeer *devChannelPeer, int connIndex, T* directBuff) {
    if (role & (ROLE_WAIT_RECV|ROLE_POST_RECV)) {
      conn = &devChannelPeer->recv[connIndex];
      minPathNum = conn->minPathNum;
      if (role & ROLE_WAIT_RECV) {
        signalMarkerPtr = conn->signalMarker;
        signalMarkerCache = signalMarkerPtr[step % NCCL_STEPS];
      }
    }
  }

  __device__ __forceinline__ void loadSendConnExtra(ncclDevChannelPeer *devChannelPeer) {
    if (role & ROLE_POST_SEND) {
      signalMarkerPtr = conn->signalMarker;
    }
  }

  using Base = PrimitivesSimpleBase<T, NRECV, NSEND, FUNC, UNROLL, SLICESPERCHUNK, SLICESTEPS, DIRECT,
                 ProtoSignal<UNROLL, SLICESPERCHUNK, SLICESTEPS>,
                 Primitives<T, FUNC, ProtoSignal<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P>, P2P>;
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
      const int tid, const int nworkers, int* recvPeers, int* sendPeers,
      T* directBuff, struct ncclKernelComm* comm, uint64_t redOpArg, struct ncclShmemGroup* ptrs,
      int group, uint8_t connIndexRecv=0, uint8_t connIndexSend=0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork=nullptr, int ncclSteps=NCCL_STEPS, bool noRed=false
    ): PrimitivesSimpleBase<T, NRECV, NSEND, FUNC, UNROLL, SLICESPERCHUNK, SLICESTEPS, DIRECT,
        ProtoSignal<UNROLL, SLICESPERCHUNK, SLICESTEPS>,
        Primitives<T, FUNC, ProtoSignal<UNROLL, SLICESPERCHUNK, SLICESTEPS>, NRECV, NSEND, DIRECT, P2P>, P2P>
        (tid, nworkers, recvPeers, sendPeers, directBuff, comm, redOpArg, ptrs, group, connIndexRecv, connIndexSend, collWork, p2pWork, ncclSteps, noRed) {
    loadRecvConnExtra(getShmemPtr(comm)->channel.peers[peer], connIndexRecv, directBuff);
    loadSendConnExtra(getShmemPtr(comm)->channel.peers[peer]);
  }

  template <int SRC, int DIRECTRECV>
  inline __device__ void waitRecv(ssize_t directOffset) {
    hangSpins = spins = 0;
    do {
      signalMarkerCache = signalMarkerPtr[step%NCCL_STEPS];
      if (internalCheckAbort(spins, hangSpins)) break;
    } while (signalMarkerCache != minPathNum); // check if marker ready for all possible route paths
    signalMarkerPtr[step%NCCL_STEPS] = 0;
    if (connOffsFifoPtr) srcs[SRC+index] = buff + loadInt(connOffsFifoPtr + (step%NCCL_STEPS))/sizeof(T);
    else srcs[SRC+index] = this->template directPtr<DIRECTRECV>(directOffset);
    step += SLICESTEPS;
  }

  template<int SEND>
  __device__ __forceinline__ void memoryFence(int size) {
    if (SEND && (role & ROLE_POST_SEND)) {
      #pragma unroll
      for (int i = 0; i < NSEND; i++) {
        if (index == i) __ppu_signal_peer(&signalMarkerPtr[step%NCCL_STEPS]);
      }
    }
  }
};

#endif
