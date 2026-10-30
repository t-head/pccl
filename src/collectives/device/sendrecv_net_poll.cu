// SPDX-License-Identifier: BSD-3-Clause
#include "sendrecv.h"
#include "common.h"
#include "collectives.h"

#if defined(__HGGC_ARCH__) && __HGGC_ARCH__ == 150
#define PCCL_NET_POLL_BOUNDS __launch_bounds__(NCCL_KERN_LAUNCH_BOUNDS_MAX_NTHREADS, NCCL_KERN_LAUNCH_BOUNDS_MIN_NBLOCKS)
#else
#define PCCL_NET_POLL_BOUNDS
#endif
__global__ void PCCL_NET_POLL_BOUNDS NCCL_KERN_NAME(SendRecvNetPoll, RING, SIMPLE, Sum, int8_t)(ncclDevKernelArgs1K const args) {
  ncclKernelMain<ncclFuncSendRecv, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE,
      FuncSum<int8_t>, int8_t, ncclCollUnroll(), 0, true>(&args.args);
}
#undef PCCL_NET_POLL_BOUNDS
