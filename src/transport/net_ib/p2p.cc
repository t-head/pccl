/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "p2p.h"
#include "common.h"
#include "connect.h"

NCCL_PARAM(IbArThreshold, "IB_AR_THRESHOLD", 8192);

ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase* base, struct ncclIbRequest** req) {
  for (int i = 0; i < NET_IB_MAX_REQUESTS; i++) {
    struct ncclIbRequest* r = base->reqs + i;
    if (r->type == NCCL_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->events = 1;
      r->sock = NULL;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/IB : unable to allocate requests");
  *req = NULL;
  return ncclInternalError;
}
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}

ncclResult_t ncclSendCheck(struct ncclIbSendComm* comm) {
  struct ncclIbQpInfo remQpInfo;

  // Do not block on this receive, return if not ready.
  int bytes = 0;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, &remQpInfo, sizeof(remQpInfo), &bytes));
  if (bytes == 0) return ncclSuccess; // Try again later
  NCCLCHECK(ncclSocketWait(NCCL_SOCKET_RECV, &comm->base.sock, &remQpInfo, sizeof(remQpInfo), &bytes));

  for (int q=0; q<comm->base.nqps; q++) {
    struct ibv_qp* qp = comm->base.qps[q];
    NCCLCHECK(ncclIbRtrQp(qp, remQpInfo.qpn[q], &remQpInfo, ncclIbGidIndexFromDevice(comm->netDev)));
    NCCLCHECK(ncclIbRtsQp(qp));
  }
  comm->base.ready = 1;
  // Block until this is done. It *should* not block indefinitely.
  NCCLCHECK(ncclSocketSend(&comm->base.sock, &comm->base.ready, sizeof(int)));

  return ncclSuccess;
}

ncclResult_t ncclRecvCheck(struct ncclIbRecvComm* comm) {
  // Do not block on this receive, return if not ready.
  int bytes = 0;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, &comm->base.ready, sizeof(int), &bytes));
  if (bytes == 0) return ncclSuccess; // Try again later
  NCCLCHECK(ncclSocketWait(NCCL_SOCKET_RECV, &comm->base.sock, &comm->base.ready, sizeof(int), &bytes));
  return ncclSuccess;
}

ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  volatile struct ncclIbSendFifo* slots = comm->ctsFifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  TRACE(NCCL_NET, "NET/IB: %s: Posting a send request (req=%p, comm=%p, slot=%d, nreqs=%d)",
        __func__, reqs[0], reqs[0]->base, slot, nreqs);
  uint64_t wr_id = 0ULL;

  for (int r = 0; r < nreqs; r++) {
    struct ibv_send_wr* wr = comm->wrs + r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge* sge = comm->sges + r;
    sge->addr = (uintptr_t)reqs[r]->send.data;
    sge->lkey = reqs[r]->send.lkey;

    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->wr.rdma.rkey = slots[r].rkey;
    wr->next = wr + 1;
    wr_id += (reqs[r] - comm->base.reqs) << (r * 8);
  }

  // Write size as immediate data. In the case of multi-send, only write
  // 0 or 1 as size to indicate whether there was data sent or received.
  uint32_t immData = 0;
  if (nreqs == 1) {
    immData = reqs[0]->send.size;
  } else {
    if (nreqs > 32) {
      WARN("Cannot store sizes of %d requests in a 32-bits field", nreqs);
      return ncclInternalError;
    }
    for (int r = 0; r < nreqs; r++) {
      immData |= (reqs[r]->send.size ? 1 : 0) << r;
    }
  }

  struct ibv_send_wr* lastWr = comm->wrs + nreqs - 1;
  if (nreqs > 1 || (/* comm->ar && */reqs[0]->send.size > ncclParamIbArThreshold())) {
    // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128 protocols still work
  const int align = 128;
  const int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
  for (int q = 0; q < nqps; q++) {
    TRACE(NCCL_NET, "NET/IB: %s: Posting send (req=%p, comm=%p, slot=%d, nreqs=%d, wr_id=%ld) on QP (qp_num=%u, qpIndex=%d)",
          __func__, reqs[0], reqs[0]->base, slot, nreqs, wr_id, comm->base.qps[comm->base.qpIndex]->qp_num, comm->base.qpIndex);
    for (int r = 0; r < nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length = std::min(reqs[r]->send.size - reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges + r;
        comm->wrs[r].num_sge = 1;
      }
    }
    struct ibv_send_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_send(comm->base.qps[comm->base.qpIndex], comm->wrs, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex + 1) % comm->base.nqps;

    for (int r = 0; r < nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
      TRACE(NCCL_NET, "NET/IB: %s: Send request (req=%p, comm=%p, slot=%d, nreqs=%d, reqIdx=%d, wr_id=%ld) posted %d bytes, total posted %d/%d bytes",
            __func__, reqs[r], reqs[0]->base, slot, nreqs, r, comm->wrs[r].wr_id, comm->wrs[r].sg_list->length, reqs[r]->send.offset, reqs[r]->send.size);
    }
  }

  TRACE(NCCL_NET, "NET/IB: %s: Send request posted (req=%p, comm=%p, slot=%d, nreqs=%d, wr_id=%ld)",
        __func__, reqs[0], reqs[0]->base, slot, nreqs, wr_id);

  return ncclSuccess;
}

ncclResult_t ncclIbIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  if (comm->base.ready == 0) { NCCLCHECK(ncclSendCheck(comm)); }
  if (comm->base.ready == 0) { *request = NULL; return ncclSuccess; }
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats, __func__));

  struct ibv_mr* mr = (struct ibv_mr*)mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct ncclIbSendFifo* slots;

  int slot = (comm->base.fifoHead) % NET_IB_MAX_REQUESTS;
  struct ncclIbRequest** reqs = comm->fifoReqs[slot];
  slots = comm->ctsFifo[slot];
  uint64_t idx = comm->base.fifoHead + 1;
  if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
  nreqs = slots[0].nreqs;

  // Wait until all data has arrived
  for (int r = 1; r < nreqs; r++) while(slots[r].idx != idx);
  __sync_synchronize(); // order the nreqsPtr load against tag/rkey/addr loads below
  for (int r = 0; r < nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag) continue;

    // Sanity checks to catch user collective call count/size mismatches
    if (size > slots[r].size) {
      char line[SOCKET_NAME_MAXLEN + 1];
      WARN("NET/IB : req %d/%d tag %x peer %s collective mismatch error, local size %d remote size %d",
           r, nreqs, tag, ncclSocketToString(&comm->base.sock.addr, line), size, slots[r].size);
      return ncclInvalidUsage;
    } // plus any potential programming errors
    else if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkey == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      WARN("NET/IB : req %d/%d tag %x peer %s posted incorrect receive info: size %d addr %lx rkey %x",
           r, nreqs, tag, ncclSocketToString(&comm->base.sock.addr, line), slots[r].size, slots[r].addr, slots[r].rkey);
      return ncclInternalError;
    }
    struct ncclIbRequest* req;
    NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
    req->base = &comm->base;
    req->devBase = &comm->dev.base;
    req->type = NCCL_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.lkey = mr->lkey;
    req->send.offset = 0;
    req->events = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
    *request = reqs[r] = req;
    TRACE(NCCL_NET, "NET/IB: %s: Send request created (req=%p, comm=%p, slot=%d, reqIdx=%d, nreqs=%d, tag=%x, size=%ld, data=0x%016" PRIx64 ")",
          __func__, req, req->base, slot, r, nreqs, tag, size, (uint64_t)data);

    // If this is a multi-recv, send only when all requests have matched.
    for (int r = 0; r < nreqs; r++) {
      if (reqs[r] == NULL) return ncclSuccess;
    }

    NCCLCHECK(ncclIbMultiSend(comm, slot));

    // Clear slots[0]->nreqs, as well as other fields to help debugging and sanity checks
    memset((void*)slots, 0, sizeof(struct ncclIbSendFifo));
    memset(reqs, 0, NCCL_NET_IB_MAX_RECVS * sizeof(struct ncclIbRequest*));
    comm->base.fifoHead++;
    return ncclSuccess;
  }

  *request = NULL;
  return ncclSuccess;
}

ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, int n, void** data, int* sizes, int* tags, void** mhandles, struct ncclIbRequest* req) {
  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->remCtsFifo.fifoTail % NET_IB_MAX_REQUESTS;
  struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];

  for (int i = 0; i < n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct ibv_mr* mr = (struct ibv_mr*)mhandles[i];
    localElem[i].rkey = mr->rkey;
    localElem[i].nreqs = n;
    localElem[i].size = sizes[i]; // Sanity/Debugging
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->remCtsFifo.fifoTail + 1;
  }
  wr.wr.rdma.remote_addr = comm->remCtsFifo.addr + slot * NCCL_NET_IB_MAX_RECVS * sizeof(struct ncclIbSendFifo);
  wr.wr.rdma.rkey = comm->remCtsFifo.rkey;
  comm->remCtsFifo.sge.addr = (uint64_t)localElem;
  comm->remCtsFifo.sge.length = n * sizeof(struct ncclIbSendFifo);
  wr.sg_list = &comm->remCtsFifo.sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remCtsFifo.flags; // IBV_SEND_INLINE

  // We need to occasionally post a request with the IBV_SEND_SIGNALED flag, otherwise
  // the send queue will never empty.
  //
  // From https://www.rdmamojo.com/2014/06/30/working-unsignaled-completions/
  // "How to use Unsignaled Completion?" / "Gotchas and Pitfalls"
  // All posted Send Requested, Signaled and Unsignaled, are considered outstanding until
  // a Work Completion that they, or Send Requests that were posted after them, was polled
  // from the Completion Queue associated with the Send Queue. This means if one works with
  // a Queue Pair that was configured to work with Unsignaled Completions, he must make
  // sure that occasionally (before the Send Queue is full with outstanding Send Requests)
  // a Send Request that generate Work Completion will be posted.
  //
  // Not following this rule may lead to a case that the Send Queue is full with Send
  // Requests that won't generate Work Completion:
  //
  //  - The Send Queue is full, so no new Send Requests can be posted to it
  //  - The Send Queue can't be emptied, since no Work Completion can be generated anymore
  //    (the reason is that no Work Completion, that can generate Work Completion that
  //    polling it will empty the Send Queue, can be posted)
  //  - The status of all posted Send Request is considered unknown
  //
  if (slot == 0) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->base.reqs;
    req->events++;
  }
  TRACE(NCCL_NET, "NET/IB: %s: Posting a CTS (req=%p, comm=%p,  slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d)",
        __func__, req, req->base, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags);

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(comm->base.qps[0], &wr, &bad_wr));
  comm->remCtsFifo.fifoTail++;
  TRACE(NCCL_NET, "NET/IB: %s: CTS posted (req=%p, comm=%p, slot=%d, nreqs=%d, wr_id=%ld, opcode=%d, send_flags=%d)",
        __func__, req, req->base, slot, req->nreqs, wr.wr_id, wr.opcode, wr.send_flags);

  return ncclSuccess;
}

ncclResult_t ncclIbIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  if (comm->base.ready == 0) NCCLCHECK(ncclRecvCheck(comm));
  if (comm->base.ready == 0) { *request = NULL; return ncclSuccess; }

  if (n > NCCL_NET_IB_MAX_RECVS) return ncclInternalError;
  NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats, __func__));

  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->base = &comm->base;
  req->devBase = &comm->dev.base;
  req->type = NCCL_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;
  TRACE(NCCL_NET, "NET/IB: %s: Recv request created (req=%p, comm=%p, nreqs=%d, tag[0]=%x)",
        __func__, req, req->base, n, tags[0]);

  for (int i = 0; i < n; i++) req->recv.sizes[i] = 0;

  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;

  wr.sg_list = NULL;
  wr.num_sge = 0;

  const int nqps = ncclIbCommBaseGetNqpsPerRequest(&comm->base);
  for (int q = 0; q < nqps; q++) {
    struct ibv_qp* qp = comm->base.qps[comm->base.qpIndex];
    struct ibv_recv_wr* bad_wr;
    NCCLCHECK(wrap_ibv_post_recv(qp, &wr, &bad_wr));
    comm->base.qpIndex = (comm->base.qpIndex + 1) % comm->base.nqps;
  }
  req->events = nqps;

  *request = req;

  // Post to FIFO to notify sender
  NCCLCHECK(ncclIbPostFifo(comm, n, data, sizes, tags, mhandles, req));
  return ncclSuccess;
}

ncclResult_t ncclIbIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  int last = -1;
  for (int i = 0; i < n; i++) if (sizes[i]) last = i;
  if (comm->gpuFlush.enabled == 0 || last == -1) return ncclSuccess;

  // Only flush once using the last non-zero receive
  struct ncclIbRequest* req;
  NCCLCHECK(ncclIbGetRequest(&comm->base, &req));
  req->base = &comm->base;
  req->devBase = &comm->dev.base;
  req->type = NCCL_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  struct ibv_mr* mr = (struct ibv_mr*)mhandles[last];

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = req - comm->base.reqs;

  struct ibv_sge sge;
  if (comm->gpuFlush.useWrite) {
    // The NIC reads one byte from GPU memory and writes it to the local host
    // buffer through the loopback QP, ordering prior RDMA Writes to the GPU.
    sge.addr = (uint64_t)data[last];
    sge.length = 1;
    sge.lkey = mr->lkey;
    wr.wr.rdma.remote_addr = (uint64_t)&comm->gpuFlush.hostMem;
    wr.wr.rdma.rkey = comm->gpuFlush.hostMr->rkey;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
  } else {
    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = mr->rkey;
    wr.sg_list = &comm->gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
  }
  wr.send_flags = IBV_SEND_SIGNALED;

  struct ibv_send_wr* bad_wr;
  NCCLCHECK(wrap_ibv_post_send(comm->gpuFlush.qp, &wr, &bad_wr));

  *request = req;
  return ncclSuccess;
}

// Log the details of a completion with error. The provided devIndex is the index
// of the IB device on which the completion was received.
static ncclResult_t ncclIbLogCompletionWithError(struct ncclIbNetCommBase* commBase, struct ibv_wc* wc, int devIndex) {
  struct ncclIbNetCommDevBase* devBase = ncclIbGetNetCommDevBase(commBase, devIndex);

  char sockStr[SOCKET_NAME_MAXLEN+1];
  union ncclSocketAddress addr;
  ncclSocketGetAddr(&commBase->sock, &addr);
  ncclSocketToString(&addr, sockStr);
  char *hcaName = devBase->pd->context->device->name;
  WARN("NET/IB: Got completion from peer %s with status=%s(%d) opcode=%s(%d) vendor_err=%u hca %s",
       sockStr, ibvWcStatusStr(wc->status), wc->status,
       ibvWcOpcodeStr(wc->opcode), wc->opcode, wc->vendor_err, hcaName);
  return ncclSuccess;
}

ncclResult_t ncclIbTest(void* request, int* done, int* sizes) {
  struct ncclIbRequest *r = (struct ncclIbRequest*)request;
  *done = 0;

  while (1) {
    NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats, __func__));
    if (r->events == 0) {
      *done = 1;
      if (sizes && r->type == NCCL_NET_IB_REQ_RECV) {
        for (int i = 0; i < r->nreqs; i++) sizes[i] = r->recv.sizes[i];
      }
      NCCLCHECK(ncclIbFreeRequest(r));
      return ncclSuccess;
    }

    int wrDone = 0;
    struct ibv_wc wcs[4];
    NCCLCHECK(wrap_ibv_poll_cq(r->devBase->cq, 4, wcs, &wrDone));
    if (wrDone == 0) return ncclSuccess;

    for (int w = 0; w < wrDone; w++) {
      struct ibv_wc *wc = wcs + w;
      if (wc->status != IBV_WC_SUCCESS) {
        union ncclSocketAddress addr;
        ncclSocketGetAddr(r->sock, &addr);
        WARN("NET/IB: %s: Got CQE with error (req=%p, comm=%p (%s), wr_id=%lu, qp_num=%d)",
             __func__, r, r->base, r->type == NCCL_NET_IB_REQ_SEND ? "send" : "recv", wc->wr_id, wc->qp_num);
        ncclIbLogCompletionWithError(r->base, wc, r->devBase->ibDevN);
        return ncclRemoteError;
      }
      TRACE(NCCL_NET, "NET/IB: %s: Processing a completion event (comm=%p (%s), req=%p, wr_id=%lu, qp_num=%d)",
            __func__, r->base, r->type == NCCL_NET_IB_REQ_SEND ? "send" : "recv", r, wc->wr_id, wc->qp_num);

      struct ncclIbRequest* req = r->base->reqs + (wc->wr_id & 0xff);
      if (req->type == NCCL_NET_IB_REQ_SEND) {
        for (int i = 0; i < req->nreqs; i++) {
          struct ncclIbRequest* sendReq = r->base->reqs + ((wc->wr_id >> (i * 8)) & 0xff);
          if ((sendReq->events <= 0)) return ncclInternalError;
          sendReq->events--;
        }
      } else {
        if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          if (req->type != NCCL_NET_IB_REQ_RECV) return ncclInternalError;
          if (req->nreqs > 1) {
            // In the case of a multi recv, we only set sizes to 0 or 1.
            for (int i = 0; i < req->nreqs; i++) {
              req->recv.sizes[i] = (wc->imm_data >> i) & 0x1;
            }
          } else {
            req->recv.sizes[0] += wc->imm_data;
          }
        }
        req->events--;
      }
    }
    // Once the IB fatal event is reported in the async thread, we want to propagate this error
    // to communicator and prevent further polling to reduce error pollution.
    NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBase->ibDevN].stats, __func__));
  }
}
