//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <dirent.h>
#include <infiniband/verbs.h>
#ifdef NUMA
#include <numa.h>
#endif

#include <hggc_runtime.h>

#include "helper_timer.h"
#include "logger.h"

#define RET_CHECK(cmd, func_tag) do { \
    ret = cmd; \
    if (ret < 0) { \
        printf("%s: %s\n", strerror(errno), func_tag); \
        return ret; \
    } \
} while (0)

#define RNIC_NAME_MAX_LEN 16
#define ELEM_PRINT_LEN 10
#define ELEM_MAX_LEN 128

typedef enum { UNI_BW = 0, BI_BW = 1 } PERF_TYPE;

struct runArgs {
  struct ibv_device **dev_list;
  int bytes;
  int iters;
  int warmup;
  int gid_index;
  int nMems;
  int nDevs;
  int nRnics;
  int nQps; // qp num
  PERF_TYPE type;
};

struct qpInfo {
  uint32_t lid;
  uint8_t ib_port;
  uint32_t qpn;

  // For RoCE
  uint64_t spn;
  uint64_t iid;
  enum ibv_mtu mtu;

  // RDMA buffer info
  uint32_t buffer_key;
  uint64_t buffer_addr;
};

struct rnicDev {
  struct ibv_device *dev;
  struct ibv_context *ctx;
  struct ibv_pd *pd;
  struct ibv_cq **cq_list;
  struct ibv_qp **qp_list;

  float *send_buf;
  float *recv_buf;
  struct ibv_mr *send_mr;
  struct ibv_mr *recv_mr;
  struct qpInfo *ib_info_list;
};

const double EPSILON = 1e-2;
bool isCloseToZero(double value, double epsilon = EPSILON) {
    return abs(value) < epsilon;
}

testResult_t getDevList(struct runArgs *args) {
  testResult_t res = testSuccess;
  args->dev_list = ibv_get_device_list(&args->nRnics);

  if (args->nRnics <= 0) {
    ERROR("Did not detect devices \n");
    ERROR("If device exists, check if driver is up\n");
    res = testRDMAError;
  }

  return res;
}

testResult_t getNumaNodesNum(struct runArgs *args) {
  testResult_t res = testSuccess;
  int numa_node_count = 0;
#ifdef NUMA
  if (numa_available() >= 0) {
    numa_node_count = numa_max_node() + 1;
  } else {
    DIR *dir;
    struct dirent *ent;

    if ((dir = opendir("/sys/devices/system/node/")) != NULL) {
      while ((ent = readdir(dir)) != NULL) {
        if (strncmp(ent->d_name, "node", 4) == 0) {
          numa_node_count++;
        }
      }
      closedir(dir);
    } else {
      ERROR("Unable to open /sys/devices/system/node/\n");
      res = testRDMAError;
    }
  }
#endif
  args->nMems = numa_node_count;
  return res;
}

struct ibv_context* ctxOpenDevice(struct ibv_device *ib_dev) {
  struct ibv_context *context;

  context = ibv_open_device(ib_dev);

  if (!context) {
    ERROR("Couldn't get context for the device\n");
    return NULL;
  }

  return context;
}

void printHeader(struct runArgs *args) {
  int nDevs = args->nDevs;
  int nMems = args->nMems;

  // length of each cell
  int cellLen = ELEM_PRINT_LEN;

  // content may exceed the limit of ELEM_PRINT_LEN
  char *line = (char *)malloc((nMems + nDevs + 1) * ELEM_MAX_LEN);
  if (!line) {
    ERROR("malloc print header failed");
    return;
  }
  size_t offset = 0;

  sprintf(line + offset, "%-*s", RNIC_NAME_MAX_LEN, " ");
  for (int i = 0; i < nMems + nDevs; i++) {
    char tmpStr[16];
    offset = strlen(line);
    if (i < nMems) {
      sprintf(tmpStr, "MEM%d", i);
    } else {
      sprintf(tmpStr, "DEV%d", i-nMems);
    }
    sprintf(line + offset, "%-*s", cellLen, tmpStr);
  }
  INFO("%s", line);
  free(line);
}

void printMatrix(struct runArgs *args, double *matrix, char **rnic_name_list) {
  int nRnics = args->nRnics;
  int nDevs = args->nDevs;
  int nMems = args->nMems;

  printHeader(args);
  // length of each cell
  int cellLen = ELEM_PRINT_LEN;

  // content may exceed the limit of ELEM_PRINT_LEN
  char *line = (char *)malloc((nMems + nDevs + 1) * ELEM_MAX_LEN);
  if (!line) {
    ERROR("malloc print matrix failed");
    return;
  }
  size_t offset = 0;
  for (int i = 0; i < nRnics; i++) {
    char tmpStr[16];
    offset = 0;
    sprintf(tmpStr, "%-*s", RNIC_NAME_MAX_LEN, rnic_name_list[i]);
    sprintf(line + offset, "%-*s", cellLen, tmpStr);
    offset = strlen(line);
    for (int j = 0; j < nMems + nDevs; j++) {
      if (isinf(matrix[i * (nMems + nDevs) + j]) || isCloseToZero(abs(matrix[i * (nMems + nDevs) + j] - 0.0))) {
        sprintf(line + offset, "%-*s", cellLen, " x");
      } else {
        sprintf(line + offset, "%-*.02f", cellLen, matrix[i * (nMems + nDevs) + j]);
      }
      offset = strlen(line);
    }
    INFO("%s", line);
  }
  free(line);
}

bool initRnic(struct runArgs *args, struct rnicDev *rnic, char *rnic_name, int rnic_id) {
  int ep_num = args->nMems + args->nDevs;
  int repeat = args->warmup + args->iters;

  memset(rnic, 0, sizeof(struct rnicDev));

  struct ibv_port_attr port_attr;
  int ib_port = 1; // use default port 1
  int gid_index = args->gid_index;
  union ibv_gid gid;

  rnic->ctx =  ctxOpenDevice(args->dev_list[rnic_id]);
  ibv_query_port(rnic->ctx, ib_port, &port_attr);
  ibv_query_gid(rnic->ctx, ib_port, gid_index, &gid);
  snprintf(rnic_name, RNIC_NAME_MAX_LEN, "%s", ibv_get_device_name(args->dev_list[rnic_id]));
  if (port_attr.state != IBV_PORT_ACTIVE) {
    ERROR("Device %s port %d isn't active", rnic_name, ib_port)
    return false;
  }

  rnic->pd = ibv_alloc_pd(rnic->ctx);
  rnic->cq_list = (struct ibv_cq **)calloc(args->nQps, sizeof(struct ibv_cq *));
  rnic->qp_list = (struct ibv_qp **)calloc(args->nQps, sizeof(struct ibv_qp *));
  rnic->ib_info_list = (struct qpInfo *)calloc(args->nQps, sizeof(struct qpInfo));
  if (!rnic->cq_list || !rnic->qp_list || !rnic->ib_info_list) {
    ERROR("inic cq/qp/ib_info list failed");
    return false;
  }

  for (int i = 0; i < args->nQps; i++) {
    rnic->cq_list[i] = ibv_create_cq(rnic->ctx, ep_num * repeat * 2, NULL, NULL, 0);

    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(struct ibv_qp_init_attr));
    qp_init_attr.send_cq = rnic->cq_list[i];
    qp_init_attr.recv_cq = rnic->cq_list[i];
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr = ep_num * repeat;
    qp_init_attr.cap.max_recv_wr = ep_num * repeat;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    rnic->qp_list[i] = ibv_create_qp(rnic->pd, &qp_init_attr);
    if (!rnic->qp_list[i]) {
      ERROR("create qp[%d] failed", i);
      return false;
    }

    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = ib_port;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    ibv_modify_qp(rnic->qp_list[i], &qp_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

    // fill qpInfo
    rnic->ib_info_list[i].ib_port = ib_port;
    rnic->ib_info_list[i].qpn = rnic->qp_list[i]->qp_num;
    rnic->ib_info_list[i].mtu = port_attr.active_mtu;
    rnic->ib_info_list[i].lid = port_attr.lid;
    if (!rnic->ib_info_list[i].lid) {
        // RoCE
        union ibv_gid gid;
        ibv_query_gid(rnic->ctx, ib_port, gid_index, &gid);
        rnic->ib_info_list[i].spn = gid.global.subnet_prefix;
        rnic->ib_info_list[i].iid = gid.global.interface_id;
    }
  }

  return true;
}

testResult_t initEndpointMem(struct runArgs *args, struct rnicDev *rnic, int rnic_id, int ep_id) {
  testResult_t res = testSuccess;
  if (ep_id < args->nMems) {
#ifdef NUMA
    if (numa_available() >= 0) {
      numa_run_on_node(ep_id);

      rnic->send_buf = (float*)numa_alloc_local(args->bytes);
      rnic->recv_buf = (float*)numa_alloc_local(args->bytes);
      if (!rnic->send_buf || !rnic->recv_buf) {
        ERROR("numa_alloc_local failed");
        res = testRDMAError;
      } else {
        // set default msg 5.0f
        for (int i = 0; i < args->bytes / sizeof(float); i++) {
          rnic->send_buf[i] = 5.0f;
          rnic->recv_buf[i] = 0.0f;
        }
      }
    }
#endif
  } else {
    HGGCCHECK(hggcSetDevice(ep_id - args->nMems));
    HGGCCHECK(hggcMalloc(&rnic->send_buf, args->bytes));
    HGGCCHECK(hggcMalloc(&rnic->recv_buf, args->bytes));
    float *h_send_buf = (float *)calloc(args->bytes / sizeof(float), sizeof(float));
    float *h_recv_buf = (float *)calloc(args->bytes / sizeof(float), sizeof(float));
    for (int i = 0; i < args->bytes / sizeof(float); i++) {
      h_send_buf[i] = 5.0f;
      h_recv_buf[i] = 0.0f;
    }
    HGGCCHECK(hggcMemcpy(rnic->send_buf, h_send_buf, args->bytes, hggcMemcpyHostToDevice));
    HGGCCHECK(hggcMemcpy(rnic->recv_buf, h_recv_buf, args->bytes, hggcMemcpyHostToDevice));
  }

  if (res == testSuccess) {
    rnic->send_mr = ibv_reg_mr(rnic->pd, rnic->send_buf, args->bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    rnic->recv_mr = ibv_reg_mr(rnic->pd, rnic->recv_buf, args->bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!rnic->send_mr || !rnic->recv_mr) {
      ERROR("ibv_reg_mr failed");
      res = testRDMAError;
    }
  }

  return res;
}

struct ibv_send_wr *** initSendWrList(struct runArgs *args, struct rnicDev *rnic) {
  int repeat = args->warmup + args->iters;

  struct ibv_send_wr ***send_wr_list = (struct ibv_send_wr ***)calloc(args->nQps, sizeof(struct ibv_send_wr **));
  for (int i = 0; i < args->nQps; i++) {
    struct ibv_send_wr **send_wr = (struct ibv_send_wr **)calloc(repeat, sizeof(struct ibv_send_wr *));
    struct ibv_sge *send_sge = (struct ibv_sge *)calloc(1, sizeof(struct ibv_sge));
    // offset
    send_sge->addr = (uint64_t)rnic->send_buf + i * args->bytes / args->nQps;
    send_sge->length = args->bytes / args->nQps;
    send_sge->lkey = rnic->send_mr->lkey;

    for (int j = 0; j < repeat; j++) {
      send_wr[j] = (struct ibv_send_wr *)calloc(1, sizeof(struct ibv_send_wr));

      send_wr[j]->wr_id = (uint64_t)(&rnic);
      send_wr[j]->sg_list = send_sge;
      send_wr[j]->num_sge = 1;
      if (args->type == BI_BW) {
        send_wr[j]->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      } else {
        send_wr[j]->opcode = IBV_WR_RDMA_WRITE;
      }
      send_wr[j]->send_flags = IBV_SEND_SIGNALED;
      send_wr[j]->wr.rdma.remote_addr = (uint64_t)rnic->recv_mr->addr + i * args->bytes / args->nQps;
      send_wr[j]->wr.rdma.rkey = rnic->recv_mr->rkey;
      if (args->type == BI_BW) send_wr[j]->imm_data = args->bytes; // Send the message size via imm_data
    }
    send_wr_list[i] = send_wr;
  }

  // INFO("initSendWrList success");
  return send_wr_list;
}

struct ibv_recv_wr *** initRecvWrList(struct runArgs *args, struct rnicDev *rnic) {
  int repeat = args->warmup + args->iters;

  struct ibv_recv_wr ***recv_wr_list = (struct ibv_recv_wr ***)calloc(args->nQps, sizeof(struct ibv_recv_wr **));
  for (int i = 0; i < args->nQps; i++) {
    struct ibv_recv_wr **recv_wr = (struct ibv_recv_wr **)calloc(repeat, sizeof(struct ibv_recv_wr *));
    struct ibv_sge *recv_sge = (struct ibv_sge *)calloc(1, sizeof(struct ibv_sge));
    // offset
    recv_sge->addr = (uint64_t)rnic->recv_buf + i * args->bytes / args->nQps;
    recv_sge->length = args->bytes / args->nQps;
    recv_sge->lkey = rnic->recv_mr->lkey;

    for (int j = 0; j < repeat; j++) {
      recv_wr[j] = (struct ibv_recv_wr *)calloc(1, sizeof(struct ibv_recv_wr));

      recv_wr[j]->wr_id = (uint64_t)(&rnic);
      recv_wr[j]->sg_list = recv_sge;
      recv_wr[j]->num_sge = 1;
    }
    recv_wr_list[i] = recv_wr;
  }
  return recv_wr_list;
}

int qpSetupRtr(struct runArgs *args, struct ibv_qp *qp, struct qpInfo *info) {
  int ret = 0;

  struct ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.path_mtu = info->mtu;
  qp_attr.dest_qp_num = info->qpn;
  qp_attr.rq_psn = 0;
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 12;
  if (info->lid == 0) {
      qp_attr.ah_attr.is_global = 1;
      qp_attr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
      qp_attr.ah_attr.grh.dgid.global.interface_id = info->iid;
      qp_attr.ah_attr.grh.flow_label = 0;
      qp_attr.ah_attr.grh.sgid_index = args->gid_index;
      qp_attr.ah_attr.grh.hop_limit = 255;
      qp_attr.ah_attr.grh.traffic_class = 0;
  } else {
      qp_attr.ah_attr.is_global = 0;
      qp_attr.ah_attr.dlid = info->lid;
  }
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = info->ib_port;
  RET_CHECK(ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
          "ibv_modify_qp rtr");
  return ret;
}

int qpSetupRts(struct ibv_qp* qp) {
  int ret = 0;

  struct ibv_qp_attr qp_attr;
  memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.timeout = 14;
  qp_attr.retry_cnt = 7;
  qp_attr.rnr_retry = 7;
  qp_attr.sq_psn = 0;
  qp_attr.max_rd_atomic = 1;
  RET_CHECK(ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC),
          "ibv_modify_qp rts");
  return ret;
}

void freeWrAndSge(struct runArgs *args, struct ibv_send_wr ***send_wr_list, struct ibv_recv_wr ***recv_wr_list) {
  for (int i = 0; i < args->nQps; i++) {
    if (send_wr_list[i][0]->sg_list) free(send_wr_list[i][0]->sg_list);
    if (recv_wr_list[i][0]->sg_list) free(recv_wr_list[i][0]->sg_list);
    for (int j = 0; j < args->warmup + args->iters; j++) {
      free(send_wr_list[i][j]);
      free(recv_wr_list[i][j]);
    }
    free(send_wr_list[i]);
    free(recv_wr_list[i]);
  }
  free(send_wr_list);
  free(recv_wr_list);
}

void freeRnic(struct runArgs *args, struct rnicDev *rnic) {
  for (int i = 0; i < args->nQps; i++) {
    if (rnic->qp_list[i]) ibv_destroy_qp(rnic->qp_list[i]);
    if (rnic->cq_list[i]) ibv_destroy_cq(rnic->cq_list[i]);
  }
  if (rnic->pd) ibv_dealloc_pd(rnic->pd);
  if (rnic->ctx) ibv_close_device(rnic->ctx);
}

void freeEndpoint(struct runArgs *args, struct rnicDev *rnic, int ep_id) {
  if (ep_id < args->nMems) {
#ifdef NUMA
    if (rnic->send_buf) numa_free(rnic->send_buf, args->bytes);
    if (rnic->recv_buf) numa_free(rnic->recv_buf, args->bytes);
#endif
  } else {
    if (rnic->send_buf) hggcFree(rnic->send_buf);
    if (rnic->recv_buf) hggcFree(rnic->recv_buf);
  }
  if (rnic->send_mr) ibv_dereg_mr(rnic->send_mr);
  if (rnic->recv_mr) ibv_dereg_mr(rnic->recv_mr);
}

testResult_t testSend(struct runArgs *args, struct rnicDev *rnic, struct ibv_send_wr ***send_wr_list, int repeat_rank) {
  for (int i = 0; i < args->nQps; i++) {
      if (ibv_post_send(rnic->qp_list[i], send_wr_list[i][repeat_rank], NULL)) {
        ERROR("ibv_post_send failed. QP: %d, Repeat: %d", i, repeat_rank);
        return testRDMAError;
      }
  }
  return testSuccess;
}

testResult_t testRecv(struct runArgs *args, struct rnicDev *rnic, struct ibv_recv_wr ***recv_wr_list, int repeat_rank) {
  for (int i = 0; i < args->nQps; i++) {
    if (ibv_post_recv(rnic->qp_list[i], recv_wr_list[i][repeat_rank], NULL)) {
      ERROR("ibv_post_recv failed. QP: %d, Repeat: %d", i, repeat_rank);
      return testRDMAError;
    }
  }
  return testSuccess;
}

testResult_t cqPolling(struct runArgs *args, struct rnicDev *rnic, int repeat_rank) {
  for (int i = 0; i < args->nQps; i++) {
    struct ibv_wc wc;
    bool send_poll = true;
    bool recv_poll = args->type == BI_BW ? true : false;
    int wr_done = 0;
    while (send_poll || recv_poll) {
      wr_done = ibv_poll_cq(rnic->cq_list[i], 1, &wc);
      if (wr_done < 0) {
        ERROR("ibv_poll_cq failed. QP: %d, Repeat: %d", i, repeat_rank);
        return testRDMAError;
      } else if (wr_done == 0) {
        // INFO("send and recv polling. QP: %d, Repeat: %d", i, repeat_rank);
        // sleep(1);
        continue;
      } else {
        if (wc.status != IBV_WC_SUCCESS) {
          ERROR("polling operation failed: %d. QP: %d, Repeat: %d", wc.status, i, repeat_rank);
          return testRDMAError;
        }
        if (wc.opcode == IBV_WC_RDMA_WRITE) {
          // INFO("IBV_WC_RDMA_WRITE success");
          send_poll = false;
        } else if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
          // INFO("IBV_WC_RECV_RDMA_WITH_IMM success");
          recv_poll = false;
        }
      }
    }
  }
  return testSuccess;
}

testResult_t testBandwidth(struct runArgs *args, struct rnicDev *rnic,
                            double *intraHostBandWidthMatrix, int rnic_id, int ep_id) {
  testResult_t res = testSuccess;
  StopWatchInterface *stopWatch = NULL;
  if (!sdkCreateTimer(&stopWatch)) {
    ERROR("Failed to create stop watch");
    exit(EXIT_FAILURE);
  }
  sdkStartTimer(&stopWatch);

  struct ibv_send_wr ***send_wr_list = initSendWrList(args, rnic);
  struct ibv_recv_wr ***recv_wr_list = initRecvWrList(args, rnic);

  for (int i = 0; i < args->nQps; i++) {
    qpSetupRtr(args, rnic->qp_list[i], &rnic->ib_info_list[i]);
    qpSetupRts(rnic->qp_list[i]);
  }
  // warm-up
  for (int i = 0; i < args->warmup; i++) {
    if (args->type == BI_BW) testRecv(args, rnic, recv_wr_list, i);
    testSend(args, rnic, send_wr_list, i);
    cqPolling(args, rnic, i);
  }

  sdkResetTimer(&stopWatch);
  for (int i = 0; i < args->iters; i++) {
    if (args->type == BI_BW) testRecv(args, rnic, recv_wr_list, i+args->warmup);
    testSend(args, rnic, send_wr_list, i+args->warmup);
    cqPolling(args, rnic, i+args->warmup);
  }
  double elapsed_time = sdkGetTimerValue(&stopWatch) / args->iters / 1e3;
  if (args->type == BI_BW) {
    intraHostBandWidthMatrix[rnic_id * (args->nMems + args->nDevs) + ep_id] = args->bytes * 2 / elapsed_time / 1e9;
  } else {
    intraHostBandWidthMatrix[rnic_id * (args->nMems + args->nDevs) + ep_id] = args->bytes / elapsed_time / 1e9;
  }

  freeWrAndSge(args, send_wr_list, recv_wr_list);
  return res;
}

testResult_t runTest(struct runArgs *args) {
  testResult_t res = testSuccess;
  int nRnics = args->nRnics;
  int nDevs = args->nDevs;
  int nMems = args->nMems;
  double *intraHostBandWidthMatrix = (double *)calloc((nMems + nDevs) * nRnics, sizeof(double));
  char **rnic_name_list = (char **)malloc(nRnics * sizeof(char *));

  for (int i = 0; i < nRnics; i++) {
    struct rnicDev rnic;
    rnic_name_list[i] = (char *)malloc(RNIC_NAME_MAX_LEN * sizeof(char));
    if (!initRnic(args, &rnic, rnic_name_list[i], i)) {
      continue;
    }
    for (int j = 0; j < nMems + nDevs; j++) {
      // INFO("-------Rnic: %d----Endpoint: %d-------", i, j);
      res = (res == testSuccess) ? initEndpointMem(args, &rnic, i, j) : res;
      res = (res == testSuccess) ? testBandwidth(args, &rnic, intraHostBandWidthMatrix, i, j) : res;

      freeEndpoint(args, &rnic, j);
    }
    freeRnic(args, &rnic);
  }
  if (res == testSuccess) {
    if (args->type == BI_BW) {
      INFO("Intra-Host Bidirectional Bandwidth Matrix (GB/s)");
    } else {
      INFO("Intra-Host Unidirectional Bandwidth Matrix (GB/s)");
    }
    printMatrix(args, intraHostBandWidthMatrix, rnic_name_list);
  }

  free(intraHostBandWidthMatrix);
  for (int i = 0; i < nRnics; i++) {
    free(rnic_name_list[i]);
  }
  free(rnic_name_list);
  ibv_free_device_list(args->dev_list);

  return res;
}

void initArgs(struct runArgs *args) {
  args->bytes = 10*1e3*1e3;
  args->iters = 10;
  args->warmup = 2;
  args->dev_list = NULL;
  args->gid_index = 3;
  args->nMems = 0;
  args->nDevs = 0;
  args->nRnics = 0;
  args->nQps = 8;
  args->type = UNI_BW;
}

int main(int argc, char *argv[]) {
  testResult_t res = testSuccess;

  // Parse args
  int longindex;
  static struct option longopts[] = {
    {"bytes", required_argument, 0, 'b'},
    {"iters", required_argument, 0, 'n'},
    {"gid_index", required_argument, 0, 'g'},
    {"qp_num", required_argument, 0, 'q'},
    {"type", required_argument, 0, 't'},
    {"help", no_argument, 0, 'h'}
  };

  struct runArgs args;
  initArgs(&args);

  while(1) {
    int c;
    c = getopt_long(argc, argv, "b:n:g:q:t:h", longopts, &longindex);
    if (c == -1)
      break;

    switch(c) {
      case 'b':
        args.bytes = strtol(optarg, NULL, 0);
        break;
      case 'n':
        args.iters = strtol(optarg, NULL, 0);
        break;
      case 'g':
        args.gid_index = strtol(optarg, NULL, 0);
        break;
      case 'q':
        args.nQps = strtol(optarg, NULL, 0);
        break;
      case 't':
        args.type = (PERF_TYPE)strtol(optarg, NULL, 0);
        break;
      case 'h':
        printf("USAGE: %s \n\t"
              "[-b,--bytes <data size>] \n\t"
              "[-n,--iters <iteration>] \n\t"
              "[-g,--gid_index <gid index>] \n\t"
              "[-q,--qp_num <qp number>] \n\t"
              "[-t,--type <0(UNI_BW)/1(BI_BW)>] \n\t"
              "[-h,--help]\n",
              basename(argv[0]));
        return 0;
      default:
        printf("invalid option \n");
        exit(EXIT_FAILURE);
    }
  }

  // float aligned
  args.bytes = ((args.bytes + sizeof(float) - 1) / sizeof(float)) * sizeof(float);

  res = getDevList(&args);
  HGGCCHECK(hggcGetDeviceCount(&args.nDevs));
  res = (res == testSuccess) ? getNumaNodesNum(&args) : res;
  res = (res == testSuccess) ? runTest(&args) : res;

  if (res == testSuccess) {
    INFO("Intra-host bandwidth and latency test finished.");
  } else {
    ERROR("Intra-host bandwidth and latency test failed.");
  }
  return res;
}
