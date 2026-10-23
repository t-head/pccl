//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <getopt.h>
#include <libgen.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <net/if.h>

#include <infiniband/verbs.h>

#include <hggc_runtime.h>
#include <iostream>

#define HG_MEMHOSTALLOC_64KB_PAGE 0x80000
#define MAX_REQUESTS 8
const char *lo = "127.0.0.1";

enum {
    HGGC_MEM = 0,
    HOST_MEM = 1,
    HGGC_HOST = 2
};

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define RET_CHECK(cmd, func_tag) do { \
    ret = cmd; \
    if (ret < 0) { \
        printf("%s: %s\n", strerror(errno), func_tag); \
        return ret; \
    } \
} while (0)

#define EXIT_CHECK(cmd, func_tag) do { \
    ret = cmd; \
    if (ret < 0) { \
        printf("%s: %s\n", strerror(errno), func_tag); \
        goto exit; \
    } \
} while (0)

// Check HGGC calls
#define HGGCCHECK(cmd) do {                                 \
    hggcError_t err = cmd;                                  \
    if( err != hggcSuccess ) {                              \
        printf("HGGC failure '%s'\n", hggcGetErrorString(err)); \
        return -1;                      \
    }                                                       \
} while(0)

/* Common socket address storage structure for IPv4/IPv6 */
union socketAddress {
    struct sockaddr sa;
    struct sockaddr_in sin;
    struct sockaddr_in6 sin6;
};

struct params {
    // runtime params
    char *ib_dev;
    char *server_ip;
    int port;
    int gid_index;
    int bytes;
    int type; // 0 GDR, 1 host, 2 hggcHost
    int iters;
    int role; // 0 is server, and 1 is for client
};

static struct params user_params;

struct qp_info {
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

struct ib_context {

    struct ibv_device *dev;
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;

    void *buf; // hggc buffer
    float *ref; // golden buffer
    struct ibv_mr *mr;

    struct qp_info ib_info;
};

static int hostMalloc(void** ptr, size_t size) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    void* p;
    int size_aligned = ROUNDUP(size, page_size);
    int ret = posix_memalign(&p, page_size, size_aligned);
    if (ret != 0) return ret;
    memset(p, 0, size);
    *ptr = p;
    return ret;
}

static inline const char *socketToString(struct sockaddr *saddr, char *buf) {
    if (buf == NULL || saddr == NULL) return NULL;
    if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) { buf[0]='\0'; return buf; }
    char host[NI_MAXHOST], service[NI_MAXSERV];
    (void) getnameinfo(saddr, sizeof(union socketAddress), host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICHOST|NI_NUMERICSERV);
    sprintf(buf, "%s<%s>", host, service);
    return buf;
}

struct ibv_device* ctx_find_dev(char **ib_devname)
{
    int num_of_device;
    struct ibv_device **dev_list;
    struct ibv_device *ib_dev = NULL;

    dev_list = ibv_get_device_list(&num_of_device);

    if (num_of_device <= 0) {
        fprintf(stderr," Did not detect devices \n");
        fprintf(stderr," If device exists, check if driver is up\n");
        return NULL;
    }

    if (!ib_devname) {
        fprintf(stderr," Internal error, existing.\n");
        return NULL;
    }

    if (!*ib_devname) {
        ib_dev = dev_list[0];
        if (!ib_dev) {
            fprintf(stderr, "No IB devices found\n");
            exit (1);
        }
    } else {
        for (; (ib_dev = *dev_list); ++dev_list)
            if (!strcmp(ibv_get_device_name(ib_dev), *ib_devname))
                break;
        if (!ib_dev) {
            fprintf(stderr, "IB device %s not found\n", *ib_devname);
            return NULL;
        }
    }

    return ib_dev;
}

struct ibv_context* ctx_open_device(struct ibv_device *ib_dev)
{
    struct ibv_context *context;

    context = ibv_open_device(ib_dev);

    if (!context) {
        fprintf(stderr, " Couldn't get context for the device\n");
        return NULL;
    }

    return context;
}

int init_ib_context(struct ib_context *context, struct params *user_params) {

    memset(context, 0, sizeof(struct ib_context));

    struct ibv_device *dev = ctx_find_dev((char **)&user_params->ib_dev);
    struct ibv_port_attr portAttr;
    int ib_port = 1; // use default port 1
    int gid_index = user_params->gid_index;
    union ibv_gid gid;

    context->ctx =  ctx_open_device(dev);
    ibv_query_port(context->ctx, ib_port, &portAttr);
    ibv_query_gid(context->ctx, ib_port, gid_index, &gid);

    context->pd = ibv_alloc_pd(context->ctx);
    context->cq = ibv_create_cq(context->ctx, MAX_REQUESTS, NULL, NULL, 0);

    struct ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
    qpInitAttr.send_cq = context->cq;
    qpInitAttr.recv_cq = context->cq;
    qpInitAttr.qp_type = IBV_QPT_RC;
    // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
    qpInitAttr.cap.max_send_wr = 2*MAX_REQUESTS;
    qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;
    qpInitAttr.cap.max_send_sge = 1;
    qpInitAttr.cap.max_recv_sge = 1;
    qpInitAttr.cap.max_inline_data = 0;
    context->qp = ibv_create_qp(context->pd, &qpInitAttr);

    struct ibv_qp_attr qpAttr;
    memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
    qpAttr.qp_state = IBV_QPS_INIT;
    qpAttr.pkey_index = 0;
    qpAttr.port_num = ib_port;
    qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
    ibv_modify_qp(context->qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);

    // fill qp_info
    context->ib_info.ib_port = ib_port;
    context->ib_info.qpn = context->qp->qp_num;
    context->ib_info.mtu = portAttr.active_mtu;
    context->ib_info.lid = portAttr.lid;
    if (!context->ib_info.lid) {
        // RoCE
        union ibv_gid gid;
        ibv_query_gid(context->ctx, ib_port, gid_index, &gid);
        context->ib_info.spn = gid.global.subnet_prefix;
        context->ib_info.iid = gid.global.interface_id;
    }

    // init ref data
    context->ref = (float *)malloc(user_params->bytes);
    if (!context->ref) {
        printf("malloc ref failed: %s\n", strerror(errno));
        return -1;
    }
    int count = user_params->bytes / sizeof(float);
    for (int i = 0; i < count; i++) {
        context->ref[i] = 1.0f / (i + 1);
    }

    // buffer
    if (user_params->type == HOST_MEM) {
        printf("Using Host Memory\n");
        hostMalloc(&context->buf, user_params->bytes);
        if (user_params->role)
            memcpy(context->buf, context->ref, user_params->bytes);
    } else if (user_params->type == HGGC_HOST) {
        printf("Using HGGC Host Memory\n");
        HGGCCHECK(hggcHostAlloc(&context->buf, user_params->bytes, hggcHostAllocMapped | HG_MEMHOSTALLOC_64KB_PAGE));
        memset(context->buf, 0, user_params->bytes);
        if (user_params->role)
            memcpy(context->buf, context->ref, user_params->bytes);

    } else {
        printf("Using GDR\n");
        HGGCCHECK(hggcMalloc(&context->buf, user_params->bytes));
        hggcMemset(context->buf, 0, user_params->bytes);
        if (user_params->role)
            hggcMemcpy(context->buf, context->ref, user_params->bytes, hggcMemcpyHostToDevice);
    }
#define REG_ALIGN (4096)
    ////uint64_t addr = (uint64_t)context->buf;
    ////int size = user_params->bytes;
    ////uint64_t regAddr = addr & (~(REG_ALIGN-1));
    ////uint64_t regSize = addr+size - regAddr;
    ////regSize = ((regSize + REG_ALIGN-1) / REG_ALIGN ) * REG_ALIGN;
    ////context->mr = ibv_reg_mr(context->pd, (void*)regAddr, regSize,
    ////IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
    context->mr = ibv_reg_mr(context->pd, context->buf, user_params->bytes,
            IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ);
    if (!context->mr) {
        printf("ibv_reg_mr failed: %s\n", strerror(errno));
        return -1;
    }

    if (!user_params->role) {
        // server
        context->ib_info.buffer_addr = (uint64_t)context->mr->addr;
        context->ib_info.buffer_key  = context->mr->rkey;
    }

    return 0;
}

void destroy_ib_context(struct ib_context *context, struct params *user_params) {

    if (context->qp)
        ibv_destroy_qp(context->qp);
    if (context->mr)
        ibv_dereg_mr(context->mr);

    ibv_destroy_cq(context->cq);
    ibv_dealloc_pd(context->pd);

    if (context->buf && user_params->type != HOST_MEM)
        hggcFree(context->buf);
    if (context->buf && user_params->type == HOST_MEM)
        free(context->buf);

    if (context->ref)
        free(context->ref);
}

#define DUMP 0

void check(struct ib_context *context, struct params *user_params) {
    // malloc host buffer
    float *host_buf = NULL;
    if (user_params->type == HGGC_MEM) {
        host_buf = (float *)malloc(user_params->bytes);
        hggcMemcpy(host_buf, context->buf, user_params->bytes, hggcMemcpyDeviceToHost);
    } else {
        host_buf = (float *)(context->buf);
    }

    // compare
    int float_cnt = user_params->bytes / sizeof(float);
    for(int i = 0; i < float_cnt; i++) {
#if DUMP
        printf("[%d]: %f\n", i, host_buf[i]);
#endif
        if (host_buf[i] != context->ref[i])
            printf("Mismatch at %d: %f vs %f\n", i, host_buf[i], context->ref[i]);
    }

    if (user_params->type == HGGC_MEM)
        free(host_buf);
}

int qp_setup_rtr(struct ibv_qp *qp, struct qp_info *info, struct params *user_params) {

    int ret = 0;

    struct ibv_qp_attr qpAttr;
    memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
    qpAttr.qp_state = IBV_QPS_RTR;
    qpAttr.path_mtu = info->mtu;
    qpAttr.dest_qp_num = info->qpn;
    qpAttr.rq_psn = 0;
    qpAttr.max_dest_rd_atomic = 1;
    qpAttr.min_rnr_timer = 12;
    if (info->lid == 0) {
        qpAttr.ah_attr.is_global = 1;
        qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
        qpAttr.ah_attr.grh.dgid.global.interface_id = info->iid;
        qpAttr.ah_attr.grh.flow_label = 0;
        qpAttr.ah_attr.grh.sgid_index = user_params->gid_index;
        qpAttr.ah_attr.grh.hop_limit = 255;
        qpAttr.ah_attr.grh.traffic_class = 0;
    } else {
        qpAttr.ah_attr.is_global = 0;
        qpAttr.ah_attr.dlid = info->lid;
    }
    qpAttr.ah_attr.sl = 0;
    qpAttr.ah_attr.src_path_bits = 0;
    qpAttr.ah_attr.port_num = info->ib_port;
    RET_CHECK(ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
            "ibv_modify_qp rtr");

    return ret;
}

int qp_setup_rts(struct ibv_qp* qp) {

    int ret = 0;

    struct ibv_qp_attr qpAttr;
    memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
    qpAttr.qp_state = IBV_QPS_RTS;
    qpAttr.timeout = 14; // TODO: obtain it from user_params later
    qpAttr.retry_cnt = 7;
    qpAttr.rnr_retry = 7;
    qpAttr.sq_psn = 0;
    qpAttr.max_rd_atomic = 1;
    RET_CHECK(ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC),
            "ibv_modify_qp rts");
    return ret;
}

int socket_send(int fd, const void *ptr, const int size) {
    //
    int ret = 0;
    int offset = 0;
    int bytes = 0;
    char *data = (char *)ptr;
    do {
        bytes = send(fd, data + offset, size - offset, 0);
        if (bytes == -1) {
            printf("%s failed: %s\n", __func__, strerror(errno));
            if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
                ret = -1;
                break;
            } else {
                bytes = 0; // retry
            }
        }
        offset += bytes;
    } while (offset < size);

    return ret;
}

int socket_recv(int fd, void *ptr, const int size) {
    //
    int ret = 0;
    int offset = 0;
    int bytes = 0;
    char *data = (char *)ptr;
    do {
        bytes = recv(fd, data + offset, size - offset, 0);
        if (bytes == 0) {
            printf("%s recv failed due to disconnected: %s\n", __func__, strerror(errno));
            ret = -1;
            break;
        } else if (bytes == -1) {
            printf("%s failed: %s\n", __func__, strerror(errno));
            bytes = 0; // retry
        }
        offset += bytes;
    } while (offset < size);

    return ret;
}

int run_server(struct params *user_params) {

    int ret = 0;

    // obtain socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        printf("create socket failed: %s\n", strerror(errno));
        return -1;
    }
    int opt = 1;
    RET_CHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");

    // bind to specific addr
    struct sockaddr_in sock_addr;
    sock_addr.sin_family = AF_INET;
    inet_pton(AF_INET, user_params->server_ip, &(sock_addr.sin_addr));
    sock_addr.sin_port = htons(user_params->port);
    int salen = sizeof(struct sockaddr_in);
    RET_CHECK(bind(sockfd, (struct sockaddr *)&sock_addr, salen), "bind");

    // obtain bind port
    union socketAddress localAddr;
    socklen_t size = salen;
    getsockname(sockfd, &localAddr.sa, &size);
    char line[1024];
    printf("Server: %s\n", socketToString(&localAddr.sa, line));

    // listen
    RET_CHECK(listen(sockfd, 16384), "listen");

    // accept
    RET_CHECK(accept(sockfd, (struct sockaddr *)&sock_addr, &size), "accept");
    int connfd = ret;

    // IB setup
    struct ib_context context;
    init_ib_context(&context, user_params);

    // recv remote QP info
    struct qp_info remote_qp;
    EXIT_CHECK(socket_recv(connfd, &remote_qp, sizeof(struct qp_info)), "server recv QP");
    printf("Received Client QP info\n");

    remote_qp.mtu = (remote_qp.mtu > context.ib_info.mtu) ? remote_qp.mtu : context.ib_info.mtu;
    context.ib_info.mtu = remote_qp.mtu;

    // send QP into to client
    EXIT_CHECK(socket_send(connfd, &context.ib_info, sizeof(struct qp_info)), "sever send QP");

    qp_setup_rtr(context.qp, &remote_qp, user_params);
    qp_setup_rts(context.qp);
    printf("QP Exchange done\n");

    while(user_params->iters--) {
        // simple barrier to make sure both side rtr/rts
        int ready = 1;
        EXIT_CHECK(socket_send(connfd, &ready, sizeof(int)), "server send ready");

        struct ibv_recv_wr wr;
        memset(&wr, 0, sizeof(wr));
        struct ibv_sge sge;
        sge.addr = (uintptr_t)context.buf;
        sge.length = user_params->bytes;
        sge.lkey = context.mr->lkey;
        wr.wr_id = (uint64_t)(&context);
        wr.sg_list = &sge;
        wr.num_sge = 1;

        struct ibv_recv_wr* bad_wr;
        EXIT_CHECK(ibv_post_recv(context.qp, &wr, &bad_wr), "ibv_post_recv");

        // poll result
        struct ibv_wc wcs[4];
        int wrDone = 0;
        bool poll = true;
        while(poll) {
            wrDone = ibv_poll_cq(context.cq, 4, wcs);
            if (wrDone < 0) {
                printf("ibv_poll_cq failed\n");
                goto exit;
            } else if (wrDone == 0) {
                printf("Polling %d ...\n", user_params->iters);
                sleep(1);
                continue;
            } else {
                for (int w = 0; w < wrDone; w++) {
                    struct ibv_wc *wc = wcs+w;
                    if (wc->status != IBV_WC_SUCCESS) {
                        printf("Got completion with error %d, opcode %d, len %d, vendor err %d\n",
                                wc->status, wc->opcode, wc->byte_len, wc->vendor_err);
                        poll = false;
                        break;
                    }
                    if (wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
                        printf("Server got RDMA Write %d bytes success\n", wc->byte_len);
                        poll = false;
                    }
                }
            }
        }

        // compare data
        check(&context, user_params);

        hggcMemset(context.buf, 0x5A, user_params->bytes);
        EXIT_CHECK(socket_recv(connfd, &ready, sizeof(int)), "server recv ready");
    }

exit:
    printf("Exit !\n");
    destroy_ib_context(&context, user_params);

    close(connfd);
    close(sockfd);

    return ret;
}

int run_client(struct params *user_params) {

    int ret = 0;
    int ready = 1;

    // obtain socket
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        printf("create socket failed: %s\n", strerror(errno));
        return -1;
    }
    const int one = 1;
    setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int));

    // connect to server addr
    union socketAddress remoteAddr;
    remoteAddr.sin.sin_family = AF_INET;
    inet_pton(AF_INET, user_params->server_ip, &(remoteAddr.sin.sin_addr));
    remoteAddr.sin.sin_port = htons(user_params->port);
    int salen = sizeof(struct sockaddr_in);
    RET_CHECK(connect(sockfd, &remoteAddr.sa, salen), "bind");
    printf("Connect to: %s %d\n", user_params->server_ip, user_params->port);

    // IB setup
    struct ib_context context;
    init_ib_context(&context, user_params);

    // send QP into to server
    EXIT_CHECK(socket_send(sockfd, &context.ib_info, sizeof(struct qp_info)), "client send QP");

    // recv remote QP info
    struct qp_info remote_qp;
    EXIT_CHECK(socket_recv(sockfd, &remote_qp, sizeof(struct qp_info)), "client recv QP");
    printf("Received Server QP info\n");

    qp_setup_rtr(context.qp, &remote_qp, user_params);
    qp_setup_rts(context.qp);
    context.ib_info.buffer_addr = (uint64_t)remote_qp.buffer_addr;
    context.ib_info.buffer_key  = remote_qp.buffer_key;
    printf("QP Exchange done\n");

    while(user_params->iters--) {
        // simple barrier to make sure both side rtr/rts
        EXIT_CHECK(socket_recv(sockfd, &ready, sizeof(int)), "client recv ready");

        struct ibv_send_wr wr;
        memset(&wr, 0, sizeof(wr));

        struct ibv_sge sge;
        memset(&sge, 0, sizeof(sge));
        sge.addr = (uintptr_t)context.buf;
        sge.length = user_params->bytes;
        sge.lkey = context.mr->lkey;

        wr.wr_id = (uint64_t)(&context);
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.wr.rdma.remote_addr = context.ib_info.buffer_addr;
        wr.wr.rdma.rkey = context.ib_info.buffer_key;
        wr.imm_data = user_params->bytes; // Send the message size via imm_data

        struct ibv_send_wr* bad_wr;
        printf("rdma write: %d bytes\n", user_params->bytes);
        EXIT_CHECK(ibv_post_send(context.qp, &wr, &bad_wr), "ibv_post_send");

        // poll result
        struct ibv_wc wcs[4];
        int wrDone = 0;
        bool poll = true;
        while(poll) {
            wrDone = ibv_poll_cq(context.cq, 4, wcs);
            if (wrDone < 0) {
                printf("ibv_poll_cq failed\n");
                goto exit;
            } else if (wrDone == 0) {
                printf("Polling %d ...\n", user_params->iters);
                sleep(1);
                continue;
            } else {
                for (int w = 0; w < wrDone; w++) {
                    struct ibv_wc *wc = wcs+w;
                    if (wc->status != IBV_WC_SUCCESS) {
                        printf("Got completion with error %d, opcode %d, len %d, vendor err %d\n",
                                wc->status, wc->opcode, wc->byte_len, wc->vendor_err);
                        poll = false;
                        break;
                    }
                    if (wc->opcode == IBV_WC_RDMA_WRITE) {
                        //printf("Client RDMA Write %d bytes success\n", wc->byte_len);
                        poll = false;
                    }
                }
            }
        }
        EXIT_CHECK(socket_send(sockfd, &ready, sizeof(int)), "client send ready");
    }

exit:
    printf("Exit !\n");
    destroy_ib_context(&context, user_params);
    close(sockfd);

    return ret;
}

int main(int argc, char *argv[]) {

    int ret = 0;

    // Parse args
    int longindex;
    static struct option longopts[] = {
        {"ib_dev", required_argument, 0, 'd'},
        {"server", required_argument, 0, 's'},
        {"port", required_argument, 0, 'p'},
        {"bytes", required_argument, 0, 'b'},
        {"iters", required_argument, 0, 'n'},
        {"help", no_argument, 0, 'h'}
    };

    user_params.ib_dev = NULL;
    user_params.server_ip = const_cast<char *>(lo);
    user_params.port = 12345;
    user_params.role = 0;
    user_params.gid_index = 3;
    user_params.bytes = 32;
    user_params.type = HGGC_MEM; // default GDR
    user_params.iters = 1;

    while(1) {
        int c;
        c = getopt_long(argc, argv, "d:s:p:n:g:b:t:r:h", longopts, &longindex);
        if (c == -1)
            break;

        switch(c) {
            case 'd':
                user_params.ib_dev = optarg;
                break;
            case 's':
                user_params.server_ip = optarg;
                break;
            case 'p':
                user_params.port = strtol(optarg, NULL, 0);
                break;
            case 'b':
                user_params.bytes = strtol(optarg, NULL, 0);
                break;
            case 't':
                user_params.type = strtol(optarg, NULL, 0);
                break;
            case 'g':
                user_params.gid_index = strtol(optarg, NULL, 0);
                break;
            case 'n':
                user_params.iters = strtol(optarg, NULL, 0);
                break;
            case 'r':
                user_params.role = strtol(optarg, NULL, 0);
                break;
            case 'h':
                printf("USAGE: %s \n\t"
                        "[-d,--ib_dev <ib device>] \n\t"
                        "[-s,--server <server ip address>] \n\t"
                        "[-p,--port <server port>] \n\t"
                        "[-g,--gid_index <gid index>] \n\t"
                        "[-b,--bytes <data size>] \n\t"
                        "[-t,--type <0/1/2>] \n\t"
                        "[-n,--iters <iteration>] \n\t"
                        "[-r,--role <0/1 1 is client>] \n\t"
                        "[-h,--help]\n",
                        basename(argv[0]));
                return 0;
            default:
                printf("invalid option \n");
                exit(EXIT_FAILURE);
        }
    }

    // float aligned
    user_params.bytes = ((user_params.bytes + sizeof(float) - 1) / sizeof(float)) * sizeof(float);

    if (user_params.role) {
        ret = run_client(&user_params);
    } else {
        ret = run_server(&user_params);
    }

    return ret;
}

