/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "connect.h"
#include "common.h"

NCCL_PARAM(IbGidIndex, "IB_GID_INDEX", -2);
NCCL_PARAM(IbRoutableFlidGidIndex, "IB_ROUTABLE_FLID_GID_INDEX", 1);
NCCL_PARAM(IbRoceVersionNum, "IB_ROCE_VERSION_NUM", 2);
PCCL_PARAM(IbAddrFamilyFallback, "IB_ADDR_FAMILY_FALLBACK", 1);
NCCL_PARAM(IbTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(IbRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(IbPkey, "IB_PKEY", 0);
NCCL_PARAM(IbUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(IbSl, "IB_SL", 0);
NCCL_PARAM(IbTc, "IB_TC", 0);
NCCL_PARAM(IbFifoTc, "IB_FIFO_TC", -1);

struct ncclIbHandle {
  union ncclSocketAddress connectAddr;
  uint64_t magic; // random number to help debugging
  struct ncclIbCommStage stage; // Used by the other side when connecting
  // TODO: Remove this compatibility field after porting NCCL's per-device metadata
  // exchange to PCCL; OOO RQ capability should then come from that metadata.
  uint32_t oooRqSize;
};

NCCL_PARAM(IbQpsPerConn, "IB_QPS_PER_CONNECTION", 2);
NCCL_PARAM(IbEceEnable, "IB_ECE_ENABLE", 1);
extern int64_t ncclParamIbOooRq();

ncclResult_t ncclIbInitCommDevBase(int ibDevN, struct ncclIbNetCommDevBase* base, void* cq_context) {
  base->ibDevN = ibDevN;
  ncclIbDev* ibDev = ncclIbDevs + ibDevN;
  {
    std::lock_guard<std::mutex> lock(ibDev->mutex);
    if (ibDev->pdRefs == 0) {
      NCCLCHECK(wrap_ibv_alloc_pd(&ibDev->pd, ibDev->context));
    }
    ibDev->pdRefs++;
    base->pd = ibDev->pd;
  }
  // Recv requests can generate 2 completions (one for the post FIFO, one for the Recv).
  NCCLCHECK(wrap_ibv_create_cq(&base->cq, ncclIbDevs[ibDevN].context, 2*NET_IB_MAX_REQUESTS*ncclParamIbQpsPerConn(), cq_context, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclIbDestroyBase(struct ncclIbNetCommDevBase* base) {
  // The two NULL guards are the only delta from upstream: connect/accept call this from their
  // failure paths, where InitCommDevBase may not have run, or may have failed after taking the pd.
  if (base->cq != NULL) {
    NCCLCHECK(wrap_ibv_destroy_cq(base->cq));
    base->cq = NULL;
  }
  if (base->pd != NULL) {
    base->pd = NULL;
    std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
    if (0 == --ncclIbDevs[base->ibDevN].pdRefs) {
      NCCLCHECK(wrap_ibv_dealloc_pd(ncclIbDevs[base->ibDevN].pd));
    }
  }
  return ncclSuccess;
}

static sa_family_t envIbAddrFamily(void) {
  sa_family_t family = AF_INET;
  const char* env = pcclGetEnvWithDualPrefix("PCCL_IB_ADDR_FAMILY");
  if (env == NULL || strlen(env) == 0) {
    return family;
  }

  INFO(NCCL_ENV, "PCCL_IB_ADDR_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0) {
    family = AF_INET;
  } else if (strcmp(env, "AF_INET6") == 0) {
    family = AF_INET6;
  }

  return family;
}

static void* envIbAddrRange(sa_family_t af, int* mask) {
  *mask = 0;
  static struct in_addr addr;
  static struct in6_addr addr6;
  void *ret = (af == AF_INET) ? (void *)&addr : (void *)&addr6;

  const char* env = pcclGetEnvWithDualPrefix("PCCL_IB_ADDR_RANGE");
  if (NULL == env || strlen(env) == 0) {
    return NULL;
  }

  INFO(NCCL_ENV, "PCCL_IB_ADDR_RANGE set by environment to %s", env);

  char addrString[128] = { 0 };
  snprintf(addrString, 128, "%s", env);
  char *addrStrPtr = addrString;
  char *maskStrPtr = strchr(addrString, '/');
  if (NULL == maskStrPtr) {
    return NULL;
  }
  *maskStrPtr++ = '\0';

  if (inet_pton(af, addrStrPtr, ret) == 0) {
    WARN("NET/IB: Ip address '%s' is invalid for family %s, ignoring address", addrStrPtr, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    return NULL;
  }

  *mask = (int)strtol(maskStrPtr, NULL, 10);
  if (af == AF_INET && *mask > 32) {
    WARN("NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  } else if (af == AF_INET6 && *mask > 128) {
    WARN("NET/IB: Ip address mask '%d' is invalid for family %s, ignoring mask", *mask, (af == AF_INET) ? "AF_INET" : "AF_INET6");
    *mask = 0;
    ret = NULL;
  }

  return ret;
}

static sa_family_t getGidAddrFamily(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  bool isIpV4Mapped = ((a->s6_addr32[0] | a->s6_addr32[1]) | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL;
  bool isIpV4MappedMulticast = (a->s6_addr32[0] == htonl(0xff0e0000) && ((a->s6_addr32[1] | (a->s6_addr32[2] ^ htonl(0x0000ffff))) == 0UL));
  return (isIpV4Mapped || isIpV4MappedMulticast) ? AF_INET : AF_INET6;
}

static bool matchGidAddrPrefix(sa_family_t af, void* prefix, int prefixlen, union ibv_gid* gid) {
  struct in_addr *base = NULL;
  struct in6_addr *base6 = NULL;
  struct in6_addr *addr6 = NULL;
  if (af == AF_INET) {
    base = (struct in_addr *)prefix;
  } else {
    base6 = (struct in6_addr *)prefix;
  }
  addr6 = (struct in6_addr *)gid->raw;

#define NETMASK(bits) (htonl(0xffffffff ^ ((1 << (32 - bits)) - 1)))

  int i = 0;
  while (prefixlen > 0 && i < 4) {
    if (af == AF_INET) {
      int mask = NETMASK(prefixlen);
      if ((base->s_addr & mask) ^ (addr6->s6_addr32[3] & mask)) {
        break;
      }
      prefixlen = 0;
      break;
    } else {
      if (prefixlen >= 32) {
        if (base6->s6_addr32[i] ^ addr6->s6_addr32[i]) {
          break;
        }
        prefixlen -= 32;
        ++i;
      } else {
        int mask = NETMASK(prefixlen);
        if ((base6->s6_addr32[i] & mask) ^ (addr6->s6_addr32[i] & mask)) {
          break;
        }
        prefixlen = 0;
      }
    }
  }

  return (prefixlen == 0);
}

static bool configuredGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  int trailer = (a->s6_addr32[1] | a->s6_addr32[2] | a->s6_addr32[3]);
  if (((a->s6_addr32[0] | trailer) == 0UL) || ((a->s6_addr32[0] == htonl(0xfe800000)) && (trailer == 0UL))) {
    return false;
  }
  return true;
}

static bool linkLocalGid(union ibv_gid* gid) {
  const struct in6_addr *a = (struct in6_addr *)gid->raw;
  return IN6_IS_ADDR_LINKLOCAL(a);
}

static bool validGid(union ibv_gid* gid) {
  return (configuredGid(gid) && !linkLocalGid(gid));
}

struct ncclIbGidTypeReadStats {
  int attempted;
  int succeeded;
};

static ncclResult_t ncclIbRoceGetVersionNum(const char* deviceName, int portNum, int gidIndex, int* version, struct ncclIbGidTypeReadStats* stats) {
  char gidRoceVerStr[16] = { 0 };
  char roceTypePath[PATH_MAX] = { 0 };
  sprintf(roceTypePath, "/sys/class/infiniband/%s/ports/%d/gid_attrs/types/%d", deviceName, portNum, gidIndex);

  stats->attempted++;
  int fd = open(roceTypePath, O_RDONLY);
  if (fd == -1) {
    return ncclSystemError;
  }
  int ret = read(fd, gidRoceVerStr, 15);
  int readErrno = errno;
  close(fd);

  if (ret == -1) {
    // In containerized environments, read could return EINVAL if the GID index is not mapped to the
    // container sysfs. In this case return ncclSuccess and let the caller move to next GID index.
    if (readErrno == EINVAL) return ncclSuccess;
    WARN("NET/IB: read failed in ncclIbRoceGetVersionNum: %s", strerror(readErrno));
    return ncclSystemError;
  }

  stats->succeeded++;
  if (strlen(gidRoceVerStr)) {
    if (strncmp(gidRoceVerStr, "IB/RoCE v1", strlen("IB/RoCE v1")) == 0 || strncmp(gidRoceVerStr, "RoCE v1", strlen("RoCE v1")) == 0) {
      *version = 1;
    } else if (strncmp(gidRoceVerStr, "RoCE v2", strlen("RoCE v2")) == 0) {
      *version = 2;
    }
  }

  return ncclSuccess;
}

static ncclResult_t ncclUpdateGidIndex(struct ibv_context* context, uint8_t portNum, sa_family_t af, void* prefix, int prefixlen, int roceVer, int gidIndexCandidate, int* gidIndex, bool* gidFound, struct ncclIbGidTypeReadStats* stats) {
  union ibv_gid gidCandidate;
  NCCLCHECK(wrap_ibv_query_gid(context, portNum, gidIndexCandidate, &gidCandidate));
  if (getGidAddrFamily(&gidCandidate) != af || !validGid(&gidCandidate) ||
      !matchGidAddrPrefix(af, prefix, prefixlen, &gidCandidate)) {
    return ncclSuccess;
  }

  int candidateRoceVer = 0;
  const char* deviceName = wrap_ibv_get_device_name(context->device);
  NCCLCHECK(ncclIbRoceGetVersionNum(deviceName, portNum, gidIndexCandidate, &candidateRoceVer, stats));
  if (candidateRoceVer == roceVer) {
    *gidIndex = gidIndexCandidate;
    *gidFound = true;
  }
  return ncclSuccess;
}

// The two bytes at offset four encode FLID; the final two prefix bytes
// identify the IB subnet. Use byte access to avoid unaligned loads.
static uint16_t ncclIbExtractFlid(const union ibv_gid* gid) {
  return ((uint16_t)gid->raw[4] << 8) | gid->raw[5];
}

static uint16_t ncclIbExtractSubnet(const union ibv_gid* gid) {
  return ((uint16_t)gid->raw[6] << 8) | gid->raw[7];
}

ncclResult_t ncclIbGetGidIndex(struct ibv_context *context, uint8_t portNum,
                                const struct ibv_port_attr* portAttr, int *gidIndex) {
  *gidIndex = ncclParamIbGidIndex();
  if (*gidIndex >= 0) {
    return ncclSuccess;
  }

  // Preserve an explicit IB_GID_INDEX; otherwise prefer a routable IB GID.
  int gidTblLen = portAttr->gid_tbl_len;
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    int64_t routableGidIndex = ncclParamIbRoutableFlidGidIndex();
    if (routableGidIndex >= 0 && routableGidIndex < gidTblLen) {
      union ibv_gid gid;
      NCCLCHECK(wrap_ibv_query_gid(context, portNum, (int)routableGidIndex, &gid));
      if (ncclIbExtractFlid(&gid) != 0) {
        *gidIndex = (int)routableGidIndex;
        return ncclSuccess;
      }
    }
    *gidIndex = 0;
    return ncclSuccess;
  }

  bool allowFallback = pcclParamIbAddrFamilyFallback() != 0;
  sa_family_t userAddrFamily = envIbAddrFamily();
  int64_t userRoceVersion = ncclParamIbRoceVersionNum();
  if (userRoceVersion != 1 && userRoceVersion != 2) {
    WARN("NET/IB: Invalid IB_ROCE_VERSION_NUM %lld, expected 1 or 2", (long long)userRoceVersion);
    return ncclInvalidArgument;
  }

  const char* deviceName = wrap_ibv_get_device_name(context->device);
  *gidIndex = 0;
  struct ncclIbGidTypeReadStats typeReadStats = { 0, 0 };

  // Fallback only controls whether to scan the other family after the configured family (IPv4 by default).
  for (int attempt = 0; attempt < (allowFallback ? 2 : 1); ++attempt) {
    sa_family_t family = attempt == 0 ? userAddrFamily : (userAddrFamily == AF_INET ? AF_INET6 : AF_INET);
    int prefixlen;
    // Parse the range for this family; never reuse an IPv4 prefix as IPv6.
    void* prefix = envIbAddrRange(family, &prefixlen);
    bool gidFound = false;
    for (int candidate = 0; candidate < gidTblLen && !gidFound; ++candidate) {
      NCCLCHECK(ncclUpdateGidIndex(context, portNum, family, prefix, prefixlen, static_cast<int>(userRoceVersion), candidate, gidIndex, &gidFound, &typeReadStats));
    }

    if (gidFound) {
      INFO(NCCL_NET, "NET/IB : GID auto-selection %s port %d: index %d family %s RoCE v%lld%s",
           deviceName, portNum, *gidIndex, family == AF_INET ? "AF_INET" : "AF_INET6", (long long)userRoceVersion,
           attempt == 0 ? "" : " (fallback)");
      return ncclSuccess;
    }

    if (attempt == 0 && allowFallback) {
      INFO(NCCL_NET, "NET/IB : No usable %s GID on %s port %d, trying %s",
           family == AF_INET ? "AF_INET" : "AF_INET6", deviceName, portNum,
           family == AF_INET ? "AF_INET6" : "AF_INET");
    }
  }

  if (typeReadStats.attempted > 0 && typeReadStats.succeeded == 0) {
    WARN("NET/IB: Unable to read GID type for all %d examined candidates on %s port %d (EINVAL); "
         "check /sys/class/infiniband/%s/ports/%d/gid_attrs/types visibility; retaining index 0 for compatibility, connection may fail",
         typeReadStats.attempted, deviceName, portNum, deviceName, portNum);
  }

  // Preserve the existing index-zero behavior regardless of the address-family fallback setting.
  INFO(NCCL_NET, "NET/IB : No matching RoCE GID on %s port %d, retaining index 0", deviceName, portNum);

  return ncclSuccess;
}

ncclResult_t ncclIbCreateQp(uint8_t ib_port, struct ncclIbNetCommDevBase* commDevBase, int access_flags,
                            int oooRq, struct ibv_qp** qp, void* qp_context) {
  if (oooRq) {
    struct ibv_qp_init_attr_ex qpInitAttr;
    struct mlx5dv_qp_init_attr mlx5QpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));
    memset(&mlx5QpInitAttr, 0, sizeof(mlx5QpInitAttr));
    qpInitAttr.qp_context = qp_context;
    qpInitAttr.send_cq = commDevBase->cq;
    qpInitAttr.recv_cq = commDevBase->cq;
    qpInitAttr.qp_type = IBV_QPT_RC;
    qpInitAttr.cap.max_send_wr = 2 * NET_IB_MAX_REQUESTS;
    qpInitAttr.cap.max_recv_wr = NET_IB_MAX_REQUESTS;
    qpInitAttr.cap.max_send_sge = 1;
    qpInitAttr.cap.max_recv_sge = 1;
    qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
    qpInitAttr.comp_mask = IBV_QP_INIT_ATTR_PD;
    qpInitAttr.pd = commDevBase->pd;
    mlx5QpInitAttr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
    mlx5QpInitAttr.create_flags = MLX5DV_QP_CREATE_OOO_DP;
    *qp = wrap_mlx5dv_create_qp(commDevBase->pd->context, &qpInitAttr, &mlx5QpInitAttr);
    if (*qp == NULL) {
      WARN("NET/IB : mlx5dv_create_qp failed to create an OOO QP: %s", strerror(errno));
      return ncclSystemError;
    }
  } else {
    struct ibv_qp_init_attr qpInitAttr;
    memset(&qpInitAttr, 0, sizeof(qpInitAttr));
    qpInitAttr.qp_context = qp_context;
    qpInitAttr.send_cq = commDevBase->cq;
    qpInitAttr.recv_cq = commDevBase->cq;
    qpInitAttr.qp_type = IBV_QPT_RC;
    // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
    qpInitAttr.cap.max_send_wr = 2 * NET_IB_MAX_REQUESTS;
    qpInitAttr.cap.max_recv_wr = NET_IB_MAX_REQUESTS;
    qpInitAttr.cap.max_send_sge = 1;
    qpInitAttr.cap.max_recv_sge = 1;
    qpInitAttr.cap.max_inline_data = ncclParamIbUseInline() ? sizeof(struct ncclIbSendFifo) : 0;
    NCCLCHECK(wrap_ibv_create_qp(qp, commDevBase->pd, &qpInitAttr));
  }
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = ncclParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  NCCLCHECK(wrap_ibv_modify_qp(*qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return ncclSuccess;
}

static uint32_t ncclIbGetOooRqSize(int dev) {
  // Advertise local policy so both endpoints negotiate the same receive format.
  return ncclParamIbOooRq() != 0 && ncclIbDevs[dev].ar ? ncclIbDevs[dev].oooRqSize : 0;
}

static ncclResult_t ncclIbEnableOooRq(int dev, uint32_t remOooRqSize, int* oooRq) {
  *oooRq = 0;
  if (ncclParamIbOooRq() == 0) return ncclSuccess;

  // Enable the requested OOO RQ mode only when adaptive routing is enabled and
  // both endpoints advertise sufficient OOO receive queue capacity. Otherwise,
  // leave OOO disabled and use in-order receive matching. Log explicit requests
  // at WARN when these prerequisites are not met.
  const bool explicitReq = (ncclGetEnv("PCCL_IB_OOO_RQ") != nullptr);
  struct ncclIbDev* ibDev = ncclIbDevs + dev;
  const char* reason = nullptr;
  if (ibDev->ar == 0) {
    reason = "adaptive routing disabled";
  } else if (ibDev->oooRqSize == 0 || remOooRqSize == 0) {
    reason = "not supported on both sides";
  } else if (ibDev->oooRqSize < NET_IB_MAX_REQUESTS || remOooRqSize < NET_IB_MAX_REQUESTS) {
    reason = "OOO RQ size below required receive queue size";
  }
  if (reason != nullptr) {
    if (explicitReq) {
      WARN("NET/IB : OOO RQ requested but unavailable on device %s (%s; local=%u remote=%u required>=%u); falling back to in-order",
           ibDev->devName, reason, ibDev->oooRqSize, remOooRqSize, (unsigned)NET_IB_MAX_REQUESTS);
    } else {
      INFO(NCCL_NET, "NET/IB : OOO RQ unavailable on device %s (%s; local=%u remote=%u); using in-order",
           ibDev->devName, reason, ibDev->oooRqSize, remOooRqSize);
    }
    return ncclSuccess;  // *oooRq stays 0 -> graceful in-order fallback, connect proceeds
  }
  *oooRq = 1;
  return ncclSuccess;
}

static ncclResult_t ncclIbPrepostRecvWorkRequests(struct ncclIbRecvComm* comm) {
  if (comm->base.recvMatchingScheme != BY_ID) return ncclSuccess;

  memset(&comm->ibRecvWorkRequest, 0, sizeof(comm->ibRecvWorkRequest));
  for (int q = 0; q < comm->base.nqps; q++) {
    for (uint32_t r = 0; r < NET_IB_MAX_REQUESTS; r++) {
      struct ibv_recv_wr* badWorkRequest;
      NCCLCHECK(wrap_ibv_post_recv(comm->base.qps[q], &comm->ibRecvWorkRequest, &badWorkRequest));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclIbRtrQp(ibv_qp* qp, uint32_t qpn, struct ncclIbConnectionMetadata* info, int localDev,
                         int tc, int sl, int maxDestRdAtomic) {
  struct ncclIbDev* dev = ncclIbDevs + localDev;
  int gidIndex = dev->gidIndex;
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = qpn;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = maxDestRdAtomic;
  qpAttr.min_rnr_timer = 12;
  if (info->lid == 0) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = gidIndex;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = tc;
  } else {
    union ibv_gid localGid, remoteGid;
    NCCLCHECK(wrap_ibv_query_gid(dev->context, dev->port, gidIndex, &localGid));
    remoteGid.global.subnet_prefix = info->spn;
    remoteGid.global.interface_id = info->iid;
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
    bool sameSubnet = ncclIbExtractSubnet(&localGid) == ncclIbExtractSubnet(&remoteGid);
    if (!sameSubnet) {
      uint16_t flid = ncclIbExtractFlid(&remoteGid);
      if (flid == 0) {
        WARN("NET/IB: Remote FLID is zero across IB subnets; using LID %u", info->lid);
      } else {
        qpAttr.ah_attr.dlid = flid;
      }
    }
    if (!sameSubnet) {
      qpAttr.ah_attr.is_global = 1;
      qpAttr.ah_attr.grh.dgid = remoteGid;
      qpAttr.ah_attr.grh.sgid_index = gidIndex;
      qpAttr.ah_attr.grh.hop_limit = 255;
    }
  }
  qpAttr.ah_attr.sl = sl;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = dev->port;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER));
  return ncclSuccess;
}

ncclResult_t ncclIbRtsQp(ibv_qp* qp, int maxRdAtomic) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = ncclParamIbTimeout();
  qpAttr.retry_cnt = ncclParamIbRetryCnt();
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = maxRdAtomic;
  NCCLCHECK(wrap_ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC));
  return ncclSuccess;
}

ncclResult_t ncclIbListen(int dev, void* opaqueHandle, void** listenComm) {
  ncclResult_t res = ncclSuccess;
  struct ncclIbListenComm* comm = NULL;
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclIbHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclIbHandle size too large");
  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);
  memset(handle, 0, sizeof(struct ncclIbHandle));
  comm->dev = dev;
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclIbIfAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), res, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), res, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), res, fail);
  handle->oooRqSize = ncclIbGetOooRqSize(dev);
  *listenComm = comm;
exit:
  return res;
fail:
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    NCCLCHECK(ncclFree(comm));
  }
  *listenComm = NULL;
  goto exit;
}

int ncclIbGidIndexFromDevice(int netDev) {
  int gidIndex = 0;
  if (netDev < ncclNIbDevs) {
    gidIndex = ncclIbDevs[netDev].gidIndex;
  } else {
    WARN("Invalid netDev %d", netDev);
  }
  return gidIndex;
}

ncclResult_t ncclIbConnect(int dev, void* opaqueHandle, void** sendComm) {
  ncclResult_t res = ncclSuccess;
  struct ncclIbHandle* handle = (struct ncclIbHandle*) opaqueHandle;
  struct ncclIbCommStage* stage = &handle->stage;
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)stage->comm;
  int ready;
  *sendComm = NULL;
  ibv_context* ctx = NULL;
  uint8_t ib_port;

  if (stage->state == ncclIbCommStateConnect) goto ib_connect_check;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state == ncclIbCommStateConnecting) goto ib_connect;
  if (stage->state == ncclIbCommStateConnected) goto ib_send_ready;
  if (stage->state != ncclIbCommStateStart) {
    ERROR("Error: trying to connect already connected sendComm");
    return ncclInternalError;
  }

  NCCLCHECKGOTO(ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm)), res, fail);
  NCCLCHECKGOTO(ncclIbSendCommInit(comm), res, fail);
  NCCLCHECKGOTO(ncclIbStatsInit(&comm->base.stats), res, fail);
  NCCLCHECKGOTO(ncclSocketInit(&comm->base.sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetIb, NULL, 1), res, fail);
  comm->netDev = dev;
  stage->comm = comm;
  stage->state = ncclIbCommStateConnect;

  NCCLCHECKGOTO(ncclSocketConnect(&comm->base.sock), res, fail);

ib_connect_check:
  NCCLCHECKGOTO(ncclSocketReady(&comm->base.sock, &ready), res, fail);
  if (!ready) return ncclSuccess;

  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbConnectionMetadata)), res, fail);
  comm->base.remOooRqSize = handle->oooRqSize;
  NCCLCHECKGOTO(ncclIbEnableOooRq(dev, comm->base.remOooRqSize, &comm->base.oooRq), res, fail);
  comm->base.recvMatchingScheme = comm->base.oooRq ? BY_ID : BY_INDEX;

  // IB Setup
  ctx = ncclIbDevs[dev].context;
  NCCLCHECKGOTO(ncclIbInitCommDevBase(dev, &comm->dev.base, &comm->base.stats), res, fail);
  comm->ar = ncclIbDevs[dev].ar;
  ib_port = ncclIbDevs[dev].port;
  comm->base.nqps = ncclParamIbQpsPerConn();
  comm->base.qpIndex = 0;
  for (int q = 0; q < comm->base.nqps; q++) {
    NCCLCHECKGOTO(ncclIbCreateQp(ib_port, &comm->dev.base, IBV_ACCESS_REMOTE_WRITE, comm->base.oooRq,
                                 comm->base.qps + q, &comm->base.stats), res, fail);
    INFO(NCCL_NET,
         "NET/IB: %s: QP created: port=%d dev=%d devName=%s qp_num=%u pkey=%u pd=%p oooRq=%d",
         __func__, ib_port, dev, ncclIbDevs[dev].devName, comm->base.qps[q]->qp_num,
         (uint16_t)ncclParamIbPkey(), comm->dev.base.pd, comm->base.oooRq);
  }

  // Send my QP Info to receiver through the socket. Hope this won't block.
  struct ibv_port_attr portAttr;
  NCCLCHECKGOTO(wrap_ibv_query_port(ctx, ib_port, &portAttr), res, fail);
  struct ncclIbConnectionMetadata meta;
  memset(&meta, 0, sizeof(meta));
  meta.tc = ncclParamIbTc();
  meta.sl = ncclParamIbSl();
  meta.ib_port = ib_port;
  for (int q=0; q<comm->base.nqps; q++) {
    meta.qpInfo[q].qpn = comm->base.qps[q]->qp_num;
    if (ncclParamIbEceEnable()) {
      NCCLCHECKGOTO(wrap_ibv_query_ece(comm->base.qps[q], &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), res, fail);
    }
  }
  meta.mtu = portAttr.active_mtu;

  // Prepare my fifo
  NCCLCHECKGOTO(wrap_ibv_reg_mr(&comm->fifoMr, comm->dev.base.pd, comm->ctsFifo, sizeof(struct ncclIbSendFifo) * NET_IB_MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS,
                                IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ), res, fail);
  meta.fifoRkey = comm->fifoMr->rkey;
  meta.fifoAddr = (uint64_t)comm->ctsFifo;
  meta.oooRqSize = ncclIbGetOooRqSize(dev);

  // Register a small scratchpad MR that the GIN proxy iputSignal path uses as
  // the local SGE of its FETCH_AND_ADD atomic ops. Allocating it here keeps it
  // co-located with the sendComm's PD and lifecycle.
  NCCLCHECKGOTO(wrap_ibv_reg_mr(&comm->dev.putSignalScratchpadMr, comm->dev.base.pd,
                                &comm->putSignalScratchpad, sizeof(comm->putSignalScratchpad),
                                IBV_ACCESS_LOCAL_WRITE), res, fail);

  // Both IB and RoCE peers need the GID for global addressing.
  union ibv_gid localGid;
  NCCLCHECKGOTO(wrap_ibv_query_gid(ctx, ib_port, ncclIbGidIndexFromDevice(dev), &localGid), res, fail);
  meta.spn = localGid.global.subnet_prefix;
  meta.iid = localGid.global.interface_id;
  meta.lid = portAttr.lid;
  if (meta.lid) { // IB
    for (int q=0; q<comm->base.nqps; q++)
      INFO(NCCL_NET,"NET/IB: Dev %d Port %d qpn %d mtu %d LID %d", dev, ib_port, meta.qpInfo[q].qpn, meta.mtu, meta.lid);
  } else { // RoCE
    int gidIndex = ncclIbGidIndexFromDevice(dev);
    for (int q=0; q<comm->base.nqps; q++)
      INFO(NCCL_NET,"NET/IB: Dev %d Port %d qpn %d mtu %d GID %ld (%lX/%lX)", dev, ib_port, meta.qpInfo[q].qpn, meta.mtu, gidIndex, meta.spn, meta.iid);
  }

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  memcpy(stage->buffer, &meta, sizeof(meta));

ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, stage->buffer, sizeof(meta),
                                   &stage->offset), res, fail);
  if (stage->offset != sizeof(meta))
    return ncclSuccess;

  stage->state = ncclIbCommStateConnecting;
  stage->offset = 0;
ib_connect:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &comm->base.sock, stage->buffer, sizeof(meta),
                                   &stage->offset), res, fail);
  if (stage->offset != sizeof(meta)) return ncclSuccess;

  struct ncclIbConnectionMetadata remMeta;
  memcpy(&remMeta, stage->buffer, sizeof(remMeta));
  for (int q = 0; q < comm->base.nqps; q++) {
    struct ibv_qp* qp = comm->base.qps[q];
    if (remMeta.qpInfo[q].ece_supported) {
      NCCLCHECKGOTO(wrap_ibv_set_ece(qp, &remMeta.qpInfo[q].ece, &remMeta.qpInfo[q].ece_supported), res, fail);
    }
    NCCLCHECKGOTO(ncclIbRtrQp(qp, remMeta.qpInfo[q].qpn, &remMeta, comm->netDev,
                               remMeta.tc, remMeta.sl), res, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(qp), res, fail);
  }

  comm->base.ready = 1;
  stage->state = ncclIbCommStateConnected;
  stage->offset = 0;
ib_send_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &comm->base.sock, &comm->base.ready, sizeof(int),
                                   &stage->offset), res, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  *sendComm = comm;
exit:
  NCCLCHECK(ncclFree(stage->buffer));
  stage->state = ncclIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  return res;
fail:
  if (comm) {
    // The close path releases everything acquired above and frees comm; its steps are
    // NULL-guarded, so a comm that failed early is fine too.
    NCCLCHECK(ncclIbCloseSend(comm));
    comm = NULL;
  }
  goto exit;
}

NCCL_PARAM(IbGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);
PCCL_PARAM(IbGdrFlushWrite, "GDR_FLUSH_WRITE", 0);
PCCL_PARAM(IbGdrFlushMaxRdAtomic, "GDR_FLUSH_MAX_RD_ATOMIC", 8);

static int ncclIbGetFlushReadDepth(const struct ncclIbDev* dev) {
  int64_t requested = pcclParamIbGdrFlushMaxRdAtomic();
  // The loopback flush QP uses the same depth for its requester and responder.
  if (requested < 1 || requested > dev->maxQpInitRdAtom || requested > dev->maxQpRdAtom) {
    VERBOSE(NCCL_NET, "NET/IB: Invalid READ flush depth: PCCL_GDR_FLUSH_MAX_RD_ATOMIC=%lld; "
            "must be >= 1 and not exceed device limits %d/%d (requester/responder); using depth 1",
            (long long)requested, dev->maxQpInitRdAtom, dev->maxQpRdAtom);
    requested = 1;
  }
  return (int)requested;
}

ncclResult_t ncclIbAccept(void* listenComm, void** recvComm) {
  ncclResult_t res = ncclSuccess;
  struct ncclIbListenComm* lComm = (struct ncclIbListenComm*)listenComm;
  struct ncclIbCommStage* stage = &lComm->stage;
  struct ncclIbRecvComm* rComm = (struct ncclIbRecvComm*)stage->comm;
  int ready;
  *recvComm = NULL;
  struct ncclIbConnectionMetadata remMeta;
  struct ncclIbConnectionMetadata meta;
  ibv_context* ctx = NULL;
  uint8_t ib_port;
  int gidIndex = ncclIbGidIndexFromDevice(lComm->dev);

  if (stage->state == ncclIbCommStateAccept) goto ib_accept_check;
  if (stage->state == ncclIbCommStateRecv) goto ib_recv;
  if (stage->state == ncclIbCommStateSend) goto ib_send;
  if (stage->state == ncclIbCommStatePendingReady) goto ib_recv_ready;
  if (stage->state != ncclIbCommStateStart) {
    ERROR("Listencomm in unknown state %d\n", stage->state);
    return ncclInternalError;
  }

  NCCLCHECKGOTO(ncclIbMalloc((void**)&rComm, sizeof(struct ncclIbRecvComm)), res, fail);
  NCCLCHECKGOTO(ncclIbRecvCommInit(rComm), res, fail);
  NCCLCHECKGOTO(ncclIbStatsInit(&rComm->base.stats), res, fail);
  stage->comm = rComm;
  stage->state = ncclIbCommStateAccept;
  NCCLCHECKGOTO(ncclSocketInit(&rComm->base.sock), res, fail);

  NCCLCHECKGOTO(ncclSocketAccept(&rComm->base.sock, &lComm->sock), res, fail);
ib_accept_check:
  NCCLCHECKGOTO(ncclSocketReady(&rComm->base.sock, &ready), res, fail);
  if (!ready) return ncclSuccess;

  NCCLCHECKGOTO(ncclIbMalloc((void**)&stage->buffer, sizeof(remMeta)), res, fail);
  stage->state = ncclIbCommStateRecv;
  stage->offset = 0;

ib_recv:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, stage->buffer, sizeof(remMeta),
                                   &stage->offset), res, fail);
  if (stage->offset != sizeof(remMeta))
    return ncclSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct ncclIbConnectionMetadata));
  memset(&meta, 0, sizeof(meta));
  rComm->base.remOooRqSize = remMeta.oooRqSize;
  NCCLCHECKGOTO(ncclIbEnableOooRq(lComm->dev, rComm->base.remOooRqSize, &rComm->base.oooRq), res, fail);
  rComm->base.recvMatchingScheme = rComm->base.oooRq ? BY_ID : BY_INDEX;

  // IB setup
  ctx = ncclIbDevs[lComm->dev].context;
  ib_port = ncclIbDevs[lComm->dev].port;
  struct ibv_port_attr portAttr;
  NCCLCHECKGOTO(wrap_ibv_query_port(ctx, ib_port, &portAttr), res, fail);
  union ibv_gid gid;
  NCCLCHECKGOTO(wrap_ibv_query_gid(ctx, ib_port, gidIndex, &gid), res, fail);

  // QP Creation
  NCCLCHECKGOTO(ncclIbInitCommDevBase(lComm->dev, &rComm->dev.base, &rComm->base.stats), res, fail);
  rComm->base.nqps = ncclParamIbQpsPerConn();
  rComm->base.qpIndex = 0;
  for (int q=0; q<rComm->base.nqps; q++) {
    // Remote Atomic operations are used for GIN IPutSignal (ATOMIC_FETCH_AND_ADD).
    // REMOTE_READ is required for GIN IGet / IFlush (RDMA READ).
    NCCLCHECKGOTO(ncclIbCreateQp(ib_port, &rComm->dev.base,
                                 IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_ATOMIC,
                                 rComm->base.oooRq, rComm->base.qps+q, &rComm->base.stats), res, fail);
    INFO(NCCL_NET,
         "NET/IB: %s: QP created: port=%d dev=%d devName=%s qp_num=%u pkey=%u pd=%p oooRq=%d",
         __func__, ib_port, lComm->dev, ncclIbDevs[lComm->dev].devName, rComm->base.qps[q]->qp_num,
         (uint16_t)ncclParamIbPkey(), rComm->dev.base.pd, rComm->base.oooRq);
  }

  // Adjust the MTU
  remMeta.mtu = (enum ibv_mtu)std::min(remMeta.mtu, portAttr.active_mtu);

  // Setup QP
  for (int q=0; q<rComm->base.nqps; q++) {
    struct ibv_qp* qp = rComm->base.qps[q];
    if (remMeta.qpInfo[q].ece_supported) {
      NCCLCHECKGOTO(wrap_ibv_set_ece(qp, &remMeta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), res, fail);
    }
    int fifoTc = remMeta.lid == 0 && ncclParamIbFifoTc() != -1 ? ncclParamIbFifoTc() : remMeta.tc;
    NCCLCHECKGOTO(ncclIbRtrQp(qp, remMeta.qpInfo[q].qpn, &remMeta, lComm->dev,
                               fifoTc, remMeta.sl), res, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(qp), res, fail);
    if (meta.qpInfo[q].ece_supported) {
      NCCLCHECKGOTO(wrap_ibv_query_ece(qp, &meta.qpInfo[q].ece, &meta.qpInfo[q].ece_supported), res, fail);
    }
  }
  NCCLCHECKGOTO(ncclIbPrepostRecvWorkRequests(rComm), res, fail);

  // Retain remote fifo info and prepare my RDMA ops
  rComm->remCtsFifo.rkey = remMeta.fifoRkey;
  rComm->remCtsFifo.addr = remMeta.fifoAddr;
  NCCLCHECKGOTO(wrap_ibv_reg_mr(&rComm->remCtsFifo.mr, rComm->dev.base.pd, &rComm->remCtsFifo.elems, sizeof(struct ncclIbSendFifo) * NET_IB_MAX_REQUESTS * NCCL_NET_IB_MAX_RECVS,
                                IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_READ), res, fail);
  rComm->remCtsFifo.sge.length = sizeof(struct ncclIbSendFifo);
  rComm->remCtsFifo.sge.lkey = rComm->remCtsFifo.mr->lkey;
  if (ncclParamIbUseInline()) rComm->remCtsFifo.flags = IBV_SEND_INLINE;

  // Allocate Flush dummy buffer for GPU Direct RDMA
  rComm->gpuFlush.enabled = (ncclIbGdrSupport() == 0) && (ncclParamIbGdrFlushDisable() == 0) ? 1 : 0;
  rComm->gpuFlush.useWrite = pcclParamIbGdrFlushWrite();
  if (rComm->gpuFlush.enabled) {
    int flushReadDepth = 1;
    if (!rComm->gpuFlush.useWrite) {
      flushReadDepth = ncclIbGetFlushReadDepth(ncclIbDevs + lComm->dev);
    }
    int hostMrFlags = IBV_ACCESS_LOCAL_WRITE;
    int qpFlags = IBV_ACCESS_LOCAL_WRITE;
    if (rComm->gpuFlush.useWrite) {
      // RDMA Write flush: host buffer is the remote write destination.
      hostMrFlags |= IBV_ACCESS_REMOTE_WRITE;
      qpFlags |= IBV_ACCESS_REMOTE_WRITE;
    } else {
      // RDMA Read flush: GPU memory is the remote read source.
      qpFlags |= IBV_ACCESS_REMOTE_READ;
    }
    NCCLCHECKGOTO(wrap_ibv_reg_mr(&rComm->gpuFlush.hostMr, rComm->dev.base.pd, &rComm->gpuFlush.hostMem, sizeof(int), hostMrFlags), res, fail);
    rComm->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlush.hostMem;
    rComm->gpuFlush.sge.length = 1;
    rComm->gpuFlush.sge.lkey = rComm->gpuFlush.hostMr->lkey;
    NCCLCHECKGOTO(ncclIbCreateQp(ib_port, &rComm->dev.base, qpFlags,
                                 0, &rComm->gpuFlush.qp, &rComm->base.stats), res, fail);
    struct ncclIbConnectionMetadata localMeta;
    localMeta.lid=portAttr.lid;
    localMeta.ib_port=ib_port;
    localMeta.spn=gid.global.subnet_prefix;
    localMeta.iid=gid.global.interface_id;
    localMeta.mtu=portAttr.active_mtu;
    NCCLCHECKGOTO(ncclIbRtrQp(rComm->gpuFlush.qp, rComm->gpuFlush.qp->qp_num, &localMeta, lComm->dev,
                               remMeta.tc, remMeta.sl, flushReadDepth), res, fail);
    NCCLCHECKGOTO(ncclIbRtsQp(rComm->gpuFlush.qp, flushReadDepth), res, fail);
    if (!rComm->gpuFlush.useWrite) {
      VERBOSE(NCCL_NET, "NET/IB: GDR READ flush QP dev=%d qp=%u depth=%d",
              lComm->dev, rComm->gpuFlush.qp->qp_num, flushReadDepth);
    }
    if (rComm->gpuFlush.useWrite) {
      INFO(NCCL_INIT, "NET/IB: GDR flush using RDMA Write (dev %d)", lComm->dev);
    }
  }

  // Fill Handle
  meta.tc = remMeta.tc;
  meta.sl = remMeta.sl;
  meta.lid=portAttr.lid;
  meta.ib_port=ib_port;
  for (int q=0; q<rComm->base.nqps; q++) meta.qpInfo[q].qpn=rComm->base.qps[q]->qp_num;
  meta.spn=gid.global.subnet_prefix;
  meta.iid=gid.global.interface_id;
  meta.mtu=remMeta.mtu;
  meta.oooRqSize = ncclIbGetOooRqSize(lComm->dev);

  stage->state = ncclIbCommStateSend;
  stage->offset = 0;
  //if (stage->buffer) free(stage->buffer);
  //NCCLCHECK(ncclIbMalloc((void**)&stage->buffer, sizeof(struct ncclIbConnectionMetadata)));
  memcpy(stage->buffer, &meta, sizeof(struct ncclIbConnectionMetadata));
ib_send:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_SEND, &rComm->base.sock, stage->buffer, sizeof(struct ncclIbConnectionMetadata),
                                   &stage->offset), res, fail);
  if (stage->offset < sizeof(struct ncclIbConnectionMetadata)) return ncclSuccess;

  stage->state = ncclIbCommStatePendingReady;
  stage->offset = 0;
ib_recv_ready:
  NCCLCHECKGOTO(ncclSocketProgress(NCCL_SOCKET_RECV, &rComm->base.sock, &rComm->base.ready, sizeof(int),
                                   &stage->offset), res, fail);
  if (stage->offset != sizeof(int)) return ncclSuccess;

  /* reset lComm stage */
  stage->state = ncclIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  *recvComm = rComm;
exit:
  NCCLCHECK(ncclFree(stage->buffer));
  return res;
fail:
  *recvComm = NULL;
  if (rComm) {
    // Same reasoning as pcclIbConnect.
    NCCLCHECK(ncclIbCloseRecv(rComm));
    rComm = NULL;
    // Only the success path above clears this; never leave it pointing at freed memory.
    stage->comm = NULL;
  }
  stage->state = ncclIbCommStateStart;
  stage->offset = 0;
  goto exit;
}

ncclResult_t ncclIbCloseSend(void* sendComm) {
  struct ncclIbSendComm* comm = (struct ncclIbSendComm*)sendComm;
  struct ncclIbSendCommDev* commDev = &comm->dev;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));
    for (int q=0; q<comm->base.nqps; q++)
      if (comm->base.qps[q] != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q]));
    if (comm->fifoMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->fifoMr));
    if (commDev->putSignalScratchpadMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(commDev->putSignalScratchpadMr));
    NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    NCCLCHECK(ncclFree(comm));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseRecv(void* recvComm) {
  struct ncclIbRecvComm* comm = (struct ncclIbRecvComm*)recvComm;
  struct ncclIbRecvCommDev* commDev = &comm->dev;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->base.sock));
    for (int q=0; q<comm->base.nqps; q++)
      if (comm->base.qps[q] != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->base.qps[q]));
    if (comm->gpuFlush.enabled) {
      if (comm->gpuFlush.qp != NULL) NCCLCHECK(wrap_ibv_destroy_qp(comm->gpuFlush.qp));
      if (comm->gpuFlush.hostMr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->gpuFlush.hostMr));
    }
    if (comm->remCtsFifo.mr != NULL) NCCLCHECK(wrap_ibv_dereg_mr(comm->remCtsFifo.mr));
    NCCLCHECK(ncclIbDestroyBase(&commDev->base));
    NCCLCHECK(ncclFree(comm));
  }
  return ncclSuccess;
}

ncclResult_t ncclIbCloseListen(void* listenComm) {
  struct ncclIbListenComm* comm = (struct ncclIbListenComm*)listenComm;
  if (comm) {
    NCCLCHECK(ncclSocketClose(&comm->sock));
    NCCLCHECK(ncclFree(comm));
  }
  return ncclSuccess;
}
