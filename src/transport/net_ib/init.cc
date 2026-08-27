/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

pthread_mutex_t ncclIbLock = PTHREAD_MUTEX_INITIALIZER;
int ncclIbRelaxedOrderingEnabled = 0;

NCCL_PARAM(IbDisable, "IB_DISABLE", 0);
NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 0);
NCCL_PARAM(IbDataDirect, "IB_DATA_DIRECT", 0);
NCCL_PARAM(IbPciRelaxedOrdering, "IB_PCI_RELAXED_ORDERING", 2);

PCCL_PARAM(IbDisableAccBondingPort, "IB_DISABLE_ACC_BONDING_PORT", 0);
PCCL_PARAM(IbNicSpeedScalingFactor, "IB_NIC_SPEED_SCALING_FACTOR", 1);
PCCL_PARAM(IbIgnoreTsoe, "IB_IGNORE_TSOE", 1);

static ncclResult_t ncclIbGetPciPath(char* devName, char** path, int* realPort) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
  char* p = realpath(devicePath, NULL);
  if (p == NULL) {
    WARN("Could not find real path of %s (%s)", devName, devicePath);
  } else {
    // Merge multi-port NICs into the same PCI device
    p[strlen(p)-1] = '0';
    // And keep the real port aside (the ibv port is always 1 on recent cards)
    *realPort = 0;
    for (int d=0; d<ncclNIbDevs; d++) {
      if (ncclIbDevs[d].pciPath == nullptr) continue;
      if (strcmp(p, ncclIbDevs[d].pciPath) == 0) (*realPort)++;
    }
  }
  *path = p;
  return ncclSuccess;
}

ncclResult_t ncclIbGetBondingIf(struct ibv_device** devices, char* ncclIbBondIfName, int nIbDevs) {
  char realPath[PATH_MAX];
  char *bondingMastersPath = realpath("/sys/class/net/bonding_masters", realPath);
  if (bondingMastersPath == NULL) {
    VERBOSE(NCCL_INIT|NCCL_NET, "NET/IB : Can't find /sys/class/net/bonding_masters");
    return ncclInternalError;
  }

  int nBondings = 0;
  char* bondingMasters = NULL;
  NCCLCHECK(ncclCalloc(&bondingMasters, nIbDevs*(MAX_IF_NAME_SIZE+1)));
  FILE *bondingMastersFile = fopen(bondingMastersPath, "r");
  char *_line = NULL;
  size_t _length = 0;
  ssize_t _read;
  while ((nBondings < nIbDevs) && ((_read = getline(&_line, &_length, bondingMastersFile)) != -1)) {
    if (_line[_read-1] == '\n') _line[_read-1] = '\0';
    char *contents = strtok(_line, " ");
    while(contents != NULL) {
      int l = 0;
      while (l<MAX_IF_NAME_SIZE && l<strlen(contents)) {
        bondingMasters[nBondings*(MAX_IF_NAME_SIZE+1)+l] = contents[l];
        l++;
      }
      bondingMasters[nBondings*(MAX_IF_NAME_SIZE+1)+l] = '\0';
      nBondings++;
      contents = strtok(NULL, " ");
    }
  }
  fclose(bondingMastersFile);

  if (nBondings == 0 || nBondings != nIbDevs) {
    VERBOSE(NCCL_INIT|NCCL_NET, "Empty bondings or mismatch bondings number(%d) to ibDev number(%d)", nBondings, nIbDevs);
    if (nBondings == 0) {
      goto exit;
    }
  }

  bool used[MAX_IB_DEVS];
  for (int i=0; i<MAX_IB_DEVS; i++) {
    used[i] = false;
  }

  char slavesPath[PATH_MAX];
  char devResPath[PATH_MAX];
  char netResPath[PATH_MAX];
  for (int d=0; d<nIbDevs; d++) {
    snprintf(devResPath, PATH_MAX, "/sys/class/infiniband/%s/device/resource", devices[d]->name);
    char *dev_p = realpath(devResPath, realPath);
    if (dev_p == NULL) {
      VERBOSE(NCCL_NET, "can't find %s", devResPath);
      continue;
    }
    for (int n=0; n<nBondings; n++) {
      if (used[n]) continue;
      snprintf(slavesPath, PATH_MAX, "/sys/class/net/%s/bonding/slaves", bondingMasters+n*(MAX_IF_NAME_SIZE+1));
      char* slaves_p = realpath(slavesPath, realPath);
      if (slaves_p == NULL) {
        VERBOSE(NCCL_NET, "can't find %s", slavesPath);
        continue;
      }
      FILE * slavesfile = fopen(slaves_p, "r");
      while ((_read = getline(&_line, &_length, slavesfile)) != -1) {
        if (_line[_read-1] == '\n') _line[_read-1] = '\0';
        char *contents = strtok(_line, " ");
        while(contents != NULL) {
          snprintf(netResPath, PATH_MAX, "/sys/class/net/%s/device/resource", contents);
          char *net_p = realpath(netResPath, realPath);
          if (net_p != NULL) {
            bool sameRes = true;
            FILE * devfile = fopen(devResPath, "r");
            FILE * netfile = fopen(netResPath, "r");
            int ch1, ch2;
            while ((ch1 = fgetc(devfile)) != EOF && (ch2 = fgetc(netfile)) != EOF) {
              if (ch1 != ch2) {
                sameRes = false;
                break;
              }
            }
            fclose(devfile);
            fclose(netfile);
            if (sameRes) {
              used[n] = true;
              strcpy(ncclIbBondIfName+d*(MAX_IF_NAME_SIZE+1), bondingMasters+n*(MAX_IF_NAME_SIZE+1));
            }
          }
          contents = strtok(NULL, " ");
        }
      }
      fclose(slavesfile);
    }
  }
  char info[1024];
  info[0] = '\0';
  snprintf(info+strlen(info), 1023-strlen(info), "nIbDevs %d, nBondIfs %d", nIbDevs, nBondings);
  for (int i = 0; i < nIbDevs; i++) {
    snprintf(info+strlen(info), 1023-strlen(info), ", %s:", devices[i]->name);
    if (strlen(ncclIbBondIfName+i*(MAX_IF_NAME_SIZE+1)) > 0) {
      snprintf(info+strlen(info), 1023-strlen(info), "%s", ncclIbBondIfName+i*(MAX_IF_NAME_SIZE+1));
    } else {
      snprintf(info+strlen(info), 1023-strlen(info), "N/A");
      sprintf(ncclIbBondIfName+i*(MAX_IF_NAME_SIZE+1), "N/A");
    }
  }
  INFO(NCCL_INIT|NCCL_NET, "%s", info);
exit:
  NCCLCHECK(ncclFree(bondingMasters));
  return ncclSuccess;
}

static int ibvWidths[] = { 1, 4, 8, 12, 2 };
static int ibvSpeeds[] = {
        2500,  /* SDR */
        5000,  /* DDR */
        10000, /* QDR */
        10000, /* QDR */
        14000, /* FDR */
        25000, /* EDR */
        50000, /* HDR */
        100000 /* NDR */ };

static int firstBitSet(int val, int max) {
  int i = 0;
  while (i<=max && ((val & (1<<i)) == 0)) i++;
  if (i > max) {
    i = 0;
    WARN("Invalid IB speed or width: %d, falling back to the lowest !", val);
  }
  return i;
}
static int ncclIbWidth(int width) {
  return ibvWidths[firstBitSet(width, sizeof(ibvWidths)/sizeof(int)-1)];
}
static int ncclIbSpeed(int speed) {
  return ibvSpeeds[firstBitSet(speed, sizeof(ibvSpeeds)/sizeof(int)-1)];
}

// Determine whether relaxed ordering is requested and the required
// IBVERBS_1.8 memory registration API is available.
static int ncclIbRelaxedOrderingCapable(void) {
  int roMode = ncclParamIbPciRelaxedOrdering();
  ncclResult_t result = ncclInternalError;
  if (roMode == 1 || roMode == 2) {
    result = wrap_ibv_reg_mr_iova2(NULL, NULL, NULL, 0, 0, 0);
  }
  return result == ncclInternalError ? 0 : 1;
}

static ncclResult_t ncclIbGetSpeed(char* ifName, const struct ibv_port_attr *portAttr, int *ibSpeed) {
  *ibSpeed = ncclIbSpeed(portAttr->active_speed) * ncclIbWidth(portAttr->active_width) * pcclParamIbNicSpeedScalingFactor();
  if (pcclParamIbDisableAccBondingPort() != 0 || strlen(ifName) <= 0 || strncmp("N/A", ifName, 2) == 0) return ncclSuccess;

  char realPath[PATH_MAX];
  char bondSpeedPath[PATH_MAX];
  snprintf(bondSpeedPath, PATH_MAX, "/sys/class/net/%s/speed", ifName);
  char *p = realpath(bondSpeedPath, realPath);
  if (p != NULL) {
    FILE *file = fopen(p, "r");
    char speedStr[16];
    int n = fscanf(file, "%s", speedStr);
    fclose(file);
    if (n > 0) {
      sscanf(speedStr, "%d", ibSpeed);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclIbInit(ncclDebugLogger_t logFunction) {
  static int shownIbHcaEnv = 0;
  if(wrap_ibv_symbols() != ncclSuccess) { return ncclInternalError; }
  if (ncclParamIbDisable()) return ncclInternalError;

  ncclResult_t res = ncclSuccess;
  if (ncclNIbDevs == -1) {
    pthread_mutex_lock(&ncclIbLock);
    wrap_ibv_fork_init();
    char* ncclIbBondIfName = NULL;
    if (ncclNIbDevs == -1) {
      ncclNIbDevs = 0;
      if (ncclFindInterfaces(ncclIbIfName, &ncclIbIfAddr, MAX_IF_NAME_SIZE, 1) != 1) {
        ERROR("NET/IB : No IP interface found.");
        res = ncclInternalError;
        goto exit;
      }

      // Detect IB cards
      int nIbDevs;
      struct ibv_device** devices;

      // Check if user defined which IB device:port to use
      const char* userIbEnv = pcclGetEnvWithDualPrefix("PCCL_IB_HCA");
      if (userIbEnv != NULL && shownIbHcaEnv++ == 0) INFO(NCCL_NET|NCCL_ENV, "PCCL_IB_HCA set to %s", userIbEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbEnv && userIbEnv[0] == '^';
      if (searchNot) userIbEnv++;
      bool searchExact = userIbEnv && userIbEnv[0] == '=';
      if (searchExact) userIbEnv++;
      int nUserIfs = parseStringList(userIbEnv, userIfs, MAX_IB_DEVS);
      int nMatchUserIfs = 0;

      if (ncclSuccess != wrap_ibv_get_device_list(&devices, &nIbDevs)) {
        ERROR("NET/IB : Unable to get device list");
        res = ncclInternalError;
        goto exit;
      }
      if (nIbDevs > 0 && pcclParamIbDisableAccBondingPort() == 0) {
        if (ncclCalloc(&ncclIbBondIfName, nIbDevs*(MAX_IF_NAME_SIZE+1)) != ncclSuccess
            || ncclIbGetBondingIf(devices, ncclIbBondIfName, nIbDevs) != ncclSuccess) {
          WARN("NET/IB : Unable to get bonding ifs");
        }
      }

      for (int d=0; d<nIbDevs && ncclNIbDevs<MAX_IB_DEVS; d++) {
        // Skip tsoe_ devices when PCCL_IB_IGNORE_TSOE is enabled and user didn't specify IB_HCA
        if (pcclParamIbIgnoreTsoe() && userIbEnv == NULL &&
            strncmp(devices[d]->name, "tsoe_", 5) == 0) {
          INFO(NCCL_INIT|NCCL_NET, "NET/IB : Skipping tsoe device %s (PCCL_IB_IGNORE_TSOE=1)", devices[d]->name);
          continue;
        }
        struct ibv_context * context;
        if (ncclSuccess != wrap_ibv_open_device(&context, devices[d]) || context == NULL) {
          WARN("NET/IB : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (ncclSuccess != wrap_ibv_query_device(context, &devAttr)) {
          WARN("NET/IB : Unable to query device %s", devices[d]->name);
          if (ncclSuccess != wrap_ibv_close_device(context)) {
            ERROR("NET/IB : Unable to close device %s after querying failed", devices[d]->name);
            res = ncclInternalError;
            goto exit;
          }
          continue;
        }
        for (int port = 1; port <= devAttr.phys_port_cnt; port++) {
          if (ncclNIbDevs >= MAX_IB_DEVS) {
            WARN("NET/IB : More than %d IB devices detected.", MAX_IB_DEVS);
            res = ncclInternalError;
            goto exit;
          }
          struct ibv_port_attr portAttr;
          if (ncclSuccess != wrap_ibv_query_port(context, port, &portAttr)) {
            WARN("NET/IB : Unable to query port %d", port);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE) {
            WARN("NET/IB : Inactive state of device %s port %d", devices[d]->name, port);
            continue;
          }
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND
              && portAttr.link_layer != IBV_LINK_LAYER_ETHERNET) {
            WARN("NET/IB : Unkown link layer %d of device %s port %d", portAttr.link_layer, devices[d]->name, port);
            continue;
          }

          // check against user specified HCAs/ports
          if (nUserIfs > 0) {
            if (! (matchIfList(devices[d]->name, port, userIfs, nUserIfs, searchExact) ^ searchNot)) {
              continue;
            }
            nMatchUserIfs++;
          }
          VERBOSE(NCCL_INIT|NCCL_NET,"NET/IB: [%d] %s:%d/%s ", d, devices[d]->name, port,
                  portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE");
          ncclIbDevs[ncclNIbDevs].device = d;
          ncclIbDevs[ncclNIbDevs].guid = devAttr.sys_image_guid;
          ncclIbDevs[ncclNIbDevs].port = port;
          ncclIbDevs[ncclNIbDevs].link = portAttr.link_layer;
          ncclIbDevs[ncclNIbDevs].context = context;
          strncpy(ncclIbDevs[ncclNIbDevs].devName, devices[d]->name, MAXNAMESIZE);
          if (ncclIbBondIfName != NULL) strcpy(ncclIbDevs[ncclNIbDevs].ifName, ncclIbBondIfName+d*(MAX_IF_NAME_SIZE+1));
          NCCLCHECK(ncclIbGetGidIndex(context, port, portAttr.gid_tbl_len, &ncclIbDevs[ncclNIbDevs].gidIndex));
          NCCLCHECK(ncclIbGetPciPath(ncclIbDevs[ncclNIbDevs].devName, &ncclIbDevs[ncclNIbDevs].pciPath, &ncclIbDevs[ncclNIbDevs].realPort));
          NCCLCHECK(ncclIbGetSpeed(ncclIbDevs[ncclNIbDevs].ifName, &portAttr, &ncclIbDevs[ncclNIbDevs].speed));
          ncclIbDevs[ncclNIbDevs].maxQp = devAttr.max_qp;
          ncclIbDevs[ncclNIbDevs].ibProvider = IB_PROVIDER_MLX5;
          ncclIbDevs[ncclNIbDevs].pd = NULL;
          ncclIbDevs[ncclNIbDevs].dmaBufSupported = -1;
          NCCLCHECK(ncclIbStatsInit(&ncclIbDevs[ncclNIbDevs].stats));

          pthread_create(&ncclIbAsyncThread, NULL, ncclIbAsyncThreadMain, ncclIbDevs + ncclNIbDevs);
          ncclSetThreadName(ncclIbAsyncThread, "PCCL IbAsync %2d", ncclNIbDevs);
          pthread_detach(ncclIbAsyncThread); // will not be pthread_join()'d

          ncclNIbDevs++;
          nPorts++;
        }
        if (nPorts == 0 && ncclSuccess != wrap_ibv_close_device(context)) {
          ERROR("NET/IB : Unable to close device %s", devices[d]->name);
          res = ncclInternalError;
          goto exit;
        }
      }
      if (nUserIfs > 0 && nMatchUserIfs == 0) {
        WARN("NET/IB : Mismatched user defined interface %s", userIbEnv);
      }
      if (nIbDevs && (ncclSuccess != wrap_ibv_free_device_list(devices))) {
        ERROR("NET/IB : Unable to free ibv device list");
        res = ncclInternalError;
        goto exit;
      };
    }
    if (ncclNIbDevs == 0) {
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : No device found.");
    } else {
      ncclIbRelaxedOrderingEnabled = ncclIbRelaxedOrderingCapable();
      char line[1024];
      line[0] = '\0';
      for (int d=0; d<ncclNIbDevs; d++) {
        snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s:%d/%s(%d)", d, ncclIbDevs[d].devName,
                 ncclIbDevs[d].ifName, ncclIbDevs[d].port, ncclIbDevs[d].link == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE", ncclIbDevs[d].speed);
      }
      line[1023] = '\0';
      char addrline[SOCKET_NAME_MAXLEN+1];
      INFO(NCCL_INIT|NCCL_NET, "NET/IB : Using%s %s; OOB %s:%s", line,
           ncclIbRelaxedOrderingEnabled ? "[RO]" : "", ncclIbIfName,
           ncclSocketToString(&ncclIbIfAddr, addrline));
    }
exit:
    NCCLCHECK(ncclFree(ncclIbBondIfName));
    pthread_mutex_unlock(&ncclIbLock);
  }
  return res;
}

ncclResult_t ncclIbDevices(int* ndev) {
  *ndev = ncclNIbDevs;
  return ncclSuccess;
}

ncclResult_t ncclIbGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = ncclIbDevs[dev].devName;
  props->pciPath = ncclIbDevs[dev].pciPath;
  props->guid = ncclIbDevs[dev].guid;
  props->ptrSupport = NCCL_PTR_HOST;
  if (ncclIbGdrSupport() != ncclSuccess) {
    INFO(NCCL_NET,"NET/IB : GPU Direct RDMA Disabled for HCA %d '%s' (no module)", dev, ncclIbDevs[dev].devName);
  } else {
    props->ptrSupport |= NCCL_PTR_HGGC;
  }
  props->speed = ncclIbDevs[dev].speed;
  props->port = ncclIbDevs[dev].port + ncclIbDevs[dev].realPort;
  props->maxComms = ncclIbDevs[dev].maxQp;
  props->maxRecvs = NCCL_NET_IB_MAX_RECVS;
  return ncclSuccess;
}
