/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "nvmlwrap.h"
#include "net.h"
#include "coll_net.h"
#include <sys/stat.h>
#include <fcntl.h>
#include "xml.h"
#include "cpuset.h"
#include "bootstrap.h"
#include "transport.h"
#include "check_items.h"

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char* topoNodeTypeStr[] = { "GPU", "PCI", "ICNS", "CPU", "NIC", "NET" };
const char* topoLinkTypeStr[] = { "LOC", "ICN", "", "PCI", "", "", "", "", "", "SYS", "NET" };
const char* topoPathTypeStr[] = { "LOC", "ICN", "ICR", "PIX", "PXB", "PXN", "PXR", "PHBH", "PHB", "SYS" };

#define PCI_LANES_X8  8
#define PCI_LANES_X16 16

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

// Get an int64 from a PCI path. For example, sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ will return 0x000002000.
ncclResult_t pciPathToInt64(char* path, int offset, int minOffset, int64_t* id) {
  char* str = path+offset;
  // Remove trailing "/"
  if (*str == '/') str--;
  // Find next /
  while (*str != '/') str--;
  str++;
  int64_t numid;
  NCCLCHECK(busIdToInt64(str, &numid));
  // Ignore subdevice because those should use the same PCI link so we want to merge nodes.
  numid -= numid & 0xf;
  *id = numid;
  return ncclSuccess;
}

static ncclResult_t findLocalCpu(struct ncclTopoNode* node, struct ncclTopoNode** cpu) {
  *cpu = NULL;
  if (node->type == CPU) {
    *cpu = node;
    return ncclSuccess;
  }
  for (int l=0; l<node->nlinks; l++) {
    if (node->links[l].type == LINK_PCI) NCCLCHECK(findLocalCpu(node->links[l].remNode, cpu));
    if (*cpu != NULL) return ncclSuccess;
  }
  return ncclSuccess;
}

int interCpuWidth = 0;
int cpuPciWidth = 0;

static ncclResult_t ncclTopoGetInterCpuWidth(struct ncclTopoNode* cpu, float* width) {
  *width = LOC_WIDTH;
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER) {
    *width = P9_WIDTH;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_ARM) {
    *width = ARM_WIDTH;
    return ncclSuccess;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
    *width =
      cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_ERP ? ERP_QPI_WIDTH :
      cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SRP ? SRP_QPI_WIDTH :
      cpu->cpu.model == NCCL_TOPO_CPU_MODEL_INTEL_SKL ? SKL_QPI_WIDTH :
      BDW_QPI_WIDTH;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_AMD) {
    *width = AMD_WIDTH;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
    *width = cpu->cpu.model ==  NCCL_TOPO_CPU_MODEL_YONGFENG ? YONGFENG_ZPI_BW : ZPI_BW;
  }
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 && cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_HYGON) {
    *width = HYGON_WIDTH;
  }
  return ncclSuccess;
}

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, int64_t id) {
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *node = system->nodes[type].nodes+i;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, int64_t id) {
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
    ERROR("Error : tried to create too many nodes of type %d", type);
    return ncclInternalError;
  }
  struct ncclTopoNode* n = system->nodes[type].nodes+system->nodes[type].count;
  system->nodes[type].count++;
  n->type = type;
  n->id = id;
  if (type == GPU) {
    // Create link to itself (used in some corner cases)
    n->nlinks=1;
    n->links[0].type = LINK_LOC;
    n->links[0].remNode = n;
    n->links[0].width = LOC_WIDTH;
    n->gpu.dev = NCCL_TOPO_UNDEF;
    n->gpu.rank = NCCL_TOPO_UNDEF;
    n->gpu.hggcCompCap = NCCL_TOPO_UNDEF;
    n->gpu.hostId = NCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    n->cpu.arch = NCCL_TOPO_UNDEF;
    n->cpu.vendor = NCCL_TOPO_UNDEF;
    n->cpu.model = NCCL_TOPO_UNDEF;
  } else if (type == NET) {
    n->net.asic = 0ULL;
    n->net.port = NCCL_TOPO_UNDEF;
    n->net.width = 0.0;
  } else if (type == PCI) {
    n->pci.device = 0;
  }
  *node = n;
  return ncclSuccess;
}

ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
  struct ncclTopoNode* delNode = system->nodes[type].nodes+index;
  VERBOSE(NCCL_INIT, "Trim Node type == %s, bus == %x, i == %d", topoNodeTypeStr[type], delNode->id, index);
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    NCCLCHECK(ncclFree(delNode->paths[t]));
    for (int n=0; n<system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes+n;
      if (node == delNode) continue;
      for (int l=0; l<node->nlinks; l++) {
        while (l<node->nlinks && node->links[l].remNode == delNode) {
          memmove(node->links+l, node->links+l+1, (node->nlinks-l-1)*sizeof(struct ncclTopoLink));
          node->nlinks--;
        }
        if (l<node->nlinks && node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  memmove(delNode, delNode+1, (system->nodes[type].count-index-1)*sizeof(struct ncclTopoNode));
  system->nodes[type].count--;
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, int count, float width) {
  // Aggregate links into higher width for NVLink
  struct ncclTopoLink* link;
  for (link = node->links; link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  if (link->remNode == NULL) node->nlinks++;
  link->type = type;
  link->count += count;
  link->remNode = remNode;
  link->width += width;

  // Sort links in BW descending order
  struct ncclTopoLink linkSave;
  memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
  while (link != node->links) {
    if ((link-1)->width >= linkSave.width) break;
    memcpy(link, link-1, sizeof(struct ncclTopoLink));
    link--;
  }
  memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
  return ncclSuccess;
}

// BCM Gen4 Switches present themselves as a two-level hierarchical switch
// even though they're supposed to sustain full BW across all ports.
// Flatten the switch as this extra level can break the search and make
// NCCL take wrong topology decisions.
ncclResult_t ncclTopoFlattenBcmSwitches(struct ncclTopoSystem* system) {
  ncclResult_t res = ncclSuccess;
  int64_t* subSwIds = nullptr;
  for (int s=0; s<system->nodes[PCI].count; s++) {
    struct ncclTopoNode* pciSwitch = system->nodes[PCI].nodes+s;
    uint64_t device = pciSwitch->pci.device;
    // Only flatten PEX Gen 4 switches in base mode
    if ((device & 0xfffffffffffff000) == 0x1000c0101000a000) {
      // Find sub switches with the same device ID.
      NCCLCHECK(ncclCalloc(&subSwIds, pciSwitch->nlinks));
      int subs = 0;
      for (int l=0; l<pciSwitch->nlinks; l++) {
        struct ncclTopoNode* sub = pciSwitch->links[l].remNode;
        // Only fuse sub switches with the same device ID.
        if (sub->type != PCI || sub->pci.device != device) continue;
        // Save sub switch for later
        subSwIds[subs++] = sub->id;
        // Remove link to that sub switch
        memmove(pciSwitch->links+l, pciSwitch->links+l+1, (pciSwitch->nlinks-l-1)*(sizeof(struct ncclTopoLink)));
        pciSwitch->nlinks--;
        // Don't increase l for the next iteration as we just shifted all links by one.
        l--;
      }

      for (int s=0; s<subs; s++) {
        // Find sub switch (system->nodes[PCI].nodes is changing every time we remove a node)
        int index;
        NCCLCHECKGOTO(ncclTopoIdToIndex(system, PCI, subSwIds[s], &index), res, end);
        struct ncclTopoNode* sub = system->nodes[PCI].nodes+index;
        // Connect all sub PCI devices to the parent switch
        for (int l=0; l<sub->nlinks; l++) {
          struct ncclTopoNode* remNode = sub->links[l].remNode;
          if (remNode == pciSwitch) continue;
          // Add link from parent PCI switch -> PCI device
          memcpy(pciSwitch->links+pciSwitch->nlinks, sub->links+l, sizeof(struct ncclTopoLink));
          pciSwitch->nlinks++;
          // Update link from PCI device -> parent PCI switch
          for (int rl=0; rl<remNode->nlinks; rl++) {
            if (remNode->links[rl].remNode == sub) {
              remNode->links[rl].remNode = pciSwitch;
              break;
            }
          }
        }
        NCCLCHECKGOTO(ncclTopoRemoveNode(system, PCI, index), res, end);
      }
      // Set subdevice to 0x0000 to make sure we don't merge this switch again.
      pciSwitch->pci.device = 0x1000c01010000000;
      NCCLCHECK(ncclFree(subSwIds));
      // Restart, as system->nodes[PCI].nodes has changed.
      s = 0;
    }
  }

end:
  NCCLCHECK(ncclFree(subSwIds));
  return res;
}

ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
  // And connect all CPU nodes together
  for (int n=0; n<system->nodes[CPU].count; n++) {
    for (int p=0; p<system->nodes[CPU].count; p++) {
      if (n == p) continue;
      float width;
      NCCLCHECK(ncclTopoGetInterCpuWidth(system->nodes[CPU].nodes+n, &width));
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[CPU].nodes+n, system->nodes[CPU].nodes+p, LINK_SYS, 1, width));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    sprintf(line+offset, "%s/%lX (rank %d)", topoNodeTypeStr[node->type], node->id, node->gpu.rank);
  } else if (node->type == CPU) {
    sprintf(line+offset, "%s/%lX (%d/%d/%d)", topoNodeTypeStr[node->type], node->id, node->cpu.arch, node->cpu.vendor, node->cpu.model);
  } else if (node->type == PCI) {
    sprintf(line+offset, "%s/%lX (%lx)", topoNodeTypeStr[node->type], node->id, node->pci.device);
  } else {
    sprintf(line+offset, "%s/%lX", topoNodeTypeStr[node->type], node->id);
  }
  INFO(NCCL_GRAPH, "%s", line);
  for (int i=0; i<offset; i++) line[i] = ' ';

  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_LOC) continue;
    if (link->type != LINK_PCI || link->remNode != prevNode) {
      sprintf(line+offset, "+ %s[%2.1f] - ", topoLinkTypeStr[link->type], link->width);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line+nextOffset, "%s/%lX (%lx/%d/%f)", topoNodeTypeStr[link->remNode->type], link->remNode->id, link->remNode->net.asic, link->remNode->net.port, link->remNode->net.width);
        } else if (link->remNode->type == GPU) {
          sprintf(line+nextOffset, "%s/%lX (rank %d)",
                   topoNodeTypeStr[link->remNode->type], link->remNode->id,
                   link->remNode->gpu.rank);
        } else {
          sprintf(line+nextOffset, "%s/%lX", topoNodeTypeStr[link->remNode->type], link->remNode->id);
        }
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(ncclComm_t comm, struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxWidth %2.1f totalWidth %2.1f totalNetWidth %2.1f ===", s->maxWidth, s->totalWidth, s->totalNetWidth);
  char line[1024];
  for (int n=0; n<s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes+n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  NCCLCHECK(ncclTopoPrintPaths(s));
  return ncclSuccess;
}

static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
  // Shift all links to have upLink as last link
  if (upNode) {
    int l=0;
    while (node->links[l].remNode != upNode) l++;
    struct ncclTopoLink upLink;
    memcpy(&upLink, node->links+l, sizeof(struct ncclTopoLink));
    while (node->links[l+1].remNode) {
      memcpy(node->links+l, node->links+l+1, sizeof(struct ncclTopoLink));
      l++;
    }
    memcpy(node->links+l, &upLink, sizeof(struct ncclTopoLink));
  }

  // Recursively sort the PCI tree
  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_PCI && link->remNode != upNode) NCCLCHECK(ncclTopoSort(link->remNode, node));
  }
  return ncclSuccess;
}

// We want the graph to be organized to ease/accelerate traversal :
// 1. NVLinks (already the case)
// 2. PCI down
// 3. PCI up
// 4. SYS (already the case)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n=0; n<system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes+n, NULL));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNet(struct ncclXmlNode* xmlNet, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  int64_t dev;
  NCCLCHECK(xmlGetAttrInt64(xmlNet, "dev", &dev));

  int64_t netId = NCCL_TOPO_ID(systemId, dev);
  struct ncclTopoNode* net;
  NCCLCHECK(ncclTopoCreateNode(system, &net, NET, netId));
  net->net.dev = dev;
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlNet, "guid", &str));
  net->net.asic = (str) ? strtoull(str, NULL, 16) : netId;

  ncclDebugNoWarn = NCCL_GRAPH;
  int mbps;
  if (xmlGetAttrInt(xmlNet, "speed", &mbps) != ncclSuccess) mbps = 0;
  if (mbps <= 0) mbps = 10000; // Some NICs define speed = -1
  net->net.width = mbps / 8000.0;
  if (xmlGetAttrInt(xmlNet, "port", &net->net.port) != ncclSuccess) net->net.port = 0;
  if (xmlGetAttrInt(xmlNet, "gdr", &net->net.gdrSupport) != ncclSuccess) net->net.gdrSupport = 0;
  if (xmlGetAttrInt(xmlNet, "maxconn", &net->net.maxChannels) != ncclSuccess) net->net.maxChannels = MAXCHANNELS;
  if (xmlGetAttrInt(xmlNet, "coll", &net->net.collSupport) != ncclSuccess) net->net.collSupport = 0;
  ncclDebugNoWarn = 0;

  NCCLCHECK(ncclTopoConnectNodes(nic, net, LINK_NET, 1, net->net.width));
  NCCLCHECK(ncclTopoConnectNodes(net, nic, LINK_NET, 1, net->net.width));
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNic(struct ncclXmlNode* xmlNic, struct ncclTopoSystem* system, struct ncclTopoNode* nic, int systemId) {
  for (int s=0; s<xmlNic->nSubs; s++) {
    struct ncclXmlNode* xmlNet = xmlNic->subs[s];
    if (strcmp(xmlNet->name, "net") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(xmlNet, "dev", &index));
    if (index == -1) continue;
    NCCLCHECK(ncclTopoAddNet(xmlNet, system, nic, systemId));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddGpu(struct ncclXmlNode* xmlGpu, struct ncclTopoSystem* system, struct ncclTopoNode* gpu, int systemId) {
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "sm", &gpu->gpu.hggcCompCap));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "rank", &gpu->gpu.rank));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "dev", &gpu->gpu.dev));
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "gdr", &gpu->gpu.gdrSupport));
  int netAffinity;
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "net_affinity", &netAffinity));
  gpu->gpu.netAffinity = NCCL_TOPO_ID(systemId, netAffinity);
  NCCLCHECK(xmlGetAttrInt(xmlGpu, "hostid", &gpu->gpu.hostId, true));
  // Do not go any further, nvlinks will be added in a second pass
  return ncclSuccess;
}

struct kvDict kvDictPciClass[] = {
  { "0x060400", PCI },
  { "0x068000", NVS },
  { "0x068001", CPU },
  { "0x03", GPU },
  { "0x000000", GPU}, // for ppu, fixed to 0x000000 boot by pcie, EVB/early EVT periods
  { "0xffff00", GPU}, // for ppu, fixed to 0xffff00 boot by mcu, EVB/early EVT periods
  { "0x02", NIC },
  { NULL, PCI /* Default fallback value */ }
};

struct kvDict kvDictPciGen[] = {
  { "2.5 GT/s", 15 }, { "5 GT/s", 30 }, { "8 GT/s", 60 }, { "16 GT/s", 120 }, { "32 GT/s", 240 }, /* Kernel 5.6 and earlier */
  { "2.5 GT/s PCIe", 15 }, { "5.0 GT/s PCIe", 30 }, { "8.0 GT/s PCIe", 60 }, { "16.0 GT/s PCIe", 120 }, { "32.0 GT/s PCIe", 240 }, { "64.0 GT/s PCIe", 480 },
  { NULL, 60 /* Default fallback */ } }; // x100 Mbps per lane

PCCL_PARAM(DoubleEicPciWidthDisable, "DOUBLE_EIC_PCI_WIDTH_DISABLE", 0);
ncclResult_t ncclTopoAddPci(struct ncclXmlNode* xmlPci, struct ncclTopoSystem* system, struct ncclTopoNode* parent, int systemId, int numaId) {
  const char* str;

  int type;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "class", &str));
  NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));

  int64_t busId;
  NCCLCHECK(xmlGetAttrStr(xmlPci, "busid", &str));
  NCCLCHECK(busIdToInt64(str, &busId));

  struct ncclTopoNode* node = NULL;
  struct ncclXmlNode* xmlGpu = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "gpu", &xmlGpu));
  if (xmlGpu != NULL) {
    type = GPU;
    // add gpu node by pci busid
    int index = -1;
    NCCLCHECK(xmlGetAttrIndex(xmlGpu, "rank", &index));
    if (index == -1) return ncclSuccess;
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, NCCL_TOPO_ID(systemId, busId)));
    NCCLCHECK(ncclTopoAddGpu(xmlGpu, system, node, systemId));
  }
  struct ncclXmlNode* xmlNic = NULL;
  NCCLCHECK(xmlGetSub(xmlPci, "nic", &xmlNic));
  if (xmlNic != NULL) {
    type = NIC;
    // Ignore sub device ID and merge multi-port NICs into one PCI device.
    busId &= 0xfffffffffffffff0;
    struct ncclTopoNode* nicNode = NULL;
    int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, busId);
    int64_t id = NCCL_TOPO_ID(systemId, localNicId);
    NCCLCHECK(ncclTopoGetNode(system, &nicNode, type, id));
    if (nicNode == NULL) {
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, type, id));
      node = nicNode; // Connect it to parent later on
    }
    NCCLCHECK(ncclTopoAddNic(xmlNic, system, nicNode, systemId));
  } else if (type == PCI) {
    NCCLCHECK(ncclTopoCreateNode(system, &node, type, busId));
    NCCLCHECK(xmlGetAttr(xmlPci, "vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 48;
    NCCLCHECK(xmlGetAttr(xmlPci, "device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 32;
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_vendor", &str));
    if (str) node->pci.device += strtol(str, NULL, 0) << 16;
    NCCLCHECK(xmlGetAttr(xmlPci, "subsystem_device", &str));
    if (str) node->pci.device += strtol(str, NULL, 0);

    for (int s=0; s<xmlPci->nSubs; s++) {
      struct ncclXmlNode* xmlSubPci = xmlPci->subs[s];
      NCCLCHECK(ncclTopoAddPci(xmlSubPci, system, node, systemId, numaId));
    }
  }

  if (node) {
    int width, speed, pciGenSpeed = -1;
    const int pciGen4Speed = 16;
    NCCLCHECK(xmlGetAttrInt(xmlPci, "link_width", &width));
    NCCLCHECK(xmlGetAttrStr(xmlPci, "link_speed", &str));

    // Manage cases where speed was not indicated in /sys
    if (width == 0) width = PCI_LANES_X16;
    sscanf(str, "%d GT/s", &pciGenSpeed);
    if (node->type == GPU && pciGenSpeed < pciGen4Speed) {
      WARN("PCIe speed is lower than Gen4 speed: rank %d, dev %d, speed %s", node->gpu.rank, node->gpu.dev, str);
    }

    // we can only see 1 ep of 2
    if (ncclEicNet() && type == NIC && width == PCI_LANES_X8) {
      if (pcclParamDoubleEicPciWidthDisable() == 1) {
        INFO(NCCL_GRAPH, "found EIC pci node %lx width %d, disable double it may result in sub-optimal performance", busId, width);
      } else {
        width = PCI_LANES_X16;
        INFO(NCCL_GRAPH, "double link_width of EIC pci node %lx to %d", busId, width);
      }
    }

    NCCLCHECK(kvConvertToInt(str, &speed, kvDictPciGen)); // Values in 100Mbps, per lane (we want GB/s in the end)

    NCCLCHECK(ncclTopoConnectNodes(node, parent, LINK_PCI, 1, width*speed/80.0));
    NCCLCHECK(ncclTopoConnectNodes(parent, node, LINK_PCI, 1, width*speed/80.0));
  }
  return ncclSuccess;
}

struct kvDict kvDictCpuArch[] = { { "x86_64", NCCL_TOPO_CPU_ARCH_X86 }, { "arm64", NCCL_TOPO_CPU_ARCH_ARM }, { "ppc64", NCCL_TOPO_CPU_ARCH_POWER }, { NULL, 0 } };
struct kvDict kvDictCpuVendor[] = { { "GenuineIntel", NCCL_TOPO_CPU_VENDOR_INTEL }, { "AuthenticAMD", NCCL_TOPO_CPU_VENDOR_AMD },
 { "CentaurHauls", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, { "  Shanghai  ", NCCL_TOPO_CPU_VENDOR_ZHAOXIN }, {"HygonGenuine", NCCL_TOPO_CPU_VENDOR_HYGON}, { NULL, 0 } };

ncclResult_t ncclGetSystemId(struct ncclTopoSystem* system, struct ncclXmlNode* xmlCpu, int* systemIdPtr) {
  const char* hostHashStr;
  NCCLCHECK(xmlGetAttr(xmlCpu, "host_hash", &hostHashStr));
  uint64_t hostHash = hostHashStr ? strtoull(hostHashStr, NULL, 16) : 0;
  int systemId;
  for (systemId=0; systemId<system->nHosts; systemId++) if (system->hostHashes[systemId] == hostHash) break;
  if (systemId == system->nHosts) system->hostHashes[system->nHosts++] = hostHash;
  *systemIdPtr = systemId;
  return ncclSuccess;
}

ncclResult_t ncclTopoAddCpu(struct ncclXmlNode* xmlCpu, struct ncclTopoSystem* system) {
  int64_t numaId;
  NCCLCHECK(xmlGetAttrInt64(xmlCpu, "numaid", &numaId));
  int systemId;
  NCCLCHECK(ncclGetSystemId(system, xmlCpu, &systemId));
  struct ncclTopoNode* cpu;
  NCCLCHECK(ncclTopoCreateNode(system, &cpu, CPU,  NCCL_TOPO_ID(systemId, numaId)));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlCpu, "affinity", &str));
  if (str != NULL) {
    NCCLCHECK(ncclStrToCpuset(str, &cpu->cpu.affinity));
  }

  NCCLCHECK(xmlGetAttrStr(xmlCpu, "arch", &str));
  NCCLCHECK(kvConvertToInt(str, &cpu->cpu.arch, kvDictCpuArch));
  if (cpu->cpu.arch == NCCL_TOPO_CPU_ARCH_X86) {
    NCCLCHECK(xmlGetAttrStr(xmlCpu, "vendor", &str));
    NCCLCHECK(kvConvertToInt(str, &cpu->cpu.vendor, kvDictCpuVendor));
    if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      cpu->cpu.model =
        (familyId == 6 && modelId >= 0xCF) ? NCCL_TOPO_CPU_MODEL_INTEL_ERP :
        (familyId == 6 && modelId >= 0x8F) ? NCCL_TOPO_CPU_MODEL_INTEL_SRP :
        (familyId == 6 && modelId >= 0x55) ? NCCL_TOPO_CPU_MODEL_INTEL_SKL :
        NCCL_TOPO_CPU_MODEL_INTEL_BDW;
    } else if (cpu->cpu.vendor == NCCL_TOPO_CPU_VENDOR_ZHAOXIN) {
      int familyId, modelId;
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "familyid", &familyId));
      NCCLCHECK(xmlGetAttrInt(xmlCpu, "modelid", &modelId));
      if (familyId == 7 && modelId == 0x5B) cpu->cpu.model = NCCL_TOPO_CPU_MODEL_YONGFENG;
    }
  }
  for (int s=0; s<xmlCpu->nSubs; s++) {
    struct ncclXmlNode* node = xmlCpu->subs[s];
    if (strcmp(node->name, "pci") == 0) NCCLCHECK(ncclTopoAddPci(node, system, cpu, systemId, numaId));
    if (strcmp(node->name, "nic") == 0) {
      struct ncclTopoNode* nic = NULL;
      int64_t localNicId = NCCL_TOPO_LOCAL_NIC_ID(numaId, 0);
      int64_t id = NCCL_TOPO_ID(systemId, localNicId);
      NCCLCHECK(ncclTopoGetNode(system, &nic, NIC, id));
      if (nic == NULL) {
        NCCLCHECK(ncclTopoCreateNode(system, &nic, NIC, id));
        NCCLCHECK(ncclTopoConnectNodes(cpu, nic, LINK_PCI, 1, LOC_WIDTH));
        NCCLCHECK(ncclTopoConnectNodes(nic, cpu, LINK_PCI, 1, LOC_WIDTH));
      }
      NCCLCHECK(ncclTopoAddNic(node, system, nic, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoAddNvLinks(struct ncclXmlNode* node, struct ncclTopoSystem* system, const char* parentBusId, int systemId) {
  if (strcmp(node->name, "icnlink") == 0) {
    struct ncclTopoNode* gpu = NULL;
    int64_t pBusId;
    NCCLCHECK(busIdToInt64(parentBusId, &pBusId));
    pBusId = NCCL_TOPO_ID(systemId, pBusId);
    NCCLCHECK(ncclTopoGetNode(system, &gpu, GPU, pBusId));
    if (gpu == NULL) {
      ERROR("Add ICNLink error : could not find PPU %lx", pBusId);
      return ncclInternalError;
    }
    int count;
    NCCLCHECK(xmlGetAttrInt(node, "count", &count));
    float totalBw = -1.0f;
    NCCLCHECK(xmlGetAttrFloat(node, "bandwidth", &totalBw, true));
    const char* targetClass;
    NCCLCHECK(xmlGetAttrStr(node, "tclass", &targetClass));
    int targetType;
    NCCLCHECK(kvConvertToInt(targetClass, &targetType, kvDictPciClass));
    struct ncclTopoNode* remote = NULL;
    if (targetType == GPU) {
      // NVL P2P connection to another GPU
      const char* target;
      NCCLCHECK(xmlGetAttrStr(node, "target", &target));
      int64_t busId;
      NCCLCHECK(busIdToInt64(target, &busId));
      NCCLCHECK(ncclTopoGetNode(system, &remote, GPU, NCCL_TOPO_ID(systemId, busId)));
    } else if (targetType == CPU) {
      // NVL connection to the local CPU
      NCCLCHECK(findLocalCpu(gpu, &remote));
    } else {
      if (system->nodes[NVS].count == 0) {
        NCCLCHECK(ncclTopoCreateNode(system, &remote, NVS, 0));
      } else {
        remote = system->nodes[NVS].nodes;
      }
    }
    if (remote) {
      float nvlSpeed = ncclTopoNVLinkSpeed(gpu->gpu.hggcCompCap);
      float maxSpeed = ncclTopoIcnLinkTheoreticalSpeed();
      if (totalBw > .0f) {
        float halfSpeed = nvlSpeed / 2;
        float bandWidth = totalBw / count; // assume same bandwidth perf link between 2 PPUs
        nvlSpeed = bandWidth >= maxSpeed ? nvlSpeed : halfSpeed;
      }
      NCCLCHECK(ncclTopoConnectNodes(gpu, remote, LINK_NVL, count, count*nvlSpeed));
      if (remote->type != GPU) {
        NCCLCHECK(ncclTopoConnectNodes(remote, gpu, LINK_NVL, count, count*nvlSpeed));
      }
    }
  } else {
    if (strcmp(node->name, "cpu") == 0) {
      NCCLCHECK(ncclGetSystemId(system, node, &systemId));
    }
    const char* busId;
    NCCLCHECK(xmlGetAttr(node, "busid", &busId));
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoAddNvLinks(node->subs[s], system, busId ? busId : parentBusId, systemId));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem, const uint64_t localHostHash) {
  NCCLCHECK(ncclCalloc(topoSystem, 1));
  struct ncclTopoSystem* system = *topoSystem;
  struct ncclXmlNode* topNode;
  NCCLCHECK(xmlFindTag(xml, "system", &topNode));
  for (int s=0; s<topNode->nSubs; s++) {
    struct ncclXmlNode* node = topNode->subs[s];
    if (strcmp(node->name, "cpu") == 0) NCCLCHECK(ncclTopoAddCpu(node, *topoSystem));
  }

  int systemId = 0;
  while (systemId < system->nHosts && system->hostHashes[systemId] != localHostHash) systemId++;
  system->systemId = systemId;
  if(systemId == system->nHosts){
    WARN("localHostHash = 0x%lx not found in the list of system hostHashes",localHostHash);
    return ncclInvalidArgument;
  }

  NCCLCHECK(ncclTopoAddNvLinks(topNode, *topoSystem, NULL, 0));
  NCCLCHECK(ncclTopoFlattenBcmSwitches(*topoSystem));
  NCCLCHECK(ncclTopoConnectCpus(*topoSystem));
  NCCLCHECK(ncclTopoSortSystem(*topoSystem));

  return ncclSuccess;
}

NCCL_PARAM(TopoDumpFileRank, "TOPO_DUMP_FILE_RANK", 0);

static ncclResult_t ncclTopoGetXmlPathRec(struct ncclXmlNode* node, int& ppuIndex, int& netIndex, struct ncclXmlPath* ppuPaths, struct ncclXmlPath* netPaths) {
  const char* name = node->name;
  if (name && strcmp(name, "gpu") == 0) {
    NCCLCHECK(xmlGetPathToSystem(node, ppuPaths + ppuIndex));
    ppuIndex++;
  } else if (name && strcmp(name, "net") == 0) {
    NCCLCHECK(xmlGetPathToSystem(node, netPaths + netIndex));
    netIndex++;
  } else {
    // Copy nSubs and subs as they could change as we trim recursively.
    struct ncclXmlNode* subs[MAX_SUBS];
    int nSubs = node->nSubs;
    memcpy(subs, node->subs, nSubs*sizeof(struct ncclXmlNode*));
    for (int s=0; s<nSubs; s++) {
      NCCLCHECK(ncclTopoGetXmlPathRec(subs[s], ppuIndex, netIndex, ppuPaths, netPaths));
    }
  }
  return ncclSuccess;
}

// Only set values if not already set
static ncclResult_t xmlInitAttrInt(struct ncclXmlNode* node, const char* attrName, const int value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "%d", value);
  }
  return ncclSuccess;
}
static ncclResult_t xmlInitAttrUint64(struct ncclXmlNode* node, const char* attrName, const uint64_t value) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(node, attrName, &index));
  if (index == -1) {
    index = node->nAttrs++;
    strncpy(node->attrs[index].key, attrName, MAX_STR_LEN);
    snprintf(node->attrs[index].value, MAX_STR_LEN, "0x%lx", value);
  }
  return ncclSuccess;
}

PCCL_PARAM(NetAffinityAuto, "NET_AFFINITY_AUTO", 1);
ncclResult_t pcclAffinityNetsPathAdjust(struct ncclTopoSystem* system) {
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    for (int n=0; n<system->nodes[NET].count; n++) {
      struct ncclTopoNode* net = system->nodes[NET].nodes + n;
      if (gpu->gpu.netAffinity == net->id) {
        struct ncclTopoLinkList* gPath = system->nodes[GPU].nodes[g].paths[NET] + n;
        struct ncclTopoLinkList* nPath = system->nodes[NET].nodes[n].paths[GPU] + g;
        if (gPath->type > PATH_PHBH) gPath->type = PATH_PHBH;
        if (nPath->type > PATH_PHBH) nPath->type = PATH_PHBH;
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystem(ncclComm_t comm, struct ncclTopoSystem** system) {
  ncclResult_t res = ncclSuccess;
  int ppuCount = 0;
  int netDevCount = 0;
  int ppuFilledCount = 0;
  bool enablePxbToPxn = false;

  struct ncclXml* xml = nullptr;
  char* mem = NULL;
  int* localRanks = NULL;
  struct ncclXml* rankXml;
  int localRank = -1, nLocalRanks = 0;
  NCCLCHECK(xmlAlloc(&xml, NCCL_TOPO_XML_MAX_NODES));
  const char* xmlTopoFile = pcclGetEnvWithDualPrefix("PCCL_TOPO_FILE");
  if (xmlTopoFile) {
    INFO(NCCL_ENV, "PCCL_TOPO_FILE set by environment to %s", xmlTopoFile);
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile(xmlTopoFile, xml), res, fail);
  } else {
    // Try default XML topology location
    ncclDebugNoWarn = NCCL_GRAPH;
    NCCLCHECKGOTO(ncclTopoGetXmlFromFile("/var/run/t-head-topologyd/virtualTopology.xml", xml), res, fail);
    ncclDebugNoWarn = 0;
  }

  // Fixup the cpu's host_hashes.
  struct ncclXmlNode* node;
  // Update every cpu node's host_hash attribute since those are not
  // intended to be preserved from the XML files that have been read.
  NCCLCHECKGOTO(xmlFindTag(xml, "cpu", &node), res, fail);
  while (node != nullptr) {
    NCCLCHECKGOTO(xmlSetAttrLong(node, "host_hash", getHostHash()), res, fail);
    NCCLCHECKGOTO(xmlFindNextTag(xml, "cpu", node, &node), res, fail);
  }
  if (xml->maxIndex == 0) {
    // Create top tag
    struct ncclXmlNode* top;
    NCCLCHECKGOTO(xmlAddNode(xml, NULL, "system", &top), res, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(top, "version", NCCL_TOPO_XML_VERSION), res, fail);
  }

  if (pcclHgmlEnsureInitialized() == ncclSuccess) {
    for (int n=0; n<pcclHgmlDeviceCount; n++) {
      hgmlPciInfo_t pciInfo;
      int64_t busId;
      char lowerId[HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      NCCLCHECKGOTO(pcclHgmlDeviceGetPciInfo(pcclHgmlDevices[n].handle, &pciInfo), res, fail);
      NCCLCHECKGOTO(convertLowerBusId(pciInfo.busId, HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, lowerId), res, fail);
      NCCLCHECKGOTO(busIdToInt64(pciInfo.busId, &busId), res, fail);
      NCCLCHECKGOTO(ncclTopoFillGpu(xml, lowerId, &node), res, fail);
      if (node) {
        ppuCount++;
        NCCLCHECKGOTO(xmlSetAttrInt(node, "rank", comm->rank), res, fail);
        NCCLCHECKGOTO(xmlInitAttrInt(node, "gdr", comm->peerInfo[comm->rank].gdrSupport), res, fail);
        NCCLCHECKGOTO(xmlSetAttrInt(node, "net_affinity", -1), res, fail);
        if (busId == comm->busId) {
          NCCLCHECKGOTO(xmlSetAttrInt(node, "keep", 1), res, fail);
        } else {
          NCCLCHECKGOTO(xmlSetAttrInt(node, "keep", 0), res, fail);
        }
      }
    }
  } else {
    char busId[HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    NCCLCHECKGOTO(int64ToBusId(comm->peerInfo[comm->rank].busId, busId), res, fail);
    NCCLCHECKGOTO(ncclTopoFillGpu(xml, busId, &node), res, fail);
    if (node) {
      ppuCount++;
      NCCLCHECKGOTO(xmlSetAttrInt(node, "keep", 1), res, fail);
      NCCLCHECKGOTO(xmlSetAttrInt(node, "rank", comm->rank), res, fail);
      NCCLCHECKGOTO(xmlInitAttrInt(node, "gdr", comm->peerInfo[comm->rank].gdrSupport), res, fail);
      NCCLCHECKGOTO(xmlSetAttrInt(node, "net_affinity", -1), res, fail);
    }
  }

  // Auto-detect NICs if needed. net/collnet share the same xml/graph nodes,
  // so we start with collnet so that it has precedence.
  if (ncclCollNet) {
    NCCLCHECKGOTO(collNetDevices(&netDevCount), res, fail);
    for (int n=0; n<netDevCount; n++) {
      ncclNetProperties_t props;
      NCCLCHECKGOTO(collNetGetProperties(n, &props), res, fail);
      struct ncclXmlNode* netNode;
      NCCLCHECKGOTO(ncclTopoFillNet(xml, props.pciPath, props.name, &netNode), res, fail);
      NCCLCHECKGOTO(xmlSetAttrInt(netNode, "keep", 1), res, fail);
      NCCLCHECKGOTO(xmlSetAttrInt(netNode, "dev", n), res, fail);
      NCCLCHECKGOTO(xmlInitAttrInt(netNode, "speed", props.speed), res, fail);
      NCCLCHECKGOTO(xmlInitAttrInt(netNode, "port", props.port), res, fail);
      NCCLCHECKGOTO(xmlInitAttrUint64(netNode, "guid", props.guid), res, fail);
      NCCLCHECKGOTO(xmlInitAttrInt(netNode, "maxconn", props.maxComms), res, fail);
      NCCLCHECKGOTO(xmlInitAttrInt(netNode, "gdr", props.ptrSupport & NCCL_PTR_HGGC ? 1 : 0), res, fail);
      NCCLCHECKGOTO(xmlInitAttrInt(netNode, "coll", 1), res, fail);
    }
  }
  if (netDevCount == 0) {
    NCCLCHECKGOTO(ncclNetDevices(&netDevCount), res, fail);
  }
  for (int n=0; n<netDevCount; n++) {
    ncclNetProperties_t props;
    NCCLCHECKGOTO(ncclNetGetProperties(n, &props), res, fail);
    struct ncclXmlNode* netNode;
    NCCLCHECKGOTO(ncclTopoFillNet(xml, props.pciPath, props.name, &netNode), res, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(netNode, "keep", 1), res, fail);
    NCCLCHECKGOTO(xmlSetAttrInt(netNode, "dev", n), res, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(netNode, "speed", props.speed), res, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(netNode, "port", props.port), res, fail);
    NCCLCHECKGOTO(xmlInitAttrUint64(netNode, "guid", props.guid), res, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(netNode, "maxconn", props.maxComms), res, fail);
    NCCLCHECKGOTO(xmlInitAttrInt(netNode, "gdr", props.ptrSupport & NCCL_PTR_HGGC ? 1 : 0), res, fail);
  }

  if (comm->icnRmtType != PCCL_ICNLINK_SWITCH) {
    for (int r=0; r<comm->nRanks; r++) {
      if (comm->peerInfo[r].hostHash == comm->peerInfo[comm->rank].hostHash && ppuFilledCount < ppuCount) {
        ppuFilledCount++;
      }
    }
    if (ppuFilledCount == pcclHgmlDeviceCount) {
      enablePxbToPxn = true;
    }
  }

  // hgml based group affinity settings before trim
  if (pcclHgmlEnsureInitialized() == ncclSuccess && pcclParamNetAffinityAuto()) {
    NCCLCHECKGOTO(pcclTopoSetNetAffinityPPUs(xml, ppuCount, netDevCount), res, fail);
  }

  // Remove XML branches which don't have a node with keep="1" (typically when importing a topology)
  NCCLCHECKGOTO(ncclTopoTrimXml(xml), res, fail);

  // XML topo fusion.
  if (comm->MNNVL) {
    // MNNVL clique support
    nLocalRanks = comm->clique.size;
    localRank = comm->cliqueRank;
    localRanks = comm->clique.ranks;
  } else {
    // Intra-node fusion.  Much of the comm is not initialized yet at this point so we need to do our own calculations.
    NCCLCHECKGOTO(ncclCalloc(&localRanks, comm->nRanks), res, fail);
    for (int i = 0; i < comm->nRanks; i++) {
      if (comm->peerInfo[i].hostHash == comm->peerInfo[comm->rank].hostHash) {
        if (i == comm->rank)
          localRank = nLocalRanks;
        localRanks[nLocalRanks++] = i;
      }
    }
  }
  NCCLCHECKGOTO(ncclCalloc(&mem, nLocalRanks * xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), res, fail);
  rankXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*localRank);
  memcpy(rankXml, xml, xmlMemSize(NCCL_TOPO_XML_MAX_NODES));
  NCCLCHECKGOTO(ncclTopoConvertXml(rankXml, (uintptr_t)xml->nodes, 1), res, fail);
  // nLocalRanks can't actually be 0, or we wouldn't be running at all...
  // coverity[divide_by_zero]
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, localRanks, localRank, nLocalRanks, mem, xmlMemSize(NCCL_TOPO_XML_MAX_NODES)), res, fail);
  if (comm->MNNVL) {
    // Ensure that we have enough room when fusing topos from multiple nodes.
    free(xml);
    xml = NULL;
    NCCLCHECKGOTO(xmlAlloc(&xml, nLocalRanks*NCCL_TOPO_XML_MAX_NODES), res, fail);
  } else {
    // In the intra-node case there's no need to enlarge the topo xml.
    xml->maxIndex = 0;
  }
  for (int i = 0; i < nLocalRanks; i++) {
    struct ncclXml* peerXml = (struct ncclXml*)(mem+xmlMemSize(NCCL_TOPO_XML_MAX_NODES)*i);
    NCCLCHECKGOTO(ncclTopoConvertXml(peerXml, (uintptr_t)peerXml->nodes, 0), res, fail);
    NCCLCHECKGOTO(ncclTopoFuseXml(xml, peerXml), res, fail);
  }

  xmlTopoFile = pcclGetEnvWithDualPrefix("PCCL_TOPO_DUMP_FILE");
  if (xmlTopoFile && comm->rank == ncclParamTopoDumpFileRank()) {
    char xmlTopoFilePerRank[1024];
    const char* dot = strrchr(xmlTopoFile,'.');
    int prefixLen = strlen(xmlTopoFile);
    if (dot) {
      prefixLen = dot - xmlTopoFile;
    }
    strncpy(xmlTopoFilePerRank, xmlTopoFile, prefixLen);
    sprintf(xmlTopoFilePerRank+prefixLen, "_rank%04d.xml", comm->rank);
    INFO(NCCL_ENV, "PCCL_TOPO_DUMP_FILE set by environment to %s", xmlTopoFilePerRank);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(xmlTopoFilePerRank, xml), res, fail);
  }

  NCCLCHECKGOTO(ncclTopoGetSystemFromXml(xml, system, getHostHash()), res, fail);
  (*system)->enablePxbToPxn = enablePxbToPxn;
  if (comm->intraRank == 0) {
    NCCLCHECKGOTO(pcclCheckPPUTopo(), res, fail);
  }

exit:
  if (!comm->MNNVL) NCCLCHECK(ncclFree(localRanks));
  NCCLCHECK(ncclFree(mem));
  NCCLCHECK(ncclFree(xml));
  return res;
fail:
  goto exit;
}

static ncclResult_t ncclGetNetsFromSystem(ncclTopoSystem* system, int g, int64_t* nets, int* count, int* minTypeReturn=NULL) {
  *count = 0;
  int minType = PATH_SYS;
  float maxWidth = 0;
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoLinkList* path = system->nodes[NET].nodes[n].paths[GPU]+g;
    if (path->width > maxWidth || (path->width == maxWidth && path->type < minType)) {
      maxWidth = path->width;
      minType = path->type;
      *count = 0;
    }
    if (path->width == maxWidth && path->type == minType) nets[(*count)++] = system->nodes[NET].nodes[n].id;
  }
  if (minTypeReturn != NULL) *minTypeReturn = minType;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalNet(struct ncclTopoSystem* system, int rank, int64_t* id, int* dev) {
  ncclResult_t res = ncclSuccess;
  int netDev;
  int64_t netId;
  int count = 0;
  int64_t* nets = NULL;
  int rr = 0;
  NCCLCHECK(ncclCalloc(&nets, system->nodes[NET].count));
  int g = -1;
  NCCLCHECKGOTO(ncclTopoRankToIndex(system, rank, &g), res, end);
  NCCLCHECKGOTO(ncclGetNetsFromSystem(system, g, nets, &count), res, end);
  if (count == 0) {
    if (id) *id = -1;
    if (dev) *dev = -1;
    free(nets);
    return ncclSuccess;
  }

  if (NCCL_TOPO_ID_LOCAL_ID(system->nodes[GPU].nodes[g].gpu.netAffinity) != NCCL_TOPO_ID_LOCAL_ID_MASK) {
    netId = system->nodes[GPU].nodes[g].gpu.netAffinity;
    NCCLCHECKGOTO(ncclTopoIdToNetDev(system, netId, &netDev), res, end);
  } else {
    rr = system->nodes[GPU].nodes[g].gpu.dev;
    netId = nets[rr % count];
    NCCLCHECKGOTO(ncclTopoIdToNetDev(system, netId, &netDev), res, end);
  }
  if (id) *id = netId;
  if (dev) *dev = netDev;

end:
  NCCLCHECK(ncclFree(nets));
  return res;
}

static ncclResult_t ncclTopoGetLocalGinDev(struct ncclTopoSystem* system, int rank, int channelId, int64_t* id, int* dev) {
  int gpu;
  NCCLCHECK(ncclTopoRankToIndex(system, rank, &gpu));

  int64_t affinityId = system->nodes[GPU].nodes[gpu].gpu.netAffinity;
  if (NCCL_TOPO_ID_LOCAL_ID(affinityId) != NCCL_TOPO_ID_LOCAL_ID_MASK) {
    int netDev;
    NCCLCHECK(ncclTopoIdToNetDev(system, affinityId, &netDev));
    if (id) *id = affinityId;
    if (dev) *dev = netDev;
    return ncclSuccess;
  }

  int64_t localNets[NCCL_TOPO_MAX_NODES];
  int localNetCount = 0;
  NCCLCHECK(ncclGetNetsFromSystem(system, gpu, localNets, &localNetCount));
  if (localNetCount == 0) {
    WARN("Could not find any local path from gpu %d to net.", gpu);
    return ncclInternalError;
  }

  int localGpuCount = 0;
  int64_t firstNetId = localNets[0];
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    int64_t gpuNets[NCCL_TOPO_MAX_NODES];
    int gpuNetCount = 0;
    NCCLCHECK(ncclGetNetsFromSystem(system, g, gpuNets, &gpuNetCount));
    for (int n = 0; n < gpuNetCount; n++) {
      if (gpuNets[n] == firstNetId) { localGpuCount++; break; }
    }
  }
  if (localGpuCount == 0) localGpuCount = 1;
  int netsPerGpu = (localNetCount + localGpuCount - 1) / localGpuCount;

  int net = system->nodes[GPU].nodes[gpu].gpu.dev;
  if (isPow2(localNetCount)) net = mirrorBits(net, localNetCount);
  net += channelId % netsPerGpu;
  int64_t netId = localNets[net % localNetCount];

  int netDev;
  NCCLCHECK(ncclTopoIdToNetDev(system, netId, &netDev));
  if (id) *id = netId;
  if (dev) *dev = netDev;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalGinDevs(ncclComm_t comm, int* localGinDevs, int* localGinCount) {
  for (int c = 0; c < NCCL_TOPO_MAX_NODES; c++) {
    NCCLCHECK(ncclTopoGetLocalGinDev(comm->topo, comm->rank, c, NULL, localGinDevs + c));
    if (c > 0 && localGinDevs[c] == localGinDevs[0]) {
      *localGinCount = c;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

int pcclGetPathTypeAlias(int type) {
  if (type == PATH_PHBH)
    return PATH_PIX;
  else
    return type;
}

static ncclResult_t ncclGetProxyRanksForNetFromSystem(ncclTopoSystem* system, int gpu, int n, int* ranks, int* count, int* minTypeReturn=NULL) {
  *count = 0;
  int proxyRanks = 0;
  int minType = PATH_SYS;
  float maxWidth = 0;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoLinkList* path = system->nodes[NET].nodes[n].paths[GPU]+g;
    if (path->width > maxWidth || (path->width == maxWidth && pcclGetPathTypeAlias(path->type) < pcclGetPathTypeAlias(minType))) {
      maxWidth = path->width;
      minType = pcclGetPathTypeAlias(path->type);
      proxyRanks = 0;
    }
    if (path->width == maxWidth && pcclGetPathTypeAlias(path->type) == pcclGetPathTypeAlias(minType)) ranks[proxyRanks++] = g;
  }
  if (minTypeReturn != NULL) *minTypeReturn = minType;

  int hops = INT_MAX;
  for (int i = 0; i < proxyRanks; i++) {
    int g = ranks[i];
    struct ncclTopoLinkList* path = system->nodes[GPU].nodes[gpu].paths[GPU]+g;
    if (path->count < hops && path->type <= PATH_NVB) {
      hops = path->count;
      *count = 0;
    }
    if (path->count == hops) ranks[(*count)++] = system->nodes[GPU].nodes[g].gpu.rank;
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoGetBalancedProxyRank(struct ncclTopoSystem* system, int g, int n, int* proxyRank) {
  ncclResult_t res = ncclSuccess;
  int count = 0;
  int* proxyRanks = NULL;
  int rr = system->nodes[GPU].nodes[g].gpu.dev;
  NCCLCHECK(ncclCalloc(&proxyRanks, system->nodes[GPU].count));
  NCCLCHECKGOTO(ncclGetProxyRanksForNetFromSystem(system, g, n, proxyRanks, &count), res, end);

  *proxyRank = proxyRanks[rr % count];

end:
  NCCLCHECK(ncclFree(proxyRanks));
  return res;
}

ncclResult_t ncclTopoAdjustSystem(ncclTopoSystem* system) {
  NCCLCHECK(pcclAffinityNetsPathAdjust(system));
  return ncclSuccess;
}

/****************************/
/* External query functions */
/****************************/

ncclResult_t ncclTopoCpuType(struct ncclTopoSystem* system, int* arch, int* vendor, int* model) {
  *arch = system->nodes[CPU].nodes[0].cpu.arch;
  *vendor = system->nodes[CPU].nodes[0].cpu.vendor;
  *model = system->nodes[CPU].nodes[0].cpu.model;
  return ncclSuccess;
}

NCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, cpu_set_t* affinity) {
  struct ncclTopoNode* cpu = NULL, *gpu = NULL;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      gpu = system->nodes[GPU].nodes+g;
      // Find closer CPU
      int cpuIndex = -1, minHops = 0;
      for (int c=0; c<system->nodes[CPU].count; c++) {
        int nHops = system->nodes[GPU].nodes[g].paths[CPU][c].count;
        if (cpuIndex == -1 || nHops < minHops) {
          cpuIndex = c;
          minHops = nHops;
        }
      }
      cpu = system->nodes[CPU].nodes+cpuIndex;
    }
  }
  if (cpu == NULL) {
    ERROR("Set CPU affinity : unable to find PPU/CPU for rank %d", rank);
    return ncclInternalError;
  }

  // Query the CPU affinity set we were provided
  cpu_set_t mask;
  SYSCHECK(sched_getaffinity(0, sizeof(cpu_set_t), &mask), "sched_getaffinity");

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&mask, affinityStr));
    TRACE(NCCL_INIT, "Current affinity for GPU %d is %s", gpu->gpu.dev, affinityStr);
  }
#endif

  // Get the affinity of the CPU close to our GPU.
  cpu_set_t cpuMask = cpu->cpu.affinity;

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&cpuMask, affinityStr));
    TRACE(NCCL_INIT, "CPU GPU affinity for GPU %d is %s", gpu->gpu.dev, affinityStr);
  }
#endif

  cpu_set_t finalMask;
  if (ncclParamIgnoreCpuAffinity())
    // Ignore the CPU affinity set and use the GPU one instead
    finalMask = cpuMask;
  else
    // Use a subset of the GPU affinity set
    CPU_AND(&finalMask, &mask, &cpuMask);

  memcpy(affinity, &finalMask, sizeof(cpu_set_t));

  // If there is a non empty set, use it to set affinity
  if (CPU_COUNT(&finalMask)) {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&finalMask, affinityStr));
    INFO(NCCL_INIT, "Setting affinity for GPU %d to %s", gpu->gpu.dev, affinityStr);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[NET].count;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetLocalRank(ncclComm_t comm, int rank, int* localRank) {
  *localRank = comm->rankToLocalRank[rank];
  return ncclSuccess;
}

ncclResult_t ncclTopoDevToRank(struct ncclTopoSystem* system, int systemId, int hgmlDev, int *rank) {
  *rank = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+i;
    if (NCCL_TOPO_ID_SYSTEM_ID(gpu->id) != systemId) continue;
    if (gpu->gpu.dev == hgmlDev) {
      *rank = gpu->gpu.rank;
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

ncclResult_t ncclTopoIdToNetDev(struct ncclTopoSystem* system, int64_t id, int* netDev) {
  *netDev = -1;
  for (int i=0; i<system->nodes[NET].count; i++) {
    if (system->nodes[NET].nodes[i].id == id) {
      *netDev = system->nodes[NET].nodes[i].net.dev;
      return ncclSuccess;
    }
  }
  ERROR("Could not find NET with id %lx", id);
  return ncclInternalError;
}

struct pcclHgmlDevInfo{
  hgmlDevice_t hgmlDev;
  hgmlPciInfo_t pciInfo;
};

ncclResult_t findPPUId(pcclHgmlDevInfo *hgmlDevInfoList, int ppuCount, int dev, pcclHgmlDevInfo *curHgmlDevInfo) {
  for (int i = 0; i < ppuCount; i++) {
    if (hgmlDevInfoList[i].pciInfo.ppuInfo.ppuId == dev) {
      if (curHgmlDevInfo) *curHgmlDevInfo = hgmlDevInfoList[i];
      return ncclSuccess;
    }
  }
  if (curHgmlDevInfo) {
    ERROR("Can't find current ppuId=%d in hgml device list ?", dev);
  } else {
    ERROR("Can't find remote ppuId=%d in hgml device list ?", dev);
  }
  return ncclInternalError;
}

// Topology information loaded from XML configuration, influenced by CUDA_VISIBLE_DEVICES setting
ncclResult_t pcclFillPPUTopoArray(int *ppuTopoArray, ncclComm_t comm) {
  int nRanks = comm->localRanks;
  struct ncclTopoNodeSet* gpus = &comm->topo->nodes[GPU];
  bool hasNvs = comm->topo->nodes[NVS].count > 0;

  for (int i = 0; i < nRanks; i++) {
    struct ncclTopoNode* gpu = &gpus->nodes[i];
    int srcRank = gpu->gpu.rank;
    for (int j = 0; j < nRanks; j++) {
      if (i == j) continue;
      struct ncclTopoLinkList* path = &gpu->paths[GPU][j];
      if (path->type == PATH_NVL && (path->count == 1 || hasNvs)) {
        int dstRank = gpus->nodes[j].gpu.rank;
        ppuTopoArray[srcRank*nRanks + dstRank] = path->list[0]->count;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t pcclFillPPUTopoArray(int *ppuTopoArray, ppuDeviceInfo *ppuDevInfo) {
  int maxIcnLinks = (ppuDevInfo->sm < 89) ? MAX_ICNLINK_SUBS_PPU0010 : MAX_ICNLINK_SUBS_PPU0015;
  unsigned int tailId = 2; // hgml will return 0 or 1
  int ppuCount = ppuDevInfo->ppuCount;

  ncclResult_t res = ncclSuccess;

  pcclHgmlDevInfo *hgmlDevInfoList = NULL;
  NCCLCHECK(ncclCalloc(&hgmlDevInfoList, ppuCount));
  for (int i = 0; i < ppuCount; i++) {
    NCCLCHECKGOTO((pcclHgmlDeviceGetHandleByIndex(i, &hgmlDevInfoList[i].hgmlDev)), res, clear);
    NCCLCHECKGOTO((pcclHgmlDeviceGetPciInfo(hgmlDevInfoList[i].hgmlDev, &hgmlDevInfoList[i].pciInfo)), res, clear);

    if (ppuDevInfo->ppuType == _PPU_ZW810E && ppuCount != MAX_GPUS_PER_NODE) {
      hgmlFieldValue_t values;
      int valuesCount = 1;
      values.fieldId = HGML_FI_DEV_REAR_ID;
      NCCLCHECKGOTO((pcclHgmlDeviceGetFieldValues(hgmlDevInfoList[i].hgmlDev, valuesCount, &values)), res, clear);
      VERBOSE(NCCL_INIT, "ppuid: %d , tail is %d", hgmlDevInfoList[i].pciInfo.ppuInfo.ppuId, values.value.uiVal);
      if (tailId == 2) {
        tailId = values.value.uiVal;
      } else {
        if (values.value.uiVal != tailId) {
          INFO(NCCL_INIT, "ppus within current hgml device list are in differen tails, skip to do topo check");
          goto clear;
        }
      }
    }
  }

  for (int curDev = 0; curDev < ppuCount; curDev++) {
    pcclHgmlDevInfo curHgmlDevInfo;
    NCCLCHECKGOTO((findPPUId(hgmlDevInfoList, ppuCount, (tailId == 1) ? curDev + PER_OAM_TAIL_NODE_DEV_NUM : curDev, &curHgmlDevInfo)), res, clear);
    for (int curLink = 0; curLink < maxIcnLinks; curLink++) {
      hgmlEnableState_t isActive;
      NCCLCHECKGOTO((pcclHgmlDeviceGetIcnLinkState(curHgmlDevInfo.hgmlDev, curLink, &isActive)), res, clear);
      if (isActive != HGML_FEATURE_ENABLED) continue;

      hgmlPciInfo_t remDevPciInfo;
      NCCLCHECKGOTO((pcclHgmlDeviceGetIcnLinkRemotePciInfo(curHgmlDevInfo.hgmlDev, curLink, &remDevPciInfo)), res, clear);
      NCCLCHECKGOTO((findPPUId(hgmlDevInfoList, ppuCount, remDevPciInfo.ppuInfo.ppuId, NULL)), res, clear);
      ppuTopoArray[curDev * ppuCount + remDevPciInfo.ppuInfo.ppuId]++;
    }
  }
  NCCLCHECK(ncclFree(hgmlDevInfoList));
  return ncclSuccess;

clear:
  NCCLCHECK(ncclFree(hgmlDevInfoList));
  return ncclInternalError;
}

ncclResult_t ncclTopoGetHggcDevByHgmlDev(const int hgmlDev, int *hggcDev) {
  *hggcDev = hgmlDev;
  hgmlDevice_t device;
  hgmlPciInfo_st pci;
  NCCLCHECK(pcclHgmlDeviceGetHandleByIndex(hgmlDev, &device));
  NCCLCHECK(pcclHgmlDeviceGetPciInfo(device, &pci));
  HGGCCHECK(hggcDeviceGetByPCIBusId(hggcDev, pci.busId));
  return ncclSuccess;
}

ncclResult_t pcclTopoGetNetFromNIC(struct ncclXmlNode* nic, int *netId) {
  int id = -1;
  for (int s = 0; s < nic->nSubs; s++) {
    struct ncclXmlNode* xmlNet = nic->subs[s];
    // TODO: assumption one NetNode per NicNode
    if (strcmp(xmlNet->name, "net") != 0) continue;
    NCCLCHECK(xmlGetAttrInt(xmlNet, "dev", &id));
    if (id != -1) break;
  }
  *netId = id;
  return ncclSuccess;
}

ncclResult_t pcclTopoGetNetFromPci(struct ncclXmlNode* pci, int *netId) {
  for (int s = 0; s < pci->nSubs; s++) {
    struct ncclXmlNode* xmlNic = pci->subs[s];
    // TODO: assumption one NicNode per PciNode
    if (strcmp(xmlNic->name, "nic") == 0) {
      NCCLCHECK(pcclTopoGetNetFromNIC(xmlNic, netId));
      break;
    }
  }
  return ncclSuccess;
}

#define PCCL_NET_AFFINITY_MAX_DEPTH 8

static ncclResult_t pcclTopoFindNetInSubtree(struct ncclXmlNode* node, struct ncclXmlNode* exclude, int* netDev) {
  if (node == exclude) return ncclSuccess;

  const char* str = NULL;
  int type = -1;
  NCCLCHECK(xmlGetAttr(node, "class", &str));
  if (str) {
    NCCLCHECK(kvConvertToInt(str, &type, kvDictPciClass));
    if (type == NIC) {
      NCCLCHECK(pcclTopoGetNetFromPci(node, netDev));
      if (*netDev != -1) return ncclSuccess;
    }
  }

  for (int s = 0; s < node->nSubs; s++) {
    if (node->subs[s] == exclude) continue;
    NCCLCHECK(pcclTopoFindNetInSubtree(node->subs[s], NULL, netDev));
    if (*netDev != -1) return ncclSuccess;
  }
  return ncclSuccess;
}

static ncclResult_t pcclTopoFindNearestNetForGpu(struct ncclXmlNode* gpuNode, int* netDev) {
  *netDev = -1;
  struct ncclXmlNode* child = gpuNode;
  struct ncclXmlNode* current = gpuNode->parent;

  for (int depth = 0; depth < PCCL_NET_AFFINITY_MAX_DEPTH && current != NULL; depth++) {
    if (strcmp(current->name, "cpu") == 0) break;
    NCCLCHECK(pcclTopoFindNetInSubtree(current, child, netDev));
    if (*netDev != -1) return ncclSuccess;

    child = current;
    current = current->parent;
  }
  return ncclSuccess;
}

ncclResult_t pcclTopoSetNetAffinityPPUs(struct ncclXml* xml, int ppuCount, int netCount) {
  ncclResult_t res = ncclSuccess;
  int *isAffinityNet = nullptr;
  int affinityNetNum = 0;
  int ppuNumPerNet = 0;

  const char *env = pcclGetEnvWithDualPrefix("PCCL_NET_AFFINITY");
  char *env_copy = NULL;
  if (env) {
    env_copy = strdup(env);
    char *token = strtok(env_copy, ",");
    while(token) {
      char *colon = strchr(token, ':');
      int ppuId = atoi(token);
      int netDev = atoi(colon + 1);
      if ((ppuCount > 0 && ppuId >= 0 && ppuId < ppuCount) &&
          (netCount > 0 && netDev >= 0 && netDev < netCount)) {
        struct ncclXmlNode* gpuNode;
        NCCLVERIFY(xmlFindTagKvInt(xml, "gpu", &gpuNode, "dev", ppuId));
        if (gpuNode) {
          NCCLCHECKGOTO(xmlSetAttrInt(gpuNode, "net_affinity", netDev), res, out);
          INFO(NCCL_INIT, "Affinity maping PPU %d: NET %d", ppuId, netDev);
        }
      } else {
        WARN("unsupported net affinity setting: ppu (%d/%d), net (%d/%d)",
        ppuId, ppuCount, netDev, netCount);
      }
      token = strtok(NULL, ",");
    }
    goto out;
  }

  if (ppuCount <= 0 || netCount <= 0 || (ppuCount != 8 && ppuCount != 16)) {
    INFO(NCCL_GRAPH, "Skip setting net affinity for %d PPUs, %d NETs ", ppuCount, netCount);
    goto out;
  }

  NCCLCHECKGOTO(ncclCalloc(&isAffinityNet, netCount), res, out);

  // find nearest net for each GPU via BFS up the tree (handles PIX, PHB, and SYS)
  for (int i = 0; i < xml->maxIndex; i++) {
    ncclXmlNode *node = &xml->nodes[i];
    if (strcmp(node->name, "gpu") != 0) continue;

    int dev = -1;
    NCCLVERIFY(xmlGetAttrInt(node, "dev", &dev));
    if (dev == -1) continue;

    int netDev = -1;
    NCCLVERIFY(pcclTopoFindNearestNetForGpu(node, &netDev));
    if (netDev != -1 && netDev < netCount) {
      isAffinityNet[netDev] = 1;
      NCCLVERIFY(xmlSetAttrInt(node, "net_affinity", netDev));
      INFO(NCCL_INIT, "Detect affinity for PPU %d: NET %d", dev, netDev);
    }
  }

  // count gpu affinity net count
  for (int i = 0; i < netCount; i++) {
    if (isAffinityNet[i] != 0) affinityNetNum++;
  }
  if (affinityNetNum == 0 || affinityNetNum > ppuCount) {
    WARN("Invalid affinity net count %d for PPUs !", affinityNetNum);
    goto out;
  } else {
    ppuNumPerNet = ppuCount / affinityNetNum;
  }

  // adjust for special system
  for (int i = 0; i < xml->maxIndex; i++) {
    ncclXmlNode *node = &xml->nodes[i];
    if (strcmp(node->name, "gpu") == 0) {
      int dev = -1;
      int net_affinity = -1;
      NCCLVERIFY(xmlGetAttrInt(node, "dev", &dev, true));
      NCCLVERIFY(xmlGetAttrInt(node, "net_affinity", &net_affinity, true));
      if (/* can't find dev attr of node */
          dev == -1
          /* net affinity is already known */
          || net_affinity != -1) continue;
      for (int j = 0; j < ppuNumPerNet; j++) {
        int neighbor = dev / ppuNumPerNet * ppuNumPerNet + j;
        if (neighbor != dev) {
          struct ncclXmlNode* neighborNode = NULL;
          NCCLVERIFY(xmlFindTagKvInt(xml, "gpu", &neighborNode, "dev", neighbor));
          if (!neighborNode) continue;
          int neighborNetAffinity = -1;
          NCCLVERIFY(xmlGetAttrInt(neighborNode, "net_affinity", &neighborNetAffinity, true));
          if (neighborNetAffinity != -1) {
            NCCLVERIFY(xmlSetAttrInt(node, "net_affinity", neighborNetAffinity));
            INFO(NCCL_INIT, "Affinity maping PPU %d: NET %d", dev, neighborNetAffinity);
            break;
          }
        }
      }
    }
  }

out:
  NCCLCHECK(ncclFree(env_copy));
  NCCLCHECK(ncclFree(isAffinityNet));
  return res;
}

ncclResult_t pcclTopoGetGpuPerNetBw(ncclComm_t comm, float* gpuPerNetBw) {
  if (comm->topo == NULL || comm->localRanks <= 0) {
    return ncclInternalError;
  }
  *gpuPerNetBw = comm->topo->totalNetWidth / comm->localRanks;
  return ncclSuccess;
}
