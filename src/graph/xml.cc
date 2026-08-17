/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <mutex>
#include "core.h"
#include "param.h"
#include "nvmlwrap.h"
#include "xml.h"
#include "check_items.h"

/*******************/
/* XML File Parser */
/*******************/

ncclResult_t xmlGetChar(FILE* file, char* c) {
  if (fread(c, 1, 1, file) == 0) {
    ERROR("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlGetValue(FILE* file, char* value, char* last) {
  char c;
  NCCLCHECK(xmlGetChar(file, &c));
  if (c != '"' && c != '\'') {
#if INT_OK
    int o = 0;
    do {
      value[o++] = c;
      NCCLCHECK(xmlGetChar(file, &c));
    } while (c >= '0' && c <= '9');
    value[o] = '\0';
    *last = c;
    return ncclSuccess;
#else
    ERROR("XML Parse : Expected (double) quote.");
    return ncclInternalError;
#endif
  }
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    value[o++] = c;
  } while (c != '"');
  value[o-1] = '\0';
  NCCLCHECK(xmlGetChar(file, last));
  return ncclSuccess;
}

ncclResult_t xmlGetToken(FILE* file, char* name, char* value, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  do {
    NCCLCHECK(xmlGetChar(file, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (value == NULL) {
        ERROR("XML Parse : Unexpected value with name %s", ptr);
        return ncclInternalError;
      }
      return xmlGetValue(file, value, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN-1) {
      ptr[o] = '\0';
      ERROR("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o-1] = '\0';
  *last = c;
  return ncclSuccess;
}

// Shift the 3-chars string by one char and append c at the end
#define SHIFT_APPEND(s, c) do { s[0]=s[1]; s[1]=s[2]; s[2]=c; } while(0)
ncclResult_t xmlSkipComment(FILE* file, char* start, char next) {
  // Start from something neutral with \0 at the end.
  char end[4] = "...";

  // Inject all trailing chars from previous reads. We don't need
  // to check for --> here because there cannot be a > in the name.
  for (int i=0; i<strlen(start); i++) SHIFT_APPEND(end, start[i]);
  SHIFT_APPEND(end, next);

  // Stop when we find "-->"
  while (strcmp(end, "-->") != 0) {
    int c;
    if (fread(&c, 1, 1, file) != 1) {
      ERROR("XML Parse error : unterminated comment");
      return ncclInternalError;
    }
    SHIFT_APPEND(end, c);
  }
  return ncclSuccess;
}

ncclResult_t xmlGetNode(FILE* file, struct ncclXmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (fread(&c, 1, 1, file) == 0) return ncclSuccess;
  }
  if (c != '<') {
    ERROR("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));

  // Check for comments
  if (strncmp(node->name, "!--", 3) == 0) {
    NCCLCHECK(xmlSkipComment(file, node->name+3, c));
    return xmlGetNode(file, node);
  }

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(xmlGetToken(file, node->name, NULL, &c));
    if (c != '>') {
      ERROR("XML Parse error : unexpected trailing %c in closing tag %s", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(xmlGetToken(file, node->attrs[a].key, node->attrs[a].value, &c));
    if (a == MAX_ATTR_COUNT) {
      INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)\n", MAX_ATTR_COUNT);
      // Actually we need to still consume the extra attributes so we have an extra one.
    } else a++;
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(xmlGetToken(file, str, NULL, &c));
  }
  if (c != '>') {
    ERROR("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

typedef ncclResult_t (*xmlHandlerFunc_t)(FILE*, struct ncclXml*, struct ncclXmlNode*);

struct xmlHandler {
  const char * name;
  xmlHandlerFunc_t func;
};

ncclResult_t xmlLoadSub(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head, struct xmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (xml->maxIndex == MAX_NODES) {
      ERROR("Error : XML parser is limited to 1024 nodes");
      return ncclInternalError;
    }
    struct ncclXmlNode* node = xml->nodes+xml->maxIndex;
    memset(node, 0, sizeof(struct ncclXmlNode));
    NCCLCHECK(xmlGetNode(file, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        ERROR("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        ERROR("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h=0; h<nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) head->subs[head->nSubs++] = node;
        node->parent = head;
        node->nSubs = 0;
        xml->maxIndex++;
        NCCLCHECK(handlers[h].func(file, xml, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(xmlLoadSub(file, xml, node, NULL, 0));
    }
  }
}

/**************/
/* XML Writer */
/**************/

// exp == 1 -- serialize; exp == 0 -- deserialize
ncclResult_t ncclTopoConvertXml(struct ncclXml* xml, uintptr_t base, int exp) {
  for (int n = 0; n < xml->maxIndex; n++) {
    struct ncclXmlNode *node = &xml->nodes[n];

    // For "parent", we shift the base by 1 so that we can distinguish actual
    // NULL pointers from pointers pointing to the first node.
    if (node->parent)
      node->parent = (struct ncclXmlNode *) (exp ? ((uintptr_t)node->parent - base + 1) : (base - 1 + (uintptr_t)node->parent));

    for (int s = 0; s < node->nSubs; s++) {
      node->subs[s] = (struct ncclXmlNode *) (exp ? ((uintptr_t)node->subs[s] - base) : (base + (uintptr_t)node->subs[s]));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlRec(int indent, FILE* file, struct ncclXmlNode* node) {
  for (int i=0; i<indent; i++) fprintf(file, " ");
  fprintf(file, "<%s", node->name);

  for (int a=0; a<node->nAttrs; a++) {
    fprintf(file, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].value);
  }
  if (node->nSubs == 0) {
    fprintf(file, "/>\n");
  } else {
    fprintf(file, ">\n");
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoDumpXmlRec(indent+2, file, node->subs[s]));
    }
    for (int i=0; i<indent; i++) fprintf(file, " ");
    fprintf(file, "</%s>\n", node->name);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpXmlToFile(const char* xmlTopoFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlTopoFile, "w");
  if (file == NULL) {
    WARN("Unable to open %s, not dumping topology.", xmlTopoFile);
    return ncclSuccess;
  }
  NCCLCHECK(ncclTopoDumpXmlRec(0, file, xml->nodes));
  fclose(file);
  return ncclSuccess;
}

int pcclGetFieldIdFromLinkId(int link) {
  int fieldId = HGML_FI_DEV_ICNLINK_SPEED_MBPS_L0;
  switch (link) {
    case 0:
      fieldId = HGML_FI_DEV_ICNLINK_SPEED_MBPS_L0;
      break;
    case 1:
      fieldId = HGML_FI_DEV_ICNLINK_SPEED_MBPS_L1;
      break;
    case 2:
      fieldId = HGML_FI_DEV_ICNLINK_SPEED_MBPS_L2;
      break;
    case 3:
      fieldId = HGML_FI_DEV_ICNLINK_SPEED_MBPS_L3;
      break;
    case 4:
      fieldId = HGML_FI_DEV_ICNLINK_SPEED_MBPS_L4;
      break;
    case 5:
      fieldId = HGML_FI_DEV_ICNLINK_SPEED_MBPS_L5;
      break;
    case 6:
      fieldId = HGML_FI_DEV_ICNLINK_SPEED_MBPS_L6;
      break;
    case 7:
      fieldId = HGML_FI_DEV_ICNLINK_SPEED_MBPS_L7;
      break;
    default:
      WARN("Invalid link id: %d", link);
      break;
  }

  return fieldId;
}

static ncclResult_t xmlTopoFuseXmlRecursive(struct ncclXml* dst, struct ncclXmlNode* dstParent, struct ncclXmlNode* srcParent) {
  for (int i = 0; i < srcParent->nSubs; i++) {
    struct ncclXmlNode* srcNode = srcParent->subs[i];
    struct ncclXmlNode* dstNode;
    NCCLCHECK(xmlFindNode(dstParent, srcNode, &dstNode));
    if (dstNode == NULL) {
      NCCLCHECK(xmlAddTree(dst, dstParent, srcNode));
    } else {
      NCCLCHECK(xmlTopoFuseXmlRecursive(dst, dstNode, srcNode));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoFuseXml(struct ncclXml* dst, struct ncclXml* src) {
  struct ncclXmlNode* topNodeDst;
  NCCLCHECK(xmlFindTag(dst, "system", &topNodeDst));

  if (topNodeDst == NULL) {
    xmlAddTree(dst, NULL, src->nodes);
    return ncclSuccess;
  }

  struct ncclXmlNode* topNodeSrc;
  NCCLCHECK(xmlFindTag(src, "system", &topNodeSrc));

  NCCLCHECK(xmlTopoFuseXmlRecursive(dst, topNodeDst, topNodeSrc));

  return ncclSuccess;
}

/****************************************/
/* Parser rules for our specific format */
/****************************************/

ncclResult_t ncclTopoXmlLoadNvlink(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // Normalize "nvlink" XML element to "icnlink" for internal representation
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  snprintf(head->name, MAX_STR_LEN, "icnlink");
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  // Accept both "nvlink" and "icnlink" in XML, normalize to "icnlink" internally
  struct xmlHandler handlers[] = { { "nvlink", ncclTopoXmlLoadNvlink }, { "icnlink", ncclTopoXmlLoadNvlink } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "net", ncclTopoXmlLoadNet } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "gpu", ncclTopoXmlLoadGpu }, { "nic", ncclTopoXmlLoadNic} };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 3));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "nic", ncclTopoXmlLoadNic } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_TOPO_XML_VERSION) {
    ERROR("XML Topology has wrong version %d", version);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading topology %s", name);
  else INFO(NCCL_GRAPH, "Loading unnamed topology");

  struct xmlHandler handlers[] = { { "cpu", ncclTopoXmlLoadCpu } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlTopoFile, "r");
  if (file == NULL) {
    WARN("Could not open XML topology file %s : %s", xmlTopoFile, strerror(errno));
    return ncclSuccess;
  }
  INFO(NCCL_GRAPH, "Loading topology file %s", xmlTopoFile);
  struct xmlHandler handlers[] = { { "system", ncclTopoXmlLoadSystem } };
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}

/**********************/
/* XML creation       */
/* from autodetection */
/**********************/

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))
static void memcpylower(char* dst, const char* src, const size_t size) {
  for (int i=0; i<size; i++) dst[i] = tolower(src[i]);
}
static ncclResult_t getPciPath(const char* busId, char** path) {
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpylower(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpylower(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    ERROR("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetStrFromSys(const char* path, const char* fileName, char* strValue) {
  char filePath[PATH_MAX];
  sprintf(filePath, "%s/%s", path, fileName);
  int offset = 0;
  FILE* file;
  if ((file = fopen(filePath, "r")) != NULL) {
    while (feof(file) == 0 && ferror(file) == 0 && offset < MAX_STR_LEN) {
      int len = fread(strValue+offset, 1, MAX_STR_LEN-offset, file);
      offset += len;
    }
    fclose(file);
  }
  if (offset == 0) {
    strValue[0] = '\0';
    INFO(NCCL_GRAPH, "Topology detection : could not read %s, ignoring", filePath);
  } else {
    strValue[offset-1] = '\0';
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSetAttrFromSys(struct ncclXmlNode* pciNode, const char* path, const char* fileName, const char* attrName) {
  char strValue[MAX_STR_LEN];
  NCCLCHECK(ncclTopoGetStrFromSys(path, fileName, strValue));
  if (strValue[0] != '\0') { NCCLCHECK(xmlSetAttr(pciNode, attrName, strValue)); }
  TRACE(NCCL_GRAPH, "Read from sys %s/%s -> %s=%s", path, fileName, attrName, strValue);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromCpu(struct ncclXmlNode* cpuNode, struct ncclXml* xml) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    const char* numaId;
    NCCLCHECK(xmlGetAttr(cpuNode, "numaid", &numaId));
    if (numaId == NULL) {
      ERROR("GetXmlFromCpu : could not find CPU numa ID.");
      return ncclInternalError;
    }
    // Set affinity
    char cpumaskPath[] = "/sys/devices/system/node/node0000";
    sprintf(cpumaskPath, "/sys/devices/system/node/node%s", numaId);
    NCCLCHECK(ncclTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity"));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
  if (index == -1) {
    // Fill CPU type / vendor / model
#if defined(__PPC__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__)
    NCCLCHECK(xmlSetAttr(cpuNode, "arch", "x86_64"));
#endif
  }

#if defined(__x86_64__)
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
    union {
      struct {
        // CPUID 0 String register order
        uint32_t ebx;
        uint32_t edx;
        uint32_t ecx;
      };
      char vendor[12];
    } cpuid0;

    asm volatile("cpuid" : "=b" (cpuid0.ebx), "=c" (cpuid0.ecx), "=d" (cpuid0.edx) : "a" (0) : "memory");
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    NCCLCHECK(xmlSetAttr(cpuNode, "vendor", vendor));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
    union {
      struct {
        unsigned steppingId:4;
        unsigned modelId:4;
        unsigned familyId:4;
        unsigned processorType:2;
        unsigned resv0:2;
        unsigned extModelId:4;
        unsigned extFamilyId:8;
        unsigned resv1:4;
      };
      uint32_t val;
    } cpuid1;
    asm volatile("cpuid" : "=a" (cpuid1.val) : "a" (1) : "memory");
    int familyId = cpuid1.familyId + (cpuid1.extFamilyId << 4);
    int modelId = cpuid1.modelId + (cpuid1.extModelId << 4);
    NCCLCHECK(xmlSetAttrInt(cpuNode, "familyid", familyId));
    NCCLCHECK(xmlSetAttrInt(cpuNode, "modelid", modelId));
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclTopoGetPciNode(struct ncclXml* xml, const char* busId, struct ncclXmlNode** pciNode) {
  NCCLCHECK(xmlFindTagKv(xml, "pci", pciNode, "busid", busId));
  if (*pciNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, NULL, "pci", pciNode));
  }
  NCCLCHECK(xmlSetAttr(*pciNode, "busid", busId));
  return ncclSuccess;
}

// Check whether a string is in BDF format or not.
// BDF (Bus-Device-Function) is "BBBB:BB:DD.F" where B, D and F are hex digits.
// There can be trailing chars.
int isHex(char c) { return ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')); }
int checkBDFFormat(char* bdf) {
  if (bdf[4] != ':' || bdf[7] != ':' || bdf[10] != '.') return 0;
  if (isHex(bdf[0]) == 0 || isHex(bdf[1] == 0) || isHex(bdf[2] == 0) || isHex(bdf[3] == 0) ||
      isHex(bdf[5] == 0) || isHex(bdf[6] == 0) || isHex(bdf[8] == 0) || isHex(bdf[9] == 0) ||
      isHex(bdf[11] == 0)) return 0;
  return 1;
}

ncclResult_t ncclTopoGetXmlFromSys(struct ncclXmlNode* pciNode, struct ncclXml* xml) {
  // Fill info, then parent
  const char* busId;
  NCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));
  char* path = NULL;
  ncclDebugNoWarn = NCCL_GRAPH;
  getPciPath(busId, &path);
  ncclDebugNoWarn = 0;

  if (path) {
    NCCLCHECK(ncclTopoSetAttrFromSys(pciNode, path, "class", "class"));
  }
  int index;
  ncclDebugNoWarn = NCCL_GRAPH;
  NCCLCHECK(xmlGetAttrIndex(pciNode, "vendor", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "vendor", "vendor");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "device", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "device", "device");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_vendor", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_vendor", "subsystem_vendor");
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "subsystem_device", &index));
  if (index == -1) {
    if (path) ncclTopoSetAttrFromSys(pciNode, path, "subsystem_device", "subsystem_device");
  }
  ncclDebugNoWarn = 0;
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_speed", &index));
  if (index == -1) {
    char deviceSpeedStr[MAX_STR_LEN];
    float deviceSpeed;
    NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr));  // ppu max_link_speed is "Unknown Speed" for now
    sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
    char portSpeedStr[MAX_STR_LEN];
    float portSpeed;
    NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr));
    sscanf(portSpeedStr, "%f GT/s", &portSpeed);
    NCCLCHECK(xmlSetAttr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr));
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_width", &index));
  if (index == -1) {
    if (path == NULL) NCCLCHECK(getPciPath(busId, &path));
    char strValue[MAX_STR_LEN];
    NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_width", strValue));
    int deviceWidth = strtol(strValue, NULL, 0);
    NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_width", strValue));
    int portWidth = strtol(strValue, NULL, 0);
    NCCLCHECK(xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth,portWidth)));
  }
  struct ncclXmlNode* parent = pciNode->parent;
  if (parent == NULL) {
    if (path == NULL) NCCLCHECK(getPciPath(busId, &path));

    // Save that for later in case next step is a CPU
    char numaIdStr[MAX_STR_LEN];
    NCCLCHECK(ncclTopoGetStrFromSys(path, "numa_node", numaIdStr));

    // Go up one level in the PCI tree. Rewind two "/" and follow the upper PCI
    // switch, or stop if we reach a CPU root complex.
    int slashCount = 0;
    int parentOffset;
    for (parentOffset = strlen(path)-1; parentOffset>0; parentOffset--) {
      if (path[parentOffset] == '/') {
        slashCount++;
        path[parentOffset] = '\0';
        int start = parentOffset - 1;
        while (start>0 && path[start] != '/') start--;
        // Check whether the parent path looks like "BBBB:BB:DD.F" or not.
        if (checkBDFFormat(path+start+1) == 0) {
          // This a CPU root complex. Create a CPU tag and stop there.
          struct ncclXmlNode* topNode;
          NCCLCHECK(xmlFindTag(xml, "system", &topNode));
          NCCLCHECK(xmlGetSubKv(topNode, "cpu", &parent, "numaid", numaIdStr));
          if (parent == NULL) {
            NCCLCHECK(xmlAddNode(xml, topNode, "cpu", &parent));
            NCCLCHECK(xmlSetAttrLong(parent, "host_hash", getHostHash()));
            NCCLCHECK(xmlSetAttr(parent, "numaid", numaIdStr));
          }
        } else if (slashCount == 2) {
          // Continue on the upper PCI switch
          for (int i = strlen(path)-1; i>0; i--) {
            if (path[i] == '/') {
              NCCLCHECK(xmlFindTagKv(xml, "pci", &parent, "busid", path+i+1));
              if (parent == NULL) {
                NCCLCHECK(xmlAddNode(xml, NULL, "pci", &parent));
                NCCLCHECK(xmlSetAttr(parent, "busid", path+i+1));
              }
              break;
            }
          }
        }
      }
      if (parent) break;
    }
    pciNode->parent = parent;
    if (parent) {
      parent->subs[parent->nSubs++] = pciNode;
    }
  }
  if (strcmp(parent->name, "pci") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
  } else if (strcmp(parent->name, "cpu") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromCpu(parent, xml));
  }
  NCCLCHECK(ncclFree(path));
  return ncclSuccess;
}

int ppuDiesNum;
PCCL_PARAM(Ndies, "NDIES", 0);
static std::once_flag pcclParamNdiesInitOnceFlag;
ncclResult_t ncclTopoGetXmlFromGpu(struct ncclXmlNode* pciNode, hgmlDevice_t hgmlDev, struct ncclXml* xml, struct ncclXmlNode** gpuNodeRet) {
  int topoVersion = -1;
  NCCLCHECK(xmlGetVersion(xml, &topoVersion));
  struct ncclXmlNode* gpuNode = NULL;
  NCCLCHECK(xmlGetSub(pciNode, "gpu", &gpuNode));
  if (gpuNode == NULL) NCCLCHECK(xmlAddNode(xml, pciNode, "gpu", &gpuNode));

  int index = -1;

  int dev = -1;
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "dev", &index));
  if (index == -1) {
    if (hgmlDev == NULL) {
      const char* busId;
      NCCLCHECK(xmlGetAttr(pciNode, "busid", &busId));
      if (busId == NULL || hggcDeviceGetByPCIBusId(&dev, busId) != hggcSuccess) dev = -1;
    } else {
      NCCLCHECK(pcclHgmlDeviceGetIndex(hgmlDev, (unsigned int*)&dev));
    }
    NCCLCHECK(xmlSetAttrInt(gpuNode, "dev", dev));
  }
  NCCLCHECK(xmlGetAttrInt(gpuNode, "dev", &dev));
  if (dev == -1) { *gpuNodeRet = NULL; return ncclSuccess; }

  NCCLCHECK(xmlGetAttrIndex(gpuNode, "sm", &index));
  if (index == -1) {
    int hggcMajor, hggcMinor;
    if (hgmlDev == NULL) {
      hggcDeviceProp devProp;
      HGGCCHECK(hggcGetDeviceProperties(&devProp, dev));
      hggcMajor = devProp.major; hggcMinor = devProp.minor;
    } else {
      NCCLCHECK(pcclHgmlDeviceGetHggcComputeCapability(hgmlDev, &hggcMajor, &hggcMinor));
    }
    NCCLCHECK(xmlSetAttrInt(gpuNode, "sm", hggcMajor*10+hggcMinor));
  }
  int sm;
  NCCLCHECK(xmlGetAttrInt(gpuNode, "sm", &sm));

  struct ncclXmlNode* nvlNode = NULL;
  NCCLCHECK(xmlGetSub(gpuNode, "icnlink", &nvlNode));

  int maxIcnLinks = (sm < 89) ? MAX_ICNLINK_SUBS_PPU0010 : MAX_ICNLINK_SUBS_PPU0015;
  int numDies = 0;
  hgmlIntIcnLinkDeviceType_t icnRmtType = HGML_ICNLINK_DEVICE_TYPE_UNKNOWN;
  if (nvlNode == NULL) {
    if (maxIcnLinks > 0 && hgmlDev == NULL) {
      WARN("No HGML device handle. Skipping nvlink detection.\n");
      maxIcnLinks = 0;
    }

    int maxActiveLinkIndex = 0;
    for (int l=0; l<maxIcnLinks; ++l) {
      // Check whether we can use this NVLink for P2P
      unsigned canP2P;
      if ((pcclHgmlDeviceGetIcnLinkCapability(hgmlDev, l, HGML_ICNLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

      // Make sure the Nvlink is up. The previous call should have trained the link.
      hgmlEnableState_t isActive;
      if ((pcclHgmlDeviceGetIcnLinkState(hgmlDev, l, &isActive) != ncclSuccess) || (isActive != HGML_FEATURE_ENABLED)) continue;
      maxActiveLinkIndex = l;

      if (icnRmtType == HGML_ICNLINK_DEVICE_TYPE_UNKNOWN) {
        pcclHgmlDeviceGetIcnLinkRemoteDeviceType(hgmlDev, l, &icnRmtType);
      }


      // Try to figure out what's on the other side of the NVLink
      hgmlPciInfo_t remoteProc;
      if (pcclHgmlDeviceGetIcnLinkRemotePciInfo(hgmlDev, l, &remoteProc) != ncclSuccess) continue;

      // Make a lower case copy of the bus ID for calling ncclDeviceType
      // PCI system path is in lower case
      char* p = remoteProc.busId;
      char lowerId[HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      for (int c=0; c<HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
        lowerId[c] = tolower(p[c]);
        if (p[c] == 0) break;
      }

      // query ICN bandwidth
      float bandWidth = -1.0f;
      hgmlFieldValue_t icnStat;
      icnStat.fieldId = pcclGetFieldIdFromLinkId(l);
      icnStat.scopeId = l;
      ncclResult_t ret = pcclHgmlDeviceGetFieldValues(hgmlDev, 1, &icnStat);
      if (ret == ncclSuccess && icnStat.hgmlReturn == HGML_SUCCESS) {
        bandWidth = icnStat.value.siVal / 1024.0f; // GB/s
        NCCLCHECK(pcclCheckICNLinkSpeed(dev, l, bandWidth));
      } else {
        WARN("Query link %d to PPU %s failed !", l, remoteProc.busId);
      }

      NCCLCHECK(xmlGetSubKv(gpuNode, "icnlink", &nvlNode, "target", lowerId));
      if (nvlNode == NULL) {
        NCCLCHECK(xmlAddNode(xml, gpuNode, "icnlink", &nvlNode));
        NCCLCHECK(xmlSetAttr(nvlNode, "target", lowerId));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", 1));
        NCCLCHECK(xmlSetAttrFloat(nvlNode, "bandwidth", bandWidth));
      } else {
        int count;
        NCCLCHECK(xmlGetAttrInt(nvlNode, "count", &count));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", count+1));

        float bw = .0f;
        NCCLCHECK(xmlGetAttrFloat(nvlNode, "bandwidth", &bw));
        bandWidth += bw;
        NCCLCHECK(xmlSetAttrFloat(nvlNode, "bandwidth", bandWidth));
      }
    }
    if (sm == 89) {
      numDies = (maxActiveLinkIndex >= (maxIcnLinks >> 1)) ? 2 : 1;
    }
  } else if (sm == 89) {
    int totalLinkCount = 0;
    for (int s=0; s<gpuNode->nSubs; s++) {
      if (strcmp(gpuNode->subs[s]->name, "icnlink") == 0) {
        int count = 0;
        NCCLCHECK(xmlGetAttrInt(gpuNode->subs[s], "count", &count));
        totalLinkCount += count;
      }
    }
    numDies = (totalLinkCount > (maxIcnLinks >> 1)) ? 2 : 1;
  }

  std::call_once(pcclParamNdiesInitOnceFlag, [&]() {
    ppuDiesNum = 1;
    if (sm == 89) {
      if (pcclParamNdies() == 0 || pcclParamNdies() > numDies) {
        ppuDiesNum = numDies;
      } else {
        ppuDiesNum = pcclParamNdies(); // respect env PCCL_NDIES
      }
      VERBOSE(NCCL_INIT, "Detected number of dies is %d and PCCL_NDIES=%ld", numDies, pcclParamNdies());
    }
  });

  // Fill target classes
  for (int s=0; s<gpuNode->nSubs; s++) {
    struct ncclXmlNode* sub = gpuNode->subs[s];
    if (strcmp(sub->name, "icnlink") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(sub, "tclass", &index));
    if (index == -1) {
      const char* busId;
      NCCLCHECK(xmlGetAttr(sub, "target", &busId));
      char* path;
      NOWARN(getPciPath(busId, &path), NCCL_GRAPH);
      if (icnRmtType == HGML_ICNLINK_DEVICE_TYPE_SWITCH || path == NULL || strcmp(busId, "fffffff:ffff:ff") == 0 || strcmp(busId, "0000:00:00.0") == 0) {
        // Remote NVLink device is not visible inside this VM. Assume NVSwitch.
        NCCLCHECK(xmlSetAttr(sub, "tclass", "0x068000"));
      } else {
        NCCLCHECK(ncclTopoSetAttrFromSys(sub, path, "class", "tclass"));
      }
      NCCLCHECK(ncclFree(path));
    }
  }
  *gpuNodeRet = gpuNode;
  return ncclSuccess;
}

PCCL_PARAM(IgnoreTopoDetection, "IGNORE_TOPO_DETECTION", 0);

ncclResult_t ncclTopoFillGpu(struct ncclXml* xml, const char* busId, struct ncclXmlNode** gpuNode) {
  struct ncclXmlNode* node;
  NCCLCHECK(ncclTopoGetPciNode(xml, busId, &node));
  NCCLCHECK(ncclTopoGetXmlFromSys(node, xml));
  hgmlDevice_t hgmlDev = NULL;
  if (!pcclParamIgnoreTopoDetection()) {
    if (pcclHgmlEnsureInitialized() != ncclSuccess || pcclHgmlDeviceGetHandleByPciBusId(busId, &hgmlDev) != ncclSuccess) hgmlDev = NULL;
  }
  NCCLCHECK(ncclTopoGetXmlFromGpu(node, hgmlDev, xml, gpuNode));
  return ncclSuccess;
}

// Returns the subsystem name of a path, i.e. the end of the path
// where sysPath/subsystem points to.
ncclResult_t ncclTopoGetSubsystem(const char* sysPath, char* subSys) {
  char subSysPath[PATH_MAX];
  sprintf(subSysPath, "%s/subsystem", sysPath);
  char* path = realpath(subSysPath, NULL);
  if (path == NULL) {
    subSys[0] = '\0';
  } else {
    int offset;
    for (offset = strlen(path); offset > 0 && path[offset] != '/'; offset--);
    strcpy(subSys, path+offset+1);
    NCCLCHECK(ncclFree(path));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoFillNet(struct ncclXml* xml, const char* pciPath, const char* netName, struct ncclXmlNode** netNode) {
  NCCLCHECK(xmlFindTagKv(xml, "net", netNode, "name", netName));
  if (*netNode != NULL) return ncclSuccess;

  const char* pciSysPath = pciPath;
  if (pciSysPath) {
    char subSystem[PATH_MAX];
    NCCLCHECK(ncclTopoGetSubsystem(pciSysPath, subSystem));
    // This is not a PCI device (virtual, usb, ...).
    if (strcmp(subSystem, "pci") != 0) {
      INFO(NCCL_GRAPH, "Topology detection: network path %s is not a PCI device (%s). Attaching to first CPU", pciSysPath, subSystem);
      pciSysPath = NULL;
    }
  }

  struct ncclXmlNode* parent = NULL;
  if (pciSysPath) {
    int offset;
    for (offset=strlen(pciSysPath)-1; pciSysPath[offset] != '/'; offset--);
    char busId[HGML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    strcpy(busId, pciSysPath+offset+1);
    NCCLCHECK(ncclTopoGetPciNode(xml, busId, &parent));
    NCCLCHECK(ncclTopoGetXmlFromSys(parent, xml));
  } else {
    // Virtual NIC, no PCI device, attach to first CPU
    NCCLCHECK(xmlFindTag(xml, "cpu", &parent));
  }

  struct ncclXmlNode* nicNode = NULL;
  NCCLCHECK(xmlGetSub(parent, "nic", &nicNode));
  if (nicNode == NULL) {
    NCCLCHECK(xmlAddNode(xml, parent, "nic", &nicNode));
  }

  // We know that this net does not exist yet (we searched for it at the
  // beginning of this function), so we can add it.
  NCCLCHECK(xmlAddNode(xml, nicNode, "net", netNode));
  NCCLCHECK(xmlSetAttr(*netNode, "name", netName));
  return ncclSuccess;
}

ncclResult_t ncclTopoTrimXmlRec(struct ncclXmlNode* node) {
  const char* str;
  NCCLCHECK(xmlGetAttr(node, "keep", &str));
  if (str && strcmp(str, "1") == 0) {
    NCCLCHECK(xmlUnsetAttr(node, "keep"));
  } else {
    // Copy nSubs and subs as they could change as we trim recursively.
    struct ncclXmlNode* subs[MAX_SUBS];
    int nSubs = node->nSubs;
    memcpy(subs, node->subs, node->nSubs*sizeof(struct ncclXmlNode*));
    for (int s=0; s<nSubs; s++) {
      NCCLCHECK(ncclTopoTrimXmlRec(subs[s]));
    }
    if (node->nSubs == 0) NCCLCHECK(xmlRemoveNode(node));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoTrimXml(struct ncclXml* xml) {
  NCCLCHECK(ncclTopoTrimXmlRec(xml->nodes));
  return ncclSuccess;
}

/**************************************************/
/* Parser rules for the user-defined graph search */
/**************************************************/

ncclResult_t ncclTopoXmlGraphLoadGpu(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadNet(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  NCCLCHECK(xmlLoadSub(file, xml, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadChannel(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "net", ncclTopoXmlGraphLoadNet }, { "gpu", ncclTopoXmlGraphLoadGpu } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraph(FILE* file, struct ncclXml* xml, struct ncclXmlNode* head) {
  struct xmlHandler handlers[] = { { "channel", ncclTopoXmlGraphLoadChannel } };
  NCCLCHECK(xmlLoadSub(file, xml, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlGraphLoadGraphs(FILE* file, struct ncclXml* xmlGraph, struct ncclXmlNode* head) {
  int version;
  NCCLCHECK(xmlGetAttrInt(head, "version", &version));
  if (version != NCCL_GRAPH_XML_VERSION) {
    ERROR("XML Graph has wrong version %d, %d needed", version, NCCL_GRAPH_XML_VERSION);
    return ncclInvalidUsage;
  }
  const char* name;
  NCCLCHECK(xmlGetAttr(head, "name", &name));
  if (name != NULL) INFO(NCCL_GRAPH, "Loading graphs for topology %s", name);
  else INFO(NCCL_GRAPH, "Loading graphs");

  struct xmlHandler handlers[] = { { "graph", ncclTopoXmlGraphLoadGraph } };
  NCCLCHECK(xmlLoadSub(file, xmlGraph, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlGraphFromFile(const char* xmlGraphFile, struct ncclXml* xml) {
  FILE* file = fopen(xmlGraphFile, "r");
  if (file == NULL) {
    ERROR("Could not open XML graph file %s : %s", xmlGraphFile, strerror(errno));
    return ncclSystemError;
  }
  struct xmlHandler handlers[] = { { "graphs", ncclTopoXmlGraphLoadGraphs } };
  xml->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(file, xml, NULL, handlers, 1));
  fclose(file);
  return ncclSuccess;
}
