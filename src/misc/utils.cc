/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "utils.h"
#include "checks.h"
#include "core.h"
#include "collectives.h"
#include "nvmlwrap.h"
#include "pccl_internal.h"
#include <fstream>
#include <string>
#include <typeinfo>
#include <sys/stat.h>
#include <libgen.h>
#include <dirent.h>
#include <cstring>

// Get current Compute Capability
int ncclHggcCompCap() {
  int hggcDev;
  if (hggcGetDevice(&hggcDev) != hggcSuccess) return 0;
  int ccMajor, ccMinor;
  if (hggcDeviceGetAttribute(&ccMajor, hggcDevAttrComputeCapabilityMajor, hggcDev) != hggcSuccess) return 0;
  if (hggcDeviceGetAttribute(&ccMinor, hggcDevAttrComputeCapabilityMinor, hggcDev) != hggcSuccess) return 0;
  return ccMajor*10+ccMinor;
}

ncclResult_t int64ToBusId(int64_t id, char* busId) {
  sprintf(busId, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
  return ncclSuccess;
}

ncclResult_t busIdToInt64(const char* busId, int64_t* id) {
  const int size = strlen(busId);
  constexpr int defaultSize = 16;
  char hexStr[defaultSize+1];
  int hexOffset = 0;
  for (int i=0; i<size; i++) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;
    if ((c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
      if (hexOffset>=defaultSize) break;
    } else break;
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  return ncclSuccess;
}

// Convert a logical hggcDev index to the NVML device minor number
ncclResult_t getBusId(int hggcDev, int64_t *busId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdStr[] = "00000000:00:00.0";
  HGGCCHECK(hggcDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), hggcDev));
  NCCLCHECK(busIdToInt64(busIdStr, busId));
  return ncclSuccess;
}

ncclResult_t convertLowerBusId(const char* busId, const int size, char* lowerId) {
  for (int c = 0; c < size; c++) {
    lowerId[c] = tolower(busId[c]);
    if (busId[c] == 0) break;
  }
  return ncclSuccess;
}

ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return ncclSystemError;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen-1)) i++;
  hostname[i] = '\0';
  return ncclSuccess;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the NCCL_HOSTID env var.
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
uint64_t getHostHash(void) {
  char hostHash[1024];
  const char *hostId = NULL;

  // Fall back is the full hostname if something fails
  (void) getHostName(hostHash, sizeof(hostHash), '\0');
  int offset = strlen(hostHash);

  if ((hostId = pcclGetEnvWithDualPrefix("PCCL_HOSTID")) != NULL) {
    INFO(NCCL_ENV, "PCCL_HOSTID set by environment to %s", hostId);
    strncpy(hostHash, hostId, sizeof(hostHash)-1);
  } else {
    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
      char *p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        NCCLCHECK(ncclFree(p));
      }
      fclose(file);
    }
  }

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  TRACE(NCCL_INIT,"unique hostname '%s'", hostHash);

  return getHash(hostHash, strlen(hostHash));
}

/* Generate a hash of the unique identifying string for this process
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $$ $(readlink /proc/self/ns/pid)
 */
uint64_t getPidHash(void) {
  char pname[1024];
  // Start off with our pid ($$)
  sprintf(pname, "%ld", (long) getpid());
  int plen = strlen(pname);
  int len = readlink("/proc/self/ns/pid", pname+plen, sizeof(pname)-1-plen);
  if (len < 0) len = 0;

  pname[plen+len]='\0';
  TRACE(NCCL_INIT,"unique PID '%s'", pname);

  return getHash(pname, strlen(pname));
}

char* getDateTime(char* timeStr) {
  std::time_t currentTime = std::time(NULL);
  std::strftime(timeStr, 64, "%Y%m%d-%H%M%S", localtime(&currentTime));
  return timeStr;
}

int parseStringList(const char* string, struct netIf* ifList, int maxList) {
  if (!string) return 0;

  const char* ptr = string;

  int ifNum = 0;
  int ifC = 0;
  char c;
  do {
    c = *ptr;
    if (c == ':') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = atoi(ptr+1);
        ifNum++; ifC = 0;
      }
      while (c != ',' && c != '\0') c = *(++ptr);
    } else if (c == ',' || c == '\0') {
      if (ifC > 0) {
        ifList[ifNum].prefix[ifC] = '\0';
        ifList[ifNum].port = -1;
        ifNum++; ifC = 0;
      }
    } else {
      ifList[ifNum].prefix[ifC] = c;
      ifC++;
    }
    ptr++;
  } while (ifNum < maxList && c);
  return ifNum;
}

static bool matchIf(const char* string, const char* ref, bool matchExact) {
  // Make sure to include '\0' in the exact case
  int matchLen = matchExact ? strlen(string) + 1 : strlen(ref);
  return strncmp(string, ref, matchLen) == 0;
}

static bool matchPort(const int port1, const int port2) {
  if (port1 == -1) return true;
  if (port2 == -1) return true;
  if (port1 == port2) return true;
  return false;
}


bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact) {
  // Make an exception for the case where no user list is defined
  if (listSize == 0) return true;

  for (int i=0; i<listSize; i++) {
    if (matchIf(string, ifList[i].prefix, matchExact)
        && matchPort(port, ifList[i].port)) {
      return true;
    }
  }
  return false;
}

int fillConfigName(char* string, int n, uint64_t commHash, int rank, int nRanks, int hggcDev, int hgmlDev, const char* prefix, const char* suffix, bool includeTime) {
  int offset = 0;
  char hostname[64] = {0};
  char timeStr[64] = {0};
  char devStr[64] = {0};
  pid_t pid = getpid();

#define OFFSETCHECK(cmd) \
  do {  \
    offset += cmd;  \
    if (offset > n) goto overfill;  \
  } while (0);

  if (prefix) {
    OFFSETCHECK(snprintf(string, n, "%s_", prefix));
  }

  getHostName(hostname, sizeof(hostname), '.');
  if (hgmlDev >= 0) {
    sprintf(devStr, "hgmldev-%d", hgmlDev);
  } else {
    sprintf(devStr, "hggcdev-%d", hggcDev);
  }

  OFFSETCHECK(snprintf(string+offset, n-offset, "comm-%lx_nranks-%d_rank-%d_%s_%s_pid-%d", commHash, nRanks, rank, devStr, hostname, pid));
  if (includeTime) {
    OFFSETCHECK(snprintf(string+offset, n-offset, "_%s", getDateTime(timeStr)));
  }
  if (suffix) {
    OFFSETCHECK(snprintf(string+offset, n-offset, ".%s", suffix));
  }

exit:
  return offset;
overfill:
  offset = n;
  goto exit;
}

bool createDir(char* dir) {
  struct stat buf;
  if (stat(dir, &buf) == 0) {
    if (S_ISDIR(buf.st_mode)) {
      return true;
    }
    return false;
  }

  char* temp = strdup(dir);
  char* parent = dirname(temp);
  createDir(parent);
  free(temp);

  if (mkdir(dir, 0770) == 0 || errno == EEXIST) {
    return true;
  }
  return false;
}

uint64_t countDirSize(const char* dir) {
  int64_t count = 0;
  DIR* dp = opendir(dir);
  if (dp) {
    struct dirent* entry;
    while ((entry = readdir(dp))) {
      char filepath[PATH_MAX+1];
      snprintf(filepath, PATH_MAX, "%s/%s", dir, entry->d_name);
      struct stat buf;
      stat(filepath, &buf);
      if (S_ISDIR(buf.st_mode)) {
        if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0) continue;
        uint64_t subDirSize = countDirSize(filepath);
        if (subDirSize == -1) continue;
        count += subDirSize;
      } else {
        count += buf.st_size;
      }
    }
    closedir(dp);
  }
  return count;
}

PCCL_PARAM(StateMonitorDumpDirSizeLimitMB, "STATE_MONITOR_DUMP_DIR_SIZE_LIMIT_MB",  2048 /* 2GB */);
ncclResult_t createOrFillDumpDir(char* dir, const int maxLength, const char* sub, int* pathOffset) {
  static bool allowDump = true;
  if (!allowDump) return ncclInternalError;

  const uint64_t limitation = pcclParamStateMonitorDumpDirSizeLimitMB() * 1024 * 1024;
  const char* pcclDumpParentDirEnv = pcclGetEnvWithDualPrefix("PCCL_DEBUG_DUMP_DIR");
  const char* pcclDumpDir = ".pccl";
  int offset = *pathOffset;
  if (pcclDumpParentDirEnv) {
    offset += snprintf(dir+offset, maxLength-offset, "%s/%s/", pcclDumpParentDirEnv, pcclDumpDir);
  } else {
    const char* userDir = userHomeDir();
    if (userDir) {
      offset += snprintf(dir+offset, maxLength-offset, "%s/%s/", userDir, pcclDumpDir);
    }
  }

  if (offset > maxLength) {
    allowDump = false;
    WARN("dump directory path is too long, disable dump to avoid file access related issues");
    return ncclInternalError;
  }
  if (countDirSize(dir) > limitation) {
    WARN("exceed maximum dump limitation(%d), clear dump directory %s if necessary", limitation, dir);
    return ncclInternalError;
  }

  if (sub) {
    offset += snprintf(dir+offset, maxLength-offset, "%s/", sub);
  }
  if (offset > maxLength) goto overfill;

  if (!createDir(dir)) {
    if (sub) {
      offset = snprintf(dir, maxLength-offset, "%s/%s/", pcclDumpDir, sub);
    } else {
      offset = snprintf(dir, maxLength-offset, "%s/", pcclDumpDir);
    }
    if (!createDir(dir)) {
      offset = 0;
    }
  }
  if (offset > maxLength) goto overfill;

exit:
  dir[offset] = '\0';
  *pathOffset = offset;
  return ncclSuccess;
overfill:
  offset = maxLength;
  goto exit;
}

void getKerFuncStr(const int funcIndexIn, char *funcStr, int *funcType, int *dataType) {
  const char* ncclDataTypeStr[ncclNumTypes] = { "int8_t", "uint8_t", "int32_t", "uint32_t", "int64_t", "uint64_t", "half", "float", "double", "bfloat16" , "fp8e4m3", "fp8e5m2" };
  const char* ncclRedOpStr[ncclNumDevRedOps] = { "Sum", "Prod", "Max", "Min", "PreMulSum", "SumPostDiv" };
  constexpr int funcRange = ncclNumDevRedOps * ncclNumTypes * NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS;
  constexpr int redopRange = ncclNumTypes * NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS;
  constexpr int typeRange = NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS;
  constexpr int algoRange = NCCL_NUM_PROTOCOLS;
  // p2p first
  int funcIndex = 0;
  int datatypeIndex = 0;;
  if (funcIndexIn < NUM_P2P_FUNC) {
    if (funcStr != NULL) {
      switch(funcIndexIn) {
        case FUNC_INDEX_P2P_SIMPLE:
          sprintf(funcStr, "%s", "SendRecv_Ring_Simple_Sum_int8_t");
          break;
        case FUNC_INDEX_P2P_SIMPLE_BULK:
          sprintf(funcStr, "%s", "SendRecv_Ring_SimpleBulk_Sum_int8_t");
          break;
        case FUNC_INDEX_P2P_SIGNAL:
          sprintf(funcStr, "%s", "SendRecv_Ring_Signal_Sum_int8_t");
          break;
        case FUNC_INDEX_P2P_SIGNAL_BULK:
          sprintf(funcStr, "%s", "SendRecv_Ring_SignalBulk_Sum_int8_t");
          break;
        case FUNC_INDEX_P2P_SIMPLE_BULK_AR:
          sprintf(funcStr, "%s", "SendRecv_Ring_SimpleBulkAR_Sum_int8_t");
          break;
        default:
          sprintf(funcStr, "%s", "Invalid Kernel");
          break;
      }
    }
    funcIndex = ncclFuncSendRecv;
    datatypeIndex = ncclInt8;
  } else {
    // coll compute
    int collIndex  = funcIndexIn - NUM_P2P_FUNC;
    funcIndex = collIndex / funcRange;
    int offset = collIndex - funcIndex * funcRange;
    int redopIndex  = offset / redopRange;
    offset -= redopIndex * redopRange;
    datatypeIndex = offset / typeRange;
    offset -= datatypeIndex * typeRange;
    int algoIndex = offset / algoRange;
    offset -= algoIndex * algoRange;

    if (funcStr != NULL) {
      sprintf(funcStr, "%s_%s_%s_%s_%s",
            ncclFuncStr[funcIndex],
            ncclAlgoStr[algoIndex],
            ncclProtoStr[offset],
            ncclRedOpStr[redopIndex],
            ncclDataTypeStr[datatypeIndex]);
    }
  }
  if (funcType != NULL) *funcType = funcIndex;
  if (dataType != NULL) *dataType = datatypeIndex;
}

__thread struct ncclThreadSignal ncclThreadSignalLocalInstance = ncclThreadSignalStaticInitializer();

void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align) {
  // `me->hunks` points to the top of the stack non-empty hunks. Hunks above
  // this (reachable via `->above`) are empty.
  struct Hunk* top = me->topFrame.hunk;
  size_t mallocSize = 0;

  // If we have lots of space left in hunk but that wasn't enough then we'll
  // allocate the object unhunked.
  if (me->topFrame.end - me->topFrame.bumper >= 8<<10)
    goto unhunked;

  // If we have another hunk (which must be empty) waiting above this one and
  // the object fits then use that.
  if (top && top->above) {
    struct Hunk* top1 = top->above;
    uintptr_t uobj = (reinterpret_cast<uintptr_t>(top1) + sizeof(struct Hunk) + align-1) & -uintptr_t(align);
    if (uobj + size <= reinterpret_cast<uintptr_t>(top1) + top1->size) {
      me->topFrame.hunk = top1;
      me->topFrame.bumper = uobj + size;
      me->topFrame.end = reinterpret_cast<uintptr_t>(top1) + top1->size;
      return reinterpret_cast<void*>(uobj);
    }
  }

  { // If the next hunk we're going to allocate wouldn't be big enough but the
    // Unhunk proxy fits in the current hunk then go allocate as unhunked.
    size_t nextSize = (top ? top->size : 0) + (64<<10);
    constexpr size_t maxAlign = 64;
    if (nextSize < sizeof(struct Hunk) + maxAlign + size) {
      uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk)-1) & -uintptr_t(alignof(Unhunk));
      if (uproxy + sizeof(struct Unhunk) <= me->topFrame.end)
        goto unhunked;
    }

    // At this point we must need another hunk, either to fit the object
    // itself or its Unhunk proxy.
    mallocSize = nextSize;
    INFO(NCCL_ALLOC, "%s:%d memory stack hunk malloc(%llu)", __FILENAME__, __LINE__, (unsigned long long)mallocSize);
    struct Hunk *top1 = (struct Hunk*)malloc(mallocSize);
    if (top1 == nullptr) goto malloc_exhausted;
    top1->size = nextSize;
    top1->above = nullptr;
    if (top) top->above = top1;
    top = top1;
    me->topFrame.hunk = top;
    me->topFrame.end = reinterpret_cast<uintptr_t>(top) + nextSize;
    me->topFrame.bumper = reinterpret_cast<uintptr_t>(top) + sizeof(struct Hunk);
  }

  { // Try to fit object in the new top hunk.
    uintptr_t uobj = (me->topFrame.bumper + align-1) & -uintptr_t(align);
    if (uobj + size <= me->topFrame.end) {
      me->topFrame.bumper = uobj + size;
      return reinterpret_cast<void*>(uobj);
    }
  }

unhunked:
  { // We need to allocate the object out-of-band and put an Unhunk proxy in-band
    // to keep track of it.
    uintptr_t uproxy = (me->topFrame.bumper + alignof(Unhunk)-1) & -uintptr_t(alignof(Unhunk));
    Unhunk* proxy = reinterpret_cast<Unhunk*>(uproxy);
    me->topFrame.bumper = uproxy + sizeof(Unhunk);
    proxy->next = me->topFrame.unhunks;
    me->topFrame.unhunks = proxy;
    mallocSize = size;
    proxy->obj = malloc(mallocSize);
    INFO(NCCL_ALLOC, "%s:%d memory stack non-hunk malloc(%llu)", __FILENAME__, __LINE__, (unsigned long long)mallocSize);
    if (proxy->obj == nullptr) goto malloc_exhausted;
    return proxy->obj;
  }

malloc_exhausted:
  WARN("%s:%d Unrecoverable error detected: malloc(size=%llu) returned null.", __FILENAME__, __LINE__, (unsigned long long)mallocSize);
  abort();
}

void ncclMemoryStackDestruct(struct ncclMemoryStack* me) {
  // Free unhunks first because both the frames and unhunk proxies lie within the hunks.
  struct ncclMemoryStack::Frame* f = &me->topFrame;
  while (f != nullptr) {
    struct ncclMemoryStack::Unhunk* u = f->unhunks;
    while (u != nullptr) {
      free(u->obj);
      u = u->next;
    }
    f = f->below;
  }
  // Free hunks
  struct ncclMemoryStack::Hunk* h = me->stub.above;
  while (h != nullptr) {
    struct ncclMemoryStack::Hunk *h1 = h->above;
    free(h);
    h = h1;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Hash function for pointer types (shared by address map and shadow pool)
// Fibonacci hashing with golden ratio constant.
uint64_t ncclHashPointer(int hbits, void* key) {
  uintptr_t h = reinterpret_cast<uintptr_t>(key);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13ULL;
  return (uint64_t)h >> (64 - hbits);
}

////////////////////////////////////////////////////////////////////////////////
// Intrusive address map implementation (untyped core functions)

// Helper: Read key from object at given offset.
// Key must be convertible to uintptr_t, so we read keySize bytes and zero-extend.
static inline uintptr_t readKey(void* object, int keySize, int keyFieldOffset) {
  void* keyPtr = (char*)object + keyFieldOffset;
  uintptr_t result = 0;
  memcpy(&result, keyPtr, keySize);
  return result;
}

// Helper: Read next pointer from object at given offset.
// Uses memcpy to avoid strict aliasing violations.
static inline void* readNextPtr(void* object, int nextFieldOffset) {
  void* nextPtr = (char*)object + nextFieldOffset;
  void* result = nullptr;
  memcpy(&result, nextPtr, sizeof(void*));
  return result;
}

// Helper: Write next pointer to object at given offset.
// Uses memcpy to avoid strict aliasing violations.
static inline void writeNextPtr(void* object, int nextFieldOffset, void* value) {
  void* nextPtr = (char*)object + nextFieldOffset;
  memcpy(nextPtr, &value, sizeof(void*));
}

ncclResult_t ncclIntruAddressMapInsert_untyped(
    struct ncclIntruAddressMap_untyped* map,
    int keySize, int keyFieldOffset, int nextFieldOffset,
    uintptr_t key, void* object) {
  // Runtime validation
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (object == nullptr) {
    WARN("Object pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  // Lazy initialization - create table on first insert
  if (map->hbits == 0) {
    map->hbits = 4;
    map->table = (void**)calloc((size_t)1 << map->hbits, sizeof(void*));
    if (map->table == nullptr) {
      map->hbits = 0;
      WARN("Intrusive address map initialization failed: calloc returned null");
      return ncclSystemError;
    }
  }

  int hbits = map->hbits;

  // Check for table size increase before inserting. Maintain 2:1 object:bucket ratio.
  if (map->count + 1 > 2 << hbits) {
    int oldSize = 1 << hbits;
    int newHbits = hbits + 1;
    int newSize = 1 << newHbits;

    void** newTable = (void**)malloc(newSize * sizeof(void*));
    if (newTable == nullptr) {
      WARN("Intrusive address map resize failed: malloc returned null");
      return ncclSystemError;
    }

    for (int i = 0; i < newSize; i++) {
      newTable[i] = nullptr;
    }

    // Rehash all existing entries
    for (int i = 0; i < oldSize; i++) {
      void* obj = map->table[i];
      while (obj) {
        void* next = readNextPtr(obj, nextFieldOffset);
        uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
        uint64_t b = ncclHashPointer(newHbits, (void*)objKey);
        writeNextPtr(obj, nextFieldOffset, newTable[b]);
        newTable[b] = obj;
        obj = next;
      }
    }

    free(map->table);
    map->table = newTable;
    map->hbits = newHbits;
  }

  // Insert object into appropriate bucket
  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  writeNextPtr(object, nextFieldOffset, map->table[b]);
  map->table[b] = object;
  map->count += 1;

  return ncclSuccess;
}

ncclResult_t ncclIntruAddressMapFind_untyped(
    struct ncclIntruAddressMap_untyped* map,
    int keySize, int keyFieldOffset, int nextFieldOffset,
    uintptr_t key, void** object) {
  // Runtime validation
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (object == nullptr) {
    WARN("Output object pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  *object = nullptr;

  // Empty map - key not found (not an error)
  if (map->hbits == 0) {
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* obj = map->table[b];

  while (obj) {
    uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
    if (objKey == key) {
      *object = obj;
      return ncclSuccess;
    }
    obj = readNextPtr(obj, nextFieldOffset);
  }

  // Key not found - *object is already nullptr
  return ncclSuccess;
}

ncclResult_t ncclIntruAddressMapRemove_untyped(
    struct ncclIntruAddressMap_untyped* map,
    int keySize, int keyFieldOffset, int nextFieldOffset,
    uintptr_t key) {
  // Runtime validation
  if (map == nullptr) {
    WARN("Intrusive address map pointer is NULL");
    return ncclInvalidUsage;
  }
  if (keySize <= 0 || keySize > (int)sizeof(uintptr_t)) {
    WARN("Invalid key size %d (must be 0 < keySize <= %zu)", keySize, sizeof(uintptr_t));
    return ncclInvalidUsage;
  }

  // Removing from empty map is idempotent (not an error)
  if (map->hbits == 0) {
    return ncclSuccess;
  }

  uint64_t b = ncclHashPointer(map->hbits, (void*)key);
  void* prev = nullptr;
  void* obj = map->table[b];

  while (obj) {
    uintptr_t objKey = readKey(obj, keySize, keyFieldOffset);
    if (objKey == key) {
      void* next = readNextPtr(obj, nextFieldOffset);

      if (prev == nullptr) {
        map->table[b] = next;
      } else {
        writeNextPtr(prev, nextFieldOffset, next);
      }

      map->count -= 1;

      // If last entry removed, clean up the table
      if (map->count == 0) {
        free(map->table);
        map->hbits = 0;
        map->count = 0;
        map->table = nullptr;
      }

      return ncclSuccess;
    }
    prev = obj;
    obj = readNextPtr(obj, nextFieldOffset);
  }

  // Key not found - remove is idempotent
  return ncclSuccess;
}
