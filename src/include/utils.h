/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_UTILS_H_
#define NCCL_UTILS_H_

#include "pccl_internal.h"
#include "comm_dumper.h"
#include <pthread.h>
#include <stdint.h>
#include <sys/stat.h>
#include "checks.h"
#include <algorithm>
#include "hggc_ad.h"
#include "alloc.h"

#ifndef LAUNCH_KERNEL
#define LAUNCH_KERNEL(kernel, gridDim, blockDim, args, sharedMem, stream, subDisaptch, launchConfig)                                                                  \
  do                                                                                                                                                                  \
  {                                                                                                                                                                   \
    HGlaunchConfigAD cfg = {gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, (unsigned int)sharedMem, stream, nullptr, 0};                        \
    HGlaunchAttributeAD attr[2];                                                                                                                                      \
    attr[0].id = HGAD_LAUNCH_ATTRIBUTE_SCHED_PREFERENCE;                                                                                                              \
    attr[0].value.schedPreference.blocksPerMultiprocessor = launchConfig.blocksPerMultiprocessor;                                                                     \
    attr[0].value.schedPreference.gridStepX = launchConfig.gridStepX;                                                                                                 \
    attr[0].value.schedPreference.gridStepY = launchConfig.gridStepY;                                                                                                 \
    attr[0].value.schedPreference.flags = (launchConfig.subDispatch >= 0 ? launchConfig.subDispatch : subDisaptch) ? HGAD_SCHED_PREFER_SYMM : HGAD_SCHED_PREFER_NONE; \
    cfg.attrs = attr;                                                                                                                                                 \
    cfg.numAttrs = 1;                                                                                                                                                 \
    if (launchConfig.cooperative) {                                                                                                                                   \
      attr[cfg.numAttrs].id = HGAD_LAUNCH_ATTRIBUTE_COOPERATIVE;                                                                                                      \
      attr[cfg.numAttrs].value.cooperative  = 1;                                                                                                                      \
      cfg.numAttrs += 1;                                                                                                                                              \
    }                                                                                                                                                                 \
    HGfunction hfunc;                                                                                                                                                 \
    HGGCCHECK(hggcGetFuncBySymbol(&hfunc, (const void *)kernel));                                                                                                     \
    HGGCCHECK((hggcError_t)hgLaunchKernelExAD(&cfg, hfunc, args, nullptr));                                                                                           \
  } while (0)
#endif

uint64_t clockNano();

int ncclHggcCompCap();

// Greatest common divisor (Euclidean). Used by dev_runtime.cc::computeLsaSize.
static inline int gcd(int a, int b) {
  while (b != 0) {
    int tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

// PCI Bus ID <-> int64 conversion functions
ncclResult_t int64ToBusId(int64_t id, char* busId);
ncclResult_t busIdToInt64(const char* busId, int64_t* id);

ncclResult_t getBusId(int hggcDev, int64_t *busId);
ncclResult_t convertLowerBusId(const char* busId, const int size, char* lowerId);

ncclResult_t getHostName(char* hostname, int maxlen, const char delim);
uint64_t getHostHash();
uint64_t getPidHash();
char* getDateTime(char* timeStr);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);
int fillConfigName(char* filename, int n, uint64_t commHash, int rank, int nRanks, int hggcDev, int hgmlDev, const char* prefix=NULL, const char* suffix=NULL, bool includeTime=true);
ncclResult_t createOrFillDumpDir(char* dir, const int maxLength, const char* sub, int* pathOffset);

static long log2i(long n) {
 long l = 0;
 while (n>>=1) l++;
 return l;
}

static int nextPow2(int v) {
  int pow2 = 1;
  while (pow2 < v) pow2 <<= 1;
  return pow2;
}

static int prePow2(int v) {
  int pow2 = 2;
  while (pow2 <= v) pow2 <<= 1;
  if (pow2 == v) return pow2;
  else return (pow2 >>= 1);
}

// Comparator function for qsort/bsearch to compare integers
static int compareInts(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

inline uint64_t clockNano() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec)*1000*1000*1000 + ts.tv_nsec;
}

/* get any bytes of random data from /dev/urandom, return ncclSuccess (0) if it succeeds. */
inline ncclResult_t getRandomData(void* buffer, size_t bytes) {
  ncclResult_t ret = ncclSuccess;
  if (bytes > 0) {
    const size_t one = 1UL;
    FILE* fp = fopen("/dev/urandom", "r");
    if (buffer == NULL || fp == NULL || fread(buffer, bytes, one, fp) != one) ret = ncclSystemError;
    if (fp) fclose(fp);
  }
  return ret;
}

template<typename Int>
inline void ncclAtomicRefCountIncrement(Int* refs) {
  __atomic_fetch_add(refs, 1, __ATOMIC_RELAXED);
}

template<typename Int>
inline Int ncclAtomicRefCountDecrement(Int* refs) {
  return __atomic_sub_fetch(refs, 1, __ATOMIC_ACQ_REL);
}

////////////////////////////////////////////////////////////////////////////////
/* ncclThreadSignal: Couples a pthread mutex and cond together. The "mutex"
 * and "cond" fields are part of the public interface.
 */
struct ncclThreadSignal {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};

// returns {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER}
constexpr ncclThreadSignal ncclThreadSignalStaticInitializer();

void ncclThreadSignalConstruct(struct ncclThreadSignal* me);
void ncclThreadSignalDestruct(struct ncclThreadSignal* me);

// A convenience instance per-thread.
extern __thread struct ncclThreadSignal ncclThreadSignalLocalInstance;

////////////////////////////////////////////////////////////////////////////////

constexpr ncclThreadSignal ncclThreadSignalStaticInitializer() {
  return {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER};
}

inline void ncclThreadSignalConstruct(struct ncclThreadSignal* me) {
  pthread_mutex_init(&me->mutex, nullptr);
  pthread_cond_init(&me->cond, nullptr);
}

inline void ncclThreadSignalDestruct(struct ncclThreadSignal* me) {
  pthread_mutex_destroy(&me->mutex);
  pthread_cond_destroy(&me->cond);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, T *T::*next>
struct ncclIntruQueueMpsc {
  T* head;
  uintptr_t tail;
  struct ncclThreadSignal* waiting;
};

template<typename T, T *T::*next>
void ncclIntruQueueMpscConstruct(struct ncclIntruQueueMpsc<T,next>* me) {
  me->head = nullptr;
  me->tail = 0x0;
  me->waiting = nullptr;
}

template<typename T, T *T::*next>
bool ncclIntruQueueMpscEmpty(struct ncclIntruQueueMpsc<T,next>* me) {
  return __atomic_load_n(&me->tail, __ATOMIC_RELAXED) <= 0x2;
}

template<typename T, T *T::*next>
bool ncclIntruQueueMpscEnqueue(ncclIntruQueueMpsc<T,next>* me, T* x) {
  __atomic_store_n(&(x->*next), nullptr, __ATOMIC_RELAXED);
  uintptr_t utail = __atomic_exchange_n(&me->tail, reinterpret_cast<uintptr_t>(x), __ATOMIC_ACQ_REL);
  T* prev = reinterpret_cast<T*>(utail);
  T** prevNext = utail <= 0x2 ? &me->head : &(prev->*next);
  __atomic_store_n(prevNext, x, __ATOMIC_RELAXED);
  if (utail == 0x1) { // waiting
    __atomic_thread_fence(__ATOMIC_ACQUIRE); // to see me->waiting
    // This lock/unlock is essential to ensure we don't race ahead of the consumer
    // and signal the cond before they begin waiting on it.
    struct ncclThreadSignal* waiting = me->waiting;
    pthread_mutex_lock(&waiting->mutex);
    pthread_mutex_unlock(&waiting->mutex);
    pthread_cond_broadcast(&waiting->cond);
  }
  return utail != 0x2; // not abandoned
}

template<typename T, T *T::*next>
T* ncclIntruQueueMpscDequeueAll(ncclIntruQueueMpsc<T,next>* me, bool waitSome) {
  T* head = __atomic_load_n(&me->head, __ATOMIC_RELAXED);
  if (head == nullptr) {
    if (!waitSome) return nullptr;
    uint64_t t0 = clockNano();
    bool sleeping = false;
    do {
      if (clockNano()-t0 >= 10*1000) { // spin for first 10us
        struct ncclThreadSignal* waitSignal = &ncclThreadSignalLocalInstance;
        pthread_mutex_lock(&waitSignal->mutex);
        uintptr_t expected = sleeping ? 0x1 : 0x0;
        uintptr_t desired = 0x1;
        me->waiting = waitSignal; // release done by successful compare exchange
        if (__atomic_compare_exchange_n(&me->tail, &expected, desired, /*weak=*/true, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
          sleeping = true;
          pthread_cond_wait(&waitSignal->cond, &waitSignal->mutex);
        }
        pthread_mutex_unlock(&waitSignal->mutex);
      }
      head = __atomic_load_n(&me->head, __ATOMIC_RELAXED);
    } while (head == nullptr);
  }

  __atomic_store_n(&me->head, nullptr, __ATOMIC_RELAXED);
  uintptr_t utail = __atomic_exchange_n(&me->tail, 0x0, __ATOMIC_ACQ_REL);
  T* tail = utail <= 0x2 ? nullptr : reinterpret_cast<T*>(utail);
  T *x = head;
  while (x != tail) {
    T *x1;
    int spins = 0;
    while (true) {
      x1 = __atomic_load_n(&(x->*next), __ATOMIC_RELAXED);
      if (x1 != nullptr) break;
      if (++spins == 1024) { spins = 1024-1; sched_yield(); }
    }
    x = x1;
  }
  return head;
}

template<typename T, T *T::*next>
T* ncclIntruQueueMpscAbandon(ncclIntruQueueMpsc<T,next>* me) {
  uintptr_t expected = 0x0;
  if (__atomic_compare_exchange_n(&me->tail, &expected, /*desired=*/0x2, /*weak=*/true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    return nullptr;
  } else {
    int spins = 0;
    T* head;
    while (true) {
      head = __atomic_load_n(&me->head, __ATOMIC_RELAXED);
      if (head != nullptr) break;
      if (++spins == 1024) { spins = 1024-1; sched_yield(); }
    }
    __atomic_store_n(&me->head, nullptr, __ATOMIC_RELAXED);
    uintptr_t utail = __atomic_exchange_n(&me->tail, 0x2, __ATOMIC_ACQ_REL);
    T* tail = utail <= 0x2 ? nullptr : reinterpret_cast<T*>(utail);
    T *x = head;
    while (x != tail) {
      T *x1;
      spins = 0;
      while (true) {
        x1 = __atomic_load_n(&(x->*next), __ATOMIC_RELAXED);
        if (x1 != nullptr) break;
        if (++spins == 1024) { spins = 1024-1; sched_yield(); }
      }
      x = x1;
    }
    return head;
  }
}

// pccl internal dump functions
int64_t pcclDevKernelArgsFifoDepth();
ncclResult_t pcclCommDump(ncclComm_t comm, pcclCommDumpMaskType pcclCommDumpMask,
                          struct pcclCommDiagCollector* collector = nullptr);
ncclResult_t getGlobalCommDumpMask(pcclCommDumpMaskType* pcclCommDumpMask);
void getKerFuncStr(const int funcIndexIn, char *funcStr, int *funcType=NULL, int *dataType=NULL);

////////////////////////////////////////////////////////////////////////////////
/* ncclMemoryStack: Pools memory for fast LIFO ordered allocation. Note that
 * granularity of LIFO is not per object, instead frames containing many objects
 * are pushed and popped. Therefor deallocation is extremely cheap since its
 * done at the frame granularity.
 *
 * The initial state of the stack is with one frame, the "nil" frame, which
 * cannot be popped. Therefor objects allocated in the nil frame cannot be
 * deallocated sooner than stack destruction.
 */
struct ncclMemoryStack;

void ncclMemoryStackConstruct(struct ncclMemoryStack* me);
void ncclMemoryStackDestruct(struct ncclMemoryStack* me);
void ncclMemoryStackPush(struct ncclMemoryStack* me);
void ncclMemoryStackPop(struct ncclMemoryStack* me);
void* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t size, size_t align);
template<typename T>
T* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t n=1);
template<typename Header, typename Element>
inline Header* ncclMemoryStackAllocInlineArray(struct ncclMemoryStack* me, size_t nElt);

////////////////////////////////////////////////////////////////////////////////

/* ncclMemoryPool: A free-list of same-sized allocations. It is an invalid for
 * a pool instance to ever hold objects whose type have differing
 * (sizeof(T), alignof(T)) pairs. The underlying memory is supplied by
 * a backing `ncclMemoryStack` passed during Alloc(). If memory
 * backing any currently held object is deallocated then it is an error to do
 * anything other than reconstruct it, after which it is a valid empty pool.
 */
struct ncclMemoryPool;

// Equivalent to zero-initialization
void ncclMemoryPoolConstruct(struct ncclMemoryPool* me);
template<typename T>
T* ncclMemoryPoolAlloc(struct ncclMemoryPool* me, struct ncclMemoryStack* backing);
template<typename T>
void ncclMemoryPoolFree(struct ncclMemoryPool* me, T* obj);

////////////////////////////////////////////////////////////////////////////////

/* ncclIntruQueue: A singly-linked list queue where the per-object next pointer
 * field is given via the `next` template argument.
 *
 * Example:
 *   struct Foo {
 *     struct Foo *next1, *next2; // can be a member of two lists at once
 *   };
 *   ncclIntruQueue<Foo, &Foo::next1> list1;
 *   ncclIntruQueue<Foo, &Foo::next2> list2;
 */
template<typename T, T *T::*next>
struct ncclIntruQueue;

template<typename T, T *T::*next>
void ncclIntruQueueConstruct(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
bool ncclIntruQueueEmpty(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
T* ncclIntruQueueHead(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
void ncclIntruQueueEnqueue(ncclIntruQueue<T,next> *me, T *x);
template<typename T, T *T::*next>
void ncclIntruQueueEnqueueFront(ncclIntruQueue<T,next> *me, T *x);
template<typename T, T *T::*next>
T* ncclIntruQueueDequeue(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
T* ncclIntruQueueTryDequeue(ncclIntruQueue<T,next> *me);
template<typename T, T *T::*next>
void ncclIntruQueueTransfer(ncclIntruQueue<T,next> *dst, ncclIntruQueue<T,next> *src);

////////////////////////////////////////////////////////////////////////////////

template<typename T, T *T::*next>
struct ncclIntruQueue {
  T *head, *tail;
};

template<typename T, T *T::*next>
inline void ncclIntruQueueConstruct(ncclIntruQueue<T,next> *me) {
  me->head = nullptr;
  me->tail = nullptr;
}

template<typename T, T *T::*next>
inline bool ncclIntruQueueEmpty(ncclIntruQueue<T,next> *me) {
  return me->head == nullptr;
}

template<typename T, T *T::*next>
inline T* ncclIntruQueueHead(ncclIntruQueue<T,next> *me) {
  return me->head;
}

template<typename T, T *T::*next>
inline T* ncclIntruQueueTail(ncclIntruQueue<T,next> *me) {
  return me->tail;
}

template<typename T, T *T::*next>
inline void ncclIntruQueueEnqueue(ncclIntruQueue<T,next> *me, T *x) {
  x->*next = nullptr;
  (me->head ? me->tail->*next : me->head) = x;
  me->tail = x;
}

template<typename T, T *T::*next>
inline void ncclIntruQueueEnqueueFront(ncclIntruQueue<T,next> *me, T *x) {
  if (me->head == nullptr) me->tail = x;
  x->*next = me->head;
  me->head = x;
}

template<typename T, T *T::*next>
inline T* ncclIntruQueueDequeue(ncclIntruQueue<T,next> *me) {
  T *ans = me->head;
  me->head = ans->*next;
  if (me->head == nullptr) me->tail = nullptr;
  return ans;
}

template<typename T, T *T::*next>
inline T* ncclIntruQueueTryDequeue(ncclIntruQueue<T,next> *me) {
  T *ans = me->head;
  if (ans != nullptr) {
    me->head = ans->*next;
    if (me->head == nullptr) me->tail = nullptr;
  }
  return ans;
}

template<typename T, T *T::*next>
void ncclIntruQueueTransfer(ncclIntruQueue<T,next> *dst, ncclIntruQueue<T,next> *src) {
  (dst->tail ? dst->tail->next : dst->head) = src->head;
  if (src->tail) dst->tail = src->tail;
  src->head = nullptr;
  src->tail = nullptr;
}

// Recyclable list that avoids frequent malloc/free
template<typename T>
struct ncclListElem {
  T data;
  struct ncclListElem* next;
};

template<typename T>
class ncclRecyclableList {
 private:
  struct ncclListElem<T>* head;
  struct ncclListElem<T>* tail;
  struct ncclListElem<T>* cursor;
  int n;

 public:
  ncclRecyclableList() {
    tail = cursor = head = NULL;
    n = 0;
  }

  int count() const { return n; }

  // Get a new element from the list and return pointer
  ncclResult_t getNewElem(T** dataOut) {
    if (tail != NULL) {
      *dataOut = &tail->data;
      memset(*dataOut, 0, sizeof(T));
    } else {
      NCCLCHECK(ncclCalloc(&tail, 1));
      *dataOut = &tail->data;
      cursor = head = tail;
    }
    if (tail->next == NULL) {
      NCCLCHECK(ncclCalloc(&tail->next, 1));
    }
    tail = tail->next;
    n += 1;
    return ncclSuccess;
  }

  T* begin() {
    if (head == NULL || head == tail) return NULL;
    cursor = head->next;
    return &head->data;
  }

  // Get next element from the list during an iteration
  T* getNext() {
    // tail always points to the next element to be enqueued
    // hence does not contain valid data
    if (cursor == NULL || cursor == tail) return NULL;
    T* rv = &cursor->data;
    cursor = cursor->next;
    return rv;
  }

  T* peakNext() {
    if (cursor == NULL || cursor == tail) return NULL;
    return &cursor->data;
  }

  // Recycle the list without freeing the space
  void recycle() {
    tail = cursor = head;
    n = 0;
  }

  ~ncclRecyclableList() {
    while (head != NULL) {
      struct ncclListElem<T>* temp = head;
      head = head->next;
      ncclFree(temp);
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

struct ncclMemoryStack {
  struct Hunk {
    struct Hunk* above; // reverse stack pointer
    size_t size; // size of this allocation (including this header struct)
  };
  struct Unhunk { // proxy header for objects allocated out-of-hunk
    struct Unhunk* next;
    void* obj;
  };
  struct Frame {
    struct Hunk* hunk; // top of non-empty hunks
    uintptr_t bumper, end; // points into top hunk
    struct Unhunk* unhunks;
    struct Frame* below;
  };

  static void* allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align);
  static void* allocate(struct ncclMemoryStack* me, size_t size, size_t align);

  struct Hunk stub;
  struct Frame topFrame;
};

inline void ncclMemoryStackConstruct(struct ncclMemoryStack* me) {
  me->stub.above = nullptr;
  me->stub.size = 0;
  me->topFrame.hunk = &me->stub;
  me->topFrame.bumper = 0;
  me->topFrame.end = 0;
  me->topFrame.unhunks = nullptr;
  me->topFrame.below = nullptr;
}

inline void* ncclMemoryStack::allocate(struct ncclMemoryStack* me, size_t size, size_t align) {
  uintptr_t o = (me->topFrame.bumper + align-1) & -uintptr_t(align);
  void* obj;
  if (__builtin_expect(o + size <= me->topFrame.end, true)) {
    me->topFrame.bumper = o + size;
    obj = reinterpret_cast<void*>(o);
  } else {
    obj = allocateSpilled(me, size, align);
  }
  return obj;
}

inline void* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t size, size_t align) {
  void *obj = ncclMemoryStack::allocate(me, size, align);
  memset(obj, 0, size);
  return obj;
}

template<typename T>
inline T* ncclMemoryStackAlloc(struct ncclMemoryStack* me, size_t n) {
  void *obj = ncclMemoryStack::allocate(me, n*sizeof(T), alignof(T));
  memset(obj, 0, n*sizeof(T));
  return (T*)obj;
}

template<typename Header, typename Element>
inline Header* ncclMemoryStackAllocInlineArray(struct ncclMemoryStack* me, size_t nElt) {
  size_t size = sizeof(Header);
  size = (size + alignof(Element)-1) & -alignof(Element);
  size += nElt*sizeof(Element);
  size_t align = alignof(Header) < alignof(Element) ? alignof(Element) : alignof(Header);
  void *obj = ncclMemoryStack::allocate(me, size, align);
  memset(obj, 0, size);
  return (Header*)obj;
}

inline void ncclMemoryStackPush(struct ncclMemoryStack* me) {
  using Frame = ncclMemoryStack::Frame;
  Frame tmp = me->topFrame;
  Frame* snapshot = (Frame*)ncclMemoryStack::allocate(me, sizeof(Frame), alignof(Frame));
  *snapshot = tmp; // C++ struct assignment
  me->topFrame.unhunks = nullptr;
  me->topFrame.below = snapshot;
}

inline void ncclMemoryStackPop(struct ncclMemoryStack* me) {
  ncclMemoryStack::Unhunk* un = me->topFrame.unhunks;
  while (un != nullptr) {
    free(un->obj);
    un = un->next;
  }
  me->topFrame = *me->topFrame.below; // C++ struct assignment
}

////////////////////////////////////////////////////////////////////////////////

struct ncclMemoryPool {
  struct Cell {
    Cell *next;
  };
  struct Cell* head;
  struct Cell* tail; // meaningful only when head != nullptr
};

inline void ncclMemoryPoolConstruct(struct ncclMemoryPool* me) {
  me->head = nullptr;
}

template<typename T>
inline T* ncclMemoryPoolAlloc(struct ncclMemoryPool* me, struct ncclMemoryStack* backing) {
  using Cell = ncclMemoryPool::Cell;
  Cell* cell;
  if (__builtin_expect(me->head != nullptr, true)) {
    cell = me->head;
    me->head = cell->next;
  } else {
    // Use the internal allocate() since it doesn't memset to 0 yet.
    size_t cellSize = std::max(sizeof(Cell), sizeof(T));
    size_t cellAlign = std::max(alignof(Cell), alignof(T));
    cell = (Cell*)ncclMemoryStack::allocate(backing, cellSize, cellAlign);
  }
  memset(cell, 0, sizeof(T));
  return reinterpret_cast<T*>(cell);
}

template<typename T>
inline void ncclMemoryPoolFree(struct ncclMemoryPool* me, T* obj) {
  using Cell = ncclMemoryPool::Cell;
  Cell* cell = reinterpret_cast<Cell*>(obj);
  cell->next = me->head;
  if (me->head == nullptr) me->tail = cell;
  me->head = cell;
}

constexpr size_t operator""_KB(unsigned long long val) {
  return val * 1024;
}

constexpr size_t operator""_MB(unsigned long long val) {
  return val * 1024 * 1024;
}

constexpr size_t operator""_GB(unsigned long long val) {
  return val * 1024 * 1024 * 1024;
}

////////////////////////////////////////////////////////////////////////////////
// Hash function for pointer types (shared by address map and shadow pool)
uint64_t ncclHashPointer(int hbits, void* key);

////////////////////////////////////////////////////////////////////////////////
// Intrusive address map implementation (avoids per-entry allocations)
//
// An intrusive hash map that stores next-pointers directly in user objects.
// Avoids separate malloc per entry (only allocates bucket table).
// Uses C++ templates for type safety with type-erased implementation.
//
// Construction: POD type, zero-initialization for static/global instances.
// Destruction: Map auto-cleans when last entry is removed (table freed).
// Thread Safety: NOT thread-safe, caller must provide external synchronization.

// Untyped internal structure
struct ncclIntruAddressMap_untyped {
  int hbits;  // log2 of table size
  int count;  // number of entries
  void** table;
};

// Typed wrapper (uses composition for C compatibility)
template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
struct ncclIntruAddressMap {
  static_assert(sizeof(Key) <= sizeof(uintptr_t),
    "ncclIntruAddressMap: Key type size must be <= sizeof(uintptr_t).");
  ncclIntruAddressMap_untyped base;
};

// Destructor (optional - only needed if abandoning a non-empty map)
template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
static inline void ncclIntruAddressMapDestruct(struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map) {
  if (map->base.table != nullptr) {
    free(map->base.table);
    map->base.table = nullptr;
  }
  map->base.hbits = 0;
  map->base.count = 0;
}

// Internal untyped function prototypes
ncclResult_t ncclIntruAddressMapInsert_untyped(
  struct ncclIntruAddressMap_untyped* map,
  int keySize, int keyFieldOffset, int nextFieldOffset,
  uintptr_t key, void* object);

ncclResult_t ncclIntruAddressMapFind_untyped(
  struct ncclIntruAddressMap_untyped* map,
  int keySize, int keyFieldOffset, int nextFieldOffset,
  uintptr_t key, void** object);

ncclResult_t ncclIntruAddressMapRemove_untyped(
  struct ncclIntruAddressMap_untyped* map,
  int keySize, int keyFieldOffset, int nextFieldOffset,
  uintptr_t key);

// Typed template implementations (type-erasing wrappers)
template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
static inline ncclResult_t ncclIntruAddressMapInsert(
    struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map,
    Key key, Obj* object) {
  Obj dummy{};
  int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
  int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
  return ncclIntruAddressMapInsert_untyped(
    // keySize == sizeof(Key) is exactly the key width to read (bounded by the static_assert above); not a mismatch.
    // coverity[sizeof_mismatch]
    &map->base, (int)sizeof(Key), keyFieldOffset, nextFieldOffset,
    reinterpret_cast<uintptr_t>(key), object);
}

template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
static inline ncclResult_t ncclIntruAddressMapFind(
    struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map,
    Key key, Obj** object) {
  Obj dummy{};
  int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
  int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
  void* tmp;
  ncclResult_t ret = ncclIntruAddressMapFind_untyped(
    // keySize == sizeof(Key) is exactly the key width to read (bounded by the static_assert above); not a mismatch.
    // coverity[sizeof_mismatch]
    &map->base, (int)sizeof(Key), keyFieldOffset, nextFieldOffset,
    reinterpret_cast<uintptr_t>(key), &tmp);
  *object = (Obj*)tmp;
  return ret;
}

template<typename Obj, typename Key, Key Obj::*keyField, Obj* Obj::*nextField>
static inline ncclResult_t ncclIntruAddressMapRemove(
    struct ncclIntruAddressMap<Obj, Key, keyField, nextField>* map,
    Key key) {
  Obj dummy{};
  int keyFieldOffset = (char*)&(dummy.*keyField) - (char*)&dummy;
  int nextFieldOffset = (char*)&(dummy.*nextField) - (char*)&dummy;
  return ncclIntruAddressMapRemove_untyped(
    // keySize == sizeof(Key) is exactly the key width to read (bounded by the static_assert above); not a mismatch.
    // coverity[sizeof_mismatch]
    &map->base, (int)sizeof(Key), keyFieldOffset, nextFieldOffset,
    reinterpret_cast<uintptr_t>(key));
}

#endif
