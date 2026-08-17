/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_U2MMWRAP_H_
#define NCCL_U2MMWRAP_H_

#include <stdint.h>
#include <stddef.h>
#include "pccl_internal.h"

#define U2MM_PAGE_FLAG_UNKNOWN          (1UL << 24)
#define U2MM_PAGE_FLAG_4K               (1UL << 25)
#define U2MM_PAGE_FLAG_64K              (1UL << 26)
#define U2MM_PAGE_FLAG_2M               (1UL << 27)
#define U2MM_PAGE_FLAG_1G               (1UL << 28)

#define U2MM_DIRECT_ALLOC_FLAG          (1UL << 32)

// MEKY: "MKEY@version:pid:fd:offset:len:mz_len@"
#define U2MM_MKEY_MAX_LEN               128
#define U2MM_MEM_SHARED_FLAG            (1UL << 33)

#define U2MM_DEV_FLAG_HOST              (1UL << 16)
#define U2MM_DEV_FLAG_GPU               (1UL << 17)
#define U2MM_DEV_FLAG_PPU               (1UL << 18)
#define U2MM_DEV_FLAG_UNKNOWN           (1UL << 19)

// Legacy callbacks
typedef void *(*memzone_alloc_post_cb)(void *mz, size_t mz_size);
typedef int (*memzone_release_pre_cb)(void *ibv_mr);

// general callback with user_data, allow user to pass more parameters, such as pd
typedef int (*post_alloc_cb)(void *user_data, void *addr, size_t len, void **mr);
typedef int (*pre_release_cb)(void *user_data, void *mr);

typedef struct _u2mm_memzone_hook {
  /* NOTICE(IMPORTANT): DO NOT set both two callbacks at the same time ! */
  /* legacy mode: this two callbacks are customized by kuafu (only used in hugepage domain for now) */
  memzone_alloc_post_cb      alloc_cb;                    /* call it after allocate a memzone */
  memzone_release_pre_cb     release_cb;                  /* call it before release memzone */

  /* general callbacks for post_alloc and pre_release, */
  post_alloc_cb              general_post_alloc_cb;               /* general post alloc cb */
  pre_release_cb             general_pre_release_cb;              /* general pre release cb*/

  void                       *user_data;                  /* user params such as pd */
} u2mm_memzone_callback;

enum u2mm_log_level {
  U2MM_LOG_LEVEL_OFF = 1,
  U2MM_LOG_LEVEL_FATAL,
  U2MM_LOG_LEVEL_ERROR,
  U2MM_LOG_LEVEL_WARN,
  U2MM_LOG_LEVEL_INFO,
  U2MM_LOG_LEVEL_DEBUG,
  U2MM_LOG_LEVEL_TRACE,
  U2MM_LOG_LEVEL_ALL
};

typedef struct u2mm_alloc_config {
  uint64_t dev_flags;  //indicate which dev(s) needs to be initialized
  uint32_t dev_num;
  uint64_t memzone_size;
  u2mm_memzone_callback mz_cb;
} u2mm_alloc_config_t;

extern ncclResult_t wrap_u2mm_symbols(void);
extern int (*u2mm_set_log_level)(enum u2mm_log_level level);
extern int (*u2mm_alloc_init)(u2mm_alloc_config_t *config);
extern u2mm_alloc_config_t (*u2mm_get_default_alloc_config)(void);
extern void *(*u2mm_alloc)(size_t size, uint64_t flags);
extern int (*u2mm_free)(void *addr);
extern void (*u2mm_alloc_cleanup)(void);
extern int (*u2mm_shm_share)(void *addr, size_t len, char **mkey);
extern int (*u2mm_shm_unshare)(char *mkey);
extern int (*u2mm_shm_map)(const char *mkey, void **addr, size_t len);
extern int (*u2mm_shm_unmap)(void *addr);
#endif
