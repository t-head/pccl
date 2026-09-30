//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include <pthread.h>
#include "core.h"
#include "u2mm.h"

#define MB (1UL * 1024 * 1024)
#define GB (1024 * MB)

pthread_mutex_t nccl_u2mm_lock = PTHREAD_MUTEX_INITIALIZER;
static bool u2mm_init_done;

ncclResult_t nccl_u2mm_init(void) {
  ncclResult_t res = ncclSuccess;
  int ret = 0;
  u2mm_alloc_config_t config;
  u2mm_log_level level = U2MM_LOG_LEVEL_WARN;
  const char* levelStr = NULL;
  pthread_mutex_lock(&nccl_u2mm_lock);
  if (u2mm_init_done)
    goto out;

  NCCLCHECKGOTO(wrap_u2mm_symbols(), res, out);
  NCCLCHECKGOTO(wrap_numa_symbols(), res, out);

  levelStr = pcclGetEnvWithDualPrefix("PCCL_U2MM_LOG_LEVEL");
  if (levelStr) {
    INFO(NCCL_INIT|NCCL_ENV, "PCCL_U2MM_LOG_LEVEL by environment to %s", levelStr);
    if (strcasecmp(levelStr, "OFF") == 0) {
      level = U2MM_LOG_LEVEL_OFF;
    } else if (strcasecmp(levelStr, "FATAL") == 0) {
      level = U2MM_LOG_LEVEL_FATAL;
    } else if (strcasecmp(levelStr, "ERROR") == 0) {
      level = U2MM_LOG_LEVEL_ERROR;
    } else if (strcasecmp(levelStr, "WARN") == 0) {
      level = U2MM_LOG_LEVEL_WARN;
    } else if (strcasecmp(levelStr, "INFO") == 0) {
      level = U2MM_LOG_LEVEL_INFO;
    } else if (strcasecmp(levelStr, "DEBUG") == 0) {
      level = U2MM_LOG_LEVEL_DEBUG;
    } else if (strcasecmp(levelStr, "TRACE") == 0) {
      level = U2MM_LOG_LEVEL_TRACE;
    } else {
      INFO(NCCL_INIT, "invalid u2mm log level %s, use WARN by default. please set PCCL_U2MM_LOG_LEVEL to OFF,FATAL,ERROR,WARN,INFO,DEBUG,TRACE", levelStr);
    }
  }

  ret = u2mm_set_log_level(level);
  if (ret != 0) {
    INFO(NCCL_INIT, "WARN: failed to set u2mm log level to %d", level);
  }

  config = u2mm_get_default_alloc_config();
  config.dev_flags |= U2MM_DEV_FLAG_PPU;
  config.dev_num = 16;
  ret = u2mm_alloc_init(&config);
  if (ret != 0) {
    WARN("failed to u2mm alloc init, ret %d error %d (%s)", ret, errno, strerror(errno));
    res = ncclSystemError;
    goto out;
  }

  u2mm_init_done = true;

out:
  pthread_mutex_unlock(&nccl_u2mm_lock);
  return res;
}
