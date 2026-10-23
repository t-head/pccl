//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#ifndef COMMON_LOGGER_H_
#define COMMON_LOGGER_H_

#include <cstdio>
#include <getopt.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

#define HOST_NAME_LEN 256
#define PRINT_BUFF_SIZE 10240
static char hostName[HOST_NAME_LEN];

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testHggcError = 2,
  testPcclError = 3,
  testAcomputeError = 4,
  testRDMAError = 5
} testResult_t;

typedef enum {
  LOG_ERROR = 0,
  LOG_WARN = 1,
  LOG_INFO = 2,
  LOG_DEBUG = 3
} logLevel;

#define INFO(...)                                                              \
  { LOG_INFO(__VA_ARGS__) }

#define MINFO(IS_MULTI, RANK, ...)                                             \
  {                                                                            \
    if (IS_MULTI) {                                                            \
      MLOG_INFO(RANK, __VA_ARGS__)                                             \
    } else {                                                                   \
      LOG_INFO(__VA_ARGS__)                                                    \
    }                                                                          \
  }

#define MINFO_RANK0(IS_MULTI, RANK, ...)                                       \
  {                                                                            \
    if (RANK == 0) {                                                           \
      MINFO(IS_MULTI, RANK, __VA_ARGS__)                                       \
    }                                                                          \
  }

#define DEBUG(...)                                                             \
  { LOG_DEBUG(__VA_ARGS__) }

#define MDEBUG(IS_MULTI, RANK, ...)                                            \
  {                                                                            \
    if (IS_MULTI) {                                                            \
      MLOG_DEBUG(RANK, __VA_ARGS__)                                            \
    } else {                                                                   \
      LOG_DEBUG(__VA_ARGS__)                                                   \
    }                                                                          \
  }

#define MDEBUG_RANK0(IS_MULTI, RANK, ...)                                      \
  {                                                                            \
    if (RANK == 0) {                                                           \
      MDEBUG(IS_MULTI, RANK, __VA_ARGS__)                                      \
    }                                                                          \
  }

#define WARN(...)                                                              \
  { LOG_WARNING(__VA_ARGS__) }

#define ERROR(...)                                                             \
  { LOG_ERROR(__VA_ARGS__) }

static void getHostName(char *hostName, int maxlen) {
  gethostname(hostName, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostName[i] == '.') {
      hostName[i] = '\0';
      return;
    }
  }
}

#define TESTCHECK(cmd)                                                           \
  do {                                                                           \
    testResult_t err = cmd;                                                      \
    if (err != testSuccess) {                                                    \
      LOG_ERROR("Test failure %s:%d", __FILE__, __LINE__)                        \
      exit(testHggcError);                                                       \
    }                                                                            \
  } while (0)

#define HGCHECK(cmd)                                                             \
  do {                                                                           \
    HGresult err = cmd;                                                          \
    if (err != HGGC_SUCCESS) {                                                   \
      const char *err_string;                                                    \
      hgGetErrorString(err, &err_string);                                        \
      LOG_ERROR("Test HGGC failure %s:%d '%s'", __FILE__, __LINE__, err_string)  \
      exit(testHggcError);                                                       \
    }                                                                            \
  } while (0)

#define HGGCCHECK(cmd)                                                         \
  do {                                                                         \
    hggcError_t err = cmd;                                                     \
    if (err != hggcSuccess) {                                                  \
      LOG_ERROR("Test HGGC failure %s:%d '%s'", __FILE__, __LINE__,            \
                hggcGetErrorString(err))                                       \
      exit(testHggcError);                                                     \
    }                                                                          \
  } while (0)

#define PCCLCHECK(cmd)                                                         \
  do {                                                                         \
    pcclResult_t res = cmd;                                                    \
    if (res != pcclSuccess && res != pcclInProgress) {                         \
      LOG_ERROR("Test PCCL failure %s:%d '%s'", __FILE__, __LINE__,            \
                pcclGetErrorString(res))                                       \
      exit(testPcclError);                                                     \
    }                                                                          \
  } while (0)

#define MPICHECK(cmd) do {                                                     \
  int e = cmd;                                                                 \
  if( e != MPI_SUCCESS ) {                                                     \
    printf("Failed: MPI error %s:%d '%d'\n",                                   \
        __FILE__,__LINE__, e);                                                 \
    exit(EXIT_FAILURE);                                                        \
  }                                                                            \
} while(0)

// Macro for checking hggc errors following a hggc launch or api call
#define hggcCheckError()                                                       \
  {                                                                            \
    hggcError_t e = hggcGetLastError();                                        \
    if (e != hggcSuccess) {                                                    \
      LOG_ERROR("HGGC failure %s:%d: '%s'", __FILE__, __LINE__,                \
                hggcGetErrorString(e));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

inline void format_buffer(char *buffer, const size_t length, const char *fmt,
                          ...) {
  va_list vargs;
  va_start(vargs, fmt);
  (void)vsnprintf(buffer, length, fmt, vargs);
  va_end(vargs);
}

#ifdef TRACE
// single process debug level infos
#define LOG_INFO(...)                                                          \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    pid_t pid = getpid();                                                      \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("[%s:%d][%s:%d] %s\n", hostName, pid, __FILE__, __LINE__, buffer);  \
  }
#define LOG_WARNING(...)                                                       \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    pid_t pid = getpid();                                                      \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("[%s:%d][%s:%d] \033[33m%s\033[0m\n", hostName, pid, __FILE__,      \
           __LINE__, buffer);                                                  \
  }
#define LOG_ERROR(...)                                                         \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    pid_t pid = getpid();                                                      \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("[%s:%d][%s:%d] \033[31m%s\033[0m\n", hostName, pid, __FILE__,      \
           __LINE__, buffer);                                                  \
  }
#define LOG_DEBUG(...)                                                         \
  { LOG_INFO(__VA_ARGS__) }
// multi-process debug level infos
#define MLOG_INFO(RANK, ...)                                                   \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    pid_t pid = getpid();                                                      \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("[%s:%d][%s:%d][%d] %s\n", hostName, pid, __FILE__, __LINE__, RANK, \
           buffer);                                                            \
  }
#define MLOG_WARNING(RANK, ...)                                                \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    pid_t pid = getpid();                                                      \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("[%s:%d][%s:%d][%d] \033[33m%s\033[0m\n", hostName, pid, __FILE__,  \
           __LINE__, RANK, buffer);                                            \
  }
#define MLOG_ERROR(RANK, ...)                                                  \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    pid_t pid = getpid();                                                      \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("[%s:%d][%s:%d][%d] \033[31m%s\033[0m\n", hostName, pid, __FILE__,  \
           __LINE__, RANK, buffer);                                            \
  }
#define MLOG_DEBUG(RANK, ...)                                                  \
  { MLOG_INFO(RANK, __VA_ARGS__) }
#else
#define LOG_INFO(...)                                                          \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("%s\n", buffer);                                                    \
  }
#define LOG_WARNING(...)                                                       \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("\033[33mWARNING %s:%d: %s\033[0m\n", __FILE__, __LINE__, buffer);  \
  }
#define LOG_ERROR(...)                                                         \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("\033[31m%s\033[0m\n", buffer);                                     \
  }
#define LOG_DEBUG(...)
#define MLOG_INFO(RANK, ...)                                                   \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    pid_t pid = getpid();                                                      \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("[%s:%d][%d] %s\n", hostName, pid, RANK, buffer);                   \
  }
#define MLOG_WARNING(RANK, ...)                                                \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    pid_t pid = getpid();                                                      \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("[%s:%d][%d] \033[33m%s\033[0m\n", hostName, pid, RANK, buffer);    \
  }
#define MLOG_ERROR(RANK, ...)                                                  \
  {                                                                            \
    char buffer[PRINT_BUFF_SIZE];                                              \
    pid_t pid = getpid();                                                      \
    format_buffer(buffer, sizeof(buffer), __VA_ARGS__);                        \
    printf("[%s:%d][%d] \033[31m%s\033[0m\n", hostName, pid, RANK, buffer);    \
  }
#define MLOG_DEBUG(RANK, ...)
#endif

static uint64_t getHash(const char *string, size_t n) {
  // Based on DJB2a, result = result * 33 ^ char
  uint64_t result = 5381;
  for (size_t c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static uint64_t getHostHash(const char *hostName) {
  char hostHash[1024];

  // Fall back is the hostname if something fails
  (void)strncpy(hostHash, hostName, sizeof(hostHash));
  int offset = strlen(hostHash);

  FILE *file = fopen(HOSTID_FILE, "r");
  if (file != NULL) {
    char *p;
    if (fscanf(file, "%ms", &p) == 1) {
      strncpy(hostHash + offset, p, sizeof(hostHash) - offset - 1);
      free(p);
    }
  }
  fclose(file);

  // Make sure the string is terminated
  hostHash[sizeof(hostHash) - 1] = '\0';

  return getHash(hostHash, strlen(hostHash));
}

#endif
