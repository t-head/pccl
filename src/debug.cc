/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include <stdlib.h>
#include <stdarg.h>
#include <ctime>
#include <sys/time.h>
#include <string>

int ncclDebugLevel = -1;
static uint32_t ncclDebugTimestampLevels = 0;     // bitmaps of levels that have timestamps turned on
static char ncclDebugTimestampFormat[256];        // with space for subseconds
static int ncclDebugTimestampSubsecondsStart;     // index where the subseconds starts
static uint64_t ncclDebugTimestampMaxSubseconds;  // Max number of subseconds plus 1, used in duration ratio
static int ncclDebugTimestampSubsecondDigits;     // Number of digits to display
thread_local int ncclDebugNoWarn = 0;
int pcclDebugDev = -1;
uint64_t ncclDebugMask = NCCL_INIT; // Default debug sub-system mask is INIT
FILE *ncclDebugFile = stdout;
thread_local FILE *ncclPerCommHashDebugFile = nullptr;
pthread_mutex_t ncclDebugLock = PTHREAD_MUTEX_INITIALIZER;
static bool error_yield = false;
char ncclLastError[1024] = ""; // Global string for the last error in human readable form
constexpr uint64_t COMM_HASH_UNIT = 0xffffffffffffffffl;

static FILE* getDebugFile(const char* ncclDebugFileEnv, uint64_t commHash) {
  FILE *file = NULL;
  if (ncclDebugLevel > NCCL_LOG_VERSION && ncclDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX+1] = "";
    char *dfn = debugFn;
    while (ncclDebugFileEnv[c] != '\0' && (dfn - debugFn) < PATH_MAX) {
      if (ncclDebugFileEnv[c++] != '%') {
        *dfn++ = ncclDebugFileEnv[c-1];
        continue;
      }
      switch (ncclDebugFileEnv[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          char hostname[1024];
          getHostName(hostname, 1024, '.');
          dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%d", getpid());
          break;
        case 'c': // %c = commHash
          if (commHash != COMM_HASH_UNIT) {
            dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%lu", commHash);
          }
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          if ((dfn - debugFn) < PATH_MAX) {
            *dfn++ = ncclDebugFileEnv[c-1];
          }
          break;
      }
      if ((dfn - debugFn) > PATH_MAX) {
        // snprintf wanted to overfill the buffer: set dfn to the end
        // of the buffer (for null char) and it will naturally exit
        // the loop.
        dfn = debugFn + PATH_MAX;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      file = fopen(debugFn, "a+");
    }
  }
  return file;
}

void ncclPerCommHashDebugInit(uint64_t commHash) {
  if (ncclPerCommHashDebugFile) {
    return;
  }
  const char* ncclPerCommHashDebugFileEnv = pcclGetEnvWithDualPrefix("PCCL_PER_COMM_HASH_DEBUG_FILE");
  FILE* file = getDebugFile(ncclPerCommHashDebugFileEnv, commHash);

  if (file != NULL) {
    ncclPerCommHashDebugFile = file;
  }
}

void ncclPerCommHashDebugClose() {
  if (ncclPerCommHashDebugFile == nullptr) {
    return;
  }
  pthread_mutex_lock(&ncclDebugLock);
  if (ncclPerCommHashDebugFile != nullptr) {
    fclose(ncclPerCommHashDebugFile);
    ncclPerCommHashDebugFile = nullptr;
  }
  pthread_mutex_unlock(&ncclDebugLock);
}

void ncclDebugInit() {
  pthread_mutex_lock(&ncclDebugLock);
  if (ncclDebugLevel != -1) { pthread_mutex_unlock(&ncclDebugLock); return; }
  const char* nccl_debug = pcclGetEnvWithDualPrefix("PCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NCCL_LOG_NONE;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = NCCL_LOG_VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = NCCL_LOG_WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = NCCL_LOG_ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    ncclDebugLevel = NCCL_LOG_TRACE;
  } else if (strcasecmp(nccl_debug, "VERBOSE") == 0) {
    ncclDebugLevel = PCCL_LOG_VERBOSE;
  } else {
    ncclDebugLevel = NCCL_LOG_NONE;
  }

  /* Parse the NCCL_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  const char* ncclDebugSubsysEnv = pcclGetEnvWithDualPrefix("PCCL_DEBUG_SUBSYS");
  if (ncclDebugSubsysEnv != NULL) {
    int invert = 0;
    if (ncclDebugSubsysEnv[0] == '^') { invert = 1; ncclDebugSubsysEnv++; }
    ncclDebugMask = invert ? ~0ULL : 0ULL;
    char *ncclDebugSubsys = strdup(ncclDebugSubsysEnv);
    char *subsys = strtok(ncclDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = NCCL_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = NCCL_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = NCCL_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = NCCL_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = NCCL_NET;
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = NCCL_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = NCCL_TUNING;
      } else if (strcasecmp(subsys, "ENV") == 0) {
        mask = NCCL_ENV;
      } else if (strcasecmp(subsys, "ALLOC") == 0) {
        mask = NCCL_ALLOC;
      } else if (strcasecmp(subsys, "EXT") == 0) {
        mask = NCCL_EXT;
      } else if (strcasecmp(subsys, "CALL") == 0) {
        mask = NCCL_CALL;
      } else if (strcasecmp(subsys, "PROXY") == 0) {
        mask = NCCL_PROXY;
      } else if (strcasecmp(subsys, "DUMP") == 0) {
        mask = NCCL_DUMP;
      } else if (strcasecmp(subsys, "PROFILE") == 0) {
        mask = NCCL_PROFILE;
      } else if (strcasecmp(subsys, "REG") == 0) {
        mask = NCCL_REG;
      } else if (strcasecmp(subsys, "RAS") == 0) {
        mask = NCCL_RAS;
      } else if (strcasecmp(subsys, "INSPECTOR") == 0) {
        mask = NCCL_INSPECTOR;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = NCCL_ALL;
      }
      if (mask) {
        if (invert) ncclDebugMask &= ~mask; else ncclDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(ncclDebugSubsys);
  }

  // Determine which debug levels will have timestamps.
  const char* timestamps = pcclGetEnvWithDualPrefix("PCCL_DEBUG_TIMESTAMP_LEVELS");
  if (timestamps == nullptr) {
    ncclDebugTimestampLevels = (1<<PCCL_LOG_ERROR) | (1<<NCCL_LOG_WARN);
  } else {
    int invert = 0;
    if (timestamps[0] == '^') { invert = 1; ++timestamps; }
    ncclDebugTimestampLevels = invert ? ~0U : 0U;
    char *timestampsDup = strdup(timestamps);
    char *level = strtok(timestampsDup, ",");
    while (level != NULL) {
      uint32_t mask = 0;
      if (strcasecmp(level, "ALL") == 0) {
        mask = ~0U;
      } else if (strcasecmp(level, "ERROR") == 0) {
        mask = (1<<PCCL_LOG_ERROR);
      } else if (strcasecmp(level, "VERSION") == 0) {
        mask = (1<<NCCL_LOG_VERSION);
      } else if (strcasecmp(level, "WARN") == 0) {
        mask = (1<<NCCL_LOG_WARN);
      } else if (strcasecmp(level, "INFO") == 0) {
        mask = (1<<NCCL_LOG_INFO);
      } else if (strcasecmp(level, "VERBOSE") == 0) {
        mask = (1<<PCCL_LOG_VERBOSE);
      } else if (strcasecmp(level, "ABORT") == 0) {
        mask = (1<<NCCL_LOG_ABORT);
      } else if (strcasecmp(level, "TRACE") == 0) {
        mask = (1<<NCCL_LOG_TRACE);
      } else {
        // Silently fail.
      }
      if (mask) {
        if (invert) ncclDebugTimestampLevels &= ~mask;
        else ncclDebugTimestampLevels |= mask;
      }
      level = strtok(NULL, ",");
    }
    free(timestampsDup);
  }

  // Store a copy of the timestamp format with space for the subseconds, if used.
  const char* tsFormat = pcclGetEnvWithDualPrefix("PCCL_DEBUG_TIMESTAMP_FORMAT");
  if (tsFormat == nullptr) tsFormat = "[%F %T] ";
  ncclDebugTimestampSubsecondsStart = -1;
  // Find where the subseconds are in the format.
  for (int i=0; tsFormat[i] != '\0'; ++i) {
    if (tsFormat[i]=='%' && tsFormat[i+1]=='%') { // Next two chars are "%"
      // Skip the next character, too, and restart checking after that.
      ++i;
      continue;
    }
    if (tsFormat[i]=='%' &&                               // Found a percentage
        ('1' <= tsFormat[i+1] && tsFormat[i+1] <= '9') && // Next char is a digit between 1 and 9 inclusive
        tsFormat[i+2]=='f'                                // Two characters later is an "f"
        ) {
      constexpr int replaceLen = sizeof("%Xf") - 1;
      ncclDebugTimestampSubsecondDigits = tsFormat[i+1] - '0';
      if (ncclDebugTimestampSubsecondDigits + strlen(tsFormat) - replaceLen > sizeof(ncclDebugTimestampFormat) - 1) {
        // Won't fit; fall back on the default.
        break;
      }
      ncclDebugTimestampSubsecondsStart = i;
      ncclDebugTimestampMaxSubseconds = 1;

      memcpy(ncclDebugTimestampFormat, tsFormat, i);
      for (int j=0; j<ncclDebugTimestampSubsecondDigits; ++j) {
        ncclDebugTimestampFormat[i+j] = ' ';
        ncclDebugTimestampMaxSubseconds *= 10;
      }
      strcpy(ncclDebugTimestampFormat+i+ncclDebugTimestampSubsecondDigits, tsFormat+i+replaceLen);
      break;
    }
  }
  if (ncclDebugTimestampSubsecondsStart == -1) {
    if (strlen(tsFormat) < sizeof(ncclDebugTimestampFormat)) {
      strcpy(ncclDebugTimestampFormat, tsFormat);
    } else {
      strcpy(ncclDebugTimestampFormat, "[%F %T] ");
    }
  }

  // Replace underscore with spaces... it is hard to put spaces in command line parameters.
  for (int i=0; ncclDebugTimestampFormat[i] != '\0'; ++i) {
    if (ncclDebugTimestampFormat[i]=='_') ncclDebugTimestampFormat[i] = ' ';
  }

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* ncclDebugFileEnv = pcclGetEnvWithDualPrefix("PCCL_DEBUG_FILE");
  FILE* file = getDebugFile(ncclDebugFileEnv, COMM_HASH_UNIT);

  if (file != NULL) {
    ncclDebugFile = file;
  }

  /*
   * Parse PCCL_ERROR_YIELD
   */
  const char* pccl_error_yield = pcclGetEnvWithDualPrefix("PCCL_ERROR_YIELD");
  if ((pccl_error_yield != NULL) && (strcmp(pccl_error_yield, "1") == 0)) {
    error_yield = true;
  }

  /*
   * Parse PCCL_DEBUG_DEV
   */
  const char* pcclDebugDevStr = pcclGetEnvWithDualPrefix("PCCL_DEBUG_DEV");
  if (pcclDebugDevStr != NULL) {
    pcclDebugDev = atoi(pcclDebugDevStr);
  }

#ifdef ENABLE_TRACE
  ncclEpoch = std::chrono::high_resolution_clock::now();
#endif
  pthread_mutex_unlock(&ncclDebugLock);
}

static void ncclDebugErrorYield() {
  while(error_yield) sched_yield();
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *file, const char *func, int line, bool record, const char *fmt, ...) {
  if (ncclDebugLevel == -1) ncclDebugInit();
  if (ncclDebugNoWarn != 0 && (level == PCCL_LOG_ERROR || level == NCCL_LOG_WARN)) { level = NCCL_LOG_INFO; flags = ncclDebugNoWarn; }
  if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) return;

  // Save the last error (WARN) as a human readable string
  if (record) {
    if ((level == PCCL_LOG_ERROR) || (level == NCCL_LOG_WARN ))
    {
      pthread_mutex_lock(&ncclDebugLock);
      va_list vargs;
      va_start(vargs, fmt);
      (void) vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vargs);
      va_end(vargs);
      pthread_mutex_unlock(&ncclDebugLock);
    }
  }

  // Gather the rank information. This can take > 1us so we want to make sure
  // we only do it when needed.
  int hggcDev = 0;
  hggcGetDevice(&hggcDev);
  if (pcclDebugDev >= 0 && pcclDebugDev != hggcDev) {
    return;
  }

  char hostname[1024];
  getHostName(hostname, 1024, '.');
  int pid = getpid();
  int tid = syscall(SYS_gettid);

  char buffer[LARGE_LOG_SIZE];
  char location[1024];
  if (func && func[0])
    snprintf(location, sizeof(location), "[%s:%d %s]", file ? file : "?", line, func);
  else
    snprintf(location, sizeof(location), "[%s:%d]", file ? file : "?", line);

  size_t len = 0;
  pthread_mutex_lock(&ncclDebugLock);

  // ERRORs come with an extra newline at the beginning.
  if (level == PCCL_LOG_ERROR) {
    buffer[len++] = '\n';
  };

  // Add the timestamp to the buffer if they are turned on for this level.
  if (ncclDebugTimestampLevels & (1<<level)) {
    if (ncclDebugTimestampFormat[0] != '\0') {
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);   // clock_gettime failure should never happen
      std::tm nowTm;
      localtime_r(&ts.tv_sec, &nowTm);

      // Add the subseconds portion if it is part of the format.
      char localTimestampFormat[sizeof(ncclDebugTimestampFormat)];
      const char* pformat = ncclDebugTimestampFormat;
      if (ncclDebugTimestampSubsecondsStart != -1) {
        pformat = localTimestampFormat;   // Need to use the local version which has subseconds
        memcpy(localTimestampFormat, ncclDebugTimestampFormat, ncclDebugTimestampSubsecondsStart);
        snprintf(localTimestampFormat + ncclDebugTimestampSubsecondsStart,
                 ncclDebugTimestampSubsecondDigits+1,
                 "%0*ld", ncclDebugTimestampSubsecondDigits,
                 ts.tv_nsec / (1000000000UL/ncclDebugTimestampMaxSubseconds));
        strcpy(    localTimestampFormat+ncclDebugTimestampSubsecondsStart+ncclDebugTimestampSubsecondDigits,
               ncclDebugTimestampFormat+ncclDebugTimestampSubsecondsStart+ncclDebugTimestampSubsecondDigits);
      }

      // Format the time. If it runs out of space, fall back on a simpler format.
      int adv = std::strftime(buffer+len, sizeof(buffer)-len, pformat, &nowTm);
      if (adv==0 && ncclDebugTimestampFormat[0] != '\0') {
        // Ran out of space. Fall back on the default. This should never fail.
        adv = std::strftime(buffer+len, sizeof(buffer)-len, "[%F %T] ", &nowTm);
      }
      len += adv;
    }
  }
  len = std::min(len, sizeof(buffer)-1);  // prevent overflows

  if (level == PCCL_LOG_ERROR)
    len += snprintf(buffer+len, sizeof(buffer)-len,
        "[%s][%d:%d][%d]%s PCCL ERROR ", hostname, pid, tid, hggcDev, location);
  else
  if (level == NCCL_LOG_WARN)
    len += snprintf(buffer+len, sizeof(buffer)-len,
        "[%s][%d:%d][%d]%s PCCL WARN ", hostname, pid, tid, hggcDev, location);
  else if (level == NCCL_LOG_INFO)
    len += snprintf(buffer+len, sizeof(buffer)-len,
        "[%s][%d:%d][%d]%s PCCL INFO ", hostname, pid, tid, hggcDev, location);
  else if (level == PCCL_LOG_VERBOSE)
    len += snprintf(buffer+len, sizeof(buffer)-len,
        "[%s][%d:%d][%d]%s PCCL VERBOSE ", hostname, pid, tid, hggcDev, location);

#ifdef ENABLE_TRACE
  else if (level == NCCL_LOG_TRACE) {
    auto delta = std::chrono::high_resolution_clock::now() - ncclEpoch;
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(delta).count()*1000;

    len += snprintf(buffer+len, sizeof(buffer)-len,
        "[%s][%d:%d][%d][%f]%s PCCL TRACE ", hostname, pid, tid, hggcDev, timestamp, location);

  }
#endif
  len = std::min(len, sizeof(buffer)-1);  // prevent overflows

  if (len > 0) {
    va_list vargs;
    va_start(vargs, fmt);
    len += vsnprintf(buffer+len, sizeof(buffer)-len, fmt, vargs);
    va_end(vargs);
    // vsnprintf may return len >= sizeof(buffer) in the case of a truncated output.
    len = std::min(len, sizeof(buffer)-1);  // prevent overflows
    if (ncclPerCommHashDebugFile) {
      fprintf(ncclPerCommHashDebugFile,"%s\n", buffer);
      fflush(ncclPerCommHashDebugFile);
    } else {
      fprintf(ncclDebugFile,"%s\n", buffer);
      fflush(ncclDebugFile);
    }
  }
  pthread_mutex_unlock(&ncclDebugLock);

  if (level == PCCL_LOG_ERROR) ncclDebugErrorYield();
}

void ncclDebugLogForPlugin(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...) {
#if 0
  // Method 1:
  va_list args;
  va_start(args, fmt);
  ncclDebugLog(level, flags, filefunc, line, true, fmt, args);
  va_end(args);
#else
  // Method 2:
  char buffer[1024];
  va_list vargs;
  va_start(vargs, fmt);
  (void) vsnprintf(buffer, 1024, fmt, vargs);
  va_end(vargs);
  ncclDebugLog(level, flags, file, nullptr, line, true, "%s", buffer);
#endif
}

NCCL_PARAM(SetThreadName, "SET_THREAD_NAME", 1);

void ncclSetThreadName(pthread_t thread, const char *fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (ncclParamSetThreadName() != 1) return;
  char threadName[NCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread, threadName);
#endif
}

void ncclSetThreadName(std::thread& thread, const char *fmt, ...) {
#ifdef _GNU_SOURCE
  if (ncclParamSetThreadName() != 1) return;
  char threadName[NCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread.native_handle(), threadName);
#endif
}
