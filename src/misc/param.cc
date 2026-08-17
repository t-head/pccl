/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "param.h"
#include "debug.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <pwd.h>

const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[0] == '#') continue;
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));
    envVar[std::min(1023,s)] = '\0';
    s++;
    strncpy(envValue, line+s, 1023);
    envValue[1023]='\0';
    setenv(envVar, envValue, 0);
    //printf("%s : %s->%s\n", fileName, envVar, envValue);
  }
  if (line) free(line);
  fclose(file);
}

static void initEnvFunc() {
  char confFilePath[1024];
  const char* userFile = pcclGetEnvWithDualPrefix("PCCL_CONF_FILE");
  if (userFile && strlen(userFile) > 0) {
    snprintf(confFilePath, sizeof(confFilePath), "%s", userFile);
    setEnvFile(confFilePath);
  } else {
    const char* userDir = userHomeDir();
    if (userDir) {
      snprintf(confFilePath, sizeof(confFilePath), "%s/.pccl.conf", userDir);
      setEnvFile(confFilePath);
    }
  }
  snprintf(confFilePath, sizeof(confFilePath), "/etc/pccl.conf");
  setEnvFile(confFilePath);
}

void initEnv() {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, initEnvFunc);
}

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_lock(&mutex);
  if (__atomic_load_n(cache, __ATOMIC_RELAXED) == uninitialized) {
    const char* str = pcclGetEnvWithDualPrefix(env);
    int64_t value = deftVal;
    if (str && strlen(str) > 0) {
      errno = 0;
      value = strtoll(str, nullptr, 0);
      if (errno) {
        value = deftVal;
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
      } else {
        INFO(NCCL_ENV,"%s set by environment to %lld.", env, (long long)value);
      }
    }
    __atomic_store_n(cache, value, __ATOMIC_RELAXED);
  }
  pthread_mutex_unlock(&mutex);
}

const char* pcclGetEnvWithDualPrefix(const char* name) {
  if (name == NULL) {
    return NULL;
  }

  const char *value = NULL;
  const char *ncclValue = NULL;
  const char *pcclValue = NULL;
  const int envNameLength = 64;
  const int envPrefixLength = strlen("PCCL");
  char ncclEnvName[envNameLength];
  char pcclEnvName[envNameLength];
  ncclEnvName[0] = '\0';
  pcclEnvName[0] = '\0';
  if ((name[0] == 'N' || name[0] == 'P') && (strncmp(name+1, "CCL_", 4) == 0)) {
    if (snprintf(pcclEnvName, envNameLength, "PCCL%s", name+envPrefixLength) >= envNameLength) pcclEnvName[0] = '\0';
    if (snprintf(ncclEnvName, envNameLength, "PCCL%s", name+envPrefixLength) >= envNameLength) ncclEnvName[0] = '\0';
    else ncclEnvName[0] = 'N';
  }
  if (strlen(ncclEnvName) > 0) ncclValue = getenv(ncclEnvName);
  if (strlen(pcclEnvName) > 0) pcclValue = getenv(pcclEnvName);
  value = pcclValue ? pcclValue : (ncclValue ? ncclValue : getenv(name));
  return value;
}

const char* ncclGetEnv(const char* name) {
  initEnv();
  return pcclGetEnvWithDualPrefix(name);
}
