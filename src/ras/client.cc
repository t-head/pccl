/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <getopt.h>
#include <netdb.h>
#include <sys/stat.h>
#include <unistd.h>

#include "pccl.h"
#include "ras_wire.h"
#include "analyzer.h"

#define STR2(v) #v
#define STR(v) STR2(v)

// Local timeout increment compared to the '-t' argument, in seconds.
#define TIMEOUT_INCREMENT 1

static const char* hostName = "localhost";
static const char* port = STR(NCCL_RAS_CLIENT_PORT);
static int timeout = -1;
static bool verbose = false;
static const char* format = "text";
static bool abortFlag = false;
static uint32_t dump = false;
static bool analyzeFlag = false;
static const char* offlineDir = nullptr;
static int offlineRound = 1;
static const char* offlineTime = nullptr;
static int sock = -1;

static void printUsage(const char* argv0) {
  fprintf(stderr,
          "Usage: %s [OPTION]...\n"
          "Query the state of a running PCCL job.\n"
          "\nOptions:\n"
          "  -f, --format=FMT    Output format: text or json (text by default)\n"
          "  -h, --host=HOST     Host name or IP address of the RAS client socket of the\n"
          "                      PCCL job to connect to (localhost by default)\n"
          "  -p, --port=PORT     TCP port of the RAS client socket of the PCCL job\n"
          "                      (" STR(NCCL_RAS_CLIENT_PORT) " by default)\n"
          "  -t, --timeout=SECS  Maximum time for the local PCCL process to wait for\n"
          "                      responses from other PCCL processes\n"
          "                      (" STR(RAS_COLLECTIVE_LEG_TIMEOUT_SEC) " secs by default; 0 disables the timeout)\n"
          "  -d, --dump=LEVEL    trigger debugging dump at level=LEVEL, persist the\n"
          "                      full analyzer report, and print the summary.\n"
          "                      0x1: rough, 0x2: raw, 0x4: buffer, 0x8: inout, 0x10: hwinfo\n"
          "  -q, --quit          abort pccl job\n"
          "  -a, --analyze       Run the online hang analyzer and print the summary\n"
          "                      (text format only, no files written).\n"
          "  -O, --offline=DIR   Load diag blob(s) (*_diag_*.bin) and/or per-rank comm-dump\n"
          "                      text log(s) (pccl_comm_dump*.txt) from DIR and run offline\n"
          "                      analysis (no RAS server needed)\n"
          "  -R, --round=N       With -O: analyze the Nth-newest comm-dump round per rank\n"
          "                      (1 = newest, default).  Ignored for diag blobs.\n"
          "  -T, --time=TIME     With -O: select each rank's comm-dump closest in time to\n"
          "                      TIME (YYYYMMDD-HHMMSS; shorter prefixes OK, e.g.\n"
          "                      20260729-1527).  Overrides --round.  Ignored for blobs.\n"
          "  -v, --verbose       Increase the verbosity level of the RAS output\n"
          "      --help          Print this help and exit\n"
          "      --version       Print the version number and exit\n", argv0);
}

static void parseArgs(int argc, char** argv) {
  int c;
  int optIdx = 0;
  struct option longOpts[] = {
    {"format",  required_argument, NULL, 'f'},
    {"help",    no_argument,       NULL, 'e'},
    {"host",    required_argument, NULL, 'h'},
    {"port",    required_argument, NULL, 'p'},
    {"timeout", required_argument, NULL, 't'},
    {"dump",    required_argument, NULL, 'd'},
    {"quit",         no_argument, NULL, 'q'},
    {"analyze",      no_argument, NULL, 'a'},
    {"offline",      required_argument, NULL, 'O'},
    {"round",        required_argument, NULL, 'R'},
    {"time",         required_argument, NULL, 'T'},
    {"verbose",      no_argument, NULL, 'v'},
    {"version",      no_argument, NULL, 'r'},
    {0}
  };

  char* endPtr = nullptr;
  while ((c = getopt_long(argc, argv, "f:h:p:t:d:q:vaO:R:T:", longOpts, &optIdx)) != -1) {
    switch (c) {
      case 'f':
        format = optarg;
        if (strcasecmp(format, "text") != 0 && strcasecmp(format, "json") != 0) {
          fprintf(stderr, "Invalid format: %s (must be text or json)\n", format);
          exit(1);
        }
        break;
      case 'h':
        hostName = optarg;
        break;
      case 'p':
        port = optarg;
        break;
      case 't': {
        endPtr = nullptr;
        timeout = strtol(optarg, &endPtr, 10);
        if (timeout < 0 || !endPtr || *endPtr != '\0') {
          fprintf(stderr, "Invalid timeout: %s\n", optarg);
          exit(1);
        }
        break;
      }
      case 'v':
        verbose = true;
        break;
      case 'd':
        endPtr = nullptr;
        dump = strtol(optarg, &endPtr, 16);
        if (dump <= 0 || !endPtr || *endPtr != '\0') {
          fprintf(stderr, "Invalid dump: %s\n", optarg);
          exit(1);
        }
        analyzeFlag = true;
        break;
      case 'q':
        abortFlag = true;
        break;
      case 'a':
        analyzeFlag = true;
        break;
      case 'O':
        offlineDir = optarg;
        break;
      case 'R': {
        endPtr = nullptr;
        errno = 0;
        long roundVal = strtol(optarg, &endPtr, 10);
        if (errno == ERANGE || roundVal < 1 || roundVal > INT_MAX ||
            !endPtr || *endPtr != '\0') {
          fprintf(stderr, "Invalid round: %s (must be an integer >= 1)\n", optarg);
          exit(1);
        }
        offlineRound = (int)roundVal;
        break;
      }
      case 'T':
        if (!analyzerTimeArgValid(optarg)) {
          fprintf(stderr, "Invalid time: %s (expected YYYYMMDD-HHMMSS or a shorter prefix)\n", optarg);
          exit(1);
        }
        offlineTime = optarg;
        break;
      case 'e':
        printUsage(argv[0]);
        exit(0);
      case 'r':
        fprintf(stderr, "PCCL RAS client version " STR(PCCL_MAJOR) "." STR(PCCL_MINOR) "."
                STR(PCCL_PATCH) PCCL_SUFFIX "\n");
        exit(0);
      default:
        printUsage(argv[0]);
        exit(1);
    }
  }
}

static ssize_t socketWrite(int fd, const void* buf, size_t count) {
  size_t done = 0;
  do {
    ssize_t ret;
    ret = write(fd, ((const char*)buf)+done, count-done);
    if (ret == -1) {
      if (errno != EINTR)
        return -1;
      continue;
    }
    done += ret;
  } while (done < count);

  return done;
}

// Reads a message from RAS.  Assumes that the message ends with '\n' (will continue reading until the terminating
// newline, unless false is passed as untilNewLine).
// Terminates the buffer with '\0'.  Returns the number of bytes read (excluding the added terminating '\0').
static ssize_t rasRead(int fd, void* buf, size_t count, bool untilNewline = true) {
  char* bufChar = (char*)buf;
  size_t done = 0;
  do {
    ssize_t ret;
    ret = read(fd, bufChar+done, count-1-done);
    if (ret == -1) {
      if (errno != EINTR)
        return -1;
      continue;
    }
    if (ret == 0)
      break; // EOF
    done += ret;
  } while (untilNewline && (done == 0 || bufChar[done-1] != '\n'));
  bufChar[done] = '\0';

  return done;
}

static int connectToNCCL() {
  struct addrinfo hints = {0};
  struct addrinfo* addrInfo = nullptr;
  int ret;
  char msgBuf[1024];
  int bytes;
  struct timeval tv = {TIMEOUT_INCREMENT, 0};

retry:
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  if ((ret = getaddrinfo(hostName, port, &hints, &addrInfo)) != 0) {
    fprintf(stderr, "Resolving %s:%s: %s\n", hostName, port, gai_strerror(ret));
    goto fail;
  }
  for (struct addrinfo* ai = addrInfo; ai; ai = ai->ai_next) {
    char hostBuf[NI_MAXHOST], portBuf[NI_MAXSERV];
    int err;
    sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (sock == -1) {
      perror("socket");
      continue;
    }
    // Initially start with a small, 1-sec timeout to quickly eliminate non-responsive processes...
    if (timeout && (setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv) != 0 ||
                    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) != 0)) {
      perror("setsockopt");
      // Non-fatal; fall through.
    }
    if (connect(sock, ai->ai_addr, ai->ai_addrlen) == 0)
      break;
    err = errno;
    if (getnameinfo(ai->ai_addr, ai->ai_addrlen, hostBuf, sizeof(hostBuf), portBuf, sizeof(portBuf),
                    NI_NUMERICHOST | NI_NUMERICSERV) != 0) {
      strcpy(hostBuf, hostName);
      strcpy(portBuf, port);
    }
    fprintf(stderr, "Connecting to %s:%s: %s\n", hostBuf, portBuf, strerror(err));
    close(sock);
    sock = -1;
  }
  freeaddrinfo(addrInfo);
  addrInfo = nullptr;

  if (sock == -1) {
    fprintf(stderr, "Failed to connect to the PCCL RAS service!\n"
            "Please make sure that the PCCL job has the RAS service enabled and that\n"
            "%s.\n",
            (strcmp(hostName, "localhost") || strcmp(port, STR(NCCL_RAS_CLIENT_PORT)) ?
            "the host/port arguments are correct and match PCCL_RAS_ADDR" :
            "the RAS client was started on a node where the PCCL job is running"));
    goto fail;
  }

  // Exchange the RAS client handshake.
  strcpy(msgBuf, "CLIENT PROTOCOL " STR(NCCL_RAS_CLIENT_PROTOCOL) "\n");
  if (socketWrite(sock, msgBuf, strlen(msgBuf)) != strlen(msgBuf)) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      goto timeout;
    }
    perror("write to socket");
    goto fail;
  }
  bytes = rasRead(sock, msgBuf, sizeof(msgBuf));
  if (bytes < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      goto timeout;
    }
    perror("read socket");
    goto fail;
  }
  if (bytes == 0) {
    fprintf(stderr, "PCCL unexpectedly closed the connection\n");
    goto fail;
  }
  if (strncasecmp(msgBuf, "SERVER PROTOCOL ", strlen("SERVER PROTOCOL "))) {
    fprintf(stderr, "Unexpected response from PCCL: %s\n", msgBuf);
    goto fail;
  }
  if (strtol(msgBuf+strlen("SERVER PROTOCOL "), nullptr, 10) != NCCL_RAS_CLIENT_PROTOCOL) {
    fprintf(stderr, "PCCL RAS protocol version mismatch (PCCL: %s; RAS client: %d)!\n"
            "Will try to continue in spite of that...\n", msgBuf+strlen("SERVER PROTOCOL "), NCCL_RAS_CLIENT_PROTOCOL);
  }

  if (timeout >= 0) {
    snprintf(msgBuf, sizeof(msgBuf), "TIMEOUT %d\n", timeout);
  } else if (abortFlag > 0) {
    snprintf(msgBuf, sizeof(msgBuf), "ABORT\n");
  }

  if (timeout >= 0 || abortFlag) {
    if (socketWrite(sock, msgBuf, strlen(msgBuf)) != strlen(msgBuf)) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        goto timeout;
      }
      perror("write to socket");
      goto fail;
    }
    bytes = rasRead(sock, msgBuf, sizeof(msgBuf));
    if (bytes < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        goto timeout;
      }
      perror("read socket");
      goto fail;
    }
    if (bytes == 0) {
      fprintf(stderr, "PCCL unexpectedly closed the connection\n");
      goto fail;
    }
    if (strcasecmp(msgBuf, "OK\n")) {
      fprintf(stderr, "Unexpected response from PCCL: %s\n", msgBuf);
      goto fail;
    }
  }

  if (timeout) {
    // Increase the socket timeout to accommodate NCCL timeout.
    tv.tv_sec += (timeout > 0 ? timeout : RAS_COLLECTIVE_LEG_TIMEOUT_SEC) + RAS_COLLECTIVE_EXTRA_TIMEOUT_SEC;
    if (setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv) != 0) {
      perror("setsockopt");
      // Non-fatal; fall through.
    }
  }

  return 0;
fail:
  if (addrInfo)
    freeaddrinfo(addrInfo);
  if (sock != -1)
    (void)close(sock);
  return 1;
timeout:
  fprintf(stderr, "Connection timed out; retrying...\n");
  (void)close(sock);
  goto retry;
}

static int setOutputFormat() {
  char msgBuf[4096];
  int bytes;

  // Only set format if it's not the default.
  if (strcasecmp(format, "text") != 0) {
    snprintf(msgBuf, sizeof(msgBuf), "SET FORMAT %s\n", format);
    if (socketWrite(sock, msgBuf, strlen(msgBuf)) != strlen(msgBuf)) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        fprintf(stderr, "Connection timed out\n");
      else
        perror("write to socket");
      return 1;
    }
    // Read response.
    bytes = rasRead(sock, msgBuf, sizeof(msgBuf));
    if (bytes < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        fprintf(stderr, "Connection timed out\n");
      else
        perror("read socket");
      return 1;
    }
    if (bytes == 0) {
      fprintf(stderr, "PCCL unexpectedly closed the connection\n");
      return 1;
    }
    if (strcasecmp(msgBuf, "OK\n")) {
      fprintf(stderr, "Unexpected response from PCCL: %s\n", msgBuf);
      return 1;
    }
  }
  return 0;
}

int getNCCLStatus() {
  char msgBuf[4096];
  int bytes;

  // Set the output format.
  if (setOutputFormat() != 0) {
    return 1;
  }

  // Send the status command.
  if (analyzeFlag) {
    snprintf(msgBuf, sizeof(msgBuf), "DUMP %d\n", dump);
  } else {
    snprintf(msgBuf, sizeof(msgBuf), "%sSTATUS\n", (verbose ? "VERBOSE " : ""));
  }
  if (socketWrite(sock, msgBuf, strlen(msgBuf)) != strlen(msgBuf)) {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      fprintf(stderr, "Connection timed out\n");
    else
      perror("write to socket");
    return 1;
  }
  for (;;) {
    bytes = rasRead(sock, msgBuf, sizeof(msgBuf), /*untileNewLine*/false);
    if (bytes < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        fprintf(stderr, "Connection timed out\n");
      else
        perror("read socket");
      return 1;
    }
    if (bytes == 0) // EOF
      break;
    if (fwrite(msgBuf, 1, bytes, stdout) != bytes) {
      fprintf(stderr, "fwrite to stdout failed!\n");
      return 1;
    }
    if (fflush(stdout) != 0) {
      perror("fflush stdout");
      return 1;
    }
  }
  return 0;
}

static int runOfflineAnalysis(const char* dir) {
  DIR* dp = opendir(dir);
  if (dp == nullptr) {
    fprintf(stderr, "Cannot open directory: %s (%s)\n", dir, strerror(errno));
    return 1;
  }

  int nAnalyzed = 0;
  struct dirent* ent;
  while ((ent = readdir(dp)) != nullptr) {
    const char* name = ent->d_name;
    if (strstr(name, "_diag_") == nullptr || strstr(name, ".bin") == nullptr)
      continue;

    char path[PATH_MAX + 1];
    snprintf(path, sizeof(path), "%s/%s", dir, name);

    FILE* f = fopen(path, "rb");
    if (f == nullptr) {
      fprintf(stderr, "Cannot open %s: %s\n", path, strerror(errno));
      continue;
    }
    struct stat st;
    if (fstat(fileno(f), &st) != 0 || !S_ISREG(st.st_mode)) {
      fclose(f);
      continue;
    }

    int nBytes = 0, nLeg = 0;
    struct rasCollComms* blob = analyzerLoadData(f, &nBytes, &nLeg);
    fclose(f);
    if (blob == nullptr) continue;

    fprintf(stdout, "\n========== Offline analysis: %s ==========\n", name);
    analyzerFormatReport(stdout, blob, nLeg, -1);
    free(blob);
    nAnalyzed++;
  }
  closedir(dp);

  int nCommDumpFiles = 0;
  int nCommDumps = analyzerFormatReportFromCommDumps(stdout, dir, offlineRound, offlineTime,
                                                    &nCommDumpFiles);
  if (nCommDumps < 0) {
    return 1;
  }

  if (nAnalyzed == 0 && nCommDumps == 0) {
    if (nCommDumpFiles > 0) {
      // Files were found and parsed, but the round/time selection matched
      // nothing (details and per-comm warnings were printed above).
      if (offlineTime != nullptr)
        fprintf(stderr, "%d comm-dump file(s) parsed in %s, but none matched --time %s; nothing analyzed\n",
                nCommDumpFiles, dir, offlineTime);
      else
        fprintf(stderr, "%d comm-dump file(s) parsed in %s, but none matched --round %d; nothing analyzed\n",
                nCommDumpFiles, dir, offlineRound);
    } else {
      fprintf(stderr, "No valid *_diag_*.bin or comm-dump .txt files found in %s\n", dir);
    }
    return 1;
  }
  fprintf(stdout, "\nOffline analysis complete: %d blob(s) and %d comm-dump group(s) processed.\n",
          nAnalyzed, nCommDumps);
  return 0;
}

int main(int argc, char** argv) {
  parseArgs(argc, argv);

  if (offlineDir != nullptr)
    return runOfflineAnalysis(offlineDir);

  if (connectToNCCL())
    return 1;

  if (getNCCLStatus()) {
    (void)close(sock);
    return 1;
  }

  if (close(sock) == -1) {
    perror("close socket");
    return 1;
  }
  return 0;
}
