/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "ipcsocket.h"
#include "utils.h"
#include "param.h"
#include <sys/stat.h>
#include <sys/syscall.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

NCCL_PARAM(IpcUseAbstractSocket, "IPC_USE_ABSTRACT_SOCKET", 1);

#define NCCL_IPC_SOCKNAME_STR "/tmp/pccl-socket-%d-%lx"

/*
 * Create a Unix Domain Socket
 */
ncclResult_t ncclIpcSocketInit(ncclIpcSocket *handle, int rank, uint64_t hash, volatile uint32_t* abortFlag) {
  int fd = -1;
  struct sockaddr_un cliaddr;
  char temp[NCCL_IPC_SOCKNAME_LEN] = "";

  if (handle == NULL) {
    return ncclInternalError;
  }

  handle->fd = -1;
  handle->socketName[0] = '\0';
  if ((fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    ERROR("UDS: Socket creation error : %s (%d)", strerror(errno), errno);
    return ncclSystemError;
  }

  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  // Create unique name for the socket.
  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    ERROR("UDS: Cannot bind provided name to socket. Name too large");
    close(fd);
    return ncclInternalError;
  }
  if (!ncclParamIpcUseAbstractSocket()) unlink(temp);

  TRACE(NCCL_INIT, "UDS: Creating socket %s", temp);

  strncpy(cliaddr.sun_path, temp, len);
  if (ncclParamIpcUseAbstractSocket()) cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick
  if (bind(fd, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
    ERROR("UDS: Binding to socket %s failed : %s (%d)", temp, strerror(errno), errno);
    close(fd);
    return ncclSystemError;
  }

  handle->fd = fd;
  strcpy(handle->socketName, temp);

  handle->abortFlag = abortFlag;
  // Mark socket as non-blocking
  if (handle->abortFlag) {
    int flags;
    SYSCHECK(flags = fcntl(fd, F_GETFL), "fcntl");
    SYSCHECK(fcntl(fd, F_SETFL, flags | O_NONBLOCK), "fcntl");
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket* handle, int* fd) {
  if (handle == NULL) {
    ERROR("pcclSocketGetFd: pass NULL socket");
    return ncclInvalidArgument;
  }
  if (fd) *fd = handle->fd;
  return ncclSuccess;
}

ncclResult_t ncclIpcSocketClose(ncclIpcSocket *handle) {
  if (handle == NULL) {
    return ncclInternalError;
  }
  if (handle->fd <= 0) {
    return ncclSuccess;
  }
  if (!ncclParamIpcUseAbstractSocket() && handle->socketName[0] != '\0') {
    unlink(handle->socketName);
  }
  close(handle->fd);

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, int *recvFd) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1];
  int ret;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  if (hdr == NULL) {
    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      ERROR("UDS: Receiving data over socket failed : %d", errno);
      return ncclSystemError;
    }
    if (handle->abortFlag && __atomic_load_n(handle->abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;
  }

  if (recvFd != NULL) {
    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        ERROR("UDS: Receiving data over socket failed");
      return ncclSystemError;
      }

      memmove(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
    } else {
      ERROR("UDS: Receiving data over socket %s failed", handle->socketName);
      return ncclSystemError;
    }
    TRACE(NCCL_INIT|NCCL_P2P, "UDS: Got recvFd %d from socket %s", *recvFd, handle->socketName);
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketRecvFd(ncclIpcSocket *handle, int *recvFd) {
  return ncclIpcSocketRecvMsg(handle, NULL, 0, recvFd);
}

ncclResult_t ncclIpcSocketSendMsg(ncclIpcSocket *handle, void *hdr, int hdrLen, const int sendFd, int rank, uint64_t hash) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];
  char temp[NCCL_IPC_SOCKNAME_LEN];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1] = {'\0'};
  struct sockaddr_un cliaddr;

  // Construct client address to send this shareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  int len = snprintf(temp, NCCL_IPC_SOCKNAME_LEN, NCCL_IPC_SOCKNAME_STR, rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    ERROR("UDS: Cannot connect to provided name for socket. Name too large");
    return ncclInternalError;
  }
  (void) strncpy(cliaddr.sun_path, temp, len);

  if (ncclParamIpcUseAbstractSocket()) cliaddr.sun_path[0] = '\0'; // Linux abstract socket trick

  TRACE(NCCL_INIT, "UDS: Sending hdr %p len %d fd %d to UDS socket %s", hdr, hdrLen, sendFd, temp);

  if (sendFd != -1) {
    memset(&control_un, '\0', sizeof(control_un));
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));
  }

  msg.msg_name = (void *)&cliaddr;
  msg.msg_namelen = sizeof(struct sockaddr_un);

  if (hdr == NULL) {
    iov[0].iov_base = (void *)dummy_buffer;
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  ssize_t sendResult;
  while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      ERROR("UDS: Sending data over socket %s failed : %s (%d)", temp, strerror(errno), errno);
      return ncclSystemError;
    }
    if (handle->abortFlag && __atomic_load_n(handle->abortFlag, __ATOMIC_ACQUIRE)) return ncclInternalError;
  }

  return ncclSuccess;
}

ncclResult_t ncclIpcSocketSendFd(ncclIpcSocket *handle, const int sendFd, int rank, uint64_t hash) {
  return ncclIpcSocketSendMsg(handle, NULL, 0, sendFd, rank, hash);
}

static uint64_t pcclGetPidNamespace() {
  struct stat st;
  if (stat("/proc/self/ns/pid", &st) != 0) return 0;
  return st.st_ino;
}

void pcclPidfdGetInfo(int fd, struct pcclPidfdInfo* info) {
  info->pid = getpid();
  info->fd = fd;
  info->pidNamespace = pcclGetPidNamespace();
}

ncclResult_t pcclPidfdGetFd(struct pcclPidfdInfo* info, int* fd) {
  *fd = -1;
#if defined(SYS_pidfd_open) && defined(SYS_pidfd_getfd)
  // PIDs exchanged over the proxy socket belong to the sender's PID namespace.
  if (info->pidNamespace == 0 || info->pidNamespace != pcclGetPidNamespace()) {
    INFO(NCCL_PROXY, "pidfd: PID namespaces differ or are unavailable");
    return ncclSuccess;
  }
  int pidFd = syscall(SYS_pidfd_open, info->pid, 0);
  if (pidFd == -1) {
    INFO(NCCL_PROXY, "pidfd_open(%d) failed: %s", info->pid, strerror(errno));
    return ncclSuccess;
  }
  *fd = syscall(SYS_pidfd_getfd, pidFd, info->fd, 0);
  if (*fd == -1) {
    INFO(NCCL_PROXY, "pidfd_getfd(%d, %d) failed: %s", info->pid, info->fd, strerror(errno));
  }
  ncclResult_t ret = ncclSuccess;
  SYSCHECKGOTO(close(pidFd), "close", ret, fail);
  return ncclSuccess;
fail:
  if (*fd != -1) (void)close(*fd);
  *fd = -1;
  return ret;
#else
  INFO(NCCL_PROXY, "pidfd syscalls unavailable at build time");
#endif
  return ncclSuccess;
}
