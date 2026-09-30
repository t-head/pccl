/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_P2P_INFO_H_
#define NCCL_P2P_INFO_H_

#include "hggc.h"
#include "utils.h"

struct ncclP2Pinfo {
  void* buff;
  ssize_t nbytes;
  struct ncclP2Pinfo* next;
};

typedef ncclRecyclableList<struct ncclP2Pinfo> ncclP2Plist;

static ncclResult_t ncclSaveP2pInfo(ncclP2Plist* &p2p, void* buff, ssize_t nBytes) {
  if (p2p == NULL) {
    p2p = new ncclP2Plist();
  }

  struct ncclP2Pinfo* next;
  NCCLCHECK(p2p->getNewElem(&next));
  next->buff = buff;
  next->nbytes = nBytes;
  return ncclSuccess;
}

#endif
