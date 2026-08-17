/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
// scheduler.h: Symmetric task scheduling API for multi-task symmetric kernels.

#ifndef NCCL_SCHEDULER_H_
#define NCCL_SCHEDULER_H_

#include "comm.h"

ncclResult_t ncclMakeSymmetricTaskList(struct pcclComm* comm,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* collTaskQueue,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
    struct ncclTaskColl** remainTasksHead);

ncclResult_t ncclSymmetricTaskScheduler(struct pcclComm* comm,
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue,
    struct ncclKernelPlan* plan);

#endif // NCCL_SCHEDULER_H_
