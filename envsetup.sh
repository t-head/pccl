#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

PCCL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"

MODE=REL

if [ "$1" = "dev" ];then
    MODE=DEV
fi

if [ "$MODE" = "REL" ]; then
    echo "Setup PCCL environment for Release"
    export PCCL_HOME=$PCCL_ROOT
    export PATH=$PCCL_ROOT/bin:$PATH
    export LD_LIBRARY_PATH=$PCCL_ROOT/lib:${LD_LIBRARY_PATH}
else
    echo "Setup PCCL environment for Development"
    export PCCL_HOME=$PCCL_ROOT/build
    export PATH=$PCCL_ROOT/build/bin:$PATH
    export LD_LIBRARY_PATH=$PCCL_ROOT/build/lib:${LD_LIBRARY_PATH}
fi

