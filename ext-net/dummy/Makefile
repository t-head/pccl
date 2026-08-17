#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information
#
PCCL_HOME:=../../build/
HGGC_HOME:=/usr/local/PPU_SDK/
INC:= -I$(PCCL_HOME)/include -I$(HGGC_HOME)/include
PLUGIN_SO:=libpccl-net.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c
	$(CC) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

clean:
	rm -f $(PLUGIN_SO)
