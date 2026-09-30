#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2023-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information
#

include ../makefiles/common.mk

INC:= ./pccl ../build/include ../src/include

LIB_NAME = libpccl-tuner.so
BUILDDIR ?= $(abspath ../build/lib)
PLUGIN_SO:=$(BUILDDIR)/$(LIB_NAME)

build: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c
	mkdir -p $(BUILDDIR)
	$(CXX) $(INC:%=-I%) $(PPU_INC:%=-I%) $(FILE_PREFIX_MAP)=$(PCCL_SRC_ROOT)=. -std=c++11 -fPIC -shared -o $@ -Wl,-soname,$(LIB_NAME) $^
ifeq ($(REL_NO_SYM), 1)
	$(OBJCOPY) --only-keep-debug $@ $(BUILDDIR)/$(LIB_NAME).dbginfo
	$(STRIP) --strip-all $@
	$(OBJCOPY) --add-gnu-debuglink=$(BUILDDIR)/$(LIB_NAME).dbginfo $@
endif

clean:
	rm -f $(PLUGIN_SO)
