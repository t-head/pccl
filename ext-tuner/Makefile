#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2023-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information
#

include ../makefiles/common.mk

PPU_SDK ?= /usr/local/PPU_SDK/
PPU_HOME ?= /usr/local/ppu
PPU_SOURCE_HOME ?= $(PPU_HOME)
PCCL_SOURCE_HOME ?= ../
INC:= ./pccl ../build/include ../src/include

ifeq ($(USE_SDK), 1)
  PPU_INC ?= $(PPU_SDK)/include $(PCCL_SOURCE_HOME)/src/include
else
  PPU_INC ?= $(PPU_SOURCE_HOME)/include/common $(PPU_SOURCE_HOME)/pccl/build/include $(PCCL_SOURCE_HOME)/src/include
endif

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
