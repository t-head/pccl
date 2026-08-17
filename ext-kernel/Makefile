#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

PPU_SDK ?= /usr/local/PPU_SDK/
PPU_HOME ?= /usr/local/ppu
PPU_SOURCE_HOME ?= $(PPU_HOME)
PCCL_SOURCE_HOME ?= ../
INC := src/ ./
SPLIT_ARCHS ?= $(shell echo $(PPU_ARCHS) | tr ',' ' ')

ifeq ($(USE_SDK), 1)
  HGCC ?= $(PPU_SDK)/bin/hgcc
  PPU_INC ?= $(PPU_SDK)/include
  PPU_LIB ?= $(PPU_SDK)/lib
else
  HGCC ?= $(PPU_SOURCE_HOME)/hgcc/build/hgcc
  PPU_INC ?= $(PPU_SOURCE_HOME)/include/common
  PPU_LIB ?= $(PPU_SOURCE_HOME)/api_loader/build
  $(shell ln -sf $(PPU_LIB)/libhg_wrapper.so $(PPU_LIB)/libhggc.so)
  $(shell ln -sf $(PPU_LIB)/libhggc_wrapper.so $(PPU_LIB)/libhggcrt1.so)
endif

HGGCRT_LDFLAG = -lhggc_wrapper

LIB_NAME = libpccl-ext-kernel.so
BUILDDIR ?= $(abspath ../build/lib)
PLUGIN_SO := $(BUILDDIR)/$(LIB_NAME)

CROSS_COMPILE_OPTION =
ifneq ($(PPU_HOST_ARCH),$(PPU_TARGET_ARCH))
  CROSS_COMPILE_OPTION = -ccbin ${PPU_TARGET_ARCH}-linux-gnu-g++
endif

CXXFLAGS += -Xcompiler -Werror,-Wno-address-of-temporary,-fPIC
ifeq ($(REL_NO_SYM), 1)
CXXFLAGS += -Xcompiler "-fvisibility=hidden -fvisibility-inlines-hidden"
endif

CXXFLAGS += -Xcompiler $(FILE_PREFIX_MAP)=$(PCCL_SRC_ROOT)=.

ifeq ($(HGTX), 0)
  CXXFLAGS  += -Xcompiler -DHGTX_DISABLE
else
  alios = $(shell cat /etc/os-release | grep -i alios)
  ifneq ($(alios),)    #  add -DHGTX3_USE_CHECKED_OVERLOADS_FOR_GET=0 for alios, compile issue
    CXXFLAGS += -Xcompiler -DHGTX3_USE_CHECKED_OVERLOADS_FOR_GET=0
  endif
endif

define append_arch
$(eval HGFLAGS += -arch=$(1))
$(eval HGLDFLAGS += -arch=$(1))
endef
TMP := $(foreach arch,$(SPLIT_ARCHS), $(call append_arch,$(arch)))

$(info "PPU_ARCHS: $(PPU_ARCHS)")

build: $(PLUGIN_SO)

$(PLUGIN_SO): ./src/plugin.c ./src/customAllReduceKernels.cu ./src/customAllGatherKernels.cu ./src/customReduceScatterKernels.cu ./src/customKernels.cu ./src/customDeviceUtils.cu ./src/ext_kernel_dispatch.cc ./src/ext_topo.c
	mkdir -p $(BUILDDIR)
	$(HGCC) -dopt 3 -O3 -g -x hg $(HGGCRT_FLAG) --no-simt -Xcompiler $(INC:%=-I%) $(PPU_INC:%=-I%) \
    -L$(BUILDDIR) -L$(PPU_LIB) $(CROSS_COMPILE_OPTION) $(CXXFLAGS) $(HGFLAGS) -lpccl $(HGGCRT_LDFLAG) -shared -o $@ -Xlinker -soname=$(LIB_NAME) $^
ifeq ($(REL_NO_SYM), 1)
	objcopy --only-keep-debug $@ $(BUILDDIR)/$(LIB_NAME).dbginfo
	objcopy --strip-all --keep-symbol=pcclExtKernelPlugin_v1 $@ $@.tmp && mv $@.tmp $@
	objcopy --add-gnu-debuglink=$(BUILDDIR)/$(LIB_NAME).dbginfo $@
endif

clean:
	rm -f $(PLUGIN_SO)
