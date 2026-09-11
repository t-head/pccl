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

HGGCRT_LDFLAG = -lhggc_wrapper -lhg_wrapper

LIB_NAME = libpccl-ext-kernel.so
BUILDDIR ?= $(abspath ../build/lib)
PLUGIN_SO := $(BUILDDIR)/$(LIB_NAME)
OBJDIR := $(abspath $(BUILDDIR)/../obj)/ext-kernel

CROSS_COMPILE_OPTION =
OBJCOPY ?= objcopy
ifneq ($(PPU_HOST_ARCH),$(PPU_TARGET_ARCH))
  CROSS_COMPILE_OPTION = -ccbin ${PPU_TARGET_ARCH}-linux-gnu-g++
  OBJCOPY := ${PPU_TARGET_ARCH}-linux-gnu-objcopy
  LINKER := ${PPU_TARGET_ARCH}-linux-gnu-g++
else
  LINKER := $(CXX)
endif
ifeq ($(strip $(LINKER)),)
  LINKER := g++
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

# hgcc is only used to compile objects; the final .so is linked by gcc
# (cross ${PPU_TARGET_ARCH}-linux-gnu-g++ when cross compiling), the same
# way src/Makefile links libpccl.so.
HGCC_CFLAGS := -dopt 3 -O3 -g -x hg $(HGGCRT_FLAG) --no-simt \
               -Xcompiler $(INC:%=-I%) $(PPU_INC:%=-I%) \
               $(CROSS_COMPILE_OPTION) $(CXXFLAGS) $(HGFLAGS)

SRCS := src/plugin.c src/customAllReduceKernels.cu src/customAllGatherKernels.cu \
        src/customReduceScatterKernels.cu src/customKernels.cu src/customDeviceUtils.cu \
        src/ext_kernel_dispatch.cc src/ext_topo.c
HDRS := $(wildcard pccl/*.h src/*.h)
OBJS := $(patsubst src/%.c,$(OBJDIR)/%.o,$(filter %.c,$(SRCS))) \
        $(patsubst src/%.cu,$(OBJDIR)/%.o,$(filter %.cu,$(SRCS))) \
        $(patsubst src/%.cc,$(OBJDIR)/%.o,$(filter %.cc,$(SRCS)))

build: $(PLUGIN_SO)

$(OBJDIR)/%.o: src/%.c $(HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p $(OBJDIR)
	$(HGCC) $(HGCC_CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: src/%.cu $(HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p $(OBJDIR)
	$(HGCC) $(HGCC_CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: src/%.cc $(HDRS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p $(OBJDIR)
	$(HGCC) $(HGCC_CFLAGS) -c $< -o $@

$(PLUGIN_SO): $(OBJS)
	@printf "Linking    %-35s > %s\n" $(LIB_NAME) $@
	mkdir -p $(BUILDDIR)
	$(LINKER) -shared -fPIC -Wl,-soname=$(LIB_NAME) -o $@ $(OBJS) -L$(PPU_LIB) $(HGGCRT_LDFLAG)
ifeq ($(REL_NO_SYM), 1)
	$(OBJCOPY) --only-keep-debug $@ $(BUILDDIR)/$(LIB_NAME).dbginfo
	$(OBJCOPY) --strip-all --keep-symbol=pcclExtKernelPlugin_v1 $@ $@.tmp && mv $@.tmp $@
	$(OBJCOPY) --add-gnu-debuglink=$(BUILDDIR)/$(LIB_NAME).dbginfo $@
endif

clean:
	rm -rf $(OBJDIR)
	rm -f $(PLUGIN_SO)
