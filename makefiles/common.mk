#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information
#

# Build against the installed PPU SDK.
PPU_SDK ?= /usr/local/PPU_SDK
SPLIT_ARCHS ?= $(shell echo $(PPU_ARCHS) | tr ',' ' ')
HGGCRT_VERSION ?= 12

HGCC ?= $(PPU_SDK)/bin/hgcc
ifeq ($(wildcard $(HGCC)),)
  HGCC = hgcc
endif

ifeq (, $(shell which $(HGCC) 2>/dev/null))
  $(error "$(HGCC) Not Found. Please set PPU_SDK to a path with PPU SDK >= 2.1 installed.")
endif

# Detect HGGC version: try hgcc --version first, fall back to PPU_SDK path
HGGC_VERSION = $(strip $(shell $(HGCC) --version 2>/dev/null | grep version | sed 's/.*version //' | sed 's/\,.*//'))
ifeq ($(HGGC_VERSION),)
  # hgcc --version not supported (SDK < 2.2); parse from PPU_SDK path (e.g. PPU_SDK_v2.1.0 → 2.1.0)
  HGGC_VERSION = $(strip $(shell echo $(PPU_SDK) | sed -n 's|.*/PPU_SDK_v*\([0-9][0-9.]*\).*|\1|p'))
endif
ifeq ($(HGGC_VERSION),)
  HGGC_VERSION = 0.0.0
endif
HGGC_MAJOR = $(shell echo $(HGGC_VERSION) | cut -d "." -f 1)
HGGC_MINOR = $(shell echo $(HGGC_VERSION) | cut -d "." -f 2)
HGGC_PATCH = $(shell echo $(HGGC_VERSION) | cut -d "." -f 3 | sed 's/[^0-9].*//')
HGGC_PATCH := $(if $(HGGC_PATCH),$(HGGC_PATCH),0)
$(info HGGC_VERSION ${HGGC_MAJOR}.${HGGC_MINOR}.${HGGC_PATCH})

PCCL_HGGC_VERSION := $(shell expr $(HGGC_MAJOR) \* 10000 + $(HGGC_MINOR) \* 100 + $(HGGC_PATCH) 2>/dev/null || echo 0)
HGGC_HAS_OKI := $(shell test "$(PCCL_HGGC_VERSION)" -ge 20200 2>/dev/null && echo 1 || echo 0)

PPU_LIB ?= $(PPU_SDK)/lib
PPU_INC ?= $(PPU_SDK)/include
ifeq ($(HGGC_HAS_OKI), 1)
  ifeq ($(USE_SIMT), 1)
    OKI_DEV_LIB ?= $(PPU_SDK)/lib/liblt_oki_device_simt.a
  else
    OKI_DEV_LIB ?= $(PPU_SDK)/lib/liblt_oki_device_simd.a
  endif
endif

PREFIX ?= /usr/local
VERBOSE ?= 0
KEEP ?= 0
DEBUG ?= 0
TRACE ?= 0
PROFAPI ?= 0
HGTX ?= 1
KERNEL_LOOPBACK ?= 0
NO_EXTRA_WARP ?= 0
MAX_VREGS ?= 96
DISABLE_TESTS ?= 1
NO_DLTO ?= 1

# Maxrregcount needs to be set accordingly to NCCL_MAX_NTHREADS (otherwise it will cause kernel launch errors)
# 512 : 120, 640 : 96, 768 : 80, 1024 : 60
#
# We would not have to set this if we used __launch_bounds__, but this only works on kernels, not on functions.

CXXFLAGS   := -DHGGC_MAJOR=$(HGGC_MAJOR) -DHGGC_MINOR=$(HGGC_MINOR) \
              -DPCCL_HGGC_VERSION=$(PCCL_HGGC_VERSION) -fPIC \
              -Wall -Wno-unused-function -Wno-sign-compare -Wno-deprecated-declarations \
              -std=c++14 -Wvla \
              $(PPU_INC:%=-I%) \
              -DPCCL_OS_LINUX \
              $(CXXFLAGS)

HGFLAGS    := -std=c++14 -Xcompiler -fPIC \
              -DPCCL_OS_LINUX -DPCCL_HGGC_VERSION=$(PCCL_HGGC_VERSION) $(PPU_INC:%=-I%)

HGCC_THREADS ?= 1
HGFLAGS += --threads $(HGCC_THREADS)
HGLDFLAGS += --threads $(HGCC_THREADS)

ifneq ($(PCCL_SRC_ROOT),)
  CXXFLAGS += $(FILE_PREFIX_MAP)=$(PCCL_SRC_ROOT)=.
  HGFLAGS += -Xcompiler $(FILE_PREFIX_MAP)=$(PCCL_SRC_ROOT)=.
endif

HGFLAGS += $(shell echo "$(PPU_OPTION)" | sed "s/;/ /g")
HGLDFLAGS  := $(PPU_LIB:%=-L%)

ifeq ($(NO_DLTO), 1)
  RDC_OPTION := -rdc=true
else
  RDC_OPTION := -rdc=true -dlto
  HGFLAGS += -O3
  HGLDFLAGS += -dlto
endif

ifneq ($(PPU_HOST_ARCH),$(PPU_TARGET_ARCH))
  HGFLAGS += -ccbin ${PPU_TARGET_ARCH}-linux-gnu-g++
  HGLDFLAGS += -ccbin ${PPU_TARGET_ARCH}-linux-gnu-g++
endif

ifeq ($(CC), 1)
  CXXFLAGS += -fprofile-arcs -ftest-coverage
  LDFLAGS +=  -fprofile-arcs
endif

define append_arch
$(eval HGFLAGS += -arch=$(1))
$(eval HGLDFLAGS += -arch=$(1))
endef
TMP := $(foreach arch,$(SPLIT_ARCHS), $(call append_arch,$(arch)))

########## GCOV ##########
GCOV ?= 0 # disable by default.
GCOV_FLAGS := $(if $(filter 0,${GCOV} ${DEBUG}),,--coverage) # only gcov=1 and debug =1
CXXFLAGS += ${GCOV_FLAGS}
HGFLAGS += ${GCOV_FLAGS:%=-Xcompiler %}
LDFLAGS += ${GCOV_FLAGS}
HGLDFLAGS += ${GCOV_FLAGS:%=-Xcompiler %}
########## GCOV ##########

ifeq ($(DEBUG), 0)
  CXXFLAGS += -O3 -g
  HGFLAGS  += -dopt 3
else
  CXXFLAGS += -O0 -g -ggdb3
  HGFLAGS  += -dopt 0 -G -g
endif

# TODO: Currently "--llvm-options -v" is not supported
ifneq ($(VERBOSE), 0)
  HGFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-parameter # --llvm-options -v
  CXXFLAGS  += -Wall -Wextra -fpermissive
else
  CXXFLAGS  += -Werror
.SILENT:
endif

ifneq ($(TRACE), 0)
  CXXFLAGS += -DENABLE_TRACE
endif

ifneq ($(NO_EXTRA_WARP), 0)
  HGFLAGS += -DNO_EXTRA_WARP
  CXXFLAGS  += -DNO_EXTRA_WARP
endif

ifneq ($(KERNEL_LOOPBACK), 0)
  HGFLAGS += -DKERNEL_LOOPBACK
  CXXFLAGS  += -DKERNEL_LOOPBACK
endif

ifeq ($(NO_TSM), 1)
  CXXFLAGS  += -DNO_TSM
  HGFLAGS += -DNO_TSM
endif

ifeq ($(NET_BYPASS), 1)
  CXXFLAGS  += -DNET_BYPASS
endif

ifeq ($(LOAD_BYPASS), 1)
  CXXFLAGS  += -DLOAD_BYPASS
  HGFLAGS += -DLOAD_BYPASS
endif

ifeq ($(STORE_BYPASS), 1)
  CXXFLAGS  += -DSTORE_BYPASS
  HGFLAGS += -DSTORE_BYPASS
endif

ifeq ($(HGTX), 0)
  CXXFLAGS  += -DHGTX_DISABLE
else
  alios = $(shell cat /etc/os-release | grep -i alios)
  ifneq ($(alios),)    #  add -DHGTX3_USE_CHECKED_OVERLOADS_FOR_GET=0 for alios, compile issue
    CXXFLAGS += -DHGTX3_USE_CHECKED_OVERLOADS_FOR_GET=0
  endif
endif

ifeq ($(KEEP), 1)
  HGFLAGS += -keep -lineinfo
endif

ifeq ($(DOCA_DEBUG), 1)
  HGFLAGS += -DPCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
endif

ifneq ($(PROFAPI), 0)
  CXXFLAGS += -DPROFAPI
endif
