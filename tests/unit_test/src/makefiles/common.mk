#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

HGCC ?= $(PPU_SDK)/bin/hgcc
PPU_INC ?= $(PPU_SDK)/include
PPU_LIB ?= $(PPU_SDK)/lib

HGGC_VERSION = $(strip $(shell $(HGCC) --version 2>/dev/null | grep version | sed 's/.*version //' | sed 's/\,.*//'))
ifeq ($(HGGC_VERSION),)
  HGGC_VERSION = $(strip $(shell echo $(PPU_SDK) | sed -n 's|.*/PPU_SDK_v*\([0-9][0-9.]*\).*|\1|p'))
endif
ifeq ($(HGGC_VERSION),)
  HGGC_VERSION = 0.0.0
endif
HGGC_MAJOR = $(shell echo $(HGGC_VERSION) | cut -d "." -f 1)
HGGC_MINOR = $(shell echo $(HGGC_VERSION) | cut -d "." -f 2)
HGGC_PATCH = $(shell echo $(HGGC_VERSION) | cut -d "." -f 3 | sed 's/[^0-9].*//')
HGGC_PATCH := $(if $(HGGC_PATCH),$(HGGC_PATCH),0)
HGGC_VERSION_CODE := $(shell expr $(HGGC_MAJOR) \* 10000 + $(HGGC_MINOR) \* 100 + $(HGGC_PATCH) 2>/dev/null || echo 0)
HGGC_HAS_OKI := $(shell test "$(HGGC_VERSION_CODE)" -ge 20200 2>/dev/null && echo 1 || echo 0)

# PPU_ARCHS can be any combination of ppu_10,ppu_15
SPLIT_ARCHS ?= $(shell echo $(PPU_ARCHS) | tr ',' ' ')
VERBOSE ?= 0
KEEP ?= 0
DEBUG ?= 0
REL_NO_SYM ?= 0
MAX_VREGS ?= 96
UNIT_TEST_ROOT := $(abspath ../..)

CXXFLAGS   := -DHGGC_MAJOR=$(HGGC_MAJOR) -DHGGC_MINOR=$(HGGC_MINOR) -fPIC \
              -Wall -Wno-unused-function -Wno-sign-compare -Wno-deprecated-declarations  \
              -std=c++11 -Wvla \
              $(PPU_INC:%=-I%) \
              $(CXXFLAGS)

HGFLAGS    := -std=c++11 -Xcompiler -fPIC $(PPU_INC:%=-I%)
HGFLAGS += $(shell echo "$(PPU_OPTION)" | sed "s/;/ /g")

ifeq ($(REL_NO_SYM), 1)
  ifneq ($(UNIT_TEST_ROOT),)
    PREFIX_MAP_FLAGS := $(shell \
      $(HGCC) -Xcompiler -ffile-prefix-map=/tmp=. -x c /dev/null -c -o /dev/null 2>/dev/null \
      && echo "-ffile-prefix-map=$(UNIT_TEST_ROOT)=." \
      || echo "-fdebug-prefix-map=$(UNIT_TEST_ROOT)=. -fmacro-prefix-map=$(UNIT_TEST_ROOT)=.")
    CXXFLAGS += $(PREFIX_MAP_FLAGS)
    HGFLAGS += $(PREFIX_MAP_FLAGS:%=-Xcompiler %)
  endif
endif

HGGCRT_LDFLAG = -lhggc_wrapper -lhg_wrapper
# Library flags are shared by hgcc and the OKI host-compiler final link.
COMMON_LDFLAGS = $(PPU_LIB:%=-L%) $(HGGCRT_LDFLAG) -lpccl -lhgml
# Required coverage flags must survive command-line LDFLAGS overrides.
# Keep user flags intact and avoid duplicating coverage from LDFLAGS below.
HOST_LDFLAGS = $(LDFLAGS) $(filter-out $(LDFLAGS),$(GCOV_FLAGS))
HGLDFLAGS = $(COMMON_LDFLAGS) $(HOST_LDFLAGS:%=-Xcompiler %)
HOST_CXX ?= $(CXX)
STRIP ?= strip
ifneq ($(PPU_HOST_ARCH),$(PPU_TARGET_ARCH))
  HOST_CXX := $(PPU_TARGET_ARCH)-linux-gnu-g++
  STRIP := $(PPU_TARGET_ARCH)-linux-gnu-strip
  HGFLAGS += -ccbin $(HOST_CXX)
  HGLDFLAGS += -ccbin $(HOST_CXX)
endif

########## ltOki ##########
# ltOki requires RDC, explicit device linking, and a host-compiler final link.
USE_OKI ?= $(HGGC_HAS_OKI)
ifeq ($(USE_OKI), 1)
  # The LTO module has SIMT e_flags even for --no-simt builds.
  OKI_DEV_LIB ?= $(PPU_LIB)/liblt_oki_device_simt.a
  RDC_OPTION := -rdc=true -dlto
  DEVLINK_FLAGS := -Xcompiler -fPIC -dlto $(PPU_LIB:%=-L%)
  ifneq ($(PPU_HOST_ARCH),$(PPU_TARGET_ARCH))
    DEVLINK_FLAGS += -ccbin $(HOST_CXX)
  endif
  ifneq ($(USE_SIMT), 1)
    DEVLINK_FLAGS += --no-simt
  endif
  OKI_HOST_LDFLAGS = $(COMMON_LDFLAGS) $(HOST_LDFLAGS) -llt_oki -lrt
  HGFLAGS += -DOKI_SUPPORT
endif
########## ltOki ##########

########## GCOV ##########
GCOV ?= 0 # disable by default.
GCOV_FLAGS := $(if $(filter 0,${GCOV} ${DEBUG}),,--coverage) # only gcov=1 and debug =1
CXXFLAGS += ${GCOV_FLAGS}
HGFLAGS += ${GCOV_FLAGS:%=-Xcompiler %}
LDFLAGS += ${GCOV_FLAGS}
########## GCOV ##########

ifeq ($(DEBUG), 0)
  HGFLAGS   += -dopt 3
  CXXFLAGS  += -O3
else
  HGFLAGS   += -dopt 0 -G -g -lineinfo -DTRACE
  CXXFLAGS  += -O0 -g -ggdb3 -DTRACE
endif

define append_arch
$(eval HGFLAGS += -arch=$(1))
$(eval DEVLINK_FLAGS += -arch=$(1))
endef
TMP := $(foreach arch,$(SPLIT_ARCHS), $(call append_arch,$(arch)))

ifneq ($(VERBOSE), 0)
  HGFLAGS += -Xcompiler -Wall,-Wextra,-Wno-unused-parameter --resource-usage
  CXXFLAGS += -Wall -Wextra
else
.SILENT:
endif

ifeq ($(KEEP), 1)
  HGFLAGS += --keep -lineinfo
endif
