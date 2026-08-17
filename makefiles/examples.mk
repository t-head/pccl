#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
#
# PCCL example build fragment
#
# Companion to makefiles/common.mk. common.mk wires up the toolchain
# (HGCC, PPU_INC, PPU_LIB, library-build flags); this fragment adds the
# bits each example needs to find libpccl and the public headers, plus
# the recipes for compiling .cu (HGGC kernel) sources used by 06_device_api.
#

# Default to source tree layout: examples live under docs/examples/<group>/<name>,
# so PCCL_HOME = ../../../../build from the leaf example dir.
PCCL_HOME ?= $(abspath ../../../../build)

# PCCL public headers + PPU headers come from common.mk (PPU_INC).
INCLUDES = -I$(PCCL_HOME)/include $(PPU_INC:%=-I%)

# Library search paths for libpccl and the HGGC runtime.
LIBRARIES = -L$(PCCL_HOME)/lib -L$(PPU_LIB)

# Link with libpccl plus the HGGC runtime (mirrors src/Makefile).
LDFLAGS += -lpccl -Wl,-rpath,$(PCCL_HOME)/lib -Wl,-rpath,$(PPU_LIB)
LDFLAGS += -lhggc_wrapper -lhg_wrapper

HGCUFLAGS = -x hg \
            -DPPU -DUSE_HGGC -DSWITCH_TO_HGGCRT \
            -Xcompiler -fPIC \
            -Xhggcfe -Wno-c++17-extensions \
            -Xhggcfe -Wno-ignored-attributes \
            -std=c++14 \
            $(PPU_INC:%=-I%)

HGCUFLAGS += $(shell echo "$(PPU_OPTION)" | sed "s/;/ /g")

ifneq ($(PPU_HOST_ARCH),$(PPU_TARGET_ARCH))
HGCUFLAGS += -ccbin $(PPU_TARGET_ARCH)-linux-gnu-g++
endif

ifeq ($(DEBUG), 0)
HGCUFLAGS += -O3 -dopt 3
else
HGCUFLAGS += -O0 -dopt 0 -G -g
endif

ifeq ($(KEEP), 1)
HGCUFLAGS += -keep -lineinfo
endif

# Multi-arch fat binary: forward every arch in SPLIT_ARCHS (set in common.mk
# from PPU_ARCHS, which defaults to "ppu0010,ppu0015"). Each -arch= adds one
# target to the fatbin so a single binary runs on both ZW810E and ZW890.
define _examples_append_arch
$(eval HGCUFLAGS += -arch=$(1))
endef
$(foreach arch,$(SPLIT_ARCHS),$(eval $(call _examples_append_arch,$(arch))))

# MPI configuration. Examples opt in by setting MPI=1 before including this file
# (or on the make command line). The resulting binary still runs as a single
# process when invoked directly (OpenMPI singleton mode); mpirun -np N works as
# usual.
ifeq ($(MPI), 1)
ifdef MPI_HOME
MPICXX ?= $(MPI_HOME)/bin/mpicxx
MPIRUN ?= $(MPI_HOME)/bin/mpirun
else
MPICXX ?= mpicxx
MPIRUN ?= mpirun
endif

CXXFLAGS += -DMPI_SUPPORT
endif
