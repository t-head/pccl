#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information
#
.DEFAULT_GOAL: build
include ../../makefiles/common.mk
BUILDDIR ?= .
PCCL_SOURCE_HOME ?= ../..
INCDIR ?= pccl $(PCCL_SOURCE_HOME)/build/include $(PCCL_SOURCE_HOME)/src/include/plugin/profiler $(PCCL_SOURCE_HOME)/src/include

SRC_FILES := $(wildcard *.cc)
DST_DIR   := $(BUILDDIR)
OBJ_FILES := $(SRC_FILES:%.cc=${DST_DIR}/%.o)
DEP_FILES := $(OBJ_FILES:%.o=%.dep)

build: ${DST_DIR}/libpccl-profiler-example.so

${DST_DIR}/libpccl-profiler-example.so: ${SRC_FILES}
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p ${DST_DIR}
	$(CXX) $(INCDIR:%=-I%) $(CXXFLAGS) -fPIC -shared -o  $@ $^
ifeq ($(REL_NO_SYM), 1)
	objcopy --only-keep-debug $(DST_DIR)/libpccl-profiler-example.so $(DST_DIR)/libpccl-profiler-example.so.dbginfo
	strip --strip-all $(DST_DIR)/libpccl-profiler-example.so
	objcopy --add-gnu-debuglink=$(DST_DIR)/libpccl-profiler-example.so.dbginfo $(DST_DIR)/libpccl-profiler-example.so
endif

clean:
	rm -f ${DST_DIR}/libpccl-profiler-example.so
