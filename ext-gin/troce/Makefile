#
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

.DEFAULT_GOAL := build
include ../../makefiles/common.mk

BUILDDIR ?= $(abspath ../../build)
PCCL_SOURCE_HOME := $(abspath ../..)
OBJDIR := $(BUILDDIR)/obj/ext-gin/troce
PLUGIN_SO := $(BUILDDIR)/lib/libpccl-gin-troce.so
INCDIRS := $(PCCL_SOURCE_HOME)/src/include $(PCCL_SOURCE_HOME)/src/include/plugin \
           $(PCCL_SOURCE_HOME)/src $(BUILDDIR)/include
SOURCES := gin_troce.cc gin_host_troce.cc trocedvwrap.cc
OBJECTS := $(SOURCES:%.cc=$(OBJDIR)/%.o)

.PHONY: build clean
build: $(PLUGIN_SO)

$(OBJDIR)/%.o: %.cc
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCDIRS:%=-I%) -MMD -MP -c $< -o $@

$(PLUGIN_SO): $(OBJECTS) exports.map $(BUILDDIR)/lib/libpccl.so
	$(CXX) -shared -Wl,-z,defs -Wl,--version-script=exports.map \
	  -Wl,-soname,libpccl-gin-troce.so -Wl,-rpath,'$$ORIGIN' \
	  -o $@ $(OBJECTS) -L$(BUILDDIR)/lib -lpccl $(PPU_LIB:%=-L%) \
	  -lhggc_wrapper -lhg_wrapper -lpthread -ldl $(LDFLAGS)

clean:
	rm -rf $(OBJDIR)
	rm -f $(PLUGIN_SO)

-include $(OBJECTS:%.o=%.d)
