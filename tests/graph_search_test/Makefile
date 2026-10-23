#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

TESTDIR := $(abspath .)
TESTSRCDIR := $(TESTDIR)/src
TESTBUILDDIR := $(TESTDIR)/build
TESTOBJDIR := $(TESTBUILDDIR)/obj
TESTBINDIR := $(TESTBUILDDIR)

GTEST_DIR := $(TESTDIR)/3rd_party/gtest
GTEST_INCLUDE := $(GTEST_DIR)/include
GTEST_LIB := $(GTEST_DIR)/lib/libgtest.a

PCCL_HOME ?= $(abspath ../../build)
LIBDIR := $(PCCL_HOME)/lib
INCDIR := $(PCCL_HOME)/include
PCCL_SO := $(LIBDIR)/libpccl.so

PPU_SDK ?= $(shell echo $$PPU_SDK)
PPU_INCLUDE := $(PPU_SDK)/include
PPU_LIB := $(PPU_SDK)/lib

TESTSRCFILES := $(TESTSRCDIR)/graph_search_tests.cc

GRAPHSEARCHTEST := $(TESTBINDIR)/graphSearchTest

TESTCXXFLAGS := -std=c++17 -g -O3 \
                -Wno-unused-function \
                -Wno-unused-parameter \
                -Wno-sign-compare \

TESTINCFLAGS := -I$(TESTDIR)/include \
                -I$(INCDIR) \
                -I../../src \
                -I../../src/include \
                -I../../src/include/plugin \
                -I../../src/graph \
                -I$(PPU_INCLUDE) \
                -I$(GTEST_INCLUDE)

TESTLDFLAGS := -pthread \
               $(PCCL_SO) \
               -Wl,--disable-new-dtags \
               -lrt -ldl \
               -L$(PPU_LIB) -lhggcrt1 -lhggc \
               $(GTEST_LIB)

.PHONY: all build clean setup_3rdparty

all: build

setup_3rdparty: $(GTEST_LIB)

$(GTEST_LIB):
	@if [ ! -f $(GTEST_LIB) ]; then \
		echo "Setting up 3rd party dependencies..."; \
		bash $(TESTDIR)/setup_3rd.sh; \
	fi

$(PCCL_SO):
	@echo "Error: PCCL library not found at $(PCCL_SO)"
	@echo "Please build PCCL first by running: make -C ../../src lib"
	@exit 1

build: $(GRAPHSEARCHTEST)

$(GRAPHSEARCHTEST): $(TESTSRCFILES) $(GTEST_LIB) $(PCCL_SO)
	@echo "Building graphSearchTest"
	@mkdir -p $(TESTOBJDIR)
	@mkdir -p $(TESTBINDIR)
	$(CXX) $(TESTCXXFLAGS) $(TESTINCFLAGS) $(TESTSRCFILES) -o $@ $(TESTLDFLAGS)
	@echo "graphSearchTest built successfully: $@"

clean:
	rm -rf $(TESTBUILDDIR) && rm -rf $(TESTDIR)/3rd_party