#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
.DEFAULT_GOAL: build
include ../../makefiles/common.mk
# Variables
BUILDDIR ?= .
PCCL_SOURCE_HOME ?= ../..
INC := -Ipccl -I$(PCCL_SOURCE_HOME)/build/include -I$(PCCL_SOURCE_HOME)/src/include -I$(PCCL_SOURCE_HOME)/src/include/plugin -I$(PCCL_SOURCE_HOME)/src/include/plugin/profiler  -I$(PCCL_SOURCE_HOME)/src
PLUGIN_SO := libpccl-profiler-inspector.so
VERSION_FILE := version.cc

# Source files
SRCS := inspector.cc inspector_prom.cc inspector_plugin.cc inspector_hggcwrap.cc json.cc
DST_DIR   := $(BUILDDIR)

# Default target
build: ${DST_DIR}/$(PLUGIN_SO)

# Rule to build the plugin
${DST_DIR}/$(PLUGIN_SO): $(VERSION_FILE) $(SRCS)
	@echo "Compiling to create $@ from $^"
	$(CXX) $(INC) $(CXXFLAGS)  -fPIC -shared -o  $@ -Wl,-soname,$(PLUGIN_SO) $^
ifeq ($(REL_NO_SYM), 1)
	$(OBJCOPY) --only-keep-debug $@ ${DST_DIR}/$(PLUGIN_SO).dbginfo
	$(STRIP) --strip-all $@
	$(OBJCOPY) --add-gnu-debuglink=${DST_DIR}/$(PLUGIN_SO).dbginfo $@
endif

# Rule to generate version.cc
$(VERSION_FILE):
	@GIT_INFO=$$(./utils/extract_git_version.sh); \
	echo '#include "version.h"' > $(VERSION_FILE).tmp; \
	echo 'const char* get_git_version_info() { return "'$$GIT_INFO'"; }' >> $(VERSION_FILE).tmp; \
	if ! cmp $(VERSION_FILE).tmp $(VERSION_FILE); then \
		echo "updating ${VERSION_FILE} file -> $$GIT_INFO"; \
		mv $(VERSION_FILE).tmp $(VERSION_FILE); \
	else \
		echo "${VERSION_FILE} up to date -> $$GIT_INFO"; \
		rm $(VERSION_FILE).tmp; \
	fi

# Clean target
clean:
	rm -f $(VERSION_FILE) ${DST_DIR}/$(PLUGIN_SO)

# Phony targets
.PHONY: build clean
