#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Include common build rules
include ../../../../makefiles/common.mk
include ../../../../makefiles/examples.mk

# Target executable
TARGET = allreduce_lsa

# Common utilities
COMMON_INC = ../../common/include
COMMON_SRC = ../../common/src

# Build configuration
# pccl_device.h is an internal PCCL header that only the device API examples
# need. The default install at $(PCCL_HOME)/include (added by
# examples.mk) is the cuda-free, pccl-spelled tree and already ships the
# doca_gpunetio/ subtree under pccl_device/gin/providers/mlx5/, so no spelling
# override is needed here -- we just prepend the example's own COMMON_INC.
INCLUDES := -I$(COMMON_INC) $(INCLUDES)

# Source files
SOURCES = main.cu $(COMMON_SRC)/utils.cc
OBJECTS = $(SOURCES:.cu=.o)
OBJECTS := $(OBJECTS:.cc=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
ifeq ($(MPI),1)
	$(MPICXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
else
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -lpthread -o $@
endif
	@echo "Built target $@"

# Compile source files
%.o: %.cu
	$(HGCC) $(HGCUFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cc
ifeq ($(MPI),1)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
else
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
endif

# Test target
test: $(TARGET)
	@echo "Testing $(TARGET)..."
ifeq ($(MPI),1)
	@echo "Running with 2 processes"
	$(MPIRUN) -np 2 ./$(TARGET)
else
	@echo "Running with all available GPUs"
	./$(TARGET)
endif

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install target
install: $(TARGET)
	@mkdir -p $(PREFIX)/bin
	cp $(TARGET) $(PREFIX)/bin/

# Help
help:
	@echo "PCCL Example: Device API Allreduce"
	@echo "=============================================="
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the example (default)"
	@echo "  test      - Build and run test with all GPUs"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to PREFIX/bin (default: /usr/local/bin)"
	@echo "  help      - Show this help"

.PHONY: all test clean install help
