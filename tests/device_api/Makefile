# SPDX-License-Identifier: Apache-2.0

PCCL_HOME ?= ../../build
PPU_ARCH ?= ppu_15
HGCC ?= hgcc
MPICC ?= mpicc
MPI_FLAGS := $(addprefix -I,$(shell $(MPICC) --showme:incdirs)) \
             $(addprefix -L,$(shell $(MPICC) --showme:libdirs)) -lmpi

all: gin_window gin_config gin_wait

gin_window gin_config gin_wait: %: %.cu
	$(HGCC) -x hg -std=c++17 -O2 -arch=$(PPU_ARCH) -DOMPI_SKIP_MPICXX \
	  -I$(PCCL_HOME)/include $< -o $@ -L$(PCCL_HOME)/lib -lpccl \
	  $(MPI_FLAGS) -lhggc_wrapper -lhg_wrapper -lpthread

clean:
	rm -f gin_window gin_config gin_wait

.PHONY: all clean
