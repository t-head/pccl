#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

include ./makefiles/common.mk

##### dirs
PCCL_HOME ?= ../../../build
HGFLAGS += -I$(PCCL_HOME)/include/
COMMON_LDFLAGS += -L$(PCCL_HOME)/lib

BUILDDIR ?= ../build

.PHONY : build clean

##### target files
HGFLAGS += -I../include
CXXFLAGS += -I../include
HGFLAGS += -x hg \
              --maxrregcount=${MAX_VREGS} \
              $(shell echo "$(PPU_OPTION)" | sed "s/;/ /g")
ifneq ($(USE_SIMT), 1)
  HGFLAGS += --no-simt
endif

ifeq ($(MPI), 1)
  MPI_INCDIRS := $(shell mpicc --showme:incdirs)
  MPI_LIBDIRS := $(shell mpicc --showme:libdirs)
  HGFLAGS += -DMPI_SUPPORT -DOMPI_SKIP_MPICXX $(addprefix -I,$(MPI_INCDIRS))
  COMMON_LDFLAGS += $(addprefix -L,$(MPI_LIBDIRS)) -lmpi
endif

define build_func
  @printf "Compiling  %-35s > %s\n" $(2).cu $(1)
  mkdir -p ${BUILDDIR}
  $(HGCC) $(HGFLAGS) -c -o $(1).o $(2).cu
  $(HGCC) -o $(1) $(1).o $(HGLDFLAGS)
  @if [ $(KEEP) -eq 0 ]; then \
    rm $(1).o; \
  fi
endef

# Compile the same source twice; only its OKI kernel section uses RDC.
# hgcc final linking leaves RDC kernels unregistered, so use $(CXX).
define build_func_rdc
  @printf "Compiling  %-35s > %s (rdc)\n" $(2).cu $(1)
  mkdir -p ${BUILDDIR}
  $(HGCC) $(HGFLAGS) -c -o $(1).o $(2).cu
  $(HGCC) $(HGFLAGS) $(RDC_OPTION) -DP2P_OKI_KERNELS_ONLY -c -o $(1).oki.o $(2).cu
  $(HGCC) -dlink $(DEVLINK_FLAGS) $(1).oki.o $(OKI_DEV_LIB) -o $(1).devlink.o
  $(HOST_CXX) -o $(1) $(1).o $(1).oki.o $(1).devlink.o $(OKI_DEV_LIB) $(OKI_HOST_LDFLAGS)
  @if [ $(KEEP) -eq 0 ]; then \
    rm $(1).o $(1).oki.o $(1).devlink.o; \
  fi
endef

BIN_FILES_LIST := collPcclUnit sendRecvPcclUnit hggcGraphUnit pcclCommAbort pcclGroupSimulateEnd pcclCommMemManager
ifeq ($(MPI), 1)
BIN_FILES_LIST += pcclCommShrink
endif
BIN_FILES := $(BIN_FILES_LIST:%=${BUILDDIR}/%Test)
# bench_oki uses one SIMT warp per QP, independently of USE_SIMT for other tests.
ifeq ($(USE_OKI), 1)
BIN_FILES += ${BUILDDIR}/bench_oki
endif


info:
	@printf '%s\n' \
		'=================================================' \
		'  PPU_ARCHS      : $(PPU_ARCHS)'                   \
		'  HGCC           : $(HGCC)'                        \
		'=================================================' \

build: info ${BIN_FILES}

clean:
	rm -rf ${BUILDDIR}
	rm -f *.o *.s *.hgfb* *.hggci *.lk *.bc a.out* *host-* *hggc-* *splitter*

${BUILDDIR}/sendRecvPcclUnitTest: sendRecvPcclUnitTest.cu
	$(call build_func, $@, sendRecvPcclUnitTest)

${BUILDDIR}/collPcclUnitTest: collPcclUnitTest.cu
	$(call build_func, $@, collPcclUnitTest)

${BUILDDIR}/hggcGraphUnitTest: hggcGraphUnitTest.cu
	$(call build_func, $@, hggcGraphUnitTest)

ifeq ($(MPI), 1)
${BUILDDIR}/pcclCommShrinkTest: pcclCommShrinkTest.cu
	$(call build_func, $@, pcclCommShrinkTest)
endif

${BUILDDIR}/pcclCommAbortTest: pcclCommAbortTest.cu
	$(call build_func, $@, pcclCommAbortTest)

${BUILDDIR}/pcclCommMemManagerTest: pcclCommMemManagerTest.cu
	$(call build_func, $@, pcclCommMemManagerTest)

${BUILDDIR}/pcclGroupSimulateEndTest: pcclGroupSimulateEndTest.cu
	$(call build_func, $@, pcclGroupSimulateEndTest)

# All of bench_oki is one RDC translation unit. Keep SIMT on compile and dlink.
ifeq ($(USE_OKI), 1)
${BUILDDIR}/bench_oki: bench_oki.cu Makefile makefiles/common.mk
	@printf "Compiling  %-35s > %s (SIMT RDC)\n" bench_oki.cu $@
	mkdir -p ${BUILDDIR}
	$(HGCC) $(filter-out --no-simt,$(HGFLAGS)) $(RDC_OPTION) -c $< -o $@.o
	$(HGCC) -dlink $(filter-out --no-simt,$(DEVLINK_FLAGS)) $@.o $(OKI_DEV_LIB) -o $@.devlink.o
	$(HOST_CXX) -o $@ $@.o $@.devlink.o $(OKI_DEV_LIB) $(OKI_HOST_LDFLAGS)
else
${BUILDDIR}/bench_oki:
	@echo "bench_oki requires an ltOki-enabled SDK (USE_OKI=1)." >&2
	@false
endif
