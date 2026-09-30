#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information
#

.PHONY : all clean

ifeq ($(NO_EXT_KERNEL), 1)
default : src.build
else
default : src.build ext-kernel.build tuner-plugin.build ext-profiler.build
endif
install : src.install
ROORDIR ?= $(abspath .)
BUILDDIR ?= $(abspath ./build)
ABSROOTDIR := $(abspath $(ROORDIR))
ABSBUILDDIR := $(abspath $(BUILDDIR))
TARGETS := src pkg
clean: ${TARGETS:%=%.clean}
test.build: src.build
LICENSE_FILES := LICENSE.txt
LICENSE_TARGETS := $(LICENSE_FILES:%=$(BUILDDIR)/%)
lic: $(LICENSE_TARGETS)

FILE_PREFIX_MAP := $(shell \
  $(CXX) -ffile-prefix-map=/tmp=. -x c /dev/null -c -o /dev/null 2>/dev/null \
  && echo "-ffile-prefix-map" || echo "-fdebug-prefix-map")

export PCCL_SRC_ROOT := $(abspath .)
export FILE_PREFIX_MAP := $(FILE_PREFIX_MAP)

${BUILDDIR}/%.txt: %.txt
	@printf "Copying    %-35s > %s\n" $< $@
	mkdir -p ${BUILDDIR}
	cp $< $@

src.%:
	${MAKE} -C src $* BUILDDIR=${ABSBUILDDIR}

ext-kernel.build: | src.build
	${MAKE}	-C ext-kernel build BUILDDIR=${ABSBUILDDIR}/lib/

ext-kernel.clean:
	${MAKE}	-C ext-kernel clean BUILDDIR=${ABSBUILDDIR}/lib/

tuner-plugin.build: | src.build
	${MAKE}	-C ext-tuner build BUILDDIR=${ABSBUILDDIR}/lib/

tuner-plugin.clean:
	${MAKE}	-C ext-tuner clean BUILDDIR=${ABSBUILDDIR}/lib/

ext-profiler.build: | src.build
	${MAKE}	-C ext-profiler/example build BUILDDIR=${ABSBUILDDIR}/lib/
	${MAKE}	-C ext-profiler/inspector build BUILDDIR=${ABSBUILDDIR}/lib/

ext-profiler.clean:
	${MAKE}	-C ext-profiler/example clean BUILDDIR=${ABSBUILDDIR}/lib/
	${MAKE}	-C ext-profiler/inspector clean BUILDDIR=${ABSBUILDDIR}/lib/

pkg.%:
	${MAKE} -C pkg $* BUILDDIR=${ABSBUILDDIR}

pkg.debian.prep: lic
pkg.txz.prep: lic
