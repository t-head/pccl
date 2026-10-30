#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information
#

include ../../makefiles/common.mk
include ../../makefiles/version.mk
BUILDDIR ?= $(abspath ../../build)
TXZPREPDIR  := $(BUILDDIR)/srctxz
PKGDIR  := $(BUILDDIR)/pkg/srctxz/

TXZGEN_IN  := $(wildcard *.in)
TXZGEN     := $(TXZGEN_IN:.in=)
TXZTARGETS := $(patsubst %, $(TXZPREPDIR)/%, $(TXZGEN))

PKG_REVISION   ?= 3
PKG_ARCH       := $(shell uname -m)

prep: $(TXZTARGETS)

build: prep
	(cd ../..; bash ./build.sh --use_sdk)
	@printf "Building source tar.xz package\n"
	(cd $(BUILDDIR); bash srctxz/create_srctxz.sh)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../../pccl-src*.txz $(PKGDIR)

clean:
	rm -Rf $(TXZPREPDIR) $(PKGDIR)

$(TXZPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(TXZPREPDIR)
	sed -e "s/\$${pccl:Major}/$(PCCL_MAJOR)/g" \
	    -e "s/\$${pccl:Minor}/$(PCCL_MINOR)/g" \
	    -e "s/\$${pccl:Patch}/$(PCCL_PATCH)/g" \
	    -e "s/\$${pccl:Suffix}/$(PCCL_SUFFIX)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    $< > $@
