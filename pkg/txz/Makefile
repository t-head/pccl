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
TXZPREPDIR  := $(BUILDDIR)/txz
PKGDIR  := $(BUILDDIR)/pkg/txz/

TXZGEN_IN  := $(wildcard *.in)
TXZGEN     := $(TXZGEN_IN:.in=)
TXZTARGETS := $(patsubst %, $(TXZPREPDIR)/%, $(TXZGEN))

PKG_ARCH   := $(shell uname -m)

prep: $(TXZTARGETS)
	$(MAKE) -C ../.. lic BUILDDIR=$(BUILDDIR)

build: prep
	(cd ../..; bash ./build.sh --use_sdk)
	@printf "Building tar.xz package\n"
	(cd $(BUILDDIR); bash txz/create_txz.sh)
	mkdir -p $(PKGDIR)
	mv $(BUILDDIR)/../pccl*.txz $(PKGDIR)

clean:
	rm -Rf $(TXZPREPDIR) $(PKGDIR)

$(TXZPREPDIR)/% : %.in
	@printf "Generating %-35s > %s\n" $< $@
	mkdir -p $(TXZPREPDIR)
	sed -e "s/\$${pccl:Major}/$(PCCL_MAJOR)/g" \
	    -e "s/\$${pccl:Minor}/$(PCCL_MINOR)/g" \
	    -e "s/\$${pccl:Patch}/$(PCCL_PATCH)/g" \
	    -e "s/\$${pccl:Suffix}/$(PCCL_SUFFIX)/g" \
	    -e "s/\$${hggc:Major}/$(HGGC_MAJOR)/g" \
	    -e "s/\$${hggc:Minor}/$(HGGC_MINOR)/g" \
	    -e "s/\$${pkg:Revision}/$(PKG_REVISION)/g" \
	    -e "s/\$${pkg:Arch}/$(PKG_ARCH)/g" \
	    $< > $@
