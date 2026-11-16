#
# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

.DEFAULT_GOAL := build
SUBDIRS := troce
BUILDDIR ?= $(abspath ../build)

.PHONY: build clean
build clean:
	@for dir in $(SUBDIRS); do \
	  $(MAKE) -C $$dir $@ BUILDDIR=$(abspath $(BUILDDIR)) || exit $$?; \
	done
