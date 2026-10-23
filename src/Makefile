#
# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information
#

include ../makefiles/common.mk
include ../makefiles/version.mk

##### src files
SRCDIR ?= $(abspath .)
INCEXPORTS  := pccl.h pccl_device.h \
	$(patsubst include/%,%,$(wildcard include/pccl_device/*.h include/pccl_device/*/*.h include/pccl_device/*/*/*.h))
LIBSRCFILES := $(filter-out ce_coll_oki.cc,$(wildcard *.cc)) \
               $(wildcard graph/*.cc) \
               $(wildcard register/*.cc) \
               $(wildcard misc/*.cc) \
               $(filter-out param/pcclparam.cc,$(wildcard param/*.cc)) \
               $(wildcard collectives/*.cc) \
               $(wildcard transport/*.cc) \
               $(wildcard transport/net_ib/*.cc) \
               $(wildcard transport/net_ib/gdaki/*.cc) \
               $(wildcard scheduler/*.cc) \
               $(wildcard plugin/*.cc) \
               $(wildcard plugin/profiler/*.cc) \
               $(filter-out ras/client.cc,$(wildcard ras/*.cc)) \
               $(wildcard pccl_device/*.cc) \
               $(wildcard gin/*.cc) \
               $(wildcard plugin/gin/*.cc)
ifeq ($(HGGC_HAS_OKI), 1)
  LIBSRCFILES += ce_coll_oki.cc
endif

LIBSRCFILES_CPP := $(wildcard transport/net_ib/gdaki/doca-gpunetio/src/*.cpp)

BINSRCFILES := ras/client.cc ras/analyzer.cc
PARAMBINSRCFILES := param/pcclparam.cc

##### lib files
LIBNAME     := libpccl.so
STATICLIBNAME := libpccl_static.a
##### binaries
BINNAME := pcclras
PARAMBIN := pcclparam
##### pkgconfig files
#PKGCONFIGFILE := pccl.pc
##### dirs
BUILDDIR ?= $(abspath ../build)
TESTSDIR = $(abspath ../tests)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj
PKGDIR := $(BUILDDIR)/lib/pkgconfig
BINDIR := $(BUILDDIR)/bin
##### target files
INCTARGETS := $(INCEXPORTS:%=$(INCDIR)/%)

# DOCA-GPUNetIO headers ship inside src/transport/net_ib/gdaki/doca-gpunetio/
# but are exposed to downstream consumers (ext-kernel, ext-tuner) under
# build/include/pccl_device/gin/gdaki/doca_gpunetio/. Append the install
# targets and define the install rules BEFORE any rule that takes
# $(INCTARGETS) as prerequisites — Make expands the prereq list at the time
# the rule is parsed, so a later `+=` would never reach them.
DOCA_GPUNETIO_INC := $(SRCDIR)/transport/net_ib/gdaki/doca-gpunetio/include
DOCA_INC_INSTALL  := $(INCDIR)/pccl_device/gin/gdaki/doca_gpunetio
DOCA_INC_SRC      := $(DOCA_GPUNETIO_INC)/doca_gpunetio_device.h \
                     $(wildcard $(DOCA_GPUNETIO_INC)/common/*.h) \
                     $(wildcard $(DOCA_GPUNETIO_INC)/device/*.cuh)
DOCA_INC_TARGETS  := $(DOCA_INC_SRC:$(DOCA_GPUNETIO_INC)/%=$(DOCA_INC_INSTALL)/%)
INCTARGETS        += $(DOCA_INC_TARGETS)

$(DOCA_INC_INSTALL)/%.h : $(DOCA_GPUNETIO_INC)/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	install -m 644 $< $@

$(DOCA_INC_INSTALL)/%.cuh : $(DOCA_GPUNETIO_INC)/%.cuh
	@printf "Grabbing   %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	install -m 644 $< $@

LIBSONAME  := $(LIBNAME:%=%.$(PCCL_MAJOR))
LIBTARGET  := $(LIBNAME:%=%.$(PCCL_MAJOR).$(PCCL_MINOR).$(PCCL_PATCH))
STATICLIBTARGET := $(STATICLIBNAME)
#PKGTARGET  := $(PKGCONFIGFILE)
LIBOBJ_CC  := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)
LIBOBJ_CPP := $(LIBSRCFILES_CPP:%.cpp=$(OBJDIR)/%.o)
LIBOBJ     := $(LIBOBJ_CC) $(LIBOBJ_CPP)
DEPFILES   := $(LIBOBJ:%.o=%.d)
BINOBJ     := $(BINSRCFILES:%.cc=$(OBJDIR)/%.o)
PARAMBINOBJ := $(PARAMBINSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d) $(BINOBJ:%.o=%.d) $(PARAMBINOBJ:%.o=%.d)
LDFLAGS    +=  -lpthread -lrt -ldl -L$(LIBDIR) $(PPU_LIB:%=-L%)
ifeq ($(HGGC_HAS_OKI), 1)
  ifeq ($(USE_SIMT), 1)
    LDFLAGS += -llt_oki -llt_oki_device_simt
  else
    LDFLAGS += -llt_oki -llt_oki_device_simd
  endif
endif
LDFLAGS += -lhggc_wrapper -lhg_wrapper
DEVICELIB  := $(BUILDDIR)/obj/collectives/device/colldevice.a
INCPLUGIN  := include/plugin

ifeq ($(STATIC_LINK), 1)
  LDFLAGS += -static-libgcc -static-libstdc++
endif

PCCL_VERSION := $(PCCL_MAJOR).$(PCCL_MINOR).$(PCCL_PATCH)

# Fields written to release.yaml by the release_yaml target.
# If PACKAGE_VERSION (a.b.c-xxxxxx) is provided externally and its a.b.c part
# matches version.mk, use it; otherwise fall back to the version.mk a.b.c.
ifneq ($(strip $(PACKAGE_VERSION)),)
PACKAGE_VERSION_ABC := $(firstword $(subst -, ,$(PACKAGE_VERSION)))
ifeq ($(PACKAGE_VERSION_ABC),$(PCCL_VERSION))
RELEASE_YAML_VERSION := $(PACKAGE_VERSION)
else
$(warning PACKAGE_VERSION a.b.c ($(PACKAGE_VERSION_ABC)) does not match version.mk ($(PCCL_VERSION)), falling back to $(PCCL_VERSION))
RELEASE_YAML_VERSION := $(PCCL_VERSION)
endif
else
RELEASE_YAML_VERSION := $(PCCL_VERSION)
endif

RELEASE_YAML_NAME := pccl

ifeq ($(HGGCRT_VERSION),12)
RELEASE_YAML_HGGCRT_VERSION := v2
else ifeq ($(HGGCRT_VERSION),13)
RELEASE_YAML_HGGCRT_VERSION := v3
else
RELEASE_YAML_HGGCRT_VERSION := unknown
endif

##### rules
ifeq ($(DISABLE_TESTS), 1)
build : lib staticlib binary release_yaml
else
build : lib staticlib tests release_yaml
endif

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) #$(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

binary : $(BINDIR)/$(BINNAME) $(BINDIR)/$(PARAMBIN)

tests : graph_search_test unit_test

graph_search_test: lib
	$(MAKE) -C $(TESTSDIR)/graph_search_test

unit_test: lib
	cd $(TESTSDIR)/unit_test && ./build.sh --ppu_archs $(PPU_ARCHS)

$(DEVICELIB): ALWAYS_REBUILD $(INCTARGETS)
	$(MAKE) -C collectives/device

release_yaml :
	@mkdir -p $(BUILDDIR)
	@echo "Generating $(BUILDDIR)/release.yaml"
	@echo "version: $(RELEASE_YAML_VERSION)" > $(BUILDDIR)/release.yaml
	@echo "name: $(RELEASE_YAML_NAME)" >> $(BUILDDIR)/release.yaml
	@echo "hggcrt_version: $(RELEASE_YAML_HGGCRT_VERSION)" >> $(BUILDDIR)/release.yaml

# Empty target to force rebuild
ALWAYS_REBUILD:

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(INCDIR)/pccl.h : pccl.h.in
# PCCL_VERSION_CODE(X,Y,Z) ((X) * 1000 + (Y) * 100 + (Z))
	@$(eval PCCL_VERSION_CODE := $(shell printf "%d%d%02d" $(PCCL_MAJOR) $(PCCL_MINOR) $(PCCL_PATCH)))
	mkdir -p $(INCDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e "s/\$${pccl:Major}/$(PCCL_MAJOR)/g" \
	    -e "s/\$${pccl:Minor}/$(PCCL_MINOR)/g" \
	    -e "s/\$${pccl:Patch}/$(PCCL_PATCH)/g" \
	    -e "s/\$${pccl:Suffix}/$(PCCL_SUFFIX)/g" \
	    -e "s/\$${pccl:Version}/$(PCCL_VERSION_CODE)/g" \
	    $< > $@

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) $(DEVICELIB)
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $(DEVICELIB) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

null :=
space := $(null) #
comma := ,

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) $(DEVICELIB)
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	printf "create $@\naddlib $(DEVICELIB)\naddmod $(subst $(space),$(comma),$(strip $(LIBOBJ)))\nsave\nend" | ar -M

$(BINDIR)/$(BINNAME): $(BINOBJ)
	@printf "Linking    %-35s > %s\n" $(BINNAME) $@
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BINDIR)/$(PARAMBIN): $(PARAMBINOBJ) $(LIBDIR)/$(LIBTARGET)
	@printf "Linking    %-35s > %s\n" $(PARAMBIN) $@
	mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $< -o $@ -L$(LIBDIR) -lpccl $(LDFLAGS)

$(PKGDIR)/pccl.pc : pccl.pc.in
	mkdir -p $(PKGDIR)
	@printf "Generating %-35s > %s\n" $< $@
	sed -e 's|$${pccl:Prefix}|\$(PREFIX)|g' \
	    -e "s/\$${pccl:Major}/$(PCCL_MAJOR)/g" \
	    -e "s/\$${pccl:Minor}/$(PCCL_MINOR)/g" \
	    -e "s/\$${pccl:Patch}/$(PCCL_PATCH)/g" \
	    $< > $@

$(INCDIR)/%.h : %.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/pccl_%.h : $(SRCDIR)/include/pccl_%.h

	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(INCDIR)/pccl_device/%.h : $(SRCDIR)/include/pccl_device/%.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	install -m 644 $< $@

$(INCDIR)/pccl_device.h : $(SRCDIR)/include/pccl_device.h
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(INCDIR)
	install -m 644 $< $@

$(PKGDIR)/%.pc : %.pc
	@printf "Grabbing   %-35s > %s\n" $< $@
	mkdir -p $(PKGDIR)
	install -m 644 $< $@

$(OBJDIR)/%.o : %.cc $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) -I$(INCPLUGIN) $(CXXFLAGS) -I$(SRCDIR)/include -I$(DOCA_GPUNETIO_INC) -c $(SRCDIR)/$< -o $@
	@$(CXX) -I. -I$(INCDIR) -I$(INCPLUGIN) $(CXXFLAGS) -I$(SRCDIR)/include -I$(DOCA_GPUNETIO_INC) -M $(SRCDIR)/$< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

$(OBJDIR)/%.o : %.cpp $(INCTARGETS)
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) -I$(INCPLUGIN) $(CXXFLAGS) -I$(SRCDIR)/include -I$(DOCA_GPUNETIO_INC) -c $(SRCDIR)/$< -o $@
	@$(CXX) -I. -I$(INCDIR) -I$(INCPLUGIN) $(CXXFLAGS) -I$(SRCDIR)/include -I$(DOCA_GPUNETIO_INC) -M $(SRCDIR)/$< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	$(MAKE) -C collectives/device clean
	rm -rf ${INCDIR} ${LIBDIR} ${PKGDIR} ${OBJDIR} ${BINDIR}
	cd ${TESTSDIR}/graph_search_test && make clean
	cd ${TESTSDIR}/unit_test && make clean

install : lib binary
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	mkdir -p $(PREFIX)/bin
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -r -v $(BUILDDIR)/include/* $(PREFIX)/include/
	cp -v $(BUILDDIR)/bin/$(BINNAME) $(PREFIX)/bin/
	cp -v $(BUILDDIR)/bin/$(PARAMBIN) $(PREFIX)/bin/

FILESTOFORMAT := $(shell find . -name ".\#*" -prune -o \( -name "*.cc" -o -name "*.h" \) -print | grep -v -E 'ibvwrap.h|nvmlwrap.h|pccl.h')

# Note that formatting.mk defines a new target so in order to not overwrite the default target,
# it shouldn't be included at the top. Also, it uses the above definition of FILESTOFORMAT as well
# as the BUILDDIR variable.
include ../makefiles/formatting.mk
