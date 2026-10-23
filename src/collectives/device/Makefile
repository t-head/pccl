# SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information

include ../../../makefiles/common.mk
include ../../../makefiles/version.mk

SRCDIR ?= $(abspath ../../../src)
BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/collectives/device

LIBSRCFILES := all_reduce.cu broadcast.cu reduce.cu all_gather.cu reduce_scatter.cu sendrecv.cu

LIBSRCFILES += functions.cu sendrecv_net_poll.cu

DEPFILES   := $(patsubst %.cu, $(OBJDIR)/%.d, $(LIBSRCFILES))
DEPENDFILES:= $(DEPFILES:%.d=%.dep)
STATICLIB  := $(OBJDIR)/colldevice.a
DEVOBJ     := $(OBJDIR)/devlink.o
RULESFILE  := $(OBJDIR)/Makefile.rules

SYMMETRIC_GENSRC_DIR := $(OBJDIR)/gensrc/symmetric
SYMMETRIC_GENOBJ_DIR := $(OBJDIR)/genobj/symmetric
SYMMETRIC_RULES_FILE := $(SYMMETRIC_GENSRC_DIR)/rules.mk

HGFLAGS  += -I. -I.. -I./symmetric -I$(BUILDDIR)/include -I$(SRCDIR)/include -Xcompiler -fPIC -x hg -c $(RDC_OPTION)
DEVLINK_FLAGS := $(shell echo "$(PPU_OPTION)" | sed "s/;/ /g")
ifneq ($(MAX_VREGS), 0) # disable max vregs set
  HGFLAGS  += -maxrregcount=$(MAX_VREGS) -Xllvm -ppu-patch-tsm-atm-cas=false
endif
ifneq ($(USE_SIMT), 1)
  HGFLAGS += --no-simt
endif
HGFLAGS += -Xllvm -ppu-patch-fence-ppu=false
# for ppu 1.0 memory model cumulativity side effect fix
HGFLAGS += -Xllvm -ppu-disable-complexsync-cum=true

ifeq ($(KEEP), 1)
  HGFLAGS += -keep -keep-dir=$(OBJDIR)/
  HGLDFLAGS += -keep -keep-dir=$(OBJDIR)/
else
  HGFLAGS += --hgfatbinary-options -compress-all
endif

all: $(STATICLIB)

# Dummy rule so that the extra dependency (%.dep) files are preserved by make
all_deps: $(DEPENDFILES)

$(SYMMETRIC_GENSRC_DIR): symmetric/generate.py
	@printf "Generating %-35s > %s\n" symmetric-sources $@
	@mkdir -p $@
	(which python3 >/dev/null || \
	  (bar='!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'; \
	   printf "\n$${bar}\nERROR: Building NCCL requires a Python 3 installation invokable as 'python3'.\n$${bar}\n\n" 1>&2; \
	   exit 1)) \
	&& ./symmetric/generate.py $@

$(SYMMETRIC_RULES_FILE): $(SYMMETRIC_GENSRC_DIR) ;

# Auto-generating the rules per op/reduction/datatype/algorithm
ifdef TEST_DT
  DATA_TYPES = $(TEST_DT)
else
  DATA_TYPES = none
endif
ifdef TEST_OP
  COLL_OPs = $(TEST_OP)
else
  COLL_OPs = none
endif
$(RULESFILE) :
	@printf "Generating %-35s > %s\n" rules $@
	@mkdir -p $(OBJDIR)
	@./gen_rules.sh --obj_dir $(OBJDIR) --coll_ops $(COLL_OPs) --data_types $(DATA_TYPES) --keep $(KEEP) > $@

-include $(RULESFILE)

# "SYMMETRIC_RULES_FILE" populates $(LIB_OBJS_SYM_GEN)
-include $(SYMMETRIC_RULES_FILE)

LIBOBJ     := $(GENOBJS) $(LIB_OBJS_SYM_GEN) $(OBJDIR)/functions.o $(OBJDIR)/onerank.o $(OBJDIR)/ce_kernel.o $(OBJDIR)/sendrecv_net_poll.o

-include $(DEPFILES)

-include $(wildcard $(OBJDIR)/genobj/symmetric/*.d)

-include $(OBJDIR)/symmetric/ce_kernel.d

$(STATICLIB): $(LIBOBJ) $(DEVOBJ)
	@printf "Archiving  %-35s > %s\n" objects $@
	ar cr $@ $^

# We do not want make to build *.d when running make clean.
# So we only provide targets for .dep which will produce .dep and .d,
# with only .d being included, and .dep keeping track of what needs to
# be regenerated.
$(OBJDIR)/%.dep : %.cu
	@mkdir -p $(dir $@)
	$(HGCC) $(HGFLAGS) -M $< -o $@.tmp
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $@.tmp > $@
	@sed -e 's/.*://' -e 's/\\$$//' < $@.tmp | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $@
	@rm -f $@.tmp
	@cp $@ $(@:.dep=.d)

$(SYMMETRIC_GENOBJ_DIR)/%.dep: $(SYMMETRIC_GENSRC_DIR)/%
	@mkdir -p $(SYMMETRIC_GENOBJ_DIR)
	@$(HGCC) $(HGFLAGS) -M $< -o $@.tmp || $(CXX) $(CXXFLAGS) -M $< -o $@.tmp
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $@.tmp > $@
	@sed -e 's/.*://' -e 's/\\$$//' < $@.tmp | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $@
	@rm -f $@.tmp
	@cp $@ $(@:.dep=.d)

$(SYMMETRIC_GENOBJ_DIR)/%.cc.o : $(SYMMETRIC_GENSRC_DIR)/%.cc $(SYMMETRIC_GENOBJ_DIR)/%.cc.dep
	@printf "Compiling  %-35s > %s\n" $(notdir $<) $@
	@mkdir -p $(dir $@)
	$(HGCC) $(HGFLAGS) -c $< -o $@

# Compiled kernels and collectives with relocatable device code ...
$(OBJDIR)/functions.o : functions.cu $(OBJDIR)/functions.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(HGCC) $(HGFLAGS) -c $< -o ${OBJDIR}/functions.o

$(OBJDIR)/onerank.o : onerank.cu
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(HGCC) $(HGFLAGS) -c $< -o ${OBJDIR}/onerank.o

# ce_kernel.o picks up header changes through an auto-generated .dep, the same
# way functions.o does (below), so edits to ce_kernel_oki.cuh, p2p_dma_kernel.cuh
# or p2p_dma_oki.cuh trigger a rebuild without listing headers here by hand.
$(OBJDIR)/ce_kernel.o : symmetric/ce_kernel.cu $(OBJDIR)/symmetric/ce_kernel.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(HGCC) $(HGFLAGS) -c $< -o ${OBJDIR}/ce_kernel.o

$(OBJDIR)/sendrecv_net_poll.o : sendrecv_net_poll.cu $(OBJDIR)/sendrecv_net_poll.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	$(HGCC) $(HGFLAGS) -c $< -o $@

# ... and create the device-side linked object with all those.
$(DEVOBJ) : $(LIBOBJ)
	@printf "Compiling devlink.o\n"
	mkdir -p `dirname $@`
	# ld -r $^ -o $@
	$(HGCC) -dlink -Xcompiler -fPIC $(DEVLINK_FLAGS) $(HGLDFLAGS) $(OKI_DEV_LIB) $^ -o $@

clean:
	rm -f $(LIBOBJ) $(DEVOBJ) $(DEPFILES) $(DEPENDFILES) $(RULESFILE) $(STATICLIB)
	rm -f *.o *.s *.hgfb *.hggci *.lk *.bc a.out*
	rm -rf $(SYMMETRIC_GENSRC_DIR) $(SYMMETRIC_GENOBJ_DIR)

