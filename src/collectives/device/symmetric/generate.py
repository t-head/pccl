#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# See LICENSE.txt for more license information

import os
import sys

################################################################################
# The first command line argument is the path to the directory to generate and
# populate.

gensrc = sys.argv[1]

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    os.remove(os.path.join(gensrc, name))
    #os.truncate(os.path.join(gensrc, name), 0)
else:
  os.mkdir(gensrc)

def paste(sep, *args):
  return sep.join(args)

indents = 0
def emitln(f, lines):
  global indents
  for ln in ((lines,) if isinstance(lines, str) else lines):
    f.write('  '*indents + ln + '\n')

def indent(s):
  return '\n'.join('  '+l for l in s.splitlines())

class Rec(object):
  def __init__(me, **kw):
    me.__dict__.update(kw)
  def __eq__(x, y):
    return x.__dict__ == y.__dict__
  def __hash__(me):
    h = 0
    for k in me.__dict__:
      h += hash((k, me.__dict__[k]))
    return h

################################################################################
# Edit this region for introducing new algos etc

reductions = ["AllReduce","ReduceScatter"]
all_reds = ["sum"]
all_tys = ["f32","f16","bf16","f8e4m3","f8e5m2"]

coll_to_lower = {
  "AllGather": "all_gather",
  "AllReduce": "all_reduce",
  "ReduceScatter": "reduce_scatter"
}

red_to_ncclDevRedOp = {
  "sum": "ncclDevSum"
}
red_to_Func = {
  "sum": "FuncSum"
}

ty_to_ncclDataType = {
  "f32": "pcclFloat32",
  "f16": "pcclFloat16",
  "bf16": "pcclBfloat16",
  "f8e4m3": "pcclFloat8e4m3",
  "f8e5m2": "pcclFloat8e5m2"
}
ty_to_cxxtype = {
  "f32": "float",
  "f16": "half",
  "bf16": "__ppu_bfloat16",
  "f8e4m3": "__hg_fp8_e4m3",
  "f8e5m2": "__hg_fp8_e5m2"
}

def enumerate_kernels():
  for algo in ["LL","ST","ST_Ring","ST_2D"]:
    yield Rec(coll="AllGather", algo=algo)
  for red in all_reds:
    for ty in all_tys:
      for algo in ["AGxLL_R","RSxLD_AGxST"]:
        yield Rec(coll="AllReduce", algo=algo, red=red, ty=ty)
      for algo in ["LL","LD"]:
        yield Rec(coll="ReduceScatter", algo=algo, red=red, ty=ty)

def required_hggc(k):
  arch = 0
  if k.coll in reductions:
    if k.ty.startswith("f8"):
      arch = 150
  return arch

################################################################################

def kernel_fdep(k):
  return coll_to_lower[k.coll] + '.cu'

def kernel_fname(k):
  if k.coll in reductions:
    return paste('_', coll_to_lower[k.coll], k.red, k.ty) + '.cu'
  else:
    return coll_to_lower[k.coll] + '.cu'

def kernel_cname(k):
  if k.coll in reductions:
    return paste("_", "pccl_symDevKernel", k.coll, k.algo, k.red, k.ty)
  else:
    return paste("_", "pccl_symDevKernel", k.coll, k.algo)

def kernel_cname_fast(k):
  return kernel_cname(k) + "_fast"

def kernel_conds(k):
  arch = required_hggc(k)
  if arch == 0: return None
  return "(defined __HGGC_ARCH__) && (__HGGC_ARCH__ >= %d)"%arch

# Multi-task wrapper (4KB by-value); the body unpacks per-channel work from
# pcclSymkDevWorkArgs and dispatches to the inline ncclSymRun_* bodies which
# take ncclSymDevArgs.
MULTI_WRAPPER_PROLOGUE = (
    "  pcclSymkDevWorkArgs const* args = &args4K.args;\n"
    "  int blockId = blockIdx.x;\n"
    "  pcclSymkChannelWorkRange* range = args->getWorkRange();\n"
    "  pcclSymkDevWork* works = args->getWorks(args->nMaxChannels);\n"
    "  int workLo = (blockId == 0) ? 0 : (int)range[blockId-1].workHi;\n"
    "  if (blockId != 0 && (uint32_t)range[blockId-1].fracHi + 1u == 0x10000u) workLo++;\n"
    "  int workHi = (int)range[blockId].workHi;\n"
    "  for (int wi = workLo; wi <= workHi; wi++) {{\n"
    "    pcclSymkDevWork const& work = works[wi];\n"
    "    if (blockId < (int)work.sChannelId ||\n"
    "        blockId >= (int)work.sChannelId + (int)work.nChannels) continue;\n"
    "    ncclSymDevArgs perWork;\n"
    "    perWork.comm.base         = args->kcomm.base;\n"
    "    perWork.comm.stride4G     = work.inputWin->stride4G;\n"
    "    perWork.comm.nRanks       = args->kcomm.devComm.lsaSize;\n"
    "    perWork.comm.rank         = args->kcomm.devComm.lsaRank;\n"
    "    perWork.comm.nRanks_rcp32 = args->kcomm.devComm.lsaSize_rcp32;\n"
    "    perWork.rootRank          = work.rootRank;\n"
    "    perWork.nBlocks           = work.nChannels;\n"
    "    perWork.redOpArg          = work.redOpArg;\n"
    "    perWork.nElts             = work.nElts;\n"
    "    perWork.input             = (char*)::add4G((void*)work.inputWin->lsaFlatBase,\n"
    "                                  (int)(args->kcomm.devComm.lsaRank * work.inputWin->stride4G)) + work.inputOff;\n"
    "    perWork.output            = (char*)::add4G((void*)work.outputWin->lsaFlatBase,\n"
    "                                  (int)(args->kcomm.devComm.lsaRank * work.outputWin->stride4G)) + work.outputOff;\n"
    "    perWork.incPrecision      = false;\n"
)
MULTI_WRAPPER_EPILOGUE = "  }}"

def instantiate(k):
  arch_cond = kernel_conds(k)
  # Fast-path: 64B ncclSymDevArgs by value. Used when PCCL_SYMK_USE_MULTI_TASK=0
  # (default). Avoids PPU hgLaunchKernelExAD memcpying the full 4KB args
  # signature per launch.
  fast_template = (
      "__global__ void __launch_bounds__(ncclSymMaxThreads, ncclSymMinBlocksPerCU)\n"
      "{cname_fast}(ncclSymDevArgs args) {{\n"
      "{fast_call}\n"
      "}}"
  )
  # Multi-task: 4KB pcclSymkDevWorkArgs4K by value.
  # Currently no callers; re-enable via PCCL_SYMK_USE_MULTI_TASK=1 when PPU
  # SDK exposes a per-launch args-size override.
  multi_template = (
      "__global__ void __launch_bounds__(ncclSymMaxThreads, ncclSymMinBlocksPerCU)\n"
      "{cname}(pcclSymkDevWorkArgs4K const args4K) {{\n"
      + MULTI_WRAPPER_PROLOGUE
      + "{multi_call}\n"
      + MULTI_WRAPPER_EPILOGUE
      + "\n}}"
  )
  if arch_cond is None:
    fast_call_red_ty  = "  pccl_symRun_{id}<{red}, {ty}>(&args);"
    fast_call_form    = "  pccl_symRun_{id}(&args);"
    multi_call_red_ty = "    pccl_symRun_{id}<{red}, {ty}>(&perWork);"
    multi_call_form   = "    pccl_symRun_{id}(&perWork);"
  else:
    fast_call_red_ty = "  #if {arch_cond}\n    pccl_symRun_{id}<{red}, {ty}>(&args);\n  #endif"
    fast_call_form   = "  #if {arch_cond}\n    pccl_symRun_{id}(&args);\n  #endif"
    multi_call_red_ty = "    #if {arch_cond}\n      pccl_symRun_{id}<{red}, {ty}>(&perWork);\n    #endif"
    multi_call_form   = "    #if {arch_cond}\n      pccl_symRun_{id}(&perWork);\n    #endif"

  id = k.coll+'_'+k.algo
  cname = kernel_cname(k)
  cname_fast = kernel_cname_fast(k)
  if k.coll in reductions:
    fast_call  = fast_call_red_ty.format(id=id, red=red_to_Func[k.red], ty=ty_to_cxxtype[k.ty], arch_cond=arch_cond)
    multi_call = multi_call_red_ty.format(id=id, red=red_to_Func[k.red], ty=ty_to_cxxtype[k.ty], arch_cond=arch_cond)
  else:
    fast_call  = fast_call_form.format(id=id, arch_cond=arch_cond)
    multi_call = multi_call_form.format(id=id, arch_cond=arch_cond)
  return (multi_template.format(cname=cname, multi_call=multi_call)
          + "\n"
          + fast_template.format(cname_fast=cname_fast, fast_call=fast_call))

def prototype(k):
  return ("__global__ void {cname}(pcclSymkDevWorkArgs4K const);\n"
          "__global__ void {cname_fast}(ncclSymDevArgs);").format(
            cname=kernel_cname(k), cname_fast=kernel_cname_fast(k))

################################################################################

def partition(vals, keyfn):
  ans = {}
  for x in vals:
    k = keyfn(x)
    if k not in ans:
      ans[k] = []
    ans[k].append(x)
  return ans


kernels_by_file = partition(enumerate_kernels(), lambda k: (kernel_fname(k), k.coll))

# Add dependency only files (e.g. allreduce.cu)
for coll in set(k.coll for k in enumerate_kernels()):
  fname = coll_to_lower[coll]+'.cu'
  if (fname, coll) not in kernels_by_file:
    kernels_by_file[fname, coll] = []

# Generate each kernel instantiation file
for (fname, coll), ks in kernels_by_file.items():
  with open(os.path.join(gensrc, fname), "w") as f:
    emitln(f, '#include "symmetric.h"')
    emitln(f, '#include "sym_kernels.h"')
    emitln(f, '#include "pccl_device/core.h"')
    emitln(f, '#include "symmetric/kernel.cuh"')
    emitln(f, '#include "symmetric/{coll}.cuh"'.format(coll=coll_to_lower[coll]))
    for k in ks:
      emitln(f, instantiate(k))

# Generate <gensrc>/symmetric_kernels.cc
kernel_list = list(enumerate_kernels())
with open(os.path.join(gensrc, "symmetric_kernels.cc"), "w") as f:
  emitln(f, '#include "sym_kernels.h"')
  emitln(f, '#include "symmetric.h"')
  emitln(f, '#include "devcomm.h"')
  emitln(f, '#include "debug.h"')
  emitln(f, '')

  for k in kernel_list:
    emitln(f, prototype(k))
  emitln(f, '')

  emitln(f, 'extern int const ncclSymkKernelCount = %d;' % len(kernel_list))
  emitln(f, 'void* ncclSymkKernelList[] = {')
  for k in kernel_list:
    emitln(f, '  (void*){cname},'.format(cname=kernel_cname(k)))
  emitln(f, '  nullptr};')
  emitln(f, '')
  emitln(f, 'void* ncclSymkKernelList_fast[] = {')
  for k in kernel_list:
    emitln(f, '  (void*){cname},'.format(cname=kernel_cname_fast(k)))
  emitln(f, '  nullptr};')
  emitln(f, '')

  # symk kernels are unconditional (no runtime/arch gating).
  emitln(f, 'int ncclSymkKernelRequirements[] = {')
  for index, k in enumerate(kernel_list):
    emitln(f, '  0, /*%4d %s*/' % (index, kernel_cname(k)))
  emitln(f, '};')
  emitln(f, '')

  emitln(f, 'int ncclSymkGetKernelIndex(ncclSymkKernelId id, int red, ncclDataType_t ty) {')
  indents += 1
  emitln(f, 'switch (id) {')
  emitln(f, 'default: WARN("pcclSymkGetKernelIndex: unknown kernel id %d", (int)id); return -1;')
  for (coll, algo), coll_algo_ks in partition(kernel_list, lambda k: (k.coll, k.algo)).items():
    emitln(f, 'case ncclSymkKernelId_'+coll+'_'+algo+':')
    indents += 1
    if len(coll_algo_ks) == 1:
      emitln(f, 'return %d;' % kernel_list.index(coll_algo_ks[0]))
    else:
      emitln(f, 'switch ((ncclDevRedOp_t)red) {')
      emitln(f, 'default: WARN("pcclSymkGetKernelIndex: unknown red op %d for id %d", red, (int)id); return -1;')
      for red, coll_algo_red_ks in partition(coll_algo_ks, lambda k: k.red).items():
        emitln(f, 'case '+red_to_ncclDevRedOp[red]+':')
        indents += 1
        emitln(f, 'switch (ty) {')
        emitln(f, 'default: WARN("pcclSymkGetKernelIndex: unknown type %d for id %d red %d", (int)ty, (int)id, red); return -1;')
        for k in coll_algo_red_ks:
          emitln(f, 'case '+ty_to_ncclDataType[k.ty]+': return %d;' % kernel_list.index(k))
        emitln(f, '}')
        indents -= 1
      emitln(f, '}')
    indents -= 1
  emitln(f, '}')
  indents -= 1
  emitln(f, '}')

# Generate <gensrc>/rules.mk
with open(os.path.join(gensrc, "rules.mk"), "w") as f:
  inst_names = sorted(set(kernel_fname(k) for k in enumerate_kernels()))
  names = inst_names + ["symmetric_kernels.cc"]
  f.write("LIB_OBJS_SYM_GEN = $(patsubst %,$(OBJDIR)/genobj/symmetric/%.o,{names})\n"
          .format(names=" ".join(names)))
  f.write("\n")

  inst_names = sorted(set((k.coll, kernel_fname(k)) for k in enumerate_kernels()))
  for coll, name in inst_names:
    f.write(
      "$(OBJDIR)/genobj/symmetric/{name}.o: $(OBJDIR)/gensrc/symmetric $(OBJDIR)/genobj/symmetric/{coll}.cu.dep\n"

      "$(OBJDIR)/genobj/symmetric/{name}.o: SRC_FILE = $(patsubst $(OBJDIR)/genobj/symmetric/%.o,$(SYMMETRIC_GENSRC_DIR)/%,$@)\n"

      "$(OBJDIR)/genobj/symmetric/{name}.o:\n"

      "\t" "@printf \"Compiling  %-35s > %s\\n\" $(notdir $(SRC_FILE)) $@\n"
      "\t" "@mkdir -p $(dir $@)\n"
      "\t" "$(HGCC) $(HGFLAGS) -c $(SRC_FILE) -o $@\n"

      "\n"
      .format(name=name, coll=coll_to_lower[coll])
    )
