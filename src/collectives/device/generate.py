#!/usr/bin/env python3
import os
import sys

################################################################################
# Command line - the device Makefile passes the selective-build options through.

def usage():
  sys.stderr.write(
    "Usage:\n"
    "  generate.py <gensrc_dir> --obj_dir <obj_dir>\n"
    "              [--coll_ops sendrecv/all_reduce/all_gather/broadcast/reduce/reduce_scatter]\n"
    "              [--reduct_ops sum/prod/minmax/premulsum/sumpostdiv]\n"
    "              [--data_types i8/u8/i32/u32/i64/u64/f16/f32/f64/bf16/f8e4m3/f8e5m2]\n"
    "              [--keep 0|1]\n"
    "\n"
    "  gensrc_dir:  directory to (re)populate with the generated sources.\n"
    "  obj_dir:     object directory baked into the generated rules.mk.\n"
    "  coll_ops / reduct_ops / data_types:\n"
    "               selective build - combinations outside the selection are\n"
    "               still compiled, but as FAKE_KERNEL stubs, so that the object\n"
    "               list and the tables keep their shape ('none' means\n"
    "               'everything').\n")

gensrc = None
obj_dir = None
sel_colls = "none"
sel_reds = "sum prod minmax premulsum sumpostdiv"
sel_tys = "none"
keep = 0

argv = sys.argv[1:]
while argv:
  key = argv.pop(0)
  if key == "--obj_dir":     obj_dir = argv.pop(0)
  elif key == "--coll_ops":  sel_colls = argv.pop(0)
  elif key == "--reduct_ops": sel_reds = argv.pop(0)
  elif key == "--data_types": sel_tys = argv.pop(0)
  elif key == "--keep":      keep = int(argv.pop(0))
  elif key == "--help":      usage(); sys.exit(0)
  elif gensrc is None:       gensrc = key
  else:
    sys.stderr.write("Unknown option: %s\n" % key)
    usage()
    sys.exit(1)

if gensrc is None:
  usage()
  sys.exit(1)
if obj_dir is None:
  obj_dir = gensrc

################################################################################
# Dimensions.  Every list below is order-critical: FUNC_ROW() is a mixed radix
# number over them, so a reordering here silently misdirects every dispatch.
# device_table.cu carries static_asserts that pin each order to its C++ enum, so
# a mismatch fails to compile rather than misbehaving at runtime.

class Rec(object):
  def __init__(me, **kw):
    me.__dict__.update(kw)

# kind: how many instantiations a collective needs
#   "R" - one per (redop, type); the reductions
#   "C" - a single (Sum, int8_t) one, shared by every redop and type
#   "P" - like "C", but RING only and with a kernel per protocol
# The order fixes the order of the rules in rules.mk, nothing else.
all_colls = [
  Rec(lower="sendrecv",       name="SendRecv",      kind="P"),
  Rec(lower="all_reduce",     name="AllReduce",     kind="R"),
  Rec(lower="all_gather",     name="AllGather",     kind="C"),
  Rec(lower="broadcast",      name="Broadcast",     kind="C"),
  Rec(lower="reduce",         name="Reduce",        kind="R"),
  Rec(lower="reduce_scatter", name="ReduceScatter", kind="R"),
]

# Order matches ncclDevRedOp_t.
all_reds = [
  Rec(lower="sum",        op="Sum"),
  Rec(lower="prod",       op="Prod"),
  Rec(lower="minmax",     op="MinMax"),
  Rec(lower="premulsum",  op="PreMulSum"),
  Rec(lower="sumpostdiv", op="SumPostDiv"),
]

# Order matches ncclDataType_t.
all_tys = [
  Rec(lower="i8",     ctype="int8_t",         nccl="ncclInt8",       flt=False),
  Rec(lower="u8",     ctype="uint8_t",        nccl="ncclUint8",      flt=False),
  Rec(lower="i32",    ctype="int32_t",        nccl="ncclInt32",      flt=False),
  Rec(lower="u32",    ctype="uint32_t",       nccl="ncclUint32",     flt=False),
  Rec(lower="i64",    ctype="int64_t",        nccl="ncclInt64",      flt=False),
  Rec(lower="u64",    ctype="uint64_t",       nccl="ncclUint64",     flt=False),
  Rec(lower="f16",    ctype="half",           nccl="ncclFloat16",    flt=True),
  Rec(lower="f32",    ctype="float",          nccl="ncclFloat32",    flt=True),
  Rec(lower="f64",    ctype="double",         nccl="ncclFloat64",    flt=True),
  Rec(lower="bf16",   ctype="__ppu_bfloat16", nccl="ncclBfloat16",   flt=True),
  Rec(lower="f8e4m3", ctype="__hg_fp8_e4m3",  nccl="ncclFloat8e4m3", flt=True),
  Rec(lower="f8e5m2", ctype="__hg_fp8_e5m2",  nccl="ncclFloat8e5m2", flt=True),
]

# Order matches NCCL_ALGO_* / NCCL_PROTO_*.
all_algos = ["TREE", "RING", "COLLNET", "FC", "PAT"]
all_protos = ["LL", "LL128", "SIMPLE", "SIGNAL", "SIMPLE_BULK", "SIGNAL_BULK", "SIMPLE_BULK_AR"]

# SendRecv exists for these protocols only, and shares one dispatch slot.
p2p_protos = ["SIMPLE", "SIMPLE_BULK", "SIGNAL", "SIGNAL_BULK", "SIMPLE_BULK_AR"]

# Order matches ncclFunc_t, i.e. the layout of the two tables.  SendRecv is not
# in here: its five entries sit in front of the regular grid (NUM_P2P_FUNC).
table_colls = ["Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce"]

coll_by_name = dict((c.name, c) for c in all_colls)

################################################################################
# Reachability - a transcription of kernel_enable.h.
#
# Kept literal (one row per algorithm, protocols in enum order, same layout as
# the C++ table) so that the two can be diffed by eye.  They cannot drift apart
# silently either way: a combination enabled there but missing here is caught by
# the static_asserts device_table.cu emits, and a combination enabled here but
# missing there simply does not exist in the C++ table's index space.

kernel_enabled = [
  { # Broadcast
    "TREE":    [0,  0,  0,  0,  0,  0,  0 ],
    "RING":    [80, 80, 80, 80, 80, 80, 89],
    "COLLNET": [0,  0,  0,  0,  0,  0,  0 ],
    "FC":      [80, 0,  80, 0,  80, 0,  0 ],
    "PAT":     [0,  0,  0,  0,  0,  0,  0 ],
  },
  { # Reduce
    "TREE":    [0,  0,  0,  0,  0,  0,  0 ],
    "RING":    [80, 80, 80, 80, 80, 80, 89],
    "COLLNET": [0,  0,  0,  0,  0,  0,  0 ],
    "FC":      [80, 0,  80, 0,  80, 0,  0 ],
    "PAT":     [0,  0,  0,  0,  0,  0,  0 ],
  },
  { # AllGather
    "TREE":    [0,  0,  0,  0,  0,  0,  0 ],
    "RING":    [80, 80, 80, 80, 80, 80, 89],
    "COLLNET": [0,  0,  0,  0,  0,  0,  0 ],
    "FC":      [80, 0,  80, 0,  80, 0,  0 ],
    "PAT":     [0,  0,  80, 0,  0,  0,  0 ],
  },
  { # ReduceScatter
    "TREE":    [0,  0,  0,  0,  0,  0,  0 ],
    "RING":    [80, 80, 80, 80, 80, 80, 89],
    "COLLNET": [0,  0,  0,  0,  0,  0,  0 ],
    "FC":      [80, 0,  80, 0,  80, 0,  0 ],
    "PAT":     [0,  0,  80, 0,  0,  0,  0 ],
  },
  { # AllReduce
    "TREE":    [80, 80, 80, 80, 80, 80, 0 ],
    "RING":    [80, 80, 80, 80, 80, 80, 89],
    "COLLNET": [80, 80, 80, 80, 80, 80, 0 ],
    "FC":      [80, 0,  80, 0,  80, 0,  0 ],
    "PAT":     [0,  0,  0,  0,  0,  0,  0 ],
  },
]

p2p_kernel_enabled = [0, 0, 80, 80, 80, 80, 89]

enabled_by_coll = dict(zip(table_colls, kernel_enabled))

def reachable(coll, algo, proto):
  """Can the scheduler ever select this (coll, algo, proto)?  A zero entry in
  kernel_enable.h leaves the tuner's bandwidth at 0, which makes
  ncclTopoGetAlgoTime() return -1.0, which the scheduler never picks."""
  if coll.kind == "P":
    return p2p_kernel_enabled[all_protos.index(proto)] != 0
  return enabled_by_coll[coll.name][algo][all_protos.index(proto)] != 0

# The p2p protocol list above is not derived from p2p_kernel_enabled because it
# is in FUNC_INDEX_P2P_* order, not NCCL_PROTO_* order.  Check that the two at
# least agree on the set.
assert set(p2p_protos) == set(p for p in all_protos
                             if p2p_kernel_enabled[all_protos.index(p)] != 0), \
  "p2p_protos and p2p_kernel_enabled disagree on which protocols exist"
assert len(p2p_kernel_enabled) == len(all_protos)

# A launch stub is emitted per reachable (coll, algo), under one of its
# protocols.  LL is the preferred name because every algorithm reachable at all
# is reachable on it - except PAT, which upstream wires for SIMPLE only, so the
# stub falls back to the first reachable protocol.  The choice is cosmetic: the
# stub dispatches through pcclFuncs[] regardless of the protocol it is named
# for, it only has to exist and to be named the same way by host_entry().
def stub_proto(coll, algo):
  protos = [p for p in all_protos if reachable(coll, algo, p)]
  return protos[0] if protos else None

for _name, _rows in zip(table_colls, kernel_enabled):
  assert len(_rows) == len(all_algos)
  for _algo, _row in _rows.items():
    assert len(_row) == len(all_protos), "%s/%s: wrong protocol count" % (_name, _algo)

################################################################################
# Signed/unsigned merging - a transcription of NCCL's equivalent_primary().
#
# For the reduction ops whose operator is bit-identical on signed and unsigned
# integers of the same width, only the unsigned instantiation is built, and the
# signed table cells point at it.  This is not an optimization of the operator,
# it is the trick NCCL uses to keep the function table small (see
# equivalent_primary() in nccl/src/device/generate.py).
#
# MinMax is sign agnostic only because its operand is transformed first: the
# comparison is an unsigned one on (value ^ xormask), and hostToDevRedOp() puts
# the sign bit into that mask for ncclInt8/Int32/Int64.  That is keyed on the
# *caller's* datatype, so the unsigned instantiation is handed the signed
# ordering when a signed reduction is routed onto it, which is what makes the
# merge behaviour preserving rather than merely compiling.
#
# It could not be merged while the two reduction paths disagreed about how much
# of the op argument to read: the legacy MULTI<> operator took its signedness
# from T and ignored the mask, so a merged signed reduction would have compared
# negative values as unsigned on those protocols, with nothing to catch it at
# compile time.  Both paths now go through MinMaxOverloads, which applies the
# mask for every overload either of them asks for, so they can no longer take
# different views of it.  See the comment there.
#
# SumPostDiv *is* in here, but it needed one extra piece to stay behaviour
# preserving: its operator divides, and division is not sign agnostic.  NCCL
# solves this without a signed instantiation by passing the caller's signedness
# as the low bit of the op argument, so reduce_kernel.h and hostToDevRedOp() were
# taught the same encoding.  ncclAvg over negative sums therefore still rounds
# towards zero the way a signed division would.

signed_to_unsigned = {"i8": "u8", "i32": "u32", "i64": "u64"}
merged_reds = ("Sum", "Prod", "MinMax", "PreMulSum", "SumPostDiv")
ty_by_lower = dict((t.lower, t) for t in all_tys)

def equivalent_primary(coll, red, ty):
  """The (redop, type) whose instantiation this cell actually runs.  Only the
  reductions have a (redop, type) plane to merge within; "C" and "P" collectives
  already collapse theirs onto (Sum, int8_t) wholesale."""
  if coll.kind == "R" and red.op in merged_reds and ty.lower in signed_to_unsigned:
    return (red, ty_by_lower[signed_to_unsigned[ty.lower]])
  return (red, ty)

def is_primary(coll, red, ty):
  return equivalent_primary(coll, red, ty) == (red, ty)

# FUNC_ROW() from collectives.h, replicated here so that the row-to-id map can
# be filled in row order.  host_table.cc pins the replica against the macro at
# both ends of the row space, so the two cannot drift apart silently.
NUM_P2P_FUNC = 5

def func_row(coll_name, red, ty, algo, proto):
  row = table_colls.index(coll_name)
  row = row*len(all_reds) + all_reds.index(red)
  row = row*len(all_tys) + all_tys.index(ty)
  row = row*len(all_algos) + all_algos.index(algo)
  row = row*len(all_protos) + all_protos.index(proto)
  return NUM_P2P_FUNC + row

num_func_rows = NUM_P2P_FUNC + (len(table_colls)*len(all_reds)*len(all_tys)
                               *len(all_algos)*len(all_protos))

################################################################################
# What each translation unit instantiates.

def directives(coll, red, ty):
  """The IMPL_COLL_* calls that <coll>_<red.lower>_<ty.lower>.cu must contain,
  in emission order.  An empty list means no translation unit at all."""
  if coll.kind == "P":
    if red.op != "Sum" or ty.lower != "i8":
      return []
    out = []
    for proto in p2p_protos:
      if not reachable(coll, "RING", proto):
        continue
      out.append(Rec(kern=False, algo="RING", proto=proto))
      # Each protocol's kernel carries its own FUNC_INDEX_P2P_* as the fIndex
      # fast path, matching the work item's funcId (host_entry() already picks
      # the per-proto kernel per id, so host and device agree on the pair).
      out.append(Rec(kern=True, algo="RING", proto=proto,
                     findex="%d" % p2p_protos.index(proto),
                     findex_note="SendRecv/RING/%s" % proto))
    return out

  if coll.kind == "C" and (red.op != "Sum" or ty.lower != "i8"):
    return []
  if not is_primary(coll, red, ty):
    # A signed cell of a merged op: it runs the unsigned instantiation, so there
    # is nothing to build here.
    return []
  if coll.kind == "R" and red.op == "SumPostDiv" and ty.flt:
    # The host lowers ncclAvg to PreMulSum for floats, so there is no device
    # operator to instantiate here.
    return []

  out = []
  for algo in all_algos:
    protos = [p for p in all_protos if reachable(coll, algo, p)]
    for proto in protos:
      out.append(Rec(kern=False, algo=algo, proto=proto))
    if protos and red.op == "Sum":
      # Kernels are launch stubs: one per (coll, algo) is enough, because
      # ncclKernelMain dispatches through pcclFuncs[] whenever the work item
      # does not match the kernel's own fIndex.  See stub_proto() for which
      # protocol the stub is named under; only Sum gets one, so that every cell
      # of the (redop, type) plane can share it.
      out.append(Rec(kern=True, algo=algo, proto=stub_proto(coll, algo),
                     findex="%d" % kernel_findex(coll, red, ty, algo),
                     findex_note="%s/%s/%s/%s/%s" % (coll.name, algo, stub_proto(coll, algo), red.op, ty.ctype)))
  return out

def cell_nullified(coll, red, ty, algo, proto):
  """Whether this table cell has to hold nullptr.  Both tables go through this
  one predicate, so they cannot end up disagreeing about which cells are live."""
  if not reachable(coll, algo, proto):
    return True
  # "C" collectives ignore the reduction op and the data type altogether, so
  # their whole plane maps onto the single (Sum, int8_t) instantiation.
  return coll.kind == "R" and red.op == "SumPostDiv" and ty.flt

def tu_name(coll, red, ty):
  return "%s_%s_%s" % (coll.lower, red.lower, ty.lower)

################################################################################
# The id space.
#
# A row's id is the index of the instantiation that runs it.  Rows sharing an
# instantiation share an id, and a row with none gets -1; the ids themselves are
# handed out densely, in row order, which is what shrinks the two tables from
# 8405 slots to one per distinct function.
#
# SendRecv is given the front of the id space so that the FUNC_INDEX_P2P_*
# constants keep working as ids without a lookup.  That is the only thing pinning
# any id to a particular value - every other id is whatever position the row
# order happens to put it in, so re-enabling a combination renumbers the ones
# after it.  Nothing outside the generated files may assume otherwise.

def primary_cell(coll, red, ty, algo, proto):
  """The cell whose instantiation this one runs, or None if it has none.  Goes
  through cell_nullified() so that the tables and the row-to-id map cannot end up
  disagreeing about which cells are live, and through equivalent_primary() so
  that there is still only one statement of which cells are merged."""
  if cell_nullified(coll, red, ty, algo, proto):
    return None
  src_red, src_ty = ((all_reds[0], all_tys[0]) if coll.kind == "C"
                     else equivalent_primary(coll, red, ty))
  return (coll.name, src_red, src_ty, algo, proto)

def cell_row(cell):
  """The row a cell sits at.  SendRecv's five rows precede the grid, so they are
  not a func_row() at all."""
  if cell[0] == "SendRecv":
    return p2p_protos.index(cell[4])
  return func_row(*cell)

def cell_label(cell):
  """How a cell is named in the generated comments."""
  coll_name, red, ty, algo, proto = cell
  return "%s/%s/%s/%s/%s" % (coll_name, algo, proto, red.op, ty.ctype)

def assign_ids():
  row_to_id = [-1]*num_func_rows
  id_to_cell = []
  id_of = {}

  # The five SendRecv rows are its five ids, in FUNC_INDEX_P2P_* order.
  for proto in p2p_protos:
    cell = ("SendRecv", all_reds[0], all_tys[0], "RING", proto)
    id_of[cell] = len(id_to_cell)
    row_to_id[cell_row(cell)] = id_of[cell]
    id_to_cell.append(cell)

  for coll_name in table_colls:
    coll = coll_by_name[coll_name]
    for red in all_reds:
      for ty in all_tys:
        for algo in all_algos:
          for proto in all_protos:
            cell = primary_cell(coll, red, ty, algo, proto)
            if cell is None:
              continue
            if cell not in id_of:
              id_of[cell] = len(id_to_cell)
              id_to_cell.append(cell)
            row_to_id[func_row(coll_name, red, ty, algo, proto)] = id_of[cell]
  return row_to_id, id_to_cell, id_of

func_row_to_id, func_id_to_cell, func_id_of = assign_ids()
func_id_count = len(func_id_to_cell)

# The row each id reports back as, for turning an id into a name again. A merged
# row collapses onto its primary, so the printed name is unchanged.
func_id_to_row = [cell_row(cell) for cell in func_id_to_cell]

# The two directions have to invert each other, or a work item's funcId would
# name one function while the tables hold another.
for _id, _row in enumerate(func_id_to_row):
  assert func_row_to_id[_row] == _id, \
    "id %d maps to row %d, which maps back to id %d" % (_id, _row, func_row_to_id[_row])
assert func_row_to_id[:NUM_P2P_FUNC] == list(range(NUM_P2P_FUNC)), \
  "the SendRecv rows no longer are the SendRecv ids"
assert func_id_count == len(set(func_id_of.values())), "an id was handed out twice"

def kernel_findex(coll, red, ty, algo):
  """The id a (coll, algo) launch stub bakes in as its FINDEX, i.e. the id of its
  own instantiation.  ncclKernelMain compares the work item's funcId against it
  to decide whether it may run the specialization directly instead of going
  through pcclFuncs[], so it has to be that instantiation's id and no other."""
  proto = stub_proto(coll, algo)
  fid = func_row_to_id[func_row(coll.name, red, ty, algo, proto)]
  assert fid == func_id_of[(coll.name, red, ty, algo, proto)], \
    "%s/%s: the kernel's FINDEX is not its own id" % (coll.name, algo)
  return fid

def enumerate_tus():
  """Every translation unit that has something in it, in rules.mk order."""
  for coll in all_colls:
    for red in all_reds:
      for ty in all_tys:
        if directives(coll, red, ty):
          yield Rec(coll=coll, red=red, ty=ty)

def generated_symbols():
  """The device functions and the kernels that the translation units define,
  deduplicated, in definition order.  These are exactly the symbols the two
  tables may refer to - and, because the generator declares nothing else,
  referring to an unreachable combination anywhere fails to compile."""
  funcs, kerns = [], []
  for tu in enumerate_tus():
    for d in directives(tu.coll, tu.red, tu.ty):
      name = (kern_name if d.kern else func_name)(
        tu.coll.name, d.algo, d.proto, tu.red.op, tu.ty.ctype)
      into = kerns if d.kern else funcs
      if name not in into:
        into.append(name)
  return funcs, kerns

################################################################################
# Emission helpers.

BANNER = [
  "/*************************************************************************",
  " * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
  " * SPDX-FileCopyrightText: Modifications Copyright (c) 2020-2026 T-HEAD (SHANGHAI) SEMICONDUCTOR CO., Ltd. All Rights Reserved.",
  " * SPDX-License-Identifier: BSD-3-Clause",
  " *",
  " * See LICENSE.txt for more license information",
  " *",
  " * GENERATED FILE - do not edit.  Produced by collectives/device/generate.py.",
  " *************************************************************************/",
  "",
]

def emitln(f, lines):
  for ln in ((lines,) if isinstance(lines, str) else lines):
    f.write(ln + "\n")

def wrap_comment(text, width=76):
  out, line = [], "//"
  for word in text.split():
    if len(line) + 1 + len(word) > width:
      out.append(line)
      line = "//"
    line += " " + word
  out.append(line)
  return out

def func_name(coll, algo, proto, red, ctype):
  return "pcclFunction_%s_%s_%s_%s_%s" % (coll, algo, proto, red, ctype)

def kern_name(coll, algo, proto, red, ctype):
  return "pcclKernel_%s_%s_%s_%s_%s" % (coll, algo, proto, red, ctype)

if os.path.exists(gensrc):
  for name in os.listdir(gensrc):
    path = os.path.join(gensrc, name)
    if os.path.isfile(path):
      os.remove(path)
else:
  os.makedirs(gensrc)

################################################################################
# Generate the per-instantiation translation units.

for tu in enumerate_tus():
  coll, red, ty = tu.coll, tu.red, tu.ty
  with open(os.path.join(gensrc, tu_name(coll, red, ty) + ".cu"), "w") as f:
    emitln(f, BANNER)
    emitln(f, ['#include "%s.h"' % coll.lower,
               '#include "common.h"',
               '#include "collectives.h"',
               ''])
    emitln(f, "// %s / %s / %s" % (coll.name, red.op, ty.ctype))
    for d in directives(coll, red, ty):
      if d.kern:
        # The fIndex is a bare id, so say which combination it is: it is the one
        # number here that cannot be read off the macro arguments.
        emitln(f, "IMPL_COLL_KERN(%s, %s, %s, %s, %s, %s) // fIndex: %s"
                  % (coll.name, d.algo, d.proto, red.op, ty.ctype, d.findex,
                     d.findex_note))
      else:
        emitln(f, "IMPL_COLL_FUNC(%s, %s, %s, %s, %s)"
                  % (coll.name, d.algo, d.proto, red.op, ty.ctype))

################################################################################
# Generate device_table.cu (pcclFuncs).

GUARDS = [
  "// FUNC_ROW() is a mixed radix number over the five dimensions, and",
  "// pcclFuncRowToId[] is declared with the resulting extent, so C++ silently",
  "// zero-fills any shortfall: were a dimension bumped (e.g. COLLNET split into",
  "// COLLNET_DIRECT/COLLNET_CHAIN) without regenerating, the map would not shrink",
  "// or error - it would misalign, and every dispatch past that point would land",
  "// on the wrong id.  Pin the counts and the enum orders that generate.py",
  "// hard-codes so that any such change fails to compile here, next to the",
  "// generator that must be updated with it.",
  'static_assert(NCCL_NUM_ALGORITHMS == 5, "generate.py enumerates TREE/RING/COLLNET/FC/PAT");',
  'static_assert(NCCL_NUM_PROTOCOLS == 7, "generate.py enumerates LL/LL128/SIMPLE/SIGNAL/SIMPLE_BULK/SIGNAL_BULK/SIMPLE_BULK_AR");',
  'static_assert(NCCL_NUM_FUNCTIONS == 5, "generate.py enumerates Broadcast/Reduce/AllGather/ReduceScatter/AllReduce");',
  'static_assert(ncclNumDevRedOps == 5 && ncclNumTypes == 12, "generate.py enumerates 5 reduction ops and 12 data types");',
  "static_assert(NCCL_ALGO_TREE == 0 && NCCL_ALGO_RING == 1 &&",
  "              NCCL_ALGO_COLLNET == 2 && NCCL_ALGO_FC == 3 &&",
  "              NCCL_ALGO_PAT == 4,",
  '  "generate.py emits algorithms in TREE,RING,COLLNET,FC,PAT order");',
  "static_assert(NCCL_PROTO_LL == 0 && NCCL_PROTO_LL128 == 1 && NCCL_PROTO_SIMPLE == 2 &&",
  "              NCCL_PROTO_SIGNAL == 3 && NCCL_PROTO_SIMPLE_BULK == 4 &&",
  "              NCCL_PROTO_SIGNAL_BULK == 5 && NCCL_PROTO_SIMPLE_BULK_AR == 6,",
  '  "generate.py emits protocols in NCCL_PROTO_* order");',
  "static_assert(ncclFuncBroadcast == 0 && ncclFuncReduce == 1 && ncclFuncAllGather == 2 &&",
  "              ncclFuncReduceScatter == 3 && ncclFuncAllReduce == 4,",
  '  "generate.py emits collectives in ncclFunc_t order");',
]

def unreachable_cells():
  """Every (coll, algo, proto) that has no instantiation, so that device_table.cu
  can pin each one against kernel_enable.h.  algo is None for SendRecv, which is
  gated by the separate p2pKernelEnabled row."""
  out = []
  for coll_name in table_colls:
    coll = coll_by_name[coll_name]
    for algo in all_algos:
      for proto in all_protos:
        if not reachable(coll, algo, proto):
          out.append((coll_name, algo, proto))
  for proto in all_protos:
    if p2p_kernel_enabled[all_protos.index(proto)] == 0:
      out.append(("SendRecv", None, proto))
  return out

def device_entry(cell):
  """The pcclFuncs entry of one id.  The cell is already the primary one, so no
  merge resolution happens here."""
  coll_name, red, ty, algo, proto = cell
  return func_name(coll_name, algo, proto, red.op, ty.ctype)

def host_entry(cell):
  """The ncclKerns entry of one id.  The host side only ever needs *a* launch
  stub: the work item carries the real funcId and ncclKernelMain redirects through
  pcclFuncs[].  That is why every reduction op collapses onto Sum here and every
  protocol onto LL, while the data type is kept."""
  coll_name, red, ty, algo, proto = cell
  coll = coll_by_name[coll_name]
  if coll.kind == "P":
    # SendRecv keeps the per-proto kernel: its work item's funcId is the
    # FUNC_INDEX_P2P_* constant, so the launch stub and the device-side fast
    # path agree on the pair.  See directives().
    return kern_name("SendRecv", "RING", proto, "Sum", "int8_t")
  ctype = "int8_t" if coll.kind == "C" else ty.ctype
  return kern_name(coll_name, algo, stub_proto(coll, algo), "Sum", ctype)

gen_funcs, gen_kerns = generated_symbols()

with open(os.path.join(gensrc, "device_table.cu"), "w") as f:
  emitln(f, BANNER)
  emitln(f, ['#include "devcomm.h"',
             '#include "collectives.h"',
             '#include "common.h"',
             '#include "kernel_enable.h"',
             ''])
  emitln(f, GUARDS)
  emitln(f, '')
  emitln(f, wrap_comment(
    "Everything that was not instantiated has to be disabled in "
    "kernel_enable.h, or the tuner would hand the scheduler a combination with no "
    "id - which enqueue.cc has to refuse, so the operation fails instead of "
    "running. Pin each skipped combination here, so that re-enabling one without "
    "regenerating fails to compile."))
  for coll_name, algo, proto in unreachable_cells():
    if algo is None:
      emitln(f, ['static_assert(p2pKernelEnabled[NCCL_PROTO_%s] == 0,' % proto,
                 '  "SendRecv/%s is enabled in kernel_enable.h but was not instantiated");' % proto])
    else:
      emitln(f, ['static_assert(kernelEnabled[ncclFunc%s][NCCL_ALGO_%s][NCCL_PROTO_%s] == 0,'
                 % (coll_name, algo, proto),
                 '  "%s/%s/%s is enabled in kernel_enable.h but was not instantiated");'
                 % (coll_name, algo, proto)])
  emitln(f, '')
  emitln(f, wrap_comment(
    "Forward declarations of exactly what was instantiated. Declaring the "
    "generated set rather than the full cross product is what makes an "
    "unreachable combination unusable: there is no name to refer to, so a stale "
    "reference fails to compile instead of linking to nothing."))
  for name in gen_funcs:
    emitln(f, "extern __device__ void %s(void* args);" % name)
  emitln(f, '')
  emitln(f, wrap_comment(
    "One entry per id, so there are no nullptr holes left to check for.  The "
    "length is left to the initializer and published as pcclFuncIdCount; the "
    "trailing nullptr is what keeps the array non-empty when the host pass of "
    "the compiler takes the #if below out."))
  emitln(f, "__device__ ncclKern_t pcclFuncs[] = {")
  emitln(f, ["// Don't try to initialize the host shadow copy of this device-side global",
             "// variable. There is no host pointer to a device-side function, which",
             "// confuses clang. This will be fixed in the next clang release.",
             "#if defined(__HGGC_ARCH__)"])
  for fid, cell in enumerate(func_id_to_cell):
    emitln(f, "/*%4d*/ %s," % (fid, device_entry(cell)))
  emitln(f, ["#endif",
             "nullptr};",
             "",
             "// Workaround for https://reviews.llvm.org/D55580",
             "__device__ void pcclWorkaroundClangD55580() {}"])

################################################################################
# Generate host_table.cc (ncclKerns, pcclFuncIdCount, pcclFuncRowToId,
# pcclFuncIdToRow).
#
# comm.h would be the natural include for a host source, but it reaches
# pccl_profiler.h through profiler.h, and this file is compiled with the device
# directory's include path, which has no -Iinclude/plugin.  devcomm.h is where
# ncclDevKernelArgs1K actually comes from anyway.

with open(os.path.join(gensrc, "host_table.cc"), "w") as f:
  emitln(f, BANNER)
  emitln(f, ['#include "devcomm.h"',
             '#include "collectives.h"',
             ''])
  # Forward declarations of kernels.
  for name in gen_kerns:
    # __global__ below gets removed by the host compiler, which results in
    # Coverity diagnosing a specifiers inconsistency.
    emitln(f, "// coverity[declaration]")
    emitln(f, "extern __global__ void %s(ncclDevKernelArgs1K const args4K);" % name)
  emitln(f, '')
  emitln(f, wrap_comment(
    "extern, because a namespace-scope const has internal linkage in C++ and "
    "enqueue.cc has to see these definitions."))
  emitln(f, "extern int const pcclFuncIdCount = %d;" % func_id_count)
  emitln(f, '')
  emitln(f, wrap_comment(
    "One launch stub per id.  Which id gets which stub is host_entry()'s call, "
    "and it is deliberately coarse: the work item carries the real funcId and "
    "ncclKernelMain redirects through pcclFuncs[], so one kernel per (coll, algo) "
    "is enough."))
  emitln(f, "extern void* const ncclKerns[] = {")
  for fid, cell in enumerate(func_id_to_cell):
    emitln(f, "/*%4d*/ (void*)%s, // %s" % (fid, host_entry(cell), cell_label(cell)))
  emitln(f, "nullptr};")

  emitln(f, '')
  emitln(f, wrap_comment(
    "generate.py replicates FUNC_ROW()'s arithmetic to fill the map below. "
    "Pin the replica at both ends of the row space, so that a change to the macro "
    "or to the dimension order the generator hard-codes fails to compile here "
    "rather than misdirecting every row."))
  for cell in [(table_colls[0], all_reds[0], all_tys[0], all_algos[0], all_protos[0]),
               (table_colls[-1], all_reds[-1], all_tys[-1], all_algos[-1], all_protos[-1])]:
    coll_name, red, ty, algo, proto = cell
    emitln(f, ['static_assert(FUNC_ROW(ncclFunc%s, ncclDev%s, %s, NCCL_ALGO_%s, NCCL_PROTO_%s)'
               % (coll_name, red.op, ty.nccl, algo, proto),
               '              == %d, "generate.py replicates FUNC_ROW() and disagrees with it");'
               % func_row(coll_name, red, ty, algo, proto)])
  emitln(f, 'static_assert(NUM_FUNC_ROWS == %d, "generate.py mapped a row space of a different size");'
            % num_func_rows)
  emitln(f, '')
  emitln(f, wrap_comment(
    "row -> id.  -1 means nothing was built for the row, which is the scheduler "
    "picking a combination kernel_enable.h and the generated tables disagree "
    "about; enqueue.cc refuses it rather than launching. "
    "Rows sharing an instantiation share an id: the \"C\" collectives collapse "
    "their whole (redop, type) plane, and a signed integer reduction of a sign "
    "agnostic op takes the unsigned id of the same width (see "
    "equivalent_primary() in generate.py).  That collapse has to happen here "
    "rather than on the host, because the funcId in the work item has to name the "
    "instantiation that will really run: ncclKernelMain compares it against the "
    "fIndex baked into the kernel to decide whether it may call the "
    "specialization directly instead of going through pcclFuncs[], and an id "
    "naming a function that was never built would never match."))
  emitln(f, "extern int const pcclFuncRowToId[NUM_FUNC_ROWS] = {")
  emitln(f, "  // SendRecv - the five rows in front of the grid (NUM_P2P_FUNC).")
  for proto in p2p_protos:
    emitln(f, "  %d, // FUNC_INDEX_P2P_%s" % (p2p_protos.index(proto), proto))
  emitted = len(p2p_protos)
  for coll_name in table_colls:
    coll = coll_by_name[coll_name]
    for red in all_reds:
      for ty in all_tys:
        src_red, src_ty = equivalent_primary(coll, red, ty)
        # Annotate where the row's ids come from, if it is not itself the source.
        # equivalent_primary() only speaks for the reductions, so a "C"
        # collective's collapse has to be named here: its whole plane runs the
        # one instantiation at the front of the plane.
        note = ""
        if coll.kind == "C":
          if (red, ty) != (all_reds[0], all_tys[0]):
            note = " -> %s / %s" % (all_reds[0].op, all_tys[0].ctype)
        elif (src_red, src_ty) != (red, ty):
          note = " -> %s" % src_ty.ctype
        emitln(f, "  // %s / %s / %s%s" % (coll_name, red.op, ty.ctype, note))
        for algo in all_algos:
          for proto in all_protos:
            emitln(f, "  %d," % func_row_to_id[func_row(coll_name, red, ty, algo, proto)])
            emitted += 1
  assert emitted == num_func_rows, "row-to-id map is %d rows, not %d" % (emitted, num_func_rows)
  emitln(f, "};")

  emitln(f, '')
  emitln(f, wrap_comment(
    "id -> the row it reports as, purely so that an id can be turned back into "
    "the five dimensions when printing it.  Merged rows collapse, so this is the "
    "primary row and not the caller's: an id prints as the function that really "
    "runs.  Nothing dispatches on this."))
  emitln(f, "extern int const pcclFuncIdToRow[] = {")
  for fid, cell in enumerate(func_id_to_cell):
    emitln(f, "/*%4d*/ %d, // %s" % (fid, func_id_to_row[fid], cell_label(cell)))
  emitln(f, "-1};")

################################################################################
# Generate rules.mk.
#
# One dependency file per collective is enough: every translation unit of a
# collective includes the same three headers.  The representative is just the
# first one generated for it; the pattern rule that turns a generated source into
# a .dep lives in the device Makefile.

def selected(coll, red, ty):
  """FAKE_KERNEL is 0 when a combination is inside the selective-build filter."""
  def has(sel, val, full):
    return val in (full if sel == "none" else sel.split())
  if not (has(sel_colls, coll.lower, [c.lower for c in all_colls])
          and has(sel_reds, red.lower, [r.lower for r in all_reds])):
    return False
  if has(sel_tys, ty.lower, [t.lower for t in all_tys]):
    return True
  # A TU is also live when it is the merge target of a *selected* signed type:
  # equivalent_primary() points that row at this unsigned instantiation, so the
  # host table hands out this TU's funcId. Faking it would let a selected row
  # pass the funcId >= 0 check and then dispatch into the fake kernel.
  return any(has(sel_tys, t.lower, [u.lower for u in all_tys])
             and equivalent_primary(coll, red, t)[1] is ty for t in all_tys)

dep_rep = {}
for tu in enumerate_tus():
  dep_rep.setdefault(tu.coll.lower, tu_name(tu.coll, tu.red, tu.ty) + ".cu")

with open(os.path.join(gensrc, "rules.mk"), "w") as f:
  f.write("GENOBJS := \\\n")
  for tu in enumerate_tus():
    f.write("\t%s/%s.o \\\n" % (obj_dir, tu_name(tu.coll, tu.red, tu.ty)))
  f.write("\n")
  f.write("GENTABLEOBJS := %s/device_table.o %s/host_table.o\n\n" % (obj_dir, obj_dir))
  f.write("GENDEPFILES := \\\n")
  for src in list(dep_rep.values()) + ["device_table.cu", "host_table.cc"]:
    f.write("\t%s/%s.dep \\\n" % (obj_dir, src))
  f.write("\n")

  for tu in enumerate_tus():
    name = tu_name(tu.coll, tu.red, tu.ty)
    src = "%s/%s.cu" % (gensrc, name)
    obj = "%s/%s.o" % (obj_dir, name)
    fake = 0 if selected(tu.coll, tu.red, tu.ty) else 1
    save_temps = ""
    f.write("%s : %s %s/%s.dep\n" % (obj, src, obj_dir, dep_rep[tu.coll.lower]))
    f.write("\t@printf \"Compiling  %%-35s > %%s\\\\n\" %s.cu %s\n" % (name, obj))
    f.write("\tmkdir -p %s\n" % obj_dir)
    if keep == 1:
      save_temps = "--keep-dir %s/%s" % (obj_dir, name)
      f.write("\tmkdir -p %s/%s\n" % (obj_dir, name))
    f.write("\t${HGCC} -DFAKE_KERNEL=%d ${HGFLAGS} %s %s -o %s\n" % (fake, save_temps, src, obj))
    f.write("\n")

  for what, ext in (("device_table", "cu"), ("host_table", "cc")):
    src = "%s/%s.%s" % (gensrc, what, ext)
    obj = "%s/%s.o" % (obj_dir, what)
    f.write("%s : %s %s/%s.%s.dep\n" % (obj, src, obj_dir, what, ext))
    f.write("\t@printf \"Compiling  %%-35s > %%s\\\\n\" %s.%s %s\n" % (what, ext, obj))
    f.write("\tmkdir -p %s\n" % obj_dir)
    f.write("\t${HGCC} ${HGFLAGS} %s -o %s\n" % (src, obj))
    f.write("\n")
