#!/bin/bash
# Read-only runtime evidence on each worker; no SDK or driver changes.
set -euo pipefail
root=${1:?Experiment result directory is required}
mkdir -p "$root/diagnostics"
log="$root/diagnostics/worker-${NODE_RANK:-0}.log"
{
  date -Is
  hostname -f
  uname -a
  ppu-smi -L || true
  cat /proc/driver/alixpu/version 2>/dev/null || true
  lsmod | grep -E 'alixpu|mlx5|ib_core|rdma' || true
  for module in alixpu_peermem mlx5_core mlx5_ib; do
    modinfo "$module" 2>/dev/null | grep -E '^(filename|version|srcversion|vermagic):' || true
  done
  dpkg-query -W 'libibverbs*' 'rdma-core' 'ibverbs-providers' 2>/dev/null || true
  ip -br address || true
  rdma link show || true
  ibv_devinfo || true
  ulimit -a
  env | grep -E '^(PCCL_|NCCL_|HGGC_|UMD_|ALIPPU_|NVIDIA_VISIBLE|OMPI_)' | sort || true
} > "$log" 2>&1
# No host-wide diagnostics are changed or cleared.
dmesg > "$root/diagnostics/worker-${NODE_RANK:-0}-dmesg-before.log" 2>&1 || true
