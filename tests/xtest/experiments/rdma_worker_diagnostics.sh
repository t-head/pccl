#!/bin/bash
# Read-only runtime evidence on each worker; no SDK or driver changes.
set -euo pipefail
umask 022
root=${1:?Experiment result directory is required}
mkdir -p "$root/diagnostics"
log="$root/diagnostics/worker-${NODE_RANK:-0}.log"
{
  date -Is
  hostname -f
  uname -a
  ppu-smi -L || true
  cat /proc/driver/alixpu/version 2>/dev/null || true
  # /proc and /sys remain usable when kmod tools are absent in the image.
  grep -E 'alixpu|mlx5|ib_core|rdma' /proc/modules || true
  for path in /sys/module/alixpu_peermem /sys/module/alixpu_peer_mem \
      /sys/kernel/mm/memory_peers/alixpu_peer_mem; do
    if [ -d "$path" ]; then
      printf 'PEER_MEMORY_PATH_PRESENT %s\n' "$path"
      for value in version srcversion taint; do
        if [ -f "$path/$value" ]; then printf '%s=' "$value"; cat "$path/$value"; fi
      done
    else
      printf 'PEER_MEMORY_PATH_ABSENT %s\n' "$path"
    fi
  done
  for module in alixpu_peermem mlx5_core mlx5_ib; do
    modinfo "$module" 2>/dev/null | grep -E '^(filename|version|srcversion|vermagic):' || true
  done
  dpkg-query -W 'libibverbs*' 'rdma-core' 'ibverbs-providers' 2>/dev/null || true
  for directory in /usr/local/PPU_SDK/lib /usr/local/PPU_SDK/PPU_SDK/lib \
      /usr/local/openmpi/lib /usr/lib/x86_64-linux-gnu; do
    if [ -d "$directory" ]; then
      find "$directory" -maxdepth 1 -name 'libibverbs*' -exec ls -l {} \; || true
    fi
  done
  ip -br address || true
  rdma link show || true
  ibv_devinfo || true
  ulimit -a
  env | grep -E '^(PCCL_|NCCL_|HGGC_|UMD_|ALIPPU_|NVIDIA_VISIBLE|OMPI_)' | sort || true
} > "$log" 2>&1
# No host-wide diagnostics are changed or cleared.
dmesg > "$root/diagnostics/worker-${NODE_RANK:-0}-dmesg-before.log" 2>&1 || true
