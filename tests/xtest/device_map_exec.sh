#!/bin/sh
# Map Case-local PPU indices in each rank before execing its real benchmark.
if [ -z "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
    echo 'NVIDIA_VISIBLE_DEVICES is required for per-worker PPU mapping' >&2
    exit 1
fi
_pccl_logical=${PCCL_LOGICAL_VISIBLE_DEVICES:-${HGGC_VISIBLE_DEVICES:-}}
if [ -z "$_pccl_logical" ]; then
    echo 'A logical PPU selection is required' >&2
    exit 1
fi
_pccl_mapped=
_pccl_old_ifs=$IFS
IFS=,
for _pccl_index in $_pccl_logical; do
    _pccl_position=$((_pccl_index + 1))
    _pccl_count=$(printf '%s\n' "$NVIDIA_VISIBLE_DEVICES" | awk -F, '{print NF}')
    if [ "$_pccl_position" -gt "$_pccl_count" ]; then
        IFS=$_pccl_old_ifs
        echo "PPU local index $_pccl_index exceeds this worker's allocation" >&2
        exit 1
    fi
    _pccl_device=$(printf '%s\n' "$NVIDIA_VISIBLE_DEVICES" | cut -d, -f"$_pccl_position")
    if [ -z "$_pccl_device" ]; then
        IFS=$_pccl_old_ifs
        echo "PPU local index $_pccl_index exceeds this worker's allocation" >&2
        exit 1
    fi
    if [ -n "$_pccl_mapped" ]; then
        _pccl_mapped="$_pccl_mapped,$_pccl_device"
    else
        _pccl_mapped=$_pccl_device
    fi
done
IFS=$_pccl_old_ifs
export HGGC_VISIBLE_DEVICES=$_pccl_mapped
_pccl_name=${0##*/}
_pccl_wrapper_dir=${0%/*}
_pccl_old_ifs=$IFS
IFS=:
for _pccl_directory in $PATH; do
    if [ "$_pccl_directory" = "$_pccl_wrapper_dir" ]; then
        continue
    fi
    _pccl_candidate=$_pccl_directory/$_pccl_name
    if [ -f "$_pccl_candidate" ] && [ -x "$_pccl_candidate" ]; then
        IFS=$_pccl_old_ifs
        exec "$_pccl_candidate" "$@"
    fi
done
IFS=$_pccl_old_ifs
echo "Unable to find the real benchmark executable: $_pccl_name" >&2
exit 127
