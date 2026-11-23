#!/bin/sh
# Map Case-local PPU indices in each rank before execing its real benchmark.
if [ -z "${NVIDIA_VISIBLE_DEVICES:-}" ]; then
    echo 'NVIDIA_VISIBLE_DEVICES is required for per-worker PPU mapping' >&2
    exit 1
fi
# Each remote worker has its own device-plugin allocation; master validation
# does not cover it. Normalize decimal indices before detecting duplicates.
_pccl_allocation=$(awk 'BEGIN {
    allocation = ENVIRON["NVIDIA_VISIBLE_DEVICES"]
    if (allocation !~ /^[[:space:]]*[0-9]+[[:space:]]*(,[[:space:]]*[0-9]+[[:space:]]*)*$/) exit 1
    count = split(allocation, devices, ",")
    if (count == 0) exit 1
    for (position = 1; position <= count; position++) {
        device = devices[position]
        sub(/^[[:space:]]+/, "", device)
        sub(/[[:space:]]+$/, "", device)
        if (device !~ /^[0-9]+$/) exit 1
        sub(/^0+/, "", device)
        if (device == "") device = "0"
        if (seen["device:" device]++) exit 1
        normalized = normalized (position == 1 ? "" : ",") device
    }
    printf "%s", normalized
}') || {
    echo 'NVIDIA_VISIBLE_DEVICES must be a finite, unique list of numeric PPU indices' >&2
    exit 1
}
_pccl_logical=${PCCL_LOGICAL_VISIBLE_DEVICES:-${HGGC_VISIBLE_DEVICES:-}}
if [ -z "$_pccl_logical" ]; then
    echo 'A logical PPU selection is required' >&2
    exit 1
fi
_pccl_mapped=
_pccl_count=$(printf '%s\n' "$_pccl_allocation" | awk -F, '{print NF}')
_pccl_old_ifs=$IFS
IFS=,
for _pccl_index in $_pccl_logical; do
    _pccl_position=$((_pccl_index + 1))
    if [ "$_pccl_position" -gt "$_pccl_count" ]; then
        IFS=$_pccl_old_ifs
        echo "PPU local index $_pccl_index exceeds this worker's allocation" >&2
        exit 1
    fi
    _pccl_device=$(printf '%s\n' "$_pccl_allocation" | cut -d, -f"$_pccl_position")
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
export HGGC_VISIBLE_DEVICES="$_pccl_mapped"
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
