#!/bin/bash
# Place remotely launched commands into the current execution group's cgroup.
set -euo pipefail
INVOKED_AS=$1
shift
: "${PCCL_CASE_ID:?PCCL_CASE_ID is required}"
[[ $PCCL_CASE_ID =~ ^[a-f0-9]{32}$ ]] || { echo 'Invalid PCCL_CASE_ID' >&2; exit 64; }
REAL_SSH=/run/cluster/bin/cluster-ssh
[[ $INVOKED_AS == */cluster-ssh || $INVOKED_AS == */cluster-exec ]] || {
    echo 'cluster_case_ssh must be invoked as cluster-ssh or cluster-exec' >&2
    exit 64
}
options=()
host=
while (( $# )); do
    case $1 in
        --) shift; (( $# )) || exit 64; host=$1; shift; break ;;
        -v|-vv|-vvv|-q|-T|-t|-tt|-n|-x|-C) options[${#options[@]}]=$1; shift ;;
        -l|-lroot)
            if [[ $1 == -l ]]; then
                (( $# >= 2 )) && [[ $2 == root ]] || exit 64
                options[${#options[@]}]=$1; options[${#options[@]}]=$2; shift 2
            else options[${#options[@]}]=$1; shift; fi ;;
        -o|-o*)
            if [[ $1 == -o ]]; then
                (( $# >= 2 )) || exit 64
                options[${#options[@]}]=$1; options[${#options[@]}]=$2; shift 2
            else options[${#options[@]}]=$1; shift; fi ;;
        -*) echo "Unsupported SSH argument: $1" >&2; exit 64 ;;
        *) host=$1; shift; break ;;
    esac
done
[[ -n $host ]] || { echo 'usage: cluster-ssh [options] host command' >&2; exit 64; }
(( $# > 0 )) || { echo 'A remote command is required for cgroup ownership' >&2; exit 64; }
command=
for argument in "$@"; do
    quoted=${argument//\'/\'\\\'\'}
    command="$command'$quoted' "
done
cgroup="/sys/fs/cgroup/pccl-recovery-$PCCL_CASE_ID"
remote="mkdir -p '$cgroup'; printf '%s\\n' \"\$\$\" > '$cgroup/cgroup.procs'; $command"
if (( ${#options[@]} > 0 )); then
    exec "$REAL_SSH" "${options[@]}" "$host" "$remote"
else
    exec "$REAL_SSH" "$host" "$remote"
fi
