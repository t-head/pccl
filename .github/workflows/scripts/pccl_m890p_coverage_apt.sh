#!/usr/bin/env bash
# Install dependencies in the M890P coverage workflow's Ubuntu 24.04 image.
set -euo pipefail

scratch=$(mktemp -d /tmp/pccl-coverage-apt.XXXXXXXX)
trap 'rm -rf -- "$scratch"' EXIT
# Allow the unprivileged _apt downloader to reach the temporary lists.
chmod 0755 "$scratch"
mkdir -m 0755 "$scratch/sourceparts"
export DEBIAN_FRONTEND=noninteractive

mirrors=(default mirrors.aliyun.com mirrors.tuna.tsinghua.edu.cn mirrors.ustc.edu.cn)
for i in "${!mirrors[@]}"; do
  mirror=${mirrors[i]}
  attempt=$((i + 1))
  # Never install with stale indexes from the image or a failed refresh.
  lists="$scratch/lists-$attempt"
  mkdir -m 0755 "$lists"
  apt_options=(
    -o "Dir::State::lists=$lists"
    -o Acquire::http::Timeout=30
    -o Acquire::https::Timeout=30
    -o Acquire::Retries=2
    -o Acquire::Languages=none
  )
  if [[ "$mirror" != default ]]; then
    sources="$scratch/ubuntu.list"
    keyring=/usr/share/keyrings/ubuntu-archive-keyring.gpg
    for suite in noble noble-updates noble-security; do
      printf 'deb [signed-by=%s] http://%s/ubuntu/ %s main restricted universe\n' "$keyring" "$mirror" "$suite"
    done > "$sources"
    apt_options+=(-o "Dir::Etc::sourcelist=$sources" -o "Dir::Etc::sourceparts=$scratch/sourceparts")
  fi

  printf 'Refreshing APT indexes from %s (%s/%s)\n' "$mirror" "$attempt" "${#mirrors[@]}"
  if timeout -k 10s 120s apt-get "${apt_options[@]}" --error-on=any update </dev/null; then
    # Install against exactly the indexes and sources that passed verification.
    timeout -k 10s 300s apt-get "${apt_options[@]}" install -y --no-install-recommends "$@" </dev/null
    exit 0
  else
    printf '::warning::APT index refresh from %s failed (exit %s)\n' "$mirror" "$?" >&2
  fi
  if ((attempt < ${#mirrors[@]})); then
    sleep 15
  fi
done

echo '::error::APT index refresh failed for the default sources and all fallback mirrors' >&2
exit 1
