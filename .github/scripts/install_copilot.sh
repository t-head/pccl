#!/usr/bin/env bash
set -euo pipefail

export VERSION=v1.0.100
export PREFIX="$RUNNER_TEMP/copilot"
curl --fail --silent --show-error --location --retry 3 \
  https://raw.githubusercontent.com/github/copilot-cli/84ee80ea6490ce02d6000439356a609fcefbb9ae/install.sh | bash
# The installer allows missing checksums; verify the executable before use.
echo "0d0cfb14f1a214ebbe0edc20c509dce0bc4f2bd0e4310c849af5c436d4d9f321  $PREFIX/bin/copilot" | sha256sum --check -
