#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ ! -s ../3rd_party/gtest/lib/libgtest.a ]]; then
  bash ./setup_3rd.sh
fi

ppu_archs="ppu_10,ppu_15"
if [[ $# -ge 1 ]] && [[ $1 == "--ppu_archs" ]]; then
  ppu_archs="$2"
fi

if [[ -f build/build.ninja ]]; then
  meson setup --reconfigure build -Dppu_archs="${ppu_archs}" -Dcpp_std=c++17
else
  meson setup build -Dppu_archs="${ppu_archs}" -Dcpp_std=c++17
fi
ninja -C build
