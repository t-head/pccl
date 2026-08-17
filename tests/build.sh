#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************

if [[ ! -d ./3rd_party ]]; then
  bash ./setup_3rd.sh
fi

ppu_archs="ppu001,ppu0015"
if [[ $# -ge 1 ]] && [[ $1 == "--ppu_archs" ]]; then
  ppu_archs="$2"
fi

rm -rf build
meson build -Dppu_archs=${ppu_archs}
cd build
ninja