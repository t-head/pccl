//*************************************************************************
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// See LICENSE.txt for more license information
//*************************************************************************

#include "pccl.h"
#include "hgtx.h"
#include "param.h"

pcclHgtxInfo::pcclHgtxInfo(const char* message) {
  hgtxRangePush(message);
}

pcclHgtxInfo:: ~pcclHgtxInfo() {
  hgtxRangePop();
}

static constexpr const hgtxPayloadEnum_t HgtxEnumRedSchema[] = {
  {"Sum", ncclSum, 0},
  {"Product", ncclProd, 0},
  {"Max", ncclMax, 0},
  {"Min", ncclMin, 0},
  {"Avg", ncclAvg, 0}
};

NCCL_PARAM(NvtxDisable, "NVTX_DISABLE", 0);

// Must be called before the first call to any reduction operation.
void initHgtxRegisteredEnums() {
  // Register schemas and strings
  if (ncclParamNvtxDisable()) {
    return;
  }

  // Register schemas and strings
  constexpr const hgtxPayloadEnumAttr_t eAttr {
    .fieldMask = HGTX_PAYLOAD_ENUM_ATTR_FIELD_ENTRIES | HGTX_PAYLOAD_ENUM_ATTR_FIELD_NUM_ENTRIES |
      HGTX_PAYLOAD_ENUM_ATTR_FIELD_SIZE | HGTX_PAYLOAD_ENUM_ATTR_FIELD_SCHEMA_ID,
    .name = NULL,
    .entries = HgtxEnumRedSchema,
    .numEntries = std::extent<decltype(HgtxEnumRedSchema)>::value,
    .sizeOfEnum = sizeof(ncclRedOp_t),
    .schemaId = HGTX_PAYLOAD_ENTRY_PCCL_REDOP,
    .extension = nullptr
  };

  hgtxPayloadEnumRegister(hgtx3::domain::get<pccl_domain>(), &eAttr);
}
