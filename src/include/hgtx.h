/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PCCL_HGTX_H_
#define PCCL_HGTX_H_

#include "hgtx3/hgtx3.hpp"
#include "hgtx3/hgToolsExtPayload.h"
#include "pccl_internal.h"
#include "comm.h"

#if __cpp_constexpr >= 201304L && !defined(HGTX3_CONSTEXPR_IF_CPP14)
#define HGTX3_CONSTEXPR_IF_CPP14 constexpr
#else
#define HGTX3_CONSTEXPR_IF_CPP14
#endif

struct pcclHgtxInfo {
  pcclHgtxInfo(const char* message);

  ~pcclHgtxInfo();
};

// Define all PCCL-provided static schema IDs here (avoid duplicates).
#define HGTX_SID_CommInitRank         0
#define HGTX_SID_CommInitAll          1
#define HGTX_SID_CommDestroy          2 // same schema as HGTX_SID_CommInitRank
#define HGTX_SID_CommAbort            3 // same schema as HGTX_SID_CommInitRank
#define HGTX_SID_AllGather            4
#define HGTX_SID_AllReduce            5
#define HGTX_SID_Broadcast            6
#define HGTX_SID_ReduceScatter        7
#define HGTX_SID_Reduce               8
#define HGTX_SID_Send                 9
#define HGTX_SID_Recv                 10
#define HGTX_SID_CommInitRankConfig   11 // same schema as HGTX_SID_CommInitRank
#define HGTX_SID_CommInitRankScalable 12 // same schema as HGTX_SID_CommInitRank
#define HGTX_SID_CommSplit            13
#define HGTX_SID_CommFinalize         14
#define HGTX_SID_CommShrink           15
#define HGTX_SID_AlltoAll             16
#define HGTX_SID_Gather               17
#define HGTX_SID_Scatter              18
#define HGTX_SID_CommRevoke           19 // same schema as HGTX_SID_CommInitRank
#define HGTX_SID_CommGrow             21 // 20 is taken by HGTX_PAYLOAD_ENTRY_PCCL_REDOP

// Define static schema ID for the reduction operation.
#define HGTX_PAYLOAD_ENTRY_PCCL_REDOP 20 + HGTX_PAYLOAD_SCHEMA_ID_STATIC_START

#define HGTX_SID_PutSignal            22
#define HGTX_SID_Signal               23
#define HGTX_SID_WaitSignal           24

extern const hgtxDomainHandle_t pcclHgtxDomainHandle;

struct pccl_domain{static constexpr char const* name{"PCCL"};};

extern int64_t ncclParamNvtxDisable();

/// @brief Register an HGTX payload schema for static-size payloads.
class payload_schema {
 public:
  explicit payload_schema(const hgtxPayloadSchemaEntry_t entries[], size_t numEntries,
    const uint64_t schemaId, const size_t size) noexcept
  {
    schema_attr.payloadStaticSize = size;
    schema_attr.entries = entries;
    schema_attr.numEntries = numEntries;
    schema_attr.schemaId = schemaId;
    hgtxPayloadSchemaRegister(hgtx3::domain::get<pccl_domain>(), &schema_attr);
  }

  payload_schema() = delete;
  ~payload_schema() = default;
  payload_schema(payload_schema const&) = default;
  payload_schema& operator=(payload_schema const&) = default;
  payload_schema(payload_schema&&) = default;
  payload_schema& operator=(payload_schema&&) = default;

 private:
  hgtxPayloadSchemaAttr_t schema_attr{
    HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_TYPE |
    HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_ENTRIES |
    HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_NUM_ENTRIES |
    HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_STATIC_SIZE |
    HGTX_PAYLOAD_SCHEMA_ATTR_FIELD_SCHEMA_ID,
    nullptr, /* schema name is not needed */
    HGTX_PAYLOAD_SCHEMA_TYPE_STATIC,
    HGTX_PAYLOAD_SCHEMA_FLAG_NONE,
    nullptr, 0, 0, 0, 0, nullptr};
};

class pcclOptionalHgtxScopedRange
{
 public:
  void push(const hgtx3::event_attributes& attr) noexcept {
    // pushed must not be true already, but it's too expensive to check
    pushed = true;
    hgtxDomainRangePushEx(hgtx3::domain::get<pccl_domain>(), attr.get());
  }

  ~pcclOptionalHgtxScopedRange() noexcept {
    if (!pushed) {
      return;
    }
    hgtxDomainRangePop(hgtx3::domain::get<pccl_domain>());
  }

  pcclOptionalHgtxScopedRange() = default;
  pcclOptionalHgtxScopedRange(pcclOptionalHgtxScopedRange const&) = delete;
  pcclOptionalHgtxScopedRange& operator=(pcclOptionalHgtxScopedRange const&) = delete;
  pcclOptionalHgtxScopedRange(pcclOptionalHgtxScopedRange&&) = delete;
  pcclOptionalHgtxScopedRange& operator=(pcclOptionalHgtxScopedRange&&) = delete;

 private:
  bool pushed = false;
};

// Convenience macro to give the payload parameters a scope.
#define HGTX3_PAYLOAD(...) __VA_ARGS__

// Create HGTX push/pop range with parameters
// @param N PCCL API name without the `nccl` prefix.
// @param T name of the used HGTX payload schema without "Schema" suffix.
// @param P payload parameters/entries
#define HGTX3_FUNC_WITH_PARAMS(N, T, P)                                                          \
  pcclOptionalHgtxScopedRange hgtx3_range__;                                                     \
  if (!ncclParamNvtxDisable())                                                                   \
  {                                                                                              \
    constexpr uint64_t schemaId = HGTX_PAYLOAD_SCHEMA_ID_STATIC_START + HGTX_SID_##N; \
    static const payload_schema                                                                  \
        schema{T##Schema, std::extent<decltype(T##Schema)>::value - 1, schemaId, sizeof(T)};     \
    static ::hgtx3::v1::registered_string_in<pccl_domain> const hgtx3_func_name__{__func__};     \
    const T _payload = {P};                                                                      \
    hgtxPayloadData_t hgtx3_bpl__[] = {{schemaId, sizeof(_payload), &_payload}};                 \
    ::hgtx3::v1::event_attributes const hgtx3_func_attr__{hgtx3_func_name__, hgtx3_bpl__};       \
    hgtx3_range__.push(hgtx3_func_attr__);                                                       \
  }

#define PCCL_HGTX3_FUNC_RANGE \
  pcclOptionalHgtxScopedRange hgtx3_range__; \
  if (!ncclParamNvtxDisable()) { \
    static ::hgtx3::v1::registered_string_in<pccl_domain> const hgtx3_func_name__{__func__}; \
    static ::hgtx3::v1::event_attributes const hgtx3_func_attr__{hgtx3_func_name__}; \
    hgtx3_range__.push(hgtx3_func_attr__); \
  }

/// @brief Creates an HGTX range with extended payload using the RAII pattern.
/// @tparam PayloadType Data type of the payload.
template <typename PayloadType>
class pcclOptionalHgtxPayloadRange {
 public:
  void push(const hgtx3::event_attributes& attr) noexcept {
    // pushed must not be true already, but it's too expensive to check
    pushed = true;
    hgtxDomainRangePushEx(hgtx3::domain::get<pccl_domain>(), attr.get());
  }

  ~pcclOptionalHgtxPayloadRange() noexcept {
    if (!pushed) {
      return;
    }
    if (payloadData.payload) {
      hgtxRangePopPayload(hgtx3::domain::get<pccl_domain>(), &payloadData, 1);
    } else {
      hgtxDomainRangePop(hgtx3::domain::get<pccl_domain>());
    }
  }

  void setPayloadData(const uint64_t schemaId) noexcept
  {
    payloadData = {schemaId, sizeof(PayloadType), &payload};
  }

  pcclOptionalHgtxPayloadRange() = default;
  pcclOptionalHgtxPayloadRange(pcclOptionalHgtxPayloadRange const&) = delete;
  pcclOptionalHgtxPayloadRange& operator=(pcclOptionalHgtxPayloadRange const&) = delete;
  pcclOptionalHgtxPayloadRange(pcclOptionalHgtxPayloadRange&&) = delete;
  pcclOptionalHgtxPayloadRange& operator=(pcclOptionalHgtxPayloadRange&&) = delete;

  // Holds the payload data.
  PayloadType payload{};

  bool isPushed() const noexcept {
    return pushed;
  }

 private:
  bool pushed = false;
  hgtxPayloadData_t payloadData = {HGTX_PAYLOAD_ENTRY_TYPE_INVALID, 0, NULL};
};

// Create an HGTX range with the function name as the range name. Use RAII pattern.
// @param T Type ID of the HGTX payload (pointer for variable-size payloads).
#define HGTX3_RANGE(T)                                                                       \
  pcclOptionalHgtxPayloadRange<T> hgtx3_range__;                                             \
  if (!ncclParamNvtxDisable())                                                               \
  {                                                                                          \
    static ::hgtx3::v1::registered_string_in<pccl_domain> const hgtx3_func_name__{__func__}; \
    ::hgtx3::v1::event_attributes const hgtx3_func_attr__{hgtx3_func_name__};                \
    hgtx3_range__.push(hgtx3_func_attr__);                                                   \
  }

// Add static-size payload to the HGTX range created with `HGTX3_RANGE()`,
// which must be in this or an outer scope.
// @param N PCCL API name without the `pccl` prefix.
// @param S name of the used HGTX payload schema.
// @param P payload parameters/entries
#define HGTX3_RANGE_ADD_PAYLOAD(N, S, P) do { \
  if (!hgtx3_range__.isPushed()) { \
    break; \
  } \
  constexpr uint64_t schema_id = HGTX_PAYLOAD_SCHEMA_ID_STATIC_START + HGTX_SID_##N; \
  static const payload_schema schema{S, std::extent<decltype(S)>::value - 1, schema_id, \
    sizeof(hgtx3_range__.payload)}; \
  hgtx3_range__.payload = {P}; \
  hgtx3_range__.setPayloadData(schema_id); \
} while (0)

extern void initHgtxRegisteredEnums();

#endif
