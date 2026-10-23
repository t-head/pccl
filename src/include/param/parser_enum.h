/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_PARSER_ENUM_H_INCLUDED
#define PARAM_PARSER_ENUM_H_INCLUDED

#include "param/parser_common.h"
#include "param/utils.h"

namespace pccl {
namespace param {
namespace parser {

template <typename T, size_t N>
pcclResult_t oneOfResolve(const void* ctx, const char* input, T& out) {
  if (input == nullptr) return pcclInvalidArgument;
  auto& opts = *static_cast<const pcclOptionSet<T, N>*>(ctx);
  std::string token = pccl::param::utils::trim(std::string(input));
  if (token.empty()) return pcclInvalidArgument;
  for (const auto& opt : opts) {
    if (pccl::param::utils::iequals(opt.name, token)) { out = opt.value; return pcclSuccess; }
  }
  return pcclInvalidArgument;
}

template <typename T, size_t N>
bool oneOfValidate(const void*, const T&) { return true; }

template <typename T, size_t N>
std::string oneOfToString(const void* ctx, const T& value) {
  auto& opts = *static_cast<const pcclOptionSet<T, N>*>(ctx);
  for (const auto& opt : opts) {
    if (opt.value == value) return opt.name;
  }
  return "<unknown>";
}

} // namespace parser
} // namespace param
} // namespace pccl

// pcclParamOneOf: create parser for mapping enum string to value
template <typename T, size_t N>
pcclParamParser<T> pcclParamOneOf(pcclOptionSet<T, N> options) {
  using namespace pccl::param::parser;
  auto ctx = std::make_shared<pcclOptionSet<T, N>>(std::move(options));
  std::string d = "One of:";
  for (const auto& opt : *ctx) {
    d += "\n        ";
    d += opt.name;
    if (opt.desc != nullptr) { d += " - "; d += opt.desc; }
  }
  return {oneOfResolve<T, N>, oneOfValidate<T, N>, oneOfToString<T, N>,
          std::move(ctx), std::move(d)};
}

#endif /* PARAM_PARSER_ENUM_H_INCLUDED */
