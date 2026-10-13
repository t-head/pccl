/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_PARSER_LIST_H_INCLUDED
#define PARAM_PARSER_LIST_H_INCLUDED

#include "param/parser_common.h"
#include "param/utils.h"

namespace pccl {
namespace param {
namespace parser {

struct listOfCtx { char delimiter; };

template <typename ContainerT>
pcclResult_t listOfResolve(const void* ctx, const char* input, ContainerT& out) {
  auto* lc = static_cast<const listOfCtx*>(ctx);
  if (input == nullptr || *input == '\0') return pcclInvalidArgument;
  auto tokens = pccl::param::utils::split(input, lc->delimiter);
  for (const auto& tok : tokens) {
    auto trimmed = pccl::param::utils::trim(tok);
    if (!trimmed.empty()) {
      out.insert(out.end(), std::string(trimmed.data(), trimmed.size()));
    }
  }
  return pcclSuccess;
}

template <typename ContainerT>
bool listOfValidate(const void*, const ContainerT&) { return true; }

template <typename ContainerT>
std::string listOfToString(const void* ctx, const ContainerT& value) {
  auto* lc = static_cast<const listOfCtx*>(ctx);
  std::string s;
  for (const auto& elem : value) {
    if (!s.empty()) { s += lc->delimiter; }
    s += elem;
  }
  return s;
}

} // namespace parser
} // namespace param
} // namespace pccl

// pcclParamListOf: Parser for delimiter-separated strings into a container.
// ContainerT must support insert(end(), std::string) and iteration.
template <typename ContainerT>
pcclParamParser<ContainerT> pcclParamListOf(char delimiter = ',') {
  using namespace pccl::param::parser;
  auto ctx = std::make_shared<listOfCtx>(listOfCtx{delimiter});
  std::string d = std::string("Delimiter-separated list (delimiter='") + delimiter + "')";
  return {listOfResolve<ContainerT>, listOfValidate<ContainerT>, listOfToString<ContainerT>,
          std::move(ctx), std::move(d)};
}

#endif /* PARAM_PARSER_LIST_H_INCLUDED */
