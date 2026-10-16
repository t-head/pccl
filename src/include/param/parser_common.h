/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_PARSER_COMMON_H_INCLUDED
#define PARAM_PARSER_COMMON_H_INCLUDED

#include "pccl.h"

#include <string>
#include <memory>
#include <array>

// pcclParamParser: Runtime parser interface pcclParam depends on
template <typename T>
struct pcclParamParser {
  using resolveFn_t  = pcclResult_t(*)(const void*, const char*, T&);
  using validateFn_t = bool(*)(const void*, const T&);
  using toStringFn_t = std::string(*)(const void*, const T&);

  resolveFn_t  resolveFn  = nullptr;
  validateFn_t validateFn = nullptr;
  toStringFn_t toStringFn = nullptr;
  std::shared_ptr<const void> ctx;  // owns factory state; nullptr for stateless parsers
  std::string desc;  // Description of accepted values

  // Wrapper methods — preserve parser.resolve(...) call syntax
  pcclResult_t resolve(const char* input, T& out) const {
    return resolveFn(ctx.get(), input, out);
  }
  bool validate(const T& val) const {
    return validateFn(ctx.get(), val);
  }
  std::string toString(const T& val) const {
    return toStringFn(ctx.get(), val);
  }
  explicit operator bool() const { return resolveFn != nullptr; }
};

// Empty braces yield a null pcclParamParser<T>; the pcclParam constructor
// detects this and fills from pcclParamDefault<T>().
#define PCCL_PARAM_DEFAULT {}

// Option Builder for enum or bitset types
//
// Usage:
//   auto opts = makeOptions(
//     makeOption<int32_t>("OFF",  0, "Disable feature"),
//     makeOption<int32_t>("ON",   1, "Enable feature"),
//     makeOption<int32_t>("AUTO", 2)      // no description
//   );
//   // opts is pcclOptionSet<int32_t, 3> containing:
//   //   {{"OFF", 0, "Disable feature"}, {"ON", 1, "Enable feature"}, {"AUTO", 2, nullptr}}
//   auto parser = pcclParamOneOf(opts);   // or pcclParamBitsetOf<EnumT>(opts)
template <typename T>
struct pcclOption {
  const char* name;
  T value;
  const char* desc;  // Per-option description (nullptr when no description)
};

// pcclOptionSet: Fixed-size option set (compile-time N, zero heap allocation)
template <typename T, size_t N>
struct pcclOptionSet {
  std::array<pcclOption<T>, N> options;

  constexpr const pcclOption<T>* begin() const { return options.data(); }
  constexpr const pcclOption<T>* end() const { return options.data() + N; }
  constexpr size_t size() const { return N; }
};

// makeOption: Create an option (2-arg: no description)
template <typename T>
pcclOption<T> makeOption(const char* name, T value) {
  return {name, value, nullptr};
}

// makeOption: Create an option with description (3-arg)
template <typename T>
pcclOption<T> makeOption(const char* name, T value, const char* desc) {
  return {name, value, desc};
}

// makeOptions: Create a fixed-size option set from variadic arguments
template <typename T, typename... Args>
auto makeOptions(pcclOption<T> first, Args... rest) -> pcclOptionSet<T, 1 + sizeof...(Args)> {
    return pcclOptionSet<T, 1 + sizeof...(Args)>{{{first, rest...}}};
}

#endif /* PARAM_PARSER_COMMON_H_INCLUDED */
