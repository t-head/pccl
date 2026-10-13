/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_PARSER_DEFAULT_H_INCLUDED
#define PARAM_PARSER_DEFAULT_H_INCLUDED

#include "param/parser_common.h"
#include "param/utils.h"

#include <type_traits>
#include <limits>
#include <cstdlib>
#include <cerrno>

// Parsers for default types

// Primary template - catch all for unsupported types
template <typename T>
struct pcclParamParserDefault {
  static pcclResult_t resolve(const char*, T&) { return pcclInvalidArgument; }

  static bool validate(const T&) { return false; }

  static std::string toString(const T&) {
    return "<unsupported>";
  }

  static constexpr const char* desc = "Unsupported parser";
};

// Specialization for bool
template <>
struct pcclParamParserDefault<bool> {
  static pcclResult_t resolve(const char* input, bool& out) {
    if (input == nullptr) return pcclInvalidArgument;
    std::string s(input);
    using pccl::param::utils::iequals;
    if (s == "1" || iequals(s, "T") || iequals(s, "TRUE"))  { out = true;  return pcclSuccess; }
    if (s == "0" || iequals(s, "F") || iequals(s, "FALSE")) { out = false; return pcclSuccess; }
    return pcclInvalidArgument;
  }

  static bool validate(const bool&) { return true; }

  static std::string toString(const bool& value) {
    return value ? "TRUE" : "FALSE";
  }

  static constexpr const char* desc = "Boolean: 1/T/TRUE or 0/F/FALSE";
};

// Specialization for const char*
// Note: This parser returns a pointer into the provided input; pcclParam<const char*>
// owns/copies the string into internal storage in pcclParam
template <>
struct pcclParamParserDefault<const char*> {
  static pcclResult_t resolve(const char* input, const char*& out) {
    out = input;
    return pcclSuccess;
  }

  static bool validate(const char* const&) { return true; }

  static std::string toString(const char* const& value) {
    return value ? std::string(value) : std::string();
  }

  static constexpr const char* desc = "String";
};

// Helper base for integer types
template <typename T>
struct pcclIntegerParser {
  static pcclResult_t resolve(const char* input, T& out) {
    if (input == nullptr || *input == '\0') return pcclInvalidArgument;
    char* endPtr = nullptr;
    errno = 0;
    if PCCL_PARAM_IF_CONSTEXPR (std::is_signed<T>::value) {
      long long val = std::strtoll(input, &endPtr, 10);
      if (endPtr == input || *endPtr != '\0' || errno == ERANGE || errno == EINVAL)
        return pcclInvalidArgument;
      out = static_cast<T>(val);
    } else {
      unsigned long long val = std::strtoull(input, &endPtr, 10);
      if (endPtr == input || *endPtr != '\0' || errno == ERANGE || errno == EINVAL)
        return pcclInvalidArgument;
      out = static_cast<T>(val);
    }
    return pcclSuccess;
  }

  static bool validate(const T& val) {
    return val >= std::numeric_limits<T>::min() && val <= std::numeric_limits<T>::max();
  }

  static std::string toString(const T& value) {
    return std::to_string(value);
  }

  static constexpr const char* desc = "Integer";
};

// Explicit specializations for integer types
template <> struct pcclParamParserDefault<int8_t>   : pcclIntegerParser<int8_t> {};
template <> struct pcclParamParserDefault<int16_t>  : pcclIntegerParser<int16_t> {};
template <> struct pcclParamParserDefault<int32_t>  : pcclIntegerParser<int32_t> {};
template <> struct pcclParamParserDefault<int64_t>  : pcclIntegerParser<int64_t> {};
template <> struct pcclParamParserDefault<uint8_t>  : pcclIntegerParser<uint8_t> {};
template <> struct pcclParamParserDefault<uint16_t> : pcclIntegerParser<uint16_t> {};
template <> struct pcclParamParserDefault<uint32_t> : pcclIntegerParser<uint32_t> {};
template <> struct pcclParamParserDefault<uint64_t> : pcclIntegerParser<uint64_t> {};

// ============================================================================
// pccl::param::parser — adapt static methods to function-pointer signatures
// ============================================================================
namespace pccl {
namespace param {
namespace parser {

template <typename T>
pcclResult_t defaultResolve(const void*, const char* input, T& out) {
  return pcclParamParserDefault<T>::resolve(input, out);
}

template <typename T>
bool defaultValidate(const void*, const T& val) {
  return pcclParamParserDefault<T>::validate(val);
}

template <typename T>
std::string defaultToString(const void*, const T& val) {
  return pcclParamParserDefault<T>::toString(val);
}

template <typename T>
struct boundedCtx { T lower; T upper; };

template <typename T>
bool boundedValidate(const void* ctx, const T& val) {
  auto* b = static_cast<const boundedCtx<T>*>(ctx);
  return val >= b->lower && val <= b->upper;
}

} // namespace parser
} // namespace param
} // namespace pccl

// ============================================================================
// Factory for default parser of type T
// ============================================================================
template <typename T>
const pcclParamParser<T>& pcclParamDefault() {
  using namespace pccl::param::parser;
  static const pcclParamParser<T> instance{
    defaultResolve<T>,
    defaultValidate<T>,
    defaultToString<T>,
    nullptr,
    pcclParamParserDefault<T>::desc
  };
  return instance;
}

// ============================================================================
// Bounded Parser Factory
// ============================================================================

// pcclParamBounded: is based on default parser with upper and lower bounds
// using resolve and toString of the default parser, customize validate function
template <typename T>
pcclParamParser<T> pcclParamBounded(T lower, T upper) {
  using namespace pccl::param::parser;
  auto ctx = std::make_shared<boundedCtx<T>>(boundedCtx<T>{lower, upper});
  std::string d = "Integer in range [" + std::to_string(lower) + ", " +
                  std::to_string(upper) + "]";
  return {defaultResolve<T>, boundedValidate<T>, defaultToString<T>,
          std::move(ctx), std::move(d)};
}

template <typename T>
pcclParamParser<T> pcclParamBounded(T lower) {
  return pcclParamBounded(lower, std::numeric_limits<T>::max());
}

#endif /* PARAM_PARSER_DEFAULT_H_INCLUDED */
