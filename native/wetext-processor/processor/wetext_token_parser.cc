// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "processor/wetext_token_parser.h"

#include <stdexcept>

#include "utils/wetext_string.h"

namespace wetext {
const char EOS[] = "<EOS>";
const std::set<std::string> UTF8_WHITESPACE = {" ", "\t", "\n", "\r",
                                               "\x0b\x0c"};
const std::set<std::string> ASCII_LETTERS = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B",
    "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "_"};

// [patch] Order tables synced with WeTextProcessing 1.2.0
// tn/token_parser.py (TN_ORDERS / ITN_ORDERS / EN_TN_ORDERS). ITN_ORDERS is
// shared by Chinese and English ITN, exactly like the Python TokenParser.
const std::unordered_map<std::string, std::vector<std::string>> TN_ORDERS = {
    {"date", {"year", "month", "day"}},
    {"fraction", {"denominator", "numerator"}},
    {"measure", {"denominator", "numerator", "value"}},
    {"money", {"value", "currency"}},
    {"time", {"noon", "hour", "minute", "second"}}};

const std::unordered_map<std::string, std::vector<std::string>> ITN_ORDERS = {
    {"date", {"year", "month", "day", "preserve_order"}},
    {"fraction", {"sign", "numerator", "denominator"}},
    {"measure", {"numerator", "denominator", "value", "units"}},
    {"money", {"currency", "value", "decimal", "quantity"}},
    {"time", {"hour", "minute", "second", "noon", "zone"}},
    {"telephone", {"country_code", "number_part"}},
    {"electronic", {"username", "domain", "protocol"}}};

const std::unordered_map<std::string, std::vector<std::string>> EN_TN_ORDERS = {
    {"date", {"preserve_order", "text", "day", "month", "year"}},
    {"money", {"integer_part", "fractional_part", "quantity", "currency_maj"}}};

const std::unordered_map<std::string, std::vector<std::string>> JA_TN_ORDERS = {
    {"date", {"year", "month", "day"}},
    {"money", {"value", "currency"}}};

TokenParser::TokenParser(ParseType type) {
  if (type == ParseType::kZH_TN) {
    orders_ = TN_ORDERS;
  } else if (type == ParseType::kZH_ITN || type == ParseType::kEN_ITN) {
    orders_ = ITN_ORDERS;
  } else if (type == ParseType::kEN_TN) {
    orders_ = EN_TN_ORDERS;
  } else if (type == ParseType::kJA_TN) {
    orders_ = JA_TN_ORDERS;
  } else {
    // [patch] Throw instead of LOG(FATAL) so library consumers can recover.
    throw std::invalid_argument("TokenParser: unsupported parse type.");
  }
}

void TokenParser::Load(const std::string& input) {
  wetext::SplitUTF8StringToChars(input, &text_);
  if (text_.empty()) {
    throw std::invalid_argument("TokenParser: input must not be empty.");
  }
  index_ = 0;
  ch_ = text_[0];
}

bool TokenParser::Read() {
  if (index_ < static_cast<int>(text_.size()) - 1) {
    index_ += 1;
    ch_ = text_[index_];
    return true;
  }
  ch_ = EOS;
  return false;
}

bool TokenParser::ParseWs() {
  bool not_eos = ch_ != EOS;
  while (not_eos && ch_ == " ") {
    not_eos = Read();
  }
  return not_eos;
}

bool TokenParser::ParseChar(const std::string& exp) {
  if (ch_ == exp) {
    Read();
    return true;
  }
  return false;
}

bool TokenParser::ParseChars(const std::string& exp) {
  bool ok = false;
  std::vector<std::string> chars;
  wetext::SplitUTF8StringToChars(exp, &chars);
  for (const auto& x : chars) {
    ok |= ParseChar(x);
  }
  return ok;
}

std::string TokenParser::ParseKey() {
  // [patch] Throw instead of CHECK (process abort) on malformed input.
  if (ch_ == EOS) {
    throw std::runtime_error("TokenParser: unexpected end of input in key.");
  }
  if (UTF8_WHITESPACE.count(ch_) > 0) {
    throw std::runtime_error("TokenParser: unexpected whitespace in key.");
  }

  std::string key = "";
  while (ASCII_LETTERS.count(ch_) > 0) {
    key += ch_;
    Read();
  }
  return key;
}

std::string TokenParser::ParseValue() {
  // [patch] Throw instead of CHECK, and guard the loop against EOS — the
  // upstream implementation spins forever on an unterminated quoted value.
  if (ch_ == EOS) {
    throw std::runtime_error("TokenParser: unexpected end of input in value.");
  }
  bool escape = false;

  std::string value = "";
  while (ch_ != "\"") {
    if (ch_ == EOS) {
      throw std::runtime_error("TokenParser: unterminated quoted value.");
    }
    value += ch_;
    escape = ch_ == "\\";
    Read();
    if (escape) {
      escape = false;
      value += ch_;
      Read();
    }
  }
  return value;
}

void TokenParser::Parse(const std::string& input) {
  Load(input);
  while (ParseWs()) {
    std::string name = ParseKey();
    ParseChars(" { ");

    Token token(name);
    while (ParseWs()) {
      if (ch_ == "}") {
        ParseChar("}");
        break;
      }
      std::string key = ParseKey();
      ParseChars(": \"");
      std::string value = ParseValue();
      ParseChar("\"");
      token.Append(key, value);
    }
    tokens_.emplace_back(token);
  }
}

std::string TokenParser::Reorder(const std::string& input) {
  Parse(input);
  std::string output = "";
  for (auto& token : tokens_) {
    output += token.String(orders_) + " ";
  }
  return Trim(output);
}

}  // namespace wetext
