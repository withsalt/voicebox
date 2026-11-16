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

#ifndef PROCESSOR_WETEXT_TOKEN_PARSER_H_
#define PROCESSOR_WETEXT_TOKEN_PARSER_H_

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace wetext {

extern const char EOS[];
extern const std::set<std::string> UTF8_WHITESPACE;
extern const std::set<std::string> ASCII_LETTERS;
extern const std::unordered_map<std::string, std::vector<std::string>>
    TN_ORDERS;
extern const std::unordered_map<std::string, std::vector<std::string>>
    ITN_ORDERS;
extern const std::unordered_map<std::string, std::vector<std::string>>
    EN_TN_ORDERS;
extern const std::unordered_map<std::string, std::vector<std::string>>
    JA_TN_ORDERS;

struct Token {
  std::string name;
  std::vector<std::string> order;
  std::unordered_map<std::string, std::string> members;

  explicit Token(const std::string& name) : name(name) {}

  void Append(const std::string& key, const std::string& value) {
    // [patch] Match Python 1.3.0: duplicate fields are invalid.
    if (members.count(key) > 0) {
      throw std::invalid_argument("Duplicate field " + key + " in token " + name);
    }
    order.emplace_back(key);
    members[key] = value;
  }

  std::string String(
      const std::unordered_map<std::string, std::vector<std::string>>& orders) {
    std::string output = name + " {";
    // [patch] Mirror Python 1.3.0: preserve the tagger's order when requested,
    // otherwise append fields absent from the canonical schema in input order.
    auto output_order = order;
    if (orders.count(name) > 0) {
      auto it = members.find("preserve_order");
      if (it == members.end() || it->second != "true") {
        output_order = orders.at(name);
        for (const auto& key : order) {
          if (std::find(output_order.begin(), output_order.end(), key) ==
              output_order.end()) {
            output_order.emplace_back(key);
          }
        }
      }
    }

    for (const auto& key : output_order) {
      if (members.count(key) == 0) {
        continue;
      }
      output += " " + key + ": \"";
      // [patch] Python stores decoded values and escapes them on serialization.
      for (char ch : members.at(key)) {
        if (ch == '\\' || ch == '"') {
          output += '\\';
        }
        output += ch;
      }
      output += '"';
    }
    return output + " }";
  }
};

enum ParseType {
  kZH_TN = 0x00,   // Chinese Text Normalization
  kZH_ITN = 0x01,  // Chinese Inverse Text Normalization
  kEN_TN = 0x02,   // English Text Normalization
  kEN_ITN = 0x03,  // English Inverse Text Normalization
  kJA_TN = 0x04,   // Japanese Text Normalization
  kJA_ITN = 0x05   // Japanese Inverse Text Normalization
};

class TokenParser {
 public:
  explicit TokenParser(ParseType type);
  std::string Reorder(const std::string& input);

 private:
  void Load(const std::string& input);
  bool Read();
  bool ParseWs();
  bool ParseChar(const std::string& exp);
  bool ParseChars(const std::string& exp);
  std::string ParseKey();
  std::string ParseValue();
  void Parse(const std::string& input);

  size_t index_;
  std::string ch_;
  std::vector<std::string> text_;
  std::vector<Token> tokens_;
  std::unordered_map<std::string, std::vector<std::string>> orders_;
};

}  // namespace wetext

#endif  // PROCESSOR_WETEXT_TOKEN_PARSER_H_
