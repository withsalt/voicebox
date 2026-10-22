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

#include "processor/wetext_processor.h"

#include <stdexcept>

namespace wetext {
Processor::Processor(const std::string& tagger_path,
                     const std::string& verbalizer_path) {
  tagger_.reset(StdVectorFst::Read(tagger_path));
  // [patch] Fail fast with a catchable exception instead of crashing later on
  // a null FST pointer when the file is missing or not a valid FST.
  if (tagger_ == nullptr) {
    throw std::runtime_error("Failed to load tagger FST: " + tagger_path);
  }
  verbalizer_.reset(StdVectorFst::Read(verbalizer_path));
  if (verbalizer_ == nullptr) {
    throw std::runtime_error("Failed to load verbalizer FST: " +
                             verbalizer_path);
  }
  compiler_ = std::make_shared<StringCompiler<StdArc>>();
  printer_ = std::make_shared<StringPrinter<StdArc>>();

  if (tagger_path.find("zh_tn_") != tagger_path.npos) {
    parse_type_ = ParseType::kZH_TN;
  } else if (tagger_path.find("zh_itn_") != tagger_path.npos) {
    parse_type_ = ParseType::kZH_ITN;
  } else if (tagger_path.find("en_tn_") != tagger_path.npos) {
    parse_type_ = ParseType::kEN_TN;
  } else if (tagger_path.find("en_itn_") != tagger_path.npos) {
    // [patch] English ITN support (upstream runtime only recognized zh/ja).
    parse_type_ = ParseType::kEN_ITN;
  } else if (tagger_path.find("ja_tn_") != tagger_path.npos) {
    parse_type_ = ParseType::kJA_TN;
  } else {
    // [patch] Throw instead of LOG(FATAL), which would abort the host process.
    throw std::invalid_argument(
        "Invalid fst prefix, tagger file name should contain one of "
        "\"zh_tn_\", \"zh_itn_\", \"en_tn_\", \"en_itn_\", \"ja_tn_\": " +
        tagger_path);
  }
}

std::string Processor::ShortestPath(const StdVectorFst& lattice) {
  StdVectorFst shortest_path;
  fst::ShortestPath(lattice, &shortest_path, 1, true);

  std::string output;
  printer_->operator()(shortest_path, &output);
  return output;
}

std::string Processor::Compose(const std::string& input,
                               const StdVectorFst* fst) {
  StdVectorFst input_fst;
  compiler_->operator()(input, &input_fst);

  StdVectorFst lattice;
  fst::Compose(input_fst, *fst, &lattice);
  return ShortestPath(lattice);
}

std::string Processor::Tag(const std::string& input) {
  if (input.empty()) {
    return "";
  }
  return Compose(input, tagger_.get());
}

std::string Processor::Verbalize(const std::string& input) {
  if (input.empty()) {
    return "";
  }
  TokenParser parser(parse_type_);
  std::string output = parser.Reorder(input);

  output = Compose(output, verbalizer_.get());
  output.erase(std::remove(output.begin(), output.end(), '\0'), output.end());
  return output;
}

std::string Processor::Normalize(const std::string& input) {
  return Verbalize(Tag(input));
}

}  // namespace wetext
