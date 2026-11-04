// SPDX-License-Identifier: MPL-2.0
// google_ocr::FstCharLanguageModel<speech::CompactLmFst>: character n-gram LM used by the CTC beam decoder.
//
// Each recognizer class (label map entry) maps to a sequence of LM symbols (OcrLanguageModel::InitSub):
//   - class " " -> "<space>";
//   - otherwise the label is normalized (google_ocr::NormalizeOcrSymbol) and split into code points
//     (google_ocr::PreparePoints); each piece is looked up in the text symbol table (<script>_lm.syms, lines
//     "symbol<TAB>id"), unknown pieces map to "<unk>".
//   NormalizeOcrSymbol tests `name[0] <= 31` on a signed char, so every label that is a single non-ASCII code
//   point normalizes to "" and gets *no* LM symbols: such classes cost nothing and keep the LM state. Only ASCII
//   and multi-code-point labels (e.g. base letter + combining mark) are scored by the LM.
//   - the end-of-sentence class (label named "eos..." or max class id + 1) maps to "</S>" (or "<eos>").
// Costs: LookupId walks backoff arcs until the symbol is found; cost = backoff weights + arc weight (float),
// capped at max_cost = -log(1e-6). "<unk>" and dead ends cost max_cost and reset the state to Start().
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lm_compact_fst.h"

namespace screenai::lm {

class FstCharLanguageModel {
 public:
  // labels: class id -> label name (the recognizer label map, without the CTC blank).
  static std::unique_ptr<FstCharLanguageModel> Create(std::string_view fst_file, std::string_view syms_file,
                                                      const std::vector<std::pair<int, std::string>>& labels,
                                                      std::string* error);

  int BosState() const { return fst_->Start(); }
  int num_classes() const { return num_classes_; }  // max class id + 1 (+1 for an implicit eos class)
  int eos_class() const { return eos_class_; }
  double max_cost() const { return max_cost_; }
  const std::vector<int>& ClassSymbols(int class_id) const;

  // GetNLogProbabilities: for each class in `classes`, (-log p summed over its symbols, resulting state).
  void GetNLogProbabilities(const std::vector<int>& classes, int state,
                            std::vector<std::pair<double, int>>* out) const;
  // One symbol from `state`: returns the cost, updates *state.
  double LookupId(int* state, int symbol) const;

  // Symbol id of an LM token; " " is "<space>", missing -> "<unk>".
  int GetLmId(std::string_view token) const;

 private:
  std::unique_ptr<CompactLmFst> fst_;
  std::unordered_map<std::string, int> symbols_;
  std::vector<std::vector<int>> class_symbols_;
  std::vector<int> unk_symbols_;
  int num_classes_ = 0, eos_class_ = 0, unk_ = -1, eos_symbol_ = -1;
  double max_cost_ = 0;
};

// google_ocr::NormalizeOcrSymbol(s, /*use_space_token=*/true) and PreparePoints(s, /*normalize_points=*/false),
// exposed for tests.
std::string NormalizeOcrSymbol(std::string_view s);
std::vector<std::string> PreparePoints(std::string_view s);

}  // namespace screenai::lm
