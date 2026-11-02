#include "lm_char_model.h"

#include <cmath>

namespace screenai::lm {
namespace {

// Decodes one UTF-8 code point at s[i] (lenient: invalid bytes are single-byte points). Returns its length.
size_t Utf8Len(std::string_view s, size_t i) {
  const uint8_t c = static_cast<uint8_t>(s[i]);
  size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
  if (c >= 0x80 && c < 0xC0) n = 1;
  return std::min(n, s.size() - i);
}

uint32_t Utf8Decode(std::string_view cp) {
  const uint8_t c = static_cast<uint8_t>(cp[0]);
  if (cp.size() == 1) return c;
  uint32_t v = c & (0x7F >> cp.size());
  for (size_t i = 1; i < cp.size(); i++) v = (v << 6) | (static_cast<uint8_t>(cp[i]) & 0x3F);
  return v;
}

std::vector<std::string_view> CodePoints(std::string_view s) {
  std::vector<std::string_view> out;
  for (size_t i = 0; i < s.size();) {
    const size_t n = Utf8Len(s, i);
    out.push_back(s.substr(i, n));
    i += n;
  }
  return out;
}

// UnicodeProps::IsFirstCharHorizWhitespace (tab + Unicode Zs).
bool IsHorizWhitespace(uint32_t c) {
  return c == 0x09 || c == 0x20 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x202F ||
         c == 0x205F || c == 0x3000;
}

// Unicode White_Space (UTF8StringUtils::UTF8WhitespaceDelimiter).
bool IsWhitespace(uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || IsHorizWhitespace(c) || c == 0x2028 || c == 0x2029;
}

std::string ReplaceAll(std::string_view s, std::string_view from, std::string_view to) {
  std::string out;
  for (size_t i = 0; i < s.size();) {
    if (s.substr(i, from.size()) == from) {
      out.append(to);
      i += from.size();
    } else {
      out.push_back(s[i++]);
    }
  }
  return out;
}

}  // namespace

std::string NormalizeOcrSymbol(std::string_view s) {
  // LRM, ZWJ, ZWNJ, PARAGRAPH SEPARATOR -> " "
  std::string p(s);
  for (const char* from : {"\xE2\x80\x8E", "\xE2\x80\x8D", "\xE2\x80\x8C", "\xE2\x80\xA9"}) p = ReplaceAll(p, from, " ");
  if (!s.empty() && CodePoints(s).size() == 1) {
    if (IsHorizWhitespace(Utf8Decode(CodePoints(s)[0]))) return "<space>";
    // The original compares a (signed) char: every non-ASCII single code point is treated as a control char.
    if (static_cast<signed char>(s[0]) <= 31) return "";
  }
  return ReplaceAll(p, " ", "");
}

std::vector<std::string> PreparePoints(std::string_view s) {
  const std::string norm = NormalizeOcrSymbol(s);
  std::vector<std::string> out;
  std::vector<std::string_view> cps = CodePoints(norm);
  size_t i = 0;
  if (!cps.empty() && cps[0] == "\xC2\xA0") {
    out.push_back("<space>");
    i = 1;
  }
  for (; i < cps.size(); i++) {
    // absl::StrSplit(cp, UTF8WhitespaceDelimiter(), SkipEmpty()) must give exactly one piece
    if (IsWhitespace(Utf8Decode(cps[i]))) continue;
    out.emplace_back(cps[i] == " " ? std::string_view("<space>") : cps[i]);
  }
  return out;
}

std::unique_ptr<FstCharLanguageModel> FstCharLanguageModel::Create(
    std::string_view fst_file, std::string_view syms_file, const std::vector<std::pair<int, std::string>>& labels,
    std::string* error) {
  auto lm = std::unique_ptr<FstCharLanguageModel>(new FstCharLanguageModel());
  lm->fst_ = CompactLmFst::Read(fst_file, error);
  if (!lm->fst_) return nullptr;
  for (size_t i = 0; i < syms_file.size();) {
    size_t e = syms_file.find('\n', i);
    if (e == std::string_view::npos) e = syms_file.size();
    std::string_view line = syms_file.substr(i, e - i);
    i = e + 1;
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    const size_t tab = line.find_last_of("\t ");
    if (line.empty() || tab == std::string_view::npos) continue;
    lm->symbols_.emplace(std::string(line.substr(0, tab)), std::atoi(std::string(line.substr(tab + 1)).c_str()));
  }
  // OcrLanguageModel::Init(min_prob = 1e-6) / InitOpticalTokens
  lm->max_cost_ = -std::log(1e-6);
  int max_id = 0;
  bool have_eos = false;
  for (const auto& [id, name] : labels) {
    if (name.size() >= 3 && name.compare(0, 3, "eos") == 0) lm->eos_class_ = id, have_eos = true;
    max_id = std::max(max_id, id);
  }
  if (!have_eos) lm->eos_class_ = ++max_id;
  lm->num_classes_ = max_id + 1;
  // FstCharLanguageModel::InitSub
  lm->unk_ = lm->GetLmId("<unk>");
  lm->unk_symbols_ = {lm->unk_};
  lm->class_symbols_.assign(lm->num_classes_, {});
  for (const auto& [id, name] : labels) {
    if (id == lm->eos_class_) continue;
    if (name == " ") {
      lm->class_symbols_[id].push_back(lm->GetLmId("<space>"));
      continue;
    }
    for (const std::string& piece : PreparePoints(name)) lm->class_symbols_[id].push_back(lm->GetLmId(piece));
  }
  lm->eos_symbol_ = lm->GetLmId("</S>");
  if (lm->eos_symbol_ == lm->unk_) lm->eos_symbol_ = lm->GetLmId("<eos>");
  lm->class_symbols_[lm->eos_class_].push_back(lm->eos_symbol_);
  return lm;
}

int FstCharLanguageModel::GetLmId(std::string_view token) const {
  if (token == " ") return GetLmId("<space>");
  auto it = symbols_.find(std::string(token));
  return it == symbols_.end() ? unk_ : it->second;
}

const std::vector<int>& FstCharLanguageModel::ClassSymbols(int class_id) const {
  if (class_id < 0 || class_id >= static_cast<int>(class_symbols_.size())) return unk_symbols_;
  return class_symbols_[class_id];
}

double FstCharLanguageModel::LookupId(int* state, int symbol) const {
  if (symbol == unk_) {
    *state = fst_->Start();
    return max_cost_;
  }
  float cost = 0;
  int s = *state;
  CompactLmFst::Arc arc;
  while (!fst_->FindLabel(s, symbol, &arc)) {
    CompactLmFst::Arc backoff;
    if (!fst_->FindBackoff(s, &backoff)) {
      *state = fst_->Start();
      return max_cost_;
    }
    s = backoff.nextstate;
    cost += backoff.weight;
  }
  double c = static_cast<double>(cost + arc.weight);
  if (max_cost_ < c) c = max_cost_;
  *state = arc.nextstate;
  return c;
}

void FstCharLanguageModel::GetNLogProbabilities(const std::vector<int>& classes, int state,
                                                std::vector<std::pair<double, int>>* out) const {
  out->clear();
  out->reserve(classes.size());
  for (int c : classes) {
    double cost = 0.0;
    int s = state;
    for (int symbol : ClassSymbols(c)) cost += LookupId(&s, symbol);
    out->emplace_back(cost, s);
  }
}

}  // namespace screenai::lm
