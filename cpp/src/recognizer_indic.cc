#include "recognizer_indic.h"

#include <algorithm>
#include <climits>
#include <map>
#include <regex>
#include <string>
#include <utility>

#include "recognizer_bidi.h"

namespace screenai::rec {
namespace {

// A symbol during joining: its text and the labels (atoms) it is made of, as indices into the input symbols.
struct Piece {
  std::string text;
  std::vector<int> atoms;
};

using Strings = std::vector<std::string>;

std::string Concat(const Strings& v) {
  std::string s;
  for (const std::string& p : v) s += p;
  return s;
}

Strings NonEmpty(const Strings& v) {
  Strings out;
  for (const std::string& s : v)
    if (!s.empty()) out.push_back(s);
  return out;
}

size_t Utf8Len(uint32_t c) { return c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4; }

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// ---- JoinAndReorderStrings of every manipulator ----

// devanagari_symbol_manipulator.cc: the pre-base vowel sign i moves behind the following consonant cluster
// (pieces ending with virama plus the next piece), before a trailing candrabindu/anusvara of that cluster.
std::string JoinDevanagari(const Strings& v) {
  const std::string kI = "ि", kVirama = "्";
  std::string out;
  const size_t n = v.size();
  for (size_t i = 0; i < n; i++) {
    if (v[i] != kI) {
      out += v[i];
      continue;
    }
    size_t j = i + 1;
    for (; j < n; j++) {
      out += v[j];
      if (v[j].empty()) continue;
      if (v[j].size() < 3 || !EndsWith(v[j], kVirama)) break;
    }
    // MaybeInsertPenultimate
    bool inserted = false;
    for (const char* mark : {"ँ", "ं"}) {
      if (EndsWith(out, mark)) {
        out.insert(out.size() - 3, kI);
        inserted = true;
        break;
      }
    }
    if (!inserted) out += kI;
    i = j;
  }
  return out;
}

// gujarati_symbol_manipulator.cc
std::string JoinGujarati(const Strings& v) {
  const std::string kI = "િ";
  std::string out;
  const size_t n = v.size();
  for (size_t i = 0; i < n;) {
    if (i + 1 < n && v[i] == kI) {
      size_t k = i + 1;
      while (k + 1 < n && v[k].empty()) k++;
      out += v[k] + kI;
      i = k + 1;
    } else {
      out += v[i];
      i++;
    }
  }
  static const std::regex reorder_mark("(ઁ|ં)િ");
  return std::regex_replace(out, reorder_mark, "િ$1");
}

// kannada_symbol_manipulator.cc: the arkavottu (reph, recognized as a subscript ra after its consonant) moves in
// front of the consonant as RA + virama; a vowel sign aa/u recognized before it is kept behind the consonant.
std::string JoinKannada(const Strings& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); i++) {
    if (i) s += '\n';
    s += v[i];
  }
  static const std::regex r0("\n+");
  static const std::regex r1("(\n|^)(ಾ|ು)\n(್ರ)(\n|$)");
  static const std::regex r2("(^|\n)(.*)\n(್ರ)(\n|$)");
  static const std::regex r3("(\n|^)್ರ(\n|$)");
  s = std::regex_replace(s, r0, "\n");
  s = std::regex_replace(s, r1, "$1$3\n$2$4");
  s = std::regex_replace(s, r2, "$1$3\n$2$4");
  s = std::regex_replace(s, r3, "$1ರ್$2");
  s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
  return s;
}

// tamil_symbol_manipulator.cc: a left vowel sign (e, ee, ai) swaps with the following piece.
std::string JoinTamil(const Strings& in) {
  Strings v = NonEmpty(in);
  for (size_t i = 0; i < v.size(); i++) {
    const std::string& s = v[i];
    if ((s == "ெ" || s == "ே" || s == "ை") && i + 1 < v.size()) {
      std::swap(v[i], v[i + 1]);
      i++;
    }
  }
  return Concat(v);
}

// telugu_symbol_manipulator.cc: vowel signs of a piece move behind the following virama (subscript) pieces.
bool IsTeluguNonSepMark(uint32_t c) {
  return (c >= 0x0C3E && c <= 0x0C44) || (c >= 0x0C46 && c <= 0x0C4C) || c == 0x0C55 || c == 0x0C56 ||
         c == 0x0C62 || c == 0x0C63 || c == 0x0C00;
}

std::string JoinTelugu(const Strings& in) {
  Strings v = NonEmpty(in);
  const std::string kVirama = "్";
  const size_t n = v.size();
  for (size_t i = 0; i < n; i++) {
    // FullMatch "(.*?)([marks]+)": base = shortest prefix, marks = longest all-mark suffix (non-empty)
    const std::vector<uint32_t> cps = DecodeUtf8(v[i]);
    size_t m = cps.size();
    while (m > 0 && IsTeluguNonSepMark(cps[m - 1])) m--;
    if (m == cps.size()) continue;
    size_t base_bytes = 0;
    for (size_t c = 0; c < m; c++) base_bytes += Utf8Len(cps[c]);
    std::string base = v[i].substr(0, base_bytes), marks = v[i].substr(base_bytes);
    size_t j = i + 1, k = i;
    while (j < n && v[j].size() >= 3 && v[j].compare(0, 3, kVirama) == 0) {
      v[k] = base + v[j];
      base.clear();
      j++, k++;
    }
    v[k] = base + marks;
    i = k;
  }
  return Concat(v);
}

// bengali_symbol_manipulator.cc: a pre-base vowel sign (i, e, ai) moves behind the following consonant cluster
// (pieces ending with hasant [+ZWNJ/ZWJ], the consonant and a ya-phala), before a candrabindu of the cluster.
std::string JoinBengali(const Strings& in) {
  Strings v = NonEmpty(in);
  const size_t n = v.size();
  auto ends_with_hasant = [](const std::string& s) {
    return EndsWith(s, "্") || EndsWith(s, "্‌") || EndsWith(s, "্‍");
  };
  auto is_ya_phala = [](const std::string& s) {
    return s == "্য" || s == "্‌য" || s == "্‍য";
  };
  for (size_t i = 0; i < n; i++) {
    if (!(v[i] == "ি" || v[i] == "ে" || v[i] == "ৈ") || i + 1 >= n) continue;
    const std::string saved = v[i];
    size_t src = i + 1, dst = i;
    while (src < n && ends_with_hasant(v[src])) v[dst++] = v[src++];
    if (src == n) {
      v[dst] = saved;
      break;
    }
    v[dst] = v[src];
    size_t next = src;
    if (src + 1 < n && is_ya_phala(v[src + 1])) {
      v[src] = v[src + 1];
      next = src + 1;
    }
    v[next] = saved;
    std::string& prev = v[next - 1];
    if (EndsWith(prev, "ঁ")) {
      prev.erase(prev.size() - 3);
      v[next] += "ঁ";
    }
    i = next;
  }
  return Concat(v);
}

// ---- SplittingSymbolAtomManipulator::JoinSymbols ----

struct Manipulator {
  std::string (*join)(const Strings&);  // SplittingSymbolAtomManipulator::JoinAndReorderStrings
  std::vector<std::pair<std::string, std::string>> normalization;  // precomposed -> decomposed, key order
  std::string (*mapping)(const std::string&) = nullptr;  // MappingSymbolAtomManipulator::TransformFromAtomString
};

void Normalize(const Manipulator& m, std::string* s) {  // ApplyNormalization(.., to_composed = true)
  for (const auto& [composed, decomposed] : m.normalization) {
    std::string out;
    size_t pos = 0;
    for (size_t f; (f = s->find(decomposed, pos)) != std::string::npos; pos = f + decomposed.size())
      out += s->substr(pos, f - pos) + composed;
    *s = out + s->substr(pos);
  }
}

// aksara::util::AlignTextToPieces: the text must be a permutation of the pieces' code points; alignment[k] is
// the rank of piece k by the first text position its code points take.
bool AlignTextToPieces(const std::string& text, const std::vector<Piece>& pieces, size_t first, size_t last,
                       std::vector<int>* alignment) {
  std::map<uint32_t, std::vector<int>> positions;  // code point -> remaining positions (ascending)
  const std::vector<uint32_t> cps = DecodeUtf8(text);
  for (size_t i = 0; i < cps.size(); i++) positions[cps[i]].push_back(static_cast<int>(i));
  std::vector<std::pair<int, int>> order;  // (piece, min position)
  for (size_t k = first; k <= last; k++) {
    int min_pos = static_cast<int>(text.size()) + 1;
    for (uint32_t c : DecodeUtf8(pieces[k].text)) {
      auto it = positions.find(c);
      if (it == positions.end()) return false;
      min_pos = std::min(min_pos, it->second.front());
      it->second.erase(it->second.begin());
      if (it->second.empty()) positions.erase(it);
    }
    order.emplace_back(static_cast<int>(k - first), min_pos);
  }
  if (!positions.empty()) return false;
  std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
  alignment->assign(order.size(), 0);
  for (size_t r = 0; r < order.size(); r++) (*alignment)[order[r].first] = static_cast<int>(r);
  return true;
}

// MappingSymbolAtomManipulator::JoinSymbols: the line text is transformed code point for code point and handed back
// to the symbols in order (MonotonicUpdate).
bool MapSymbols(const Manipulator& m, std::vector<Piece>* symbols) {
  std::string concat;
  for (const Piece& p : *symbols) concat += p.text;
  const std::string output = m.mapping(concat);
  const std::vector<uint32_t> cps = DecodeUtf8(output);
  const std::vector<uint32_t> in = DecodeUtf8(concat);
  if (cps.size() != in.size()) return false;  // CHECK in the original
  size_t pos = 0, byte = 0;
  for (Piece& p : *symbols) {
    if (p.text.empty()) continue;
    const size_t n = DecodeUtf8(p.text).size();
    size_t len = 0;
    for (size_t k = 0; k < n; k++, pos++) len += Utf8Len(cps[pos]);
    p.text = output.substr(byte, len);
    byte += len;
  }
  return true;
}

bool JoinSymbols(const Manipulator& m, std::vector<Piece>* symbols) {
  if (m.mapping) return MapSymbols(m, symbols);
  Strings inputs;
  for (const Piece& p : *symbols) inputs.push_back(p.text);
  const std::string output = m.join(inputs);
  if (output == Concat(inputs)) {
    std::string normalized = output;
    Normalize(m, &normalized);
    if (normalized == output) return true;
  }
  std::vector<Piece> joined;
  size_t offset = 0, pos = 0;
  const size_t n = symbols->size();
  while (offset < output.size()) {
    if (pos >= n) return false;
    size_t j = pos, len = 0;
    std::vector<int> alignment;
    for (;; j++) {
      if (j >= n) return false;
      len += inputs[j].size();
      if (offset + len > output.size()) return false;  // CHECK in the original
      if (AlignTextToPieces(output.substr(offset, len), *symbols, pos, j, &alignment)) {
        // the group must not end inside a decomposed sequence of the normalization map
        const std::string rest = output.substr(offset);
        bool straddles = false;
        for (const auto& kv : m.normalization) {
          const size_t f = rest.find(kv.second);
          if (f != std::string::npos && f < len && f + kv.second.size() > len) {
            straddles = true;
            break;
          }
        }
        if (!straddles) break;
      }
    }
    if (j == pos) {
      Piece p = (*symbols)[pos];
      if (!m.normalization.empty()) Normalize(m, &p.text);
      joined.push_back(std::move(p));
    } else {
      std::vector<std::vector<int>> slots(j - pos + 1);
      for (size_t k = pos; k <= j; k++) slots[alignment[k - pos]] = (*symbols)[k].atoms;
      Piece p;
      p.text = output.substr(offset, len);
      if (!m.normalization.empty()) Normalize(m, &p.text);
      for (const auto& s : slots) p.atoms.insert(p.atoms.end(), s.begin(), s.end());
      joined.push_back(std::move(p));
    }
    offset += len;
    pos = j + 1;
  }
  *symbols = std::move(joined);
  return true;
}

const std::vector<Manipulator>& Manipulators() {
  static const std::vector<Manipulator> kAll = {
      {JoinBengali, {{"ো", "ো"}, {"ৌ", "ৌ"}}},
      {JoinDevanagari, {}},
      {JoinGujarati, {}},
      {JoinKannada, {}},
      {nullptr, {}, MirrorRtlText},  // MirroringSymbolManipulator
      {JoinTamil,
       {{"ஔ", "ஔ"}, {"ொ", "ொ"}, {"ோ", "ோ"}, {"ௌ", "ௌ"}}},
      {JoinTelugu, {}},
  };
  return kAll;
}

}  // namespace

void JoinSymbolsLogical(std::vector<RecSymbol>* symbols) {
  std::vector<Piece> pieces;
  for (size_t i = 0; i < symbols->size(); i++) pieces.push_back({(*symbols)[i].text, {static_cast<int>(i)}});
  const auto& manipulators = Manipulators();
  for (auto it = manipulators.rbegin(); it != manipulators.rend(); ++it)
    if (!JoinSymbols(*it, &pieces)) return;  // original: back to one symbol per atom
  bool changed = pieces.size() != symbols->size();
  for (size_t i = 0; !changed && i < pieces.size(); i++) changed = pieces[i].text != (*symbols)[i].text;
  if (!changed) return;
  std::vector<RecSymbol> out;
  for (const Piece& p : pieces) {
    RecSymbol s;
    s.text = p.text;
    // box: union of the atom boxes; confidence: product (double) of the label confidences in label order
    std::vector<int> labels = p.atoms;
    std::sort(labels.begin(), labels.end());
    int x0 = INT_MAX, y0 = INT_MAX, x1 = INT_MIN, y1 = INT_MIN;
    double conf = 1.0;
    for (int a : labels) {
      const RecSymbol& at = (*symbols)[a];
      x0 = std::min(x0, at.box.left), y0 = std::min(y0, at.box.top);
      x1 = std::max(x1, at.box.left + at.box.width), y1 = std::max(y1, at.box.top + at.box.height);
      conf *= static_cast<double>(at.confidence);
    }
    if (labels.size() == 1) {
      s.box = (*symbols)[labels[0]].box;
    } else if (!labels.empty()) {
      s.box = (*symbols)[labels[0]].box;
      s.box.left = x0, s.box.top = y0, s.box.width = x1 - x0, s.box.height = y1 - y0;
    }
    s.confidence = static_cast<float>(conf);
    if (labels.size() == 1) {
      const std::vector<uint32_t> now = DecodeUtf8(s.text), atom = DecodeUtf8((*symbols)[labels[0]].text);
      s.mirrored = s.text != (*symbols)[labels[0]].text && now.size() == 1 && atom.size() == 1 &&
                   BidiMirrorGlyph(now[0]) == atom[0];
    }
    out.push_back(std::move(s));
  }
  *symbols = std::move(out);
}

}  // namespace screenai::rec
