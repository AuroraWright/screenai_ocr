#include "recognizer_bidi.h"

#include <algorithm>

#include "recognizer_words.h"

namespace screenai::rec {
namespace {

// ICU DirProp order (ubidiimp.h); ENL/ENR are EN resolved by W7 / not subject to W7.
enum BidiClass : uint8_t {
  kL, kR, kEN, kES, kET, kAN, kCS, kB, kS, kWS, kON, kLRE, kLRO, kAL, kRLE, kRLO, kPDF, kNSM, kBN,
  kFSI, kLRI, kRLI, kPDI, kENL, kENR
};

#include "recognizer_bidi.inc"

template <size_t N>
const uint32_t* FindRow(const uint32_t (*table)[N], size_t rows, uint32_t cp) {
  const uint32_t(*it)[N] =
      std::lower_bound(table, table + rows, cp, [](const uint32_t(&r)[N], uint32_t c) { return r[0] < c; });
  return it != table + rows && (*it)[0] == cp ? *it : nullptr;
}

// The recognized text has no explicit embeddings or isolates: their codes are treated as BN / ON.
BidiClass ClassOf(uint32_t cp) {
  const size_t n = sizeof(kBidiClass) / sizeof(kBidiClass[0]);
  const auto* it = std::lower_bound(kBidiClass, kBidiClass + n, cp,
                                    [](const uint32_t(&r)[3], uint32_t c) { return r[1] < c; });
  if (it == kBidiClass + n || (*it)[0] > cp) return kL;
  const BidiClass c = static_cast<BidiClass>((*it)[2]);
  if (c == kLRE || c == kLRO || c == kRLE || c == kRLO || c == kPDF) return kBN;
  if (c == kFSI || c == kLRI || c == kRLI || c == kPDI) return kON;
  return c;
}


uint32_t Flag(int c) { return 1u << c; }
constexpr uint32_t kMaskLtr = (1u << kL) | (1u << kEN) | (1u << kENL) | (1u << kENR) | (1u << kAN) | (1u << kLRE) |
                              (1u << kLRO) | (1u << kLRI);
constexpr uint32_t kMaskRtl = (1u << kR) | (1u << kAL) | (1u << kRLE) | (1u << kRLO) | (1u << kRLI);
constexpr uint32_t kMaskBnExplicit =
    (1u << kBN) | (1u << kLRE) | (1u << kLRO) | (1u << kRLE) | (1u << kRLO) | (1u << kPDF);
constexpr uint32_t kMaskBS = (1u << kB) | (1u << kS);
constexpr uint32_t kMaskIso = (1u << kLRI) | (1u << kRLI) | (1u << kFSI) | (1u << kPDI);
constexpr uint32_t kMaskWs = kMaskBS | (1u << kWS) | kMaskBnExplicit | kMaskIso;
constexpr uint32_t kMaskPossibleN = (1u << kON) | (1u << kCS) | (1u << kES) | (1u << kET) | kMaskWs;

void AppendUtf8(uint32_t cp, std::string* s) {
  if (cp < 0x80) {
    s->push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s->push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    s->push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s->push_back(static_cast<char>(0xF0 | (cp >> 18)));
    s->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    s->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// ---- ICU ubidi.cpp: implicit level resolution of one paragraph ----

enum class Mode { kDefault, kInverseLikeDirect };  // UBIDI_REORDER_DEFAULT / UBIDI_REORDER_INVERSE_LIKE_DIRECT
enum class Direction { kLtr, kRtl, kMixed };

// Paired brackets (N0), BracketData with a single isolating run.
class Brackets {
 public:
  Brackets(const std::vector<uint32_t>& text, std::vector<uint8_t>* props, int level)
      : text_(text), props_(*props), level_(level) {
    last_strong_ = last_base_ = context_dir_ = static_cast<uint8_t>(level & 1);
  }

  void ProcessChar(int pos) {  // bracketProcessChar
    const uint8_t prop = props_[pos];
    uint8_t new_prop;
    if (prop == kON) {
      uint32_t c = text_[pos];
      for (int idx = static_cast<int>(openings_.size()) - 1; idx >= 0; idx--) {
        if (openings_[idx].match != static_cast<int32_t>(c)) continue;
        new_prop = ProcessClosing(idx, pos);
        if (new_prop == kON) {  // N0d
          c = 0;
          break;
        }
        last_base_ = kON;
        context_dir_ = new_prop;
        context_pos_ = pos;
        return;
      }
      if (c) {
        const uint32_t* r = FindRow(kBidiBracket, sizeof(kBidiBracket) / sizeof(kBidiBracket[0]), c);
        if (r && r[2] == 1) {
          if (r[1] == 0x232A) AddOpening(0x3009, pos);
          else if (r[1] == 0x3009) AddOpening(0x232A, pos);
          AddOpening(static_cast<int32_t>(r[1]), pos);
        }
      }
    }
    if (prop <= kR || prop == kAL) {
      new_prop = prop == kL ? kL : kR;
      last_base_ = last_strong_ = prop;
      context_dir_ = new_prop;
      context_pos_ = pos;
    } else if (prop == kEN) {
      last_base_ = kEN;
      if (last_strong_ == kL) {
        new_prop = kL;  // W7
        props_[pos] = kENL;
        context_dir_ = kL;
      } else {
        new_prop = kR;  // N0
        props_[pos] = last_strong_ == kAL ? kAN : kENR;  // W2
        context_dir_ = kR;
      }
      context_pos_ = pos;
    } else if (prop == kAN) {
      new_prop = kR;
      last_base_ = kAN;
      context_dir_ = kR;
      context_pos_ = pos;
    } else if (prop == kNSM) {
      new_prop = last_base_;
      if (new_prop == kON) props_[pos] = new_prop;
    } else {
      new_prop = prop;
      last_base_ = prop;
    }
    if (new_prop <= kR || new_prop == kAL) {
      const uint8_t flag = new_prop == kL ? kFoundL : kFoundR;
      for (Opening& o : openings_)
        if (pos > o.position) o.flags |= flag;
    }
  }

 private:
  static constexpr uint8_t kFoundL = 1, kFoundR = 2;
  struct Opening {
    int position;
    int32_t match;  // closing bracket; -position of the closing one when unstable; 0 when neutralized
    int context_pos;
    uint8_t flags;
    uint8_t context_dir;
  };

  void AddOpening(int32_t match, int pos) { openings_.push_back({pos, match, context_pos_, 0, context_dir_}); }

  void FixN0c(int opening_index, int new_prop_pos, uint8_t new_prop) {
    for (int k = opening_index + 1; k < static_cast<int>(openings_.size()); k++) {
      Opening& q = openings_[k];
      if (q.match >= 0) continue;
      if (new_prop_pos < q.context_pos) break;
      if (new_prop_pos >= q.position) continue;
      if (new_prop == q.context_dir) break;
      const int opening_pos = q.position, closing_pos = -q.match;
      props_[opening_pos] = new_prop;
      props_[closing_pos] = new_prop;
      q.match = 0;
      FixN0c(k, opening_pos, new_prop);
      FixN0c(k, closing_pos, new_prop);
    }
  }

  uint8_t ProcessClosing(int open_idx, int pos) {  // bracketProcessClosing
    Opening& o = openings_[open_idx];
    const uint8_t direction = static_cast<uint8_t>(level_ & 1);
    bool stable = true;
    uint8_t new_prop;
    if ((direction == 0 && (o.flags & kFoundL)) || (direction == 1 && (o.flags & kFoundR))) {  // N0b
      new_prop = direction;
    } else if (o.flags & (kFoundL | kFoundR)) {  // N0c
      stable = open_idx == 0;
      new_prop = direction != o.context_dir ? o.context_dir : direction;
    } else {  // N0d
      openings_.resize(open_idx);
      return kON;
    }
    const int opening_pos = o.position;
    props_[opening_pos] = new_prop;
    props_[pos] = new_prop;
    FixN0c(open_idx, opening_pos, new_prop);
    if (stable) {
      openings_.resize(open_idx);
      while (!openings_.empty() && openings_.back().position == opening_pos) openings_.pop_back();
    } else {
      openings_[open_idx].match = -pos;
      for (int k = open_idx - 1; k >= 0 && openings_[k].position == opening_pos; k--) openings_[k].match = 0;
      for (int k = open_idx + 1; k < static_cast<int>(openings_.size()); k++) {
        if (openings_[k].position >= pos) break;
        if (openings_[k].match > 0) openings_[k].match = 0;
      }
    }
    return new_prop;
  }

  const std::vector<uint32_t>& text_;
  std::vector<uint8_t>& props_;
  const int level_;
  std::vector<Opening> openings_;
  uint8_t last_strong_, last_base_, context_dir_;
  int context_pos_ = 0;
};

#define s(action, state) static_cast<uint8_t>((state) + ((action) << 5))
enum { kDirPropL, kDirPropR, kDirPropEN, kDirPropAN, kDirPropON, kDirPropS, kDirPropB };
constexpr uint8_t kGroupProp[] = {0, 1, 2, 7, 8, 3, 9, 6, 5, 4, 4, 10, 10, 12, 10, 10, 10, 11, 10, 4, 4, 4, 4, 13, 14};
constexpr uint8_t kImpTabProps[][16] = {
    {1, 2, 4, 5, 7, 15, 17, 7, 9, 7, 0, 7, 3, 18, 21, kDirPropON},
    {1, s(1, 2), s(1, 4), s(1, 5), s(1, 7), s(1, 15), s(1, 17), s(1, 7), s(1, 9), s(1, 7), 1, 1, s(1, 3), s(1, 18),
     s(1, 21), kDirPropL},
    {s(1, 1), 2, s(1, 4), s(1, 5), s(1, 7), s(1, 15), s(1, 17), s(1, 7), s(1, 9), s(1, 7), 2, 2, s(1, 3), s(1, 18),
     s(1, 21), kDirPropR},
    {s(1, 1), s(1, 2), s(1, 6), s(1, 6), s(1, 8), s(1, 16), s(1, 17), s(1, 8), s(1, 8), s(1, 8), 3, 3, 3, s(1, 18),
     s(1, 21), kDirPropR},
    {s(1, 1), s(1, 2), 4, s(1, 5), s(1, 7), s(1, 15), s(1, 17), s(2, 10), 11, s(2, 10), 4, 4, s(1, 3), 18, 21,
     kDirPropEN},
    {s(1, 1), s(1, 2), s(1, 4), 5, s(1, 7), s(1, 15), s(1, 17), s(1, 7), s(1, 9), s(2, 12), 5, 5, s(1, 3), s(1, 18),
     s(1, 21), kDirPropAN},
    {s(1, 1), s(1, 2), 6, 6, s(1, 8), s(1, 16), s(1, 17), s(1, 8), s(1, 8), s(2, 13), 6, 6, s(1, 3), 18, 21,
     kDirPropAN},
    {s(1, 1), s(1, 2), s(1, 4), s(1, 5), 7, s(1, 15), s(1, 17), 7, s(2, 14), 7, 7, 7, s(1, 3), s(1, 18), s(1, 21),
     kDirPropON},
    {s(1, 1), s(1, 2), s(1, 6), s(1, 6), 8, s(1, 16), s(1, 17), 8, 8, 8, 8, 8, s(1, 3), s(1, 18), s(1, 21),
     kDirPropON},
    {s(1, 1), s(1, 2), 4, s(1, 5), 7, s(1, 15), s(1, 17), 7, 9, 7, 9, 9, s(1, 3), 18, 21, kDirPropON},
    {s(3, 1), s(3, 2), 4, s(3, 5), s(4, 7), s(3, 15), s(3, 17), s(4, 7), s(4, 14), s(4, 7), 10, s(4, 7), s(3, 3), 18,
     21, kDirPropEN},
    {s(1, 1), s(1, 2), 4, s(1, 5), s(1, 7), s(1, 15), s(1, 17), s(1, 7), 11, s(1, 7), 11, 11, s(1, 3), 18, 21,
     kDirPropEN},
    {s(3, 1), s(3, 2), s(3, 4), 5, s(4, 7), s(3, 15), s(3, 17), s(4, 7), s(4, 14), s(4, 7), 12, s(4, 7), s(3, 3),
     s(3, 18), s(3, 21), kDirPropAN},
    {s(3, 1), s(3, 2), 6, 6, s(4, 8), s(3, 16), s(3, 17), s(4, 8), s(4, 8), s(4, 8), 13, s(4, 8), s(3, 3), 18, 21,
     kDirPropAN},
    {s(1, 1), s(1, 2), s(4, 4), s(1, 5), 7, s(1, 15), s(1, 17), 7, 14, 7, 14, 14, s(1, 3), s(4, 18), s(4, 21),
     kDirPropON},
    {s(1, 1), s(1, 2), s(1, 4), s(1, 5), s(1, 7), 15, s(1, 17), s(1, 7), s(1, 9), s(1, 7), 15, s(1, 7), s(1, 3),
     s(1, 18), s(1, 21), kDirPropS},
    {s(1, 1), s(1, 2), s(1, 6), s(1, 6), s(1, 8), 16, s(1, 17), s(1, 8), s(1, 8), s(1, 8), 16, s(1, 8), s(1, 3),
     s(1, 18), s(1, 21), kDirPropS},
    {s(1, 1), s(1, 2), s(1, 4), s(1, 5), s(1, 7), s(1, 15), 17, s(1, 7), s(1, 9), s(1, 7), 17, s(1, 7), s(1, 3),
     s(1, 18), s(1, 21), kDirPropB},
    {s(1, 1), s(1, 2), 18, s(1, 5), s(1, 7), s(1, 15), s(1, 17), s(2, 19), 20, s(2, 19), 18, 18, s(1, 3), 18, 21,
     kDirPropL},
    {s(3, 1), s(3, 2), 18, s(3, 5), s(4, 7), s(3, 15), s(3, 17), s(4, 7), s(4, 14), s(4, 7), 19, s(4, 7), s(3, 3), 18,
     21, kDirPropL},
    {s(1, 1), s(1, 2), 18, s(1, 5), s(1, 7), s(1, 15), s(1, 17), s(1, 7), 20, s(1, 7), 20, 20, s(1, 3), 18, 21,
     kDirPropL},
    {s(1, 1), s(1, 2), 21, s(1, 5), s(1, 7), s(1, 15), s(1, 17), s(2, 22), 23, s(2, 22), 21, 21, s(1, 3), 18, 21,
     kDirPropAN},
    {s(3, 1), s(3, 2), 21, s(3, 5), s(4, 7), s(3, 15), s(3, 17), s(4, 7), s(4, 14), s(4, 7), 22, s(4, 7), s(3, 3), 18,
     21, kDirPropAN},
    {s(1, 1), s(1, 2), 21, s(1, 5), s(1, 7), s(1, 15), s(1, 17), s(1, 7), 23, s(1, 7), 23, 23, s(1, 3), 18, 21,
     kDirPropAN},
};
#undef s
#define s(action, state) static_cast<uint8_t>((state) + ((action) << 4))
using ImpTab = uint8_t[8];
constexpr ImpTab kImpTabLDefault[] = {
    {0, 1, 0, 2, 0, 0, 0, 0},           {0, 1, 3, 3, s(1, 4), s(1, 4), 0, 1}, {0, 1, 0, 2, s(1, 5), s(1, 5), 0, 2},
    {0, 1, 3, 3, s(1, 4), s(1, 4), 0, 2}, {0, s(2, 1), s(3, 3), s(3, 3), 4, 4, 0, 0},
    {0, s(2, 1), 0, s(3, 2), 5, 5, 0, 0},
};
constexpr ImpTab kImpTabRDefault[] = {
    {1, 0, 2, 2, 0, 0, 0, 0}, {1, 0, 1, 3, s(1, 4), s(1, 4), 0, 1}, {1, 0, 2, 2, 0, 0, 0, 1},
    {1, 0, 1, 3, 5, 5, 0, 1}, {s(2, 1), 0, s(2, 1), 3, 4, 4, 0, 0}, {1, 0, 1, 3, 5, 5, 0, 0},
};
constexpr ImpTab kImpTabRInverseLikeDirect[] = {
    {1, 0, 2, 2, 0, 0, 0, 0},
    {1, 0, 1, 2, s(1, 3), s(1, 3), 0, 1},
    {1, 0, 2, 2, 0, 0, 0, 1},
    {s(2, 1), s(3, 0), 6, 4, 3, 3, s(3, 0), 0},
    {s(2, 1), s(3, 0), 6, 4, 5, 5, s(3, 0), 3},
    {s(2, 1), s(3, 0), 6, 4, 5, 5, s(3, 0), 2},
    {s(2, 1), s(3, 0), 6, 4, 3, 3, s(3, 0), 1},
};
#undef s
constexpr uint8_t kImpAct0[] = {0, 1, 2, 3, 4};
constexpr uint8_t kImpAct1[] = {0, 1, 13, 14};

struct LevState {
  const ImpTab* tab;
  const uint8_t* act;
  int start_on = -1, state = 0, run_level = 0;
};

void ProcessPropertySeq(LevState* ls, uint8_t prop, int start, int limit, std::vector<uint8_t>* levels_out) {
  std::vector<uint8_t>& levels = *levels_out;
  const int start0 = start;
  const uint8_t old_state = static_cast<uint8_t>(ls->state);
  const uint8_t cell = ls->tab[old_state][prop];
  ls->state = cell & 0x0F;
  const uint8_t action = ls->act[cell >> 4];
  const uint8_t add_level = ls->tab[ls->state][7];
  switch (action) {
    case 0: break;
    case 1: ls->start_on = start0; break;  // init ON seq
    case 2: start = ls->start_on; break;   // prepend ON seq to current seq
    case 3:                                // EN/AN after R+ON
      for (int k = ls->start_on; k < start0; k++) levels[k] = static_cast<uint8_t>(ls->run_level + 1);
      break;
    case 4:  // EN/AN before R for NUMBERS_SPECIAL
      for (int k = ls->start_on; k < start0; k++) levels[k] = static_cast<uint8_t>(ls->run_level + 2);
      break;
    case 13: {  // L after L+ON+EN/AN/ON
      const int level = ls->run_level;
      for (int k = start0 - 1; k >= ls->start_on; k--) {
        if (levels[k] == level + 3) {
          while (levels[k] == level + 3) levels[k--] -= 2;
          while (levels[k] == level) k--;
        }
        if (levels[k] == level + 2) {
          levels[k] = static_cast<uint8_t>(level);
          continue;
        }
        levels[k] = static_cast<uint8_t>(level + 1);
      }
      break;
    }
    case 14: {  // R after L+ON+EN/AN/ON
      const int level = ls->run_level + 1;
      for (int k = start0 - 1; k >= ls->start_on; k--)
        if (levels[k] > level) levels[k] -= 2;
      break;
    }
  }
  if (add_level || start < start0) {
    const uint8_t level = static_cast<uint8_t>(ls->run_level + add_level);
    for (int k = std::max(start, 0); k < limit; k++) levels[k] = level;
  }
}

// ubidi_setPara for one paragraph at `para` level (0/1) without explicit embeddings; `levels` gets the resolved
// level of every code point (all `para` unless the direction is mixed).
Direction ResolveLevels(const std::vector<uint32_t>& text, int para, Mode mode, std::vector<uint8_t>* levels_out) {
  const int n = static_cast<int>(text.size());
  std::vector<uint8_t> props(n);
  uint32_t flags = 0;
  int last_arabic_pos = -1;
  for (int i = 0; i < n; i++) {  // getDirProps
    props[i] = ClassOf(text[i]);
    flags |= Flag(props[i]);
    if (props[i] == kAL) last_arabic_pos = i;
  }
  flags |= Flag(para & 1 ? kR : kL);
  std::vector<uint8_t>& levels = *levels_out;
  levels.assign(n, static_cast<uint8_t>(para));
  // directionFromFlags
  if (!((flags & kMaskRtl) || ((flags & Flag(kAN)) && (flags & kMaskPossibleN)))) return Direction::kLtr;
  if (!(flags & kMaskLtr)) return Direction::kRtl;
  // resolveExplicitLevels: no embeddings; bracket matching except for inverse BiDi
  if (mode == Mode::kDefault) {
    Brackets brackets(text, &props, para);
    for (int i = 0; i < n; i++)
      if (props[i] != kBN && props[i] != kB) brackets.ProcessChar(i);
  }
  // resolveImplicitLevels(0, n, sor, eor)
  LevState ls;
  ls.run_level = para;
  if (para & 1) {
    ls.tab = mode == Mode::kDefault ? kImpTabRDefault : kImpTabRInverseLikeDirect;
    ls.act = mode == Mode::kDefault ? kImpAct0 : kImpAct1;
  } else {
    ls.tab = kImpTabLDefault;
    ls.act = kImpAct0;
  }
  const uint8_t sor = static_cast<uint8_t>(para & 1), eor = sor;
  const bool inverse_rtl = 0 < last_arabic_pos && (para & 1) && mode == Mode::kInverseLikeDirect;
  int start1 = 0, start2 = 0;
  int state_imp = n > 0 && props[0] == kNSM ? 1 + sor : 0;
  ProcessPropertySeq(&ls, sor, 0, 0, &levels);
  uint8_t next_strong_prop = kR;
  int next_strong_pos = -1;
  for (int i = 0; i <= n; i++) {
    uint8_t gprop;
    if (i >= n) {
      gprop = eor;
    } else {
      uint8_t prop = props[i];
      if (inverse_rtl) {
        if (prop == kAL) {
          prop = kR;  // AL before EN does not make it AN
        } else if (prop == kEN) {
          if (next_strong_pos <= i) {
            next_strong_prop = kR;
            next_strong_pos = n;
            for (int j = i + 1; j < n; j++) {
              if (props[j] == kL || props[j] == kR || props[j] == kAL) {
                next_strong_prop = props[j];
                next_strong_pos = j;
                break;
              }
            }
          }
          if (next_strong_prop == kAL) prop = kAN;
        }
      }
      gprop = kGroupProp[prop];
    }
    const int old_state_imp = state_imp;
    const uint8_t cell = kImpTabProps[old_state_imp][gprop];
    state_imp = cell & 0x1F;
    int action = cell >> 5;
    if (i == n && action == 0) action = 1;
    if (action) {
      const uint8_t res = kImpTabProps[old_state_imp][15];
      switch (action) {
        case 1:
          ProcessPropertySeq(&ls, res, start1, i, &levels);
          start1 = i;
          break;
        case 2: start2 = i; break;
        case 3:
          ProcessPropertySeq(&ls, res, start1, start2, &levels);
          ProcessPropertySeq(&ls, kDirPropON, start2, i, &levels);
          start1 = i;
          break;
        case 4:
          ProcessPropertySeq(&ls, res, start1, start2, &levels);
          start1 = start2;
          start2 = i;
          break;
      }
    }
  }
  ProcessPropertySeq(&ls, eor, n, n, &levels);
  // adjustWSLevels (L1)
  if (flags & kMaskWs) {
    int i = n;
    while (i > 0) {
      while (i > 0 && (Flag(props[--i]) & kMaskWs)) levels[i] = static_cast<uint8_t>(para);
      while (i > 0) {
        const uint32_t flag = Flag(props[--i]);
        if (flag & kMaskBnExplicit) {
          levels[i] = levels[i + 1];
        } else if (flag & kMaskBS) {
          levels[i] = static_cast<uint8_t>(para);
          break;
        }
      }
    }
  }
  return Direction::kMixed;
}

// Visual order (visual[j] = logical index) by L2 (ubidi_getVisualMap).
std::vector<int> VisualOrder(const std::vector<uint8_t>& level) {
  const int n = static_cast<int>(level.size());
  std::vector<int> order(n);
  for (int i = 0; i < n; i++) order[i] = i;
  int max_level = 0, min_odd = 1000;
  for (int l : level) {
    max_level = std::max(max_level, l);
    if (l & 1) min_odd = std::min(min_odd, l);
  }
  for (int lv = max_level; lv >= min_odd && lv >= 1; lv--) {
    for (int i = 0; i < n;) {
      if (level[order[i]] < lv) {
        i++;
        continue;
      }
      int j = i;
      while (j < n && level[order[j]] >= lv) j++;
      std::reverse(order.begin() + i, order.begin() + j);
      i = j;
    }
  }
  return order;
}

// ubidi_writeReordered: odd-level runs are written reversed; with `keep_base_combining` combining marks stay after
// their base character, with `mirroring` the base character of a reversed run is mirrored.
std::vector<uint32_t> WriteReordered(const std::vector<uint32_t>& cps, const std::vector<uint8_t>& level,
                                     bool keep_base_combining, bool mirroring) {
  const int n = static_cast<int>(cps.size());
  std::vector<uint32_t> out;
  out.reserve(n);
  struct Run {
    int start, end;
    uint8_t level;
  };
  std::vector<Run> runs;
  for (int i = 0; i < n;) {
    int j = i;
    while (j < n && level[j] == level[i]) j++;
    runs.push_back({i, j, level[i]});
    i = j;
  }
  std::vector<uint8_t> run_level(runs.size());
  for (size_t r = 0; r < runs.size(); r++) run_level[r] = runs[r].level;
  for (int r : VisualOrder(run_level)) {
    const Run& run = runs[r];
    if (!(run.level & 1)) {
      out.insert(out.end(), cps.begin() + run.start, cps.begin() + run.end);
      continue;
    }
    for (int end = run.end; end > run.start;) {
      int k = end - 1;
      if (keep_base_combining)  // IS_COMBINING: Mn/Me (bidi NSM)
        while (k > run.start && ClassOf(cps[k]) == kNSM) k--;
      out.push_back(mirroring ? BidiMirrorGlyph(cps[k]) : cps[k]);
      out.insert(out.end(), cps.begin() + k + 1, cps.begin() + end);
      end = k;
    }
  }
  return out;
}

// i18n_bidi::BiDiConverter::ConvertUtf8Text: `map` gets ubidi_getLogicalMap (output index of every input code
// point; identity for UBIDI_LTR text, which is returned unchanged).
std::string Convert(const std::string& text, int para, Mode mode, bool keep_base_combining, bool mirroring,
                    std::vector<int>* map) {
  const std::vector<uint32_t> cps = DecodeUtf8(text);
  const int n = static_cast<int>(cps.size());
  std::vector<uint8_t> level;
  const Direction dir = ResolveLevels(cps, para, mode, &level);
  if (map) {
    map->resize(n);
    for (int i = 0; i < n; i++) (*map)[i] = i;
  }
  if (dir == Direction::kLtr) return text;
  if (map) {
    const std::vector<int> visual = VisualOrder(level);
    for (int j = 0; j < n; j++) (*map)[visual[j]] = j;
  }
  std::string out;
  for (uint32_t cp : WriteReordered(cps, level, keep_base_combining, mirroring)) AppendUtf8(cp, &out);
  return out;
}

}  // namespace

uint32_t BidiMirrorGlyph(uint32_t cp) {
  const uint32_t* r = FindRow(kBidiMirror, sizeof(kBidiMirror) / sizeof(kBidiMirror[0]), cp);
  return r ? r[1] : cp;
}

void SymbolsVisualToLogical(const std::vector<std::string>& symbols, bool base_dir_rtl, std::string* text,
                            std::vector<int>* offsets) {
  std::string display;
  for (const std::string& s : symbols) {  // InternalReorder: LogicalToDisplay
    const int para = base_dir_rtl && HasAnyRtl(s) ? 1 : 0;
    display += Convert(s, para, Mode::kDefault, true, false, nullptr);
  }
  std::vector<int> map;
  const int para = base_dir_rtl && HasAnyRtl(display) ? 1 : 0;
  *text = Convert(display, para, Mode::kInverseLikeDirect, true, false, &map);  // DisplayToLogical
  // byte offset of every code point of the logical text
  std::vector<int> cp_offset;
  for (size_t i = 0; i < text->size(); i++)
    if ((static_cast<unsigned char>((*text)[i]) & 0xC0) != 0x80) cp_offset.push_back(static_cast<int>(i));
  offsets->assign(symbols.size(), -1);
  size_t cp = 0;
  for (size_t k = 0; k < symbols.size(); k++) {
    const std::string& s = symbols[k];
    if (s.empty()) continue;
    int first = static_cast<int>(text->size());
    for (size_t m = DecodeUtf8(s).size(); m > 0; m--, cp++)
      if (cp < map.size()) first = std::min(first, map[cp]);
    if (first >= static_cast<int>(cp_offset.size())) continue;
    const int off = cp_offset[first];
    if (text->compare(off, s.size(), s) == 0) (*offsets)[k] = off;
  }
}

std::string MirrorRtlText(const std::string& text) {
  if (!HasAnyRtl(text)) return text;
  std::vector<int> map;
  const std::string converted = Convert(text, 1, Mode::kInverseLikeDirect, false, true, &map);
  const std::vector<uint32_t> cps = DecodeUtf8(converted);
  std::string out;
  for (int m : map) AppendUtf8(cps[m], &out);
  return out;
}

}  // namespace screenai::rec
