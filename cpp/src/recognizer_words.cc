#include "recognizer_words.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <set>

namespace screenai::rec {
namespace {

#include "recognizer_unicode.inc"

template <size_t N>
bool InTable(const uint32_t (&table)[N][2], uint32_t cp) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < table[mid][0]) {
      hi = mid;
    } else if (cp > table[mid][1]) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

// Float angle normalization loops of box_util (into (-180, 180]).
float NormalizeAngleF(float a) {
  while (a <= -180.0f) a = a + 360.0f;
  while (a > 180.0f) a = a + -360.0f;
  return a;
}

// FCVTZS-style saturating conversion after the explicit +-2147500000 clamps of box_util.
int SaturatingInt(float v) {
  if (v >= 2147500000.0f) return INT_MAX;
  if (v <= -2147500000.0f) return INT_MIN;
  return static_cast<int>(v);
}

constexpr float kDegToRad = 0.0174532924f;

RotatedBox UnionStart(const RotatedBox& b) {
  RotatedBox r;
  r.left = b.left, r.top = b.top, r.width = 1, r.height = 1, r.angle = 0;
  return r;
}

}  // namespace

void SinCos(float x, float* s, float* c) {
#if defined(__APPLE__)
  const __float2 r = __sincosf_stret(x);
  *s = r.__sinval, *c = r.__cosval;
#else
  *s = sinf(x), *c = cosf(x);
#endif
}

std::vector<uint32_t> DecodeUtf8(const std::string& s) {
  std::vector<uint32_t> out;
  for (size_t i = 0; i < s.size();) {
    const uint8_t c = static_cast<uint8_t>(s[i]);
    int n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
    uint32_t cp = n == 1 ? c : n == 2 ? (c & 0x1F) : n == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < n && i + k < s.size(); k++) cp = (cp << 6) | (static_cast<uint8_t>(s[i + k]) & 0x3F);
    out.push_back(cp);
    i += n;
  }
  return out;
}

bool IsWhitespaceText(const std::string& utf8) {
  const auto cps = DecodeUtf8(utf8);
  if (cps.empty()) return false;
  for (uint32_t cp : cps)
    if (!InTable(kSeparator, cp)) return false;
  return true;
}

bool IsPunctuationText(const std::string& utf8) {
  const auto cps = DecodeUtf8(utf8);
  if (cps.empty()) return false;
  for (uint32_t cp : cps)
    if (!InTable(kPunctuation, cp)) return false;
  return true;
}

bool HasAnyRtl(const std::string& utf8) {
  for (uint32_t cp : DecodeUtf8(utf8))
    if (InTable(kBidiRtl, cp)) return true;
  return false;
}

bool HasAnyLtr(const std::string& utf8) {
  for (uint32_t cp : DecodeUtf8(utf8))
    if (InTable(kBidiLtr, cp)) return true;
  return false;
}

void ExpandBoxToContainBox(const RotatedBox& src, RotatedBox* dst) {
  int w = dst->width, h = dst->height;
  if (w < 1 || h < 1) {
    *dst = src;
    dst->angle = NormalizeAngleF(src.angle);
    return;
  }
  const float j = NormalizeAngleF(src.angle);
  const float k = NormalizeAngleF(dst->angle);
  if (j == 0.0f && k == 0.0f) {
    int left = dst->left;
    if (left - src.left >= 1) {
      w += left - src.left;
      dst->width = w;
      dst->left = left = src.left;
    }
    int top = dst->top;
    if (top - src.top >= 1) {
      h += top - src.top;
      dst->height = h;
      dst->top = top = src.top;
    }
    if (src.width + src.left > w + left) dst->width = src.width + src.left - left;
    if (src.height + src.top > h + top) dst->height = src.height + src.top - top;
    return;
  }
  // src expressed in the frame of dst (rotation by -k about dst's origin)
  const float ox = static_cast<float>(dst->left), oy = static_cast<float>(dst->top);
  const float sx = static_cast<float>(src.left), sy = static_cast<float>(src.top);
  float s1, c1;
  SinCos(k * -kDegToRad, &s1, &c1);
  const float ty = (s1 * (sx - ox) + oy) + c1 * (sy - oy);
  const int L = SaturatingInt(roundf((c1 * (sx - ox) + ox) - s1 * (sy - oy)));
  const int T = SaturatingInt(roundf(ty));
  const float m = NormalizeAngleF(src.angle - k);
  float p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y;  // origin, +width, +width+height, +height
  if (m == 0.0f) {
    p0x = static_cast<float>(L), p0y = static_cast<float>(T);
    p1x = static_cast<float>(src.width + L), p1y = static_cast<float>(T);
    p2x = p1x, p2y = static_cast<float>(src.height + T);
    p3x = static_cast<float>(L), p3y = p2y;
  } else {
    float s2, c2;
    SinCos(m * kDegToRad, &s2, &c2);
    const float fw = static_cast<float>(src.width), fh = static_cast<float>(src.height);
    p0x = static_cast<float>(L), p0y = static_cast<float>(T);
    p1x = c2 * fw + static_cast<float>(L);
    p1y = s2 * fw + static_cast<float>(T);
    const float ch = c2 * fh;
    p2x = p1x - s2 * fh;
    p2y = p1y + ch;
    p3x = static_cast<float>(L) - s2 * fh;
    p3y = ch + static_cast<float>(T);
  }
  const float fx = static_cast<float>(dst->width + dst->left), fy = static_cast<float>(dst->height + dst->top);
  auto mn = [](float a, float b) { return a > b ? b : a; };
  auto mx = [](float a, float b) { return a > b ? a : b; };
  float minx = mn(ox, p0x), miny = mn(oy, p0y);
  float maxx = mx(p0x, fx), maxy = mx(p0y, fy);
  minx = mn(minx, p1x), miny = mn(miny, p1y), maxx = mx(p1x, maxx), maxy = mx(p1y, maxy);
  minx = mn(minx, p2x), miny = mn(miny, p2y), maxx = mx(p2x, maxx), maxy = mx(p2y, maxy);
  minx = mn(minx, p3x), miny = mn(miny, p3y), maxx = mx(p3x, maxx), maxy = mx(p3y, maxy);
  float s3, c3;
  SinCos(k * kDegToRad, &s3, &c3);
  const float d = minx - ox;
  const float ny = (s3 * d + oy) + c3 * (miny - oy);
  dst->width = SaturatingInt(roundf(maxx)) - SaturatingInt(roundf(minx));
  dst->height = SaturatingInt(roundf(maxy)) - SaturatingInt(roundf(miny));
  dst->left = SaturatingInt(roundf((c3 * d + ox) - s3 * (miny - oy)));
  dst->top = SaturatingInt(roundf(ny));
}

RotatedBox ContainingBoxWithAverageAngle(const std::vector<RotatedBox>& boxes) {
  RotatedBox acc;
  if (boxes.empty()) return acc;
  float angle;
  const int n = static_cast<int>(boxes.size());
  if (n == 1) {
    angle = NormalizeAngleF(boxes[0].angle);
  } else {
    float s = 0, c = 0;
    for (const RotatedBox& b : boxes) {
      float sb, cb;
      SinCos(b.angle * kDegToRad, &sb, &cb);
      s += sb;
      c += cb;
    }
    if (std::fabs(c) > 1e-5f || std::fabs(s) > 1e-5f) {
      if (std::fabs(s) > 1e-5f || c > static_cast<float>(n) * -0.99999f)
        angle = NormalizeAngleF(atan2f(s, c) * 57.2957763671875f /* 0x42652EE0 */);
      else
        angle = 180.0f;
    } else {
      angle = boxes[0].angle;
      for (const RotatedBox& b : boxes)
        if (std::fabs(b.angle) < std::fabs(angle)) angle = b.angle;
    }
  }
  acc.angle = angle;
  for (const RotatedBox& b : boxes) {
    if (acc.width < 1 || acc.height < 1) {
      acc.left = b.left, acc.top = b.top, acc.width = 1, acc.height = 1;
    }
    ExpandBoxToContainBox(b, &acc);
  }
  return acc;
}

RotatedBox ToPage(const RotatedBox& line_box, const RotatedBox& rel) {
  const double cx = line_box.left, cy = line_box.top;
  const double px = cx + static_cast<double>(rel.left), py = cy + static_cast<double>(rel.top);
  const double dx = px - cx, dy = py - cy;
  const double rad = static_cast<double>(line_box.angle) * 3.14159265358979311600 / 180.0;
  const double sn = sin(rad), cs = cos(rad);
  const double x = cx + (cs * dx - sn * dy);
  const double y = cy + (sn * dx + cs * dy);
  RotatedBox out = rel;
  out.left = static_cast<int>(round(x));
  out.top = static_cast<int>(round(y));
  // aksara::box_utils::NormalizedAngle
  double a = fmod(static_cast<double>(static_cast<float>(line_box.angle + rel.angle)), 360.0);
  a = a > 180.0 ? a + -360.0 : (a > -180.0 ? a : a + 360.0);
  out.angle = static_cast<float>(a);
  return out;
}

std::vector<RecWord> BuildWords(const std::vector<RecSymbol>& symbols, const std::string& line_text) {
  // SplitBySpace: byte offsets where a space-delimited token ends.
  std::set<size_t> token_ends;
  for (size_t i = 0; i < line_text.size(); i++)
    if (line_text[i] != ' ' && (i + 1 == line_text.size() || line_text[i + 1] == ' ')) token_ends.insert(i + 1);

  std::vector<RecWord> words;
  int cur = -1, last = -1;
  bool rtl = false, ltr = false;
  auto finalize = [&]() {
    RecWord& w = words[cur];
    if (!w.brk) w.brk.emplace();
    w.brk->type = 1;
    if (rtl) {
      w.writing_direction = 1;
    } else if (ltr) {
      w.writing_direction = 0;
    }
    double sum = 0;
    for (int si : w.symbols) {
      w.text += symbols[si].text;
      sum += symbols[si].confidence;
    }
    w.box = UnionStart(symbols[w.symbols[0]].box);
    for (int si : w.symbols) ExpandBoxToContainBox(symbols[si].box, &w.box);
    w.confidence = static_cast<float>(sum / static_cast<double>(w.symbols.size()));
    if (!w.text.empty()) last = cur;
    cur = -1;
  };
  // Whitespace after the last word gives it no break (e.g. a recognized trailing space).
  size_t end = symbols.size();
  while (end > 0 && IsWhitespaceText(symbols[end - 1].text)) end--;
  size_t offset = 0;
  for (size_t i = 0; i < end; i++) {
    const RecSymbol& s = symbols[i];
    offset += s.text.size();
    if (cur < 0 && last >= 0 && !words[last].brk) words[last].brk.emplace();
    if (IsWhitespaceText(s.text)) {
      if (cur >= 0) finalize();
      if (last >= 0) {
        WordBreak& b = *words[last].brk;
        b.type = b.type == 2 ? 3 : 2;
        if (!b.box) b.box = RotatedBox();
        ExpandBoxToContainBox(s.box, &*b.box);
      }
      continue;
    }
    if (cur < 0) {
      words.emplace_back();
      cur = static_cast<int>(words.size()) - 1;
      rtl = ltr = false;
    }
    const bool punct = IsPunctuationText(s.text);
    rtl |= HasAnyRtl(s.text) && !punct;
    ltr |= HasAnyLtr(s.text) && !punct;
    words[cur].symbols.push_back(static_cast<int>(i));
    if (token_ends.count(offset)) finalize();
  }
  if (cur >= 0) finalize();
  return words;
}

void RecomputeSpaceWordBreakBoxes(const std::vector<RecSymbol>& symbols, std::vector<RecWord>* words) {
  for (size_t i = 0; i + 1 < words->size(); i++) {
    RecWord& a = (*words)[i];
    const RecWord& b = (*words)[i + 1];
    if (!a.brk || (a.brk->type != 2 && a.brk->type != 3) || a.symbols.empty() || b.symbols.empty() || !a.brk->box)
      continue;
    const RotatedBox& la = symbols[a.symbols.back()].box;
    const RotatedBox& fb = symbols[b.symbols.front()].box;
    RotatedBox u = UnionStart(la);
    ExpandBoxToContainBox(la, &u);
    ExpandBoxToContainBox(fb, &u);
    const float gap = std::max(0.0f, static_cast<float>(u.width - la.width - fb.width));
    a.brk->box->left = a.box.left + a.box.width;
    a.brk->box->width = std::max(1, static_cast<int>(std::ceil(gap)));
  }
}

namespace {

// ocr::photo::line_metrics_util::ComputeOtsuThresholdOfVector(InlinedVector<int,256>, n, &count_at_threshold).
int Otsu(const int* hist, int n, int* below) {
  int cum[256], mom[256];
  cum[0] = hist[0], mom[0] = 0;
  for (int i = 1; i < 256; i++) cum[i] = cum[i - 1] + hist[i], mom[i] = mom[i - 1] + hist[i] * i;
  const int total = cum[255], total_mom = mom[255];
  const float N = static_cast<float>(n <= 1 ? 1 : n);
  float best = -1.0f;
  int t = -1;
  for (int i = 0; i < 256; i++) {
    const int c = cum[i];
    if (c == 0 || c == total) continue;
    const int rest = total_mom - mom[i];
    const float d = static_cast<float>(mom[i] / c) - static_cast<float>(rest / (total - c));
    const float s = d * (((static_cast<float>(c) / N) * (static_cast<float>(rest) / N)) * d);
    if (s > best) best = s, t = i;
  }
  if (below) *below = t < 0 ? 0 : cum[t];
  return t;
}

// Sampling steps for a region of `count` pixels: alternately doubles the x then the y step until <= 5000 samples.
void Steps(int count, int* xs, int* ys) {
  *xs = 1, *ys = 1;
  if (count < 5001) return;
  int i = 0;
  const float fc = static_cast<float>(static_cast<unsigned>(count));
  do {
    *ys <<= (i & 1);
    *xs <<= !(i & 1);
    i++;
  } while (fc / static_cast<float>(*xs * *ys) > 5000.0f);
}

struct Coords {
  std::pair<int, int> at[256];
  Coords() {
    for (auto& p : at) p = {-1, -1};
  }
};

struct Clamped {
  int top, bottom, left, right;
};

Clamped Clamp(const RotatedBox& b, int W, int H) {
  auto c = [](int v, int hi) { return v < 0 ? 0 : std::min(v, hi); };
  return {c(b.top, H - 1), c(b.top + b.height, H - 1), c(b.left, W - 1), c(b.left + b.width, W - 1)};
}

}  // namespace

void EstimateWordColors(const GrayImage& img, const std::vector<RecSymbol>& symbols, std::vector<RecWord>* words) {
  const int W = img.width, H = img.height;
  for (size_t wi = 0; wi < words->size(); wi++) {
    RecWord& word = (*words)[wi];
    int bg_hist[256] = {0}, fg_hist[256] = {0};
    Coords lists[4], fg_coords;
    const Clamped cb = Clamp(word.box, W, H);
    const float m = static_cast<float>(word.box.height) * 0.3f;
    auto sample = [&](int x0, int x1, int y0, int y1, int count, int* hist, Coords* coords) {
      int xs, ys;
      Steps(count, &xs, &ys);
      int n = 0;
      if (y1 <= y0 || x1 <= x0) return 0;
      for (int y = y0; y < y1; y += ys)
        for (int x = x0; x < x1; x += xs) {
          const int v = img.data[static_cast<size_t>(y) * W + x];
          hist[v]++;
          coords->at[v] = {x, y};
          n++;
        }
      return n;
    };
    // above
    const int a0 = std::max(0, static_cast<int>(static_cast<float>(cb.top) - m));
    int bg_n = sample(cb.left, cb.right, a0, cb.top, (cb.top - a0) * (cb.right - cb.left), bg_hist, &lists[1]);
    // below
    const int b1 = std::min(H, static_cast<int>(m + static_cast<float>(cb.bottom)));
    bg_n += sample(cb.left, cb.right, cb.bottom + 1, b1, (b1 - (cb.bottom + 1)) * (cb.right - cb.left), bg_hist,
                   &lists[2]);
    // right, up to this word's break
    int lim = W;
    if (word.brk && word.brk->box) lim = std::min(lim, word.brk->box->left + word.brk->box->width);
    lim = std::min(lim, static_cast<int>(m + static_cast<float>(cb.right)));
    bg_n += sample(cb.right + 1, lim, cb.top, cb.bottom, (lim - (cb.right + 1)) * (cb.bottom - cb.top), bg_hist,
                   &lists[3]);
    // left, from the previous word's break
    int prev = 0;
    if (wi > 0 && (*words)[wi - 1].brk && (*words)[wi - 1].brk->box) prev = std::max(0, (*words)[wi - 1].brk->box->left);
    const int lo = std::max(prev, static_cast<int>(static_cast<float>(cb.left) - m));
    bg_n += sample(lo, cb.left, 0, cb.bottom, (cb.left - lo) * cb.bottom, bg_hist, &lists[2]);
    const int bg_regions = bg_n;
    if (bg_n == 0)
      bg_n = sample(cb.left, cb.right, cb.top, cb.bottom, (cb.right - cb.left) * (cb.bottom - cb.top), bg_hist,
                    &lists[0]);
    int fg_n = 0;
    for (int si : word.symbols) {
      const Clamped c = Clamp(symbols[si].box, W, H);
      fg_n += sample(c.left, c.right, c.top, c.bottom, (c.right - c.left) * (c.bottom - c.top), fg_hist, &fg_coords);
    }
    int all[256];
    for (int v = 0; v < 256; v++) all[v] = bg_hist[v] + fg_hist[v];
    const int T = Otsu(all, fg_n + bg_n, nullptr);
    int bg_below = 0, fg_below = 0;
    for (int v = 0; v <= T && v < 256; v++) bg_below += bg_hist[v], fg_below += fg_hist[v];
    const float fbg = static_cast<float>(bg_n);
    bool dark = static_cast<float>(bg_n) / 2.0f < static_cast<float>(bg_below);
    if (bg_regions != 0 && fbg * 0.3f < static_cast<float>(bg_below) && fbg * 0.6f >= static_cast<float>(bg_below)) {
      int word_hist[256] = {0};
      Coords unused;
      const int cnt = sample(cb.left, cb.right, cb.top, cb.bottom, (cb.right - cb.left) * (cb.bottom - cb.top),
                             word_hist, &unused);
      int below2 = 0;
      const int T2 = Otsu(word_hist, cnt, &below2);
      int fg_below2 = 0;
      for (int v = 0; v <= T2 && v < 256; v++) fg_below2 += fg_hist[v];
      const float r1 = static_cast<float>(below2) / static_cast<float>(cnt);
      const float r2 = static_cast<float>(fg_below2) / static_cast<float>(fg_n);
      dark = r1 >= r2 + 0.1f || (dark && r1 > r2 + -0.1f);
    }
    const int nb = dark ? bg_below : bg_n - bg_below;
    const int nf = dark ? fg_n - fg_below : fg_below;
    const int hb = nb / 2, hf = nf / 2;
    int cbg = 0, cfg = 0, bg = -1, fg = -1;
    std::vector<std::pair<int, int>> bg_samples;
    std::pair<int, int> fg_sample{-1, -1};
    for (int v = 0; v < 256; v++) {
      if (dark) {
        if (v > T)
          cfg += fg_hist[v];
        else
          cbg += bg_hist[v];
      } else {
        if (v > T)
          cbg += bg_hist[v];
        else
          cfg += fg_hist[v];
      }
      if (cbg >= hb && bg == -1) {
        for (const Coords& l : lists)
          if (l.at[v].first != -1 && l.at[v].second != -1) bg_samples.push_back(l.at[v]);
        bg = v;
      }
      if (cfg >= hf && fg == -1) {
        fg_sample = fg_coords.at[v];
        fg = v;
      }
    }
    if (bg == -1 || fg == -1) continue;
    Colors colors;
    colors.foreground_gray = fg;
    colors.background_gray = bg;
    colors.foreground_gray_threshold = T;
    colors.foreground_samples.push_back(fg_sample);
    colors.background_samples = bg_samples;
    word.colors = colors;
  }
}

}  // namespace screenai::rec
