#include "layout_analyzer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <tuple>

#include "box_overlap.h"
#include "layout_graph.h"
#include "proto_wire.h"
#include "tflite_model.h"

namespace screenai {
namespace {

using Type = EntityType;

// ---- math helpers mirroring the original's libm calls ----

struct SinCos {
  float s, c;
};
SinCos SinCosf(float x) {
#if defined(__APPLE__)
  const __float2 r = __sincosf_stret(x);
  return {r.__sinval, r.__cosval};
#else
  return {sinf(x), cosf(x)};
#endif
}

struct SinCos2 {
  double s, c;
};

SinCos2 SinCosD(double x) {
#if defined(__APPLE__)
  const __double2 r = __sincos_stret(x);
  return {r.__sinval, r.__cosval};
#else
  return {sin(x), cos(x)};
#endif
}

constexpr float BitsF(uint32_t u) { return __builtin_bit_cast(float, u); }
constexpr float kDegToRad = BitsF(0x3C8EFA35);  // 0.017453292
constexpr float kRadToDeg = BitsF(0x42652EE0);  // 57.295776

float NormalizeAngle(float a) {  // (-180, 180]
  while (a <= -180.0f) a += 360.0f;
  while (a > 180.0f) a -= 360.0f;
  return a;
}

// float -> int as the original's absl-style saturating round: roundf, then clamp to the int range.
int SaturatingInt(float v) {
  if (v >= 2147483648.0f) return 0x7FFFFFFF;
  if (v <= -2147483648.0f) v = -2147483648.0f;
  return static_cast<int>(v);
}
int RoundToInt(float v) { return SaturatingInt(roundf(v)); }

// ---- tree helpers (goodoc::PageLayout) ----

std::vector<std::vector<int>> ChildLists(const PageLayout& l) {
  std::vector<std::vector<int>> kids(l.entities.size());
  for (size_t i = 0; i < l.entities.size(); i++) {
    const int p = l.entities[i].parent;
    if (p >= 0 && p < static_cast<int>(l.entities.size())) kids[p].push_back(static_cast<int>(i));
  }
  return kids;
}

bool IsRoot(const PageLayout& l, int i) {
  const int p = l.entities[i].parent;
  return p < 0 || p >= static_cast<int>(l.entities.size());
}

// PageLayout::Descendants(nullptr): level order starting from the roots.
std::vector<int> AllDescendants(const PageLayout& l, const std::vector<std::vector<int>>& kids) {
  std::vector<int> out;
  for (size_t i = 0; i < l.entities.size(); i++)
    if (IsRoot(l, static_cast<int>(i))) out.push_back(static_cast<int>(i));
  for (size_t k = 0; k < out.size(); k++)
    for (int c : kids[out[k]]) out.push_back(c);
  return out;
}

// PageLayout::Descendants(e): level order below e (e excluded).
std::vector<int> DescendantsOf(int e, const std::vector<std::vector<int>>& kids) {
  std::vector<int> out(kids[e]);
  for (size_t k = 0; k < out.size(); k++)
    for (int c : kids[out[k]]) out.push_back(c);
  return out;
}

std::vector<int> FilterBy(const PageLayout& l, const std::vector<int>& ids, Type t) {
  std::vector<int> out;
  for (int i : ids)
    if (l.entities[i].type == t) out.push_back(i);
  return out;
}

// Keeps entities in `order` (each index once), renumbering parent references. Parents not kept become -1.
// Returns the old -> new index map (-1 for dropped entities).
std::vector<int> Reorder(PageLayout* l, const std::vector<int>& order) {
  std::vector<int> remap(l->entities.size(), -1);
  for (size_t k = 0; k < order.size(); k++) remap[order[k]] = static_cast<int>(k);
  std::vector<Entity> out;
  out.reserve(order.size());
  for (int i : order) {
    Entity e = std::move(l->entities[i]);
    if (e.parent >= 0) e.parent = e.parent < static_cast<int>(remap.size()) ? remap[e.parent] : -1;
    out.push_back(std::move(e));
  }
  l->entities = std::move(out);
  return remap;
}

// PageLayout::Delete: removes the marked entities and all their descendants; the rest keeps its order.
// Returns the old -> new index map.
std::vector<int> DeleteWithDescendants(PageLayout* l, std::vector<bool> del) {
  const auto kids = ChildLists(*l);
  for (size_t i = 0; i < del.size(); i++)
    if (del[i])
      for (int d : DescendantsOf(static_cast<int>(i), kids)) del[d] = true;
  std::vector<int> keep;
  for (size_t i = 0; i < del.size(); i++)
    if (!del[i]) keep.push_back(static_cast<int>(i));
  return Reorder(l, keep);
}

// ReorderUsingDFS: preorder over the roots and their children (index order). Returns the old -> new index map.
std::vector<int> ReorderUsingDfs(PageLayout* l) {
  const auto kids = ChildLists(*l);
  std::vector<int> order;
  std::vector<int> stack;
  for (int i = static_cast<int>(l->entities.size()) - 1; i >= 0; i--)
    if (IsRoot(*l, i)) stack.push_back(i);
  while (!stack.empty()) {
    const int e = stack.back();
    stack.pop_back();
    order.push_back(e);
    for (auto it = kids[e].rbegin(); it != kids[e].rend(); ++it) stack.push_back(*it);
  }
  return Reorder(l, order);
}

// ---- box utilities (google_ocr::box_util) ----

// ExpandBoxToContainBox<goodoc::RotatedBoundingBox, goodoc::RotatedBoundingBox> (0x10081ccbc).
void ExpandBoxToContainBox(const RotatedBox& src, RotatedBox* dst) {
  if (dst->width < 1 || dst->height <= 0) {
    const auto curve = std::move(dst->curve);
    *dst = src;
    dst->curve = curve;  // only the rotated box is copied
    dst->angle = NormalizeAngle(src.angle);
    return;
  }
  const float j = NormalizeAngle(src.angle);
  const float k = NormalizeAngle(dst->angle);
  if (j == 0.0f && k == 0.0f) {
    int l = dst->left, t = dst->top, w = dst->width, h = dst->height;
    if (l - src.left >= 1) w += l - src.left, l = src.left;
    if (t - src.top >= 1) h += t - src.top, t = src.top;
    dst->left = l, dst->top = t, dst->width = w, dst->height = h;
    if (src.width + src.left > w + l) dst->width = src.width + src.left - l;
    if (src.height + src.top > h + t) dst->height = src.height + src.top - t;
    return;
  }
  const float dl = static_cast<float>(dst->left), dt = static_cast<float>(dst->top);
  // Source origin in the destination's rotated frame.
  const SinCos r0 = SinCosf(k * -kDegToRad);
  const float sl = static_cast<float>(src.left), st = static_cast<float>(src.top);
  const float ny = ((r0.s * (sl - dl)) + dt) + (r0.c * (st - dt));
  const int x0 = RoundToInt(((r0.c * (sl - dl)) + dl) - (r0.s * (st - dt)));
  const int y0 = RoundToInt(ny);
  const float m = NormalizeAngle(src.angle - k);
  float p0x = static_cast<float>(x0), p0y = static_cast<float>(y0);
  float p1x, p1y, p2x, p2y, p3x, p3y;
  if (m == 0.0f) {
    p1x = static_cast<float>(src.width + x0), p1y = static_cast<float>(y0);
    p2x = p1x, p2y = static_cast<float>(src.height + y0);
    p3x = static_cast<float>(x0), p3y = p2y;
  } else {
    const SinCos r = SinCosf(m * kDegToRad);
    const float w = static_cast<float>(src.width), h = static_cast<float>(src.height);
    p1x = (r.c * w) + p0x;
    p1y = (r.s * w) + p0y;
    const float ch = r.c * h;
    p2x = p1x - (r.s * h);
    p2y = p1y + ch;
    p3x = p0x - (r.s * h);
    p3y = ch + p0y;
  }
  const float qx = static_cast<float>(dst->width + dst->left), qy = static_cast<float>(dst->height + dst->top);
  auto mn = [](float a, float b) { return a > b ? b : a; };  // vbsl(a > b, b, a)
  auto mx = [](float a, float b) { return a > b ? a : b; };
  float minx = mn(dl, p0x), miny = mn(dt, p0y);
  float maxx = mx(p0x, qx), maxy = mx(p0y, qy);
  minx = mn(minx, p1x), miny = mn(miny, p1y), maxx = mx(p1x, maxx), maxy = mx(p1y, maxy);
  minx = mn(minx, p2x), miny = mn(miny, p2y), maxx = mx(p2x, maxx), maxy = mx(p2y, maxy);
  minx = mn(minx, p3x), miny = mn(miny, p3y), maxx = mx(p3x, maxx), maxy = mx(p3y, maxy);
  const SinCos r1 = SinCosf(k * kDegToRad);
  const float ex = minx - dl;
  const float top = ((r1.s * ex) + dt) + (r1.c * (miny - dt));
  dst->width = RoundToInt(maxx) - RoundToInt(minx);
  dst->height = RoundToInt(maxy) - RoundToInt(miny);
  dst->left = RoundToInt(((r1.c * ex) + dl) - (r1.s * (miny - dt)));
  dst->top = RoundToInt(top);
}

// FindContainingBoxWithAverageAngle<goodoc::RotatedBoundingBox const> (0x100896340).
RotatedBox FindContainingBoxWithAverageAngle(const std::vector<RotatedBox>& boxes) {
  float angle = 0.0f;
  const size_t n = boxes.size();
  if (n == 1) {
    angle = NormalizeAngle(boxes[0].angle);
  } else if (n > 1) {
    float s = 0.0f, c = 0.0f;
    for (const auto& b : boxes) {
      const SinCos r = SinCosf(b.angle * kDegToRad);
      s = s + r.s;
      c = c + r.c;
    }
    const float as = fabsf(s);
    if (fabsf(c) > 1e-5f || as > 1e-5f) {
      if (as > 1e-5f || c > static_cast<float>(n) * -0.99999f)
        angle = atan2f(s, c) * kRadToDeg;
      else
        angle = 180.0f;
    } else {
      float best = 180.0f;
      angle = 180.0f;
      for (const auto& b : boxes) {
        const float a = NormalizeAngle(b.angle);
        const float aa = fabsf(a);
        const float alt = a == -angle ? fabsf(angle) : angle;
        if (aa < best) {
          angle = a;
          best = aa;
        } else {
          angle = alt;
        }
      }
    }
  }
  RotatedBox out;
  out.angle = angle;
  for (const auto& b : boxes) {
    if (out.width < 1 || out.height <= 0) out.left = b.left, out.top = b.top, out.width = 1, out.height = 1;
    ExpandBoxToContainBox(b, &out);
  }
  return out;
}

// goodoc::RecomputeBoxFromChildren: containing box of the children's boxes and break boxes.
void RecomputeBoxFromChildren(PageLayout* l, int e, const std::vector<std::vector<int>>& kids) {
  std::vector<RotatedBox> boxes;
  for (int c : kids[e]) {
    const Entity& ch = l->entities[c];
    if (ch.box) boxes.push_back(*ch.box);
    if (ch.word_break && ch.word_break->box) boxes.push_back(*ch.word_break->box);
  }
  if (boxes.empty()) return;
  // A curved entity keeps its curve; only its rotated_bounding_box is replaced.
  std::optional<Curve> curve;
  if (l->entities[e].box) curve = std::move(l->entities[e].box->curve);
  l->entities[e].box = FindContainingBoxWithAverageAngle(boxes);
  l->entities[e].box->curve = std::move(curve);
}

void RecomputeBoxesFromSymbols(PageLayout* l) {
  const auto kids = ChildLists(*l);
  const auto all = AllDescendants(*l, kids);
  for (auto it = all.rbegin(); it != all.rend(); ++it) RecomputeBoxFromChildren(l, *it, kids);
}

// goodoc::RecomputeConfidenceFromDescendants; returns (count, sum).
std::pair<int, double> RecomputeConfidenceFromDescendants(PageLayout* l, int e,
                                                          const std::vector<std::vector<int>>& kids,
                                                          bool only_if_missing) {
  int n = 0;
  double sum = 0.0;
  for (int c : kids[e]) {
    const auto r = RecomputeConfidenceFromDescendants(l, c, kids, only_if_missing);
    n += r.first;
    sum = sum + r.second;
  }
  Entity& ent = l->entities[e];
  if (n < 1) {
    if (ent.confidence) return {1, static_cast<double>(*ent.confidence)};
    return {0, 0.0};
  }
  if (!(ent.confidence && only_if_missing))
    ent.confidence = static_cast<float>(sum / static_cast<double>(static_cast<unsigned>(n)));
  return {n, sum};
}

void RecomputeConfidencesFromSymbols(PageLayout* l, bool only_if_missing = false) {
  const auto kids = ChildLists(*l);
  for (size_t i = 0; i < l->entities.size(); i++)
    if (IsRoot(*l, static_cast<int>(i)))
      RecomputeConfidenceFromDescendants(l, static_cast<int>(i), kids, only_if_missing);
}

// ---- text (goodoc::PageLayout::Text with separator " ", no direction marks) ----

std::string EntityText(const PageLayout& l, int e, const std::vector<std::vector<int>>& kids) {
  const Entity& ent = l.entities[e];
  if (ent.text && !ent.text->empty()) return *ent.text;
  std::string out;
  for (int c : kids[e]) {
    const std::string t = EntityText(l, c, kids);
    if (t.empty()) continue;
    out += t;
    const Entity& ch = l.entities[c];
    switch (ch.type) {
      case Type::kParagraph:
      case Type::kBlock:
        out += '\n';
        break;
      case Type::kLine:
        out += ' ';
        break;
      case Type::kSymbol:
        break;
      default: {
        const int bt = ch.word_break ? ch.word_break->type : 0;
        if (bt == 1) break;              // NONE
        if (bt == 4) out += '-';         // SOFT_HYPHEN
        else out += ' ';                 // UNKNOWN (logged), SPACE, SURE_SPACE, others: separator
      }
    }
  }
  size_t n = out.size();
  while (n > 0 && (out[n - 1] == ' ' || out[n - 1] == '\t' || out[n - 1] == '\n')) n--;
  out.resize(n);
  return out;
}

// goodoc::RecomputeWordTextContentFromSymbols (0x100c06260): a word's text is the concatenation of its symbols'.
// (Content types 6/7 join through LatexTextReorderer instead; the recognizer never emits them.)
void RecomputeWordTextContent(PageLayout* l) {
  const auto kids = ChildLists(*l);
  for (size_t i = 0; i < l->entities.size(); i++) {
    if (l->entities[i].type != Type::kWord) continue;
    std::string t;
    for (int s : FilterBy(*l, DescendantsOf(static_cast<int>(i), kids), Type::kSymbol))
      t += l->entities[s].text.value_or("");
    l->entities[i].text = t;
  }
}

// Line text_content = PageLayout::Text of the line (word texts are kept as they are, even when they differ from
// their symbols; not goodoc::RecomputeLineTextContentFromSymbols, which would recompute them first).
void RecomputeLineTextContent(PageLayout* l) {
  const auto kids = ChildLists(*l);
  for (size_t i = 0; i < l->entities.size(); i++) {
    Entity& e = l->entities[i];
    if (e.type != Type::kLine) continue;
    e.text.reset();
    e.text = EntityText(*l, static_cast<int>(i), kids);
  }
}

// ---- mutator pre-step ----

void SetDetectionConfidence(PageLayout* l) {
  for (Entity& e : l->entities) {
    if (e.type != Type::kLine) continue;
    if (e.detection)
      e.detection_confidence = e.detection->sum_raw_boxes_scores / static_cast<float>(e.detection->num_raw_boxes);
    else
      e.detection_confidence.reset();
  }
}

// ---- ConfidenceFilter (0x10083f7dc) ----

void ConfidenceFilter(PageLayout* l) {
  constexpr float kWRec = 3.1155417f, kWDet = 7.5824437f, kWGarbage = -8.527104f, kWDir = 3.6975038f;
  constexpr float kBias = -8.170908f, kThreshold = 0.45f;
  const float thr = kThreshold <= 0.0f   ? -FLT_MAX
                    : kThreshold >= 1.0f ? FLT_MAX
                                         : -logf((1.0f / kThreshold) - 1.0f);
  const auto kids = ChildLists(*l);
  std::vector<bool> del(l->entities.size(), false);
  for (size_t i = 0; i < l->entities.size(); i++) {
    const Entity& e = l->entities[i];
    if (e.type != Type::kLine) continue;
    const float rec = e.confidence.value_or(0.0f);
    bool has_word = false;
    for (int c : kids[i]) has_word |= l->entities[c].type == Type::kWord;
    if (rec == 0.0f || !has_word) {
      del[i] = true;
      continue;
    }
    if (!e.detection) continue;
    const float garbage = e.detection->garbage_prob.value_or(0.0f);
    const float score = kBias + (((rec * kWRec + garbage * kWGarbage) + e.detection_confidence.value_or(0.0f) * kWDet) +
                                 e.direction_confidence.value_or(0.0f) * kWDir);
    if (!(score >= thr)) del[i] = true;
  }
  DeleteWithDescendants(l, del);
}

// ---- RemoveOverlapsWordPruningStep (0x1008713c0) ----

struct LineWords {
  int line;
  std::vector<int> words;
  std::vector<int> counts;  // symbols with a confidence per word
};
void PruneOverlappingLines(PageLayout* l, const std::vector<LineWords>& lines,
                           const std::vector<std::pair<float, int>>& order);

void WordPruning(PageLayout* l) {
  const auto kids = ChildLists(*l);
  const auto lines = FilterBy(*l, AllDescendants(*l, kids), Type::kLine);
  std::vector<std::pair<float, int>> line_conf;
  std::vector<LineWords> line_words;
  for (int li : lines) {
    LineWords lw{li, FilterBy(*l, DescendantsOf(li, kids), Type::kWord), {}};
    double total = 0.0;
    int total_count = 0;
    for (int wi : lw.words) {
      int count = 0;
      double sum = 0.0;
      for (int si : FilterBy(*l, DescendantsOf(wi, kids), Type::kSymbol)) {
        const Entity& s = l->entities[si];
        if (s.confidence) count++, sum = sum + static_cast<double>(*s.confidence);
      }
      l->entities[wi].confidence = count <= 0 ? 0.0f : static_cast<float>(sum / static_cast<double>(static_cast<unsigned>(count)));
      lw.counts.push_back(count);
      total = total + sum;
      total_count += count;
    }
    l->entities[li].confidence = total > 0.0 ? static_cast<float>(total / static_cast<double>(total_count)) : 0.0f;
    line_conf.emplace_back(static_cast<float>(total), static_cast<int>(line_words.size()));
    line_words.push_back(std::move(lw));
  }
  std::sort(line_conf.begin(), line_conf.end(), std::greater<std::pair<float, int>>());
  PruneOverlappingLines(l, line_words, line_conf);
}

// ---- FillSymbolGaps (0x10085a318): RefineLinesByNuggets, then boxes from symbols ----

// PageLayout::Orientation(e).writing_direction: the nearest entity (e or an ancestor) with an orientation.
int WritingDirection(const PageLayout& l, int e) {
  for (int i = e, guard = 0; i >= 0 && i < static_cast<int>(l.entities.size()) && guard < 64; i = l.entities[i].parent, guard++)
    if (l.entities[i].writing_direction || l.entities[i].orientation) return l.entities[i].writing_direction.value_or(0);
  return 0;
}

// page_layout_utils::LineLength: extent along the writing direction.
int LineLength(const RotatedBox& b, int writing_direction) { return writing_direction == 2 ? b.height : b.width; }

// box_util::ResizeBox<goodoc::RotatedBoundingBox> (0x10085fb38): grows by (dw, dh) keeping the center.
void ResizeBox(int dw, int dh, RotatedBox* b) {
  b->width = std::max(b->width + dw, 0);
  b->height = std::max(b->height + dh, 0);
  const float a = NormalizeAngle(b->angle);
  if (a == 0.0f) {
    b->left = static_cast<int>(static_cast<float>(b->left) + static_cast<float>(dw) * -0.5f);
    b->top = static_cast<int>(static_cast<float>(b->top) + static_cast<float>(dh) * -0.5f);
    return;
  }
  const float l = static_cast<float>(b->left), t = static_cast<float>(b->top);
  const float nl = static_cast<float>(static_cast<double>(b->left) - static_cast<double>(dw) * 0.5);
  const float nt = static_cast<float>(static_cast<double>(b->top) - static_cast<double>(dh) * 0.5);
  const SinCos r = SinCosf(a * kDegToRad);
  const float y = ((r.s * (nl - l)) + t) + (r.c * (nt - t));
  b->left = RoundToInt(((r.c * (nl - l)) + l) - (r.s * (nt - t)));
  b->top = RoundToInt(y);
}

// DepthGapBetweenEntities (0x100ab4920) for two symbols: free length between them along the writing direction.
double DepthGap(const PageLayout& l, int a, int b) {
  const RotatedBox ba = l.entities[a].box.value_or(RotatedBox()), bb = l.entities[b].box.value_or(RotatedBox());
  RotatedBox u = ba;
  ExpandBoxToContainBox(bb, &u);
  const int wa = WritingDirection(l, a);
  const int d = LineLength(u, wa) - (LineLength(ba, wa) + LineLength(bb, WritingDirection(l, b)));
  return static_cast<double>(d > 0 ? d : 0);
}

// RecomputeSpaceWordBreakBoxes(words, layout, false) (0x100ab2c08): the SPACE break box of each word spans the gap
// to the next word's first symbol.
void RecomputeSpaceWordBreakBoxes(PageLayout* l, const std::vector<int>& words, const std::vector<std::vector<int>>& kids) {
  for (size_t k = 0; k + 1 < words.size(); k++) {
    Entity& w = l->entities[words[k]];
    if (!w.word_break || (w.word_break->type & ~1) != 2) continue;
    const auto &ka = kids[words[k]], &kb = kids[words[k + 1]];
    if (ka.empty() || kb.empty()) continue;
    const double gap = DepthGap(*l, ka.back(), kb.front());
    if (!w.word_break->box) continue;
    RotatedBox* box = &*w.word_break->box;
    const int wd = WritingDirection(*l, words[k]);
    const int delta = static_cast<int>(gap - static_cast<double>(LineLength(*box, wd)));
    if (wd == 2) {
      ResizeBox(0, delta, box);
      box->height = std::max(box->height, 1);
    } else {
      ResizeBox(delta, 0, box);
      box->width = std::max(box->width, 1);
    }
  }
}

}  // namespace

// PageLayoutAnalyzerSpec.FillSymbolGapsSpec: 1 enabled, 2 use_nuggets_refinement, 3
// minimum_single_symbol_length_as_fraction_of_thickness (double), 4 vertically_expand_words_and_symbols, 5
// ignored_scripts, 6 loose_lines. FillSymbolGapsStep passes has_minimum_single_symbol_length... (spec has-bit 0x20,
// checked at runtime: has_bits 0x27 with the shipped config) as RefineLinesByNuggets' first argument, so single-symbol
// words are only widened when the field is present.
void ParseFillSymbolGapsSpec(std::string_view bytes, FillSymbolGapsOptions* o) {
  pb::Reader r(bytes);
  while (r.Next()) {
    switch (r.field()) {
      case 2: o->use_nuggets_refinement = r.varint() != 0; break;
      case 3: o->has_min_single_symbol_length = true, o->min_single_symbol_length = r.float64(); break;
      case 4: o->vertically_expand = r.varint() != 0; break;
      case 5: o->ignored_scripts.insert(std::string(r.bytes())); break;
    }
  }
}

namespace {

// Finds the FillSymbolGapsSpec in the analysis config: google.protobuf.Any{value = PageLayoutAnalysisMutatorConfig}
// -> analyzer_spec (1) -> spec (1) -> fill_symbol_boxes (1).
bool FindFillSymbolGapsSpec(std::string_view any, FillSymbolGapsOptions* o) {
  auto field = [](std::string_view msg, int f, std::string_view* out) {
    pb::Reader r(msg);
    while (r.Next())
      if (r.field() == f) return *out = r.bytes(), true;
    return false;
  };
  std::string_view cfg, analyzer, spec, fsg;
  if (!field(any, 2, &cfg) || !field(cfg, 1, &analyzer)) return false;
  pb::Reader r(analyzer);
  while (r.Next()) {
    if (r.field() != 1) continue;
    spec = r.bytes();
    if (field(spec, 1, &fsg)) {
      *o = FillSymbolGapsOptions();
      ParseFillSymbolGapsSpec(fsg, o);
      return true;
    }
  }
  return false;
}

// box_util::RotateBoundingBox<goodoc::RotatedBoundingBox, false> (0x10085f5dc): rotates the origin corner about
// (cx, cy) by a degrees.
void RotateBoxInt(int cx, int cy, float a, RotatedBox* b) {
  const SinCos r = SinCosf(a * kDegToRad);
  const float x = static_cast<float>(b->left), y = static_cast<float>(b->top);
  const float fx = static_cast<float>(cx), fy = static_cast<float>(cy);
  const float ny = ((r.s * (x - fx)) + fy) + (r.c * (y - fy));
  b->left = RoundToInt(((r.c * (x - fx)) + fx) - (r.s * (y - fy)));
  b->top = RoundToInt(ny);
  b->angle = NormalizeAngle(a + b->angle);
}

// Integer RotateOrientationClockwise by +1 quadrant (top-to-bottom text handled as left-to-right) and its inverse.
void TurnClockwise(RotatedBox* b) {
  const int w = b->width;
  std::swap(b->width, b->height);
  const SinCos r = SinCosf(b->angle * kDegToRad);
  const float ny = (r.s * static_cast<float>(w)) + static_cast<float>(b->top);
  b->left = RoundToInt((r.c * static_cast<float>(w)) + static_cast<float>(b->left));
  b->top = RoundToInt(ny);
  b->angle = NormalizeAngle(b->angle + 90.0f);
}
void TurnCounterClockwise(RotatedBox* b) {
  const int h = b->height;
  std::swap(b->width, b->height);
  const SinCos r = SinCosf(b->angle * kDegToRad);
  const float ny = (r.c * static_cast<float>(h)) + static_cast<float>(b->top);
  b->left = RoundToInt(static_cast<float>(b->left) - (r.s * static_cast<float>(h)));
  b->top = RoundToInt(ny);
  b->angle = NormalizeAngle(b->angle + 270.0f);
}

// AxisAlignedNuggets without nuggets (ExtractAndRotateNuggets, 0x100ab133c): the line box (turned for
// top-to-bottom writing) and the detection's padding before the line start.
struct LineFrame {
  RotatedBox box;
  float pad_start = 0;
};
LineFrame ExtractLineFrame(const Entity& line, int writing_direction) {
  LineFrame f;
  f.box = line.box.value_or(RotatedBox());
  if (!line.detection) return f;
  float pads[2] = {line.detection->left_padding, line.detection->right_padding};
  if (writing_direction == 2) TurnClockwise(&f.box);
  const float d = fabsf(fabsf(f.box.angle - line.detection->detection_box.angle) + -180.0f);
  if (d < 0.01f) std::swap(pads[0], pads[1]);  // box flipped against the detection
  f.pad_start = pads[0];
  return f;
}


// DeltaSymbolExpansionWithConstraints (0x10085ea40) without nuggets: in the line frame, each symbol grows by 0.4 of
// the gaps to its neighbours (0.8 towards the inside for the first/last symbol), centered, but not past the line
// start (box left + padding); symbols already starting before it stay unchanged.
void DeltaSymbolExpansion(PageLayout* l, const std::vector<int>& symbols, int wd, const LineFrame& f,
                          const FillSymbolGapsOptions& o) {
  const int n = static_cast<int>(symbols.size());
  if (n < 1) return;
  const int min_left = static_cast<int>(f.pad_start + static_cast<float>(f.box.left));
  std::vector<RotatedBox> b(n);
  for (int k = 0; k < n; k++) {
    b[k] = l->entities[symbols[k]].box.value_or(RotatedBox());
    if (wd == 2) TurnClockwise(&b[k]);
    RotateBoxInt(f.box.left, f.box.top, -f.box.angle, &b[k]);
  }
  auto gap = [&](int a, int c) {
    const RotatedBox* first = b[a].left > b[c].left ? &b[c] : &b[a];
    const int g = std::max(b[a].left, b[c].left) - std::min(b[a].left, b[c].left) - first->width;
    return static_cast<float>(g > 0 ? g : 0);
  };
  for (int k = 0; k < n; k++) {
    const int left_nb = wd == 1 ? k + 1 : k - 1, right_nb = wd == 1 ? k - 1 : k + 1;
    float delta = 0.0f;
    if (n == 1) {
      if (o.has_min_single_symbol_length) {
        const int min_len = static_cast<int>(o.min_single_symbol_length * static_cast<double>(b[k].height));
        if (b[k].width < min_len) delta = static_cast<float>(min_len - b[k].width);
      }
    } else {
      if (left_nb >= 0 && left_nb < n) delta = (k == n - 1 ? 0.8f : 0.4f) * gap(k, left_nb);
      if (right_nb >= 0 && right_nb < n) delta = delta + (k != 0 ? 0.4f : 0.8f) * gap(k, right_nb);
    }
    int left = static_cast<int>(static_cast<int64_t>(rint(static_cast<double>(b[k].left) + delta * -0.5)));
    if (min_left > left) {
      delta = delta - static_cast<float>(min_left - left);
      left = min_left;
    }
    if (b[k].left < min_left) continue;
    b[k].left = left;
    b[k].width = std::max(static_cast<int>(delta + static_cast<float>(b[k].width)), 1);
  }
  for (int k = 0; k < n; k++) {
    RotateBoxInt(f.box.left, f.box.top, f.box.angle, &b[k]);
    if (wd == 2) TurnCounterClockwise(&b[k]);
    l->entities[symbols[k]].box = b[k];
  }
}

// RefineLinesByNuggets (0x10085b0c0). Our layouts carry no detection nuggets, so per line (unless its script is
// ignored) DeltaSymbolExpansionWithConstraints per word, then RecomputeSpaceWordBreakBoxes.
void RefineLinesByNuggets(PageLayout* l, const FillSymbolGapsOptions& o) {
  const auto kids = ChildLists(*l);
  for (size_t i = 0; i < l->entities.size(); i++) {
    const Entity& line = l->entities[i];
    if (line.type != Type::kLine) continue;
    const auto words = FilterBy(*l, DescendantsOf(static_cast<int>(i), kids), Type::kWord);
    if (words.empty()) continue;
    if (!(line.recognizer_script_id && o.ignored_scripts.count(*line.recognizer_script_id))) {
      const LineFrame f = ExtractLineFrame(line, WritingDirection(*l, static_cast<int>(i)));
      for (int w : words) DeltaSymbolExpansion(l, kids[w], WritingDirection(*l, w), f, o);
    }
    RecomputeSpaceWordBreakBoxes(l, words, kids);
  }
}

// FillSymbolGapsStep::AnalyzeInternal. Only the use_nuggets_refinement path (the config's) is implemented; the
// other path and vertically_expand_words_and_symbols are not.
void FillSymbolGaps(PageLayout* l, const FillSymbolGapsOptions& o) {
  RefineLinesByNuggets(l, o);
  RecomputeBoxesFromSymbols(l);
}

// ---- EstimateEntitiesColors (0x1008298c0) -> line_metrics_util::EstimateColorsOfWord (0x100220340) ----

// ComputeOtsuThresholdOfVector (0x10021f198), including the original's moment formula.
int OtsuThreshold(const int hist[256], int total, int* num_low) {
  int cum[256], mom[256];
  cum[0] = hist[0];
  mom[0] = 0;
  for (int i = 0; i < 255; i++) {
    cum[i + 1] = cum[i] + hist[i + 1];
    mom[i + 1] = hist[i + 1] * (i + 1) + mom[i];
  }
  const int n = cum[255], m = mom[255];
  const float s0 = static_cast<float>(static_cast<unsigned>(total > 1 ? total : 1));
  float best = -1.0f;
  int t_best = -1;
  for (int t = 0; t < 256; t++) {
    const int c = cum[t];
    if (c == 0 || c == n) continue;
    const int m0 = mom[t] / c;
    const int rest = m - mom[t];
    const int m1 = rest / (n - c);
    const float d = static_cast<float>(m0) - static_cast<float>(m1);
    const float w = (static_cast<float>(c) / s0) * (static_cast<float>(rest) / s0);
    const float v = d * (w * d);
    if (v > best) best = v, t_best = t;
  }
  *num_low = t_best >= 0 ? cum[t_best] : 0;
  return t_best;
}

bool EstimateColorsOfWord(const RotatedBox& box, const GrayImage& g, int page_w, int page_h, float pad,
                          Colors* out) {
  if (box.width <= 2) return false;
  const float fr[4] = {0.0f - pad, 0.33f, 0.67f, pad + 1.0f};
  int hist[256] = {0}, pos[256] = {0};
  const SinCos r = SinCosf(box.angle * kDegToRad);
  const int max_x = std::min(g.width, page_w) - 1, max_y = std::min(g.height, page_h) - 1;
  auto clamp = [](int64_t v, int hi) {
    int x = static_cast<int>(v) & ~(static_cast<int>(v) >> 31);
    return x >= hi ? hi : x;
  };
  for (float f : fr) {
    const float d = f * static_cast<float>(box.height);
    const int sx = clamp(static_cast<int64_t>(rintf(static_cast<float>(box.left) - (r.s * d))), max_x);
    const int sy = clamp(static_cast<int64_t>(rintf((r.c * d) + static_cast<float>(box.top))), max_y);
    const float fx = static_cast<float>(sx), fy = static_cast<float>(sy);
    for (int k = 0; k < box.width; k++) {
      const int x = clamp(static_cast<int64_t>(rintf((r.c * static_cast<float>(k)) + fx)), max_x);
      const int y = clamp(static_cast<int64_t>(rintf((r.s * static_cast<float>(k)) + fy)), max_y);
      const int v = g.data[static_cast<size_t>(y) * g.width + x];
      hist[v]++;
      pos[v] = x + y * g.width;
    }
  }
  const int total = box.width * 4;
  int num_low = 0;
  const int t = OtsuThreshold(hist, total, &num_low);
  const int num_high = total - num_low;
  if (num_low == 0 || total == num_low) return false;
  int low_med = -1, high_med = -1, cl = 0, ch = 0;
  for (int i = 0; i < 256; i++) {
    cl += i <= t ? hist[i] : 0;
    ch += i <= t ? 0 : hist[i];
    if (cl > num_low / 2 && low_med < 0) low_med = i;
    if (ch > num_high / 2 && high_med < 0) high_med = i;
  }
  auto rgb = [&](int med) { return med < 0 ? 0 : static_cast<int>(16843008u * g.data[pos[med]]); };
  const bool low_fg = num_low <= num_high;
  out->foreground_gray = low_fg ? low_med : high_med;
  out->background_gray = low_fg ? high_med : low_med;
  out->foreground_rgb = low_fg ? rgb(low_med) : rgb(high_med);
  out->background_rgb = low_fg ? rgb(high_med) : rgb(low_med);
  out->foreground_gray_threshold = t;
  return true;
}

void EstimateWordColors(PageLayout* l, const GrayImage& g) {
  constexpr float kPadding = 0.1f;  // EstimateWordColorsSpec.vertical_box_padding
  for (Entity& e : l->entities) {
    if (e.type != Type::kWord) continue;
    Colors c = e.colors.value_or(Colors());
    if (e.box && EstimateColorsOfWord(*e.box, g, l->width, l->height, kPadding, &c))
      e.colors = c;
    else
      e.colors.reset();
  }
}

// ---- PopulatePageLanguage ----

void PopulatePageLanguage(PageLayout* l) {
  const auto kids = ChildLists(*l);
  std::vector<std::pair<std::string, double>> weights;  // first-seen order (the original iterates a hash map)
  for (size_t i = 0; i < l->entities.size(); i++) {
    const Entity& e = l->entities[i];
    if (e.type != Type::kWord) continue;
    const float w = e.confidence.value_or(0.0f) * static_cast<float>(kids[i].size());
    for (const auto& lang : e.languages) {
      auto it = std::find_if(weights.begin(), weights.end(), [&](const auto& p) { return p.first == lang; });
      if (it == weights.end()) weights.emplace_back(lang, 0.0), it = weights.end() - 1;
      it->second += static_cast<double>(w);
    }
  }
  std::stable_sort(weights.begin(), weights.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  double total = 0.0;
  for (const auto& p : weights) total += p.second;
  l->language_mix.emplace();
  for (const auto& p : weights)
    l->language_mix->emplace_back(p.first, total == 0.0 ? 1.0f : static_cast<float>(p.second / total));
}

// ---- ClusterSortGcn line features (google_ocr::LineClusteringFeatures, GetBoxEdgeTensors) ----

// UnicodeProps::IsFirstCharCJKLetter: utf8propcjk property of the first code point (lead byte >= 0xE1).
bool IsFirstCharCjkLetter(const std::string& s) {
  static constexpr std::pair<int, int> kCjk[] = {
      {0x1100, 0x11FF},   {0x3005, 0x3007},   {0x3021, 0x302F},   {0x3031, 0x3035},   {0x3038, 0x303C},
      {0x3041, 0x3096},   {0x3099, 0x30FA},   {0x30FC, 0x30FF},   {0x3105, 0x312F},   {0x3131, 0x318E},
      {0x31A0, 0x31BF},   {0x31F0, 0x31FF},   {0x3400, 0x4DBF},   {0x4E00, 0x9FFF},   {0xA960, 0xA97C},
      {0xAC00, 0xD7A3},   {0xD7B0, 0xD7C6},   {0xD7CB, 0xD7FB},   {0xF900, 0xFA6D},   {0xFA70, 0xFAD9},
      {0xFF66, 0xFFBE},   {0xFFC2, 0xFFC7},   {0xFFCA, 0xFFCF},   {0xFFD2, 0xFFD7},   {0xFFDA, 0xFFDC},
      {0x16FE3, 0x16FE3}, {0x16FF0, 0x16FF6}, {0x1AFF0, 0x1AFF3}, {0x1AFF5, 0x1AFFB}, {0x1AFFD, 0x1AFFE},
      {0x1B000, 0x1B122}, {0x1B132, 0x1B132}, {0x1B150, 0x1B152}, {0x1B155, 0x1B155}, {0x1B164, 0x1B167},
      {0x20000, 0x2A6DF}, {0x2A700, 0x2B81D}, {0x2B820, 0x2CEAD}, {0x2CEB0, 0x2EBE0}, {0x2EBF0, 0x2EE5D},
      {0x2F800, 0x2FA1D}, {0x30000, 0x3134A}, {0x31350, 0x323AF}};
  if (s.empty() || static_cast<unsigned char>(s[0]) < 0xE1) return false;
  const auto* u = reinterpret_cast<const unsigned char*>(s.data());
  int cp;
  if (u[0] < 0xF0) {
    if (s.size() < 3) return false;
    cp = ((u[0] & 0x0F) << 12) | ((u[1] & 0x3F) << 6) | (u[2] & 0x3F);
  } else {
    if (s.size() < 4) return false;
    cp = ((u[0] & 0x07) << 18) | ((u[1] & 0x3F) << 12) | ((u[2] & 0x3F) << 6) | (u[3] & 0x3F);
  }
  for (const auto& [lo, hi] : kCjk)
    if (cp >= lo && cp <= hi) return true;
  return false;
}

struct LineGraph {
  std::vector<int> lines;  // LINE entities, level order
  std::vector<BoxF> boxes;
  std::vector<float> extra;
  std::vector<GraphEdge> edges;
};

LineGraph BuildLineGraph(const PageLayout& l) {
  constexpr int kPointsHorizontal = 10, kPointsVertical = 2;  // GcnLayoutModelSpec defaults
  LineGraph g;
  const auto kids = ChildLists(l);
  g.lines = FilterBy(l, AllDescendants(l, kids), Type::kLine);
  for (int li : g.lines) {
    const Entity& e = l.entities[li];
    RotatedBox b = e.box.value_or(RotatedBox());
    if (e.writing_direction && *e.writing_direction == 2) {  // vertical text: box from the line's side
      const SinCos r = SinCosf(b.angle * kDegToRad);
      const float w = static_cast<float>(b.width);
      b.left = RoundToInt((r.c * w) + static_cast<float>(b.left));
      b.top = RoundToInt((r.s * w) + static_cast<float>(b.top));
      std::swap(b.width, b.height);
      b.angle = NormalizeAngle(b.angle + 90.0f);
    }
    g.boxes.push_back({static_cast<float>(b.left), static_cast<float>(b.top), static_cast<float>(b.width),
                       static_cast<float>(b.height), b.angle});
    const auto words = FilterBy(l, DescendantsOf(li, kids), Type::kWord);
    float extra = 0.0f;
    if (!words.empty()) {
      const Entity& w = l.entities[words[0]];
      if (IsFirstCharCjkLetter(w.text.value_or("")))
        extra = -1.0f;
      else
        extra = static_cast<float>(w.box ? w.box->width : 0) / static_cast<float>(l.width);
    }
    g.extra.push_back(extra);
  }
  g.edges = GenerateBoxNeighborEdges(g.boxes, kPointsHorizontal, kPointsVertical);
  return g;
}

// GetBoxEdgeTensors: per box w, h (scaled), angle (rad), cos, sin, then per corner x, cos*x, sin*x, y, cos*y, sin*y,
// then the extra feature.
std::vector<float> NodeFeatures(const LineGraph& g, float scale) {
  std::vector<float> f;
  for (size_t i = 0; i < g.boxes.size(); i++) {
    const BoxF& b = g.boxes[i];
    const float rad = static_cast<float>(static_cast<double>(b.angle) * 3.141592653589793 / 180.0);
    const SinCos r = SinCosf(rad);
    f.push_back(b.w / scale);
    f.push_back(b.h / scale);
    f.push_back(rad);
    f.push_back(r.c);
    f.push_back(r.s);
    const BoxF sb{b.x / scale, b.y / scale, b.w / scale, b.h / scale, b.angle};
    float p[8];
    BoxCorners(sb, p);
    for (int k = 0; k < 4; k++) {
      const float x = p[2 * k], y = p[2 * k + 1];
      f.push_back(x);
      f.push_back(r.c * x);
      f.push_back(r.s * x);
      f.push_back(y);
      f.push_back(r.c * y);
      f.push_back(r.s * y);
    }
    f.push_back(g.extra[i]);
  }
  return f;
}

// GCN node outputs ("sort" logits, one per line); empty when there are no edges (the model is not run).
bool GcnNodeOutputs(TfliteModel* model, const LineGraph& g, float page_width, std::vector<float>* raw) {
  raw->clear();
  if (g.edges.empty()) return true;
  const std::vector<float> f = NodeFeatures(g, page_width);
  const int n = static_cast<int>(g.lines.size()), e = static_cast<int>(g.edges.size());
  if (!model->Resize(0, {e}) || !model->Resize(1, {e}) || !model->Resize(2, {n, 30}) || !model->AllocateTensors())
    return false;
  int64_t* snd = model->input(0)->data.i64;
  int64_t* rcv = model->input(1)->data.i64;
  for (int k = 0; k < e; k++) snd[k] = g.edges[k].a, rcv[k] = g.edges[k].b;
  std::memcpy(model->input(2)->data.f, f.data(), f.size() * sizeof(float));
  if (!model->Invoke()) return false;
  const TfLiteTensor* out = model->output(1);  // StatefulPartitionedCall:0, [n]
  if (!out || out->bytes / sizeof(float) < static_cast<size_t>(n)) return false;
  raw->assign(out->data.f, out->data.f + n);
  return true;
}

// ---- google_ocr::box_util on Box<float> (no curved boxes / polygons) ----

// CopyBoundingBox<goodoc::RotatedBoundingBox, Box<float>> / BoundingPolygonToBoundingBox.
BoxF ToBoxF(const RotatedBox& b) {
  return {static_cast<float>(b.left), static_cast<float>(b.top), static_cast<float>(b.width),
          static_cast<float>(b.height), NormalizeAngle(b.angle)};
}

// RotateBoundingBox<Box<float>, false>: rotates the origin corner about (cx, cy).
void RotateBoxF(BoxF* b, float cx, float cy, float a) {
  const SinCos r = SinCosf(a * kDegToRad);
  const float x = b->x, y = b->y;
  b->x = (cx + (r.c * (x - cx))) - (r.s * (y - cy));
  b->y = (cy + (r.s * (x - cx))) + (r.c * (y - cy));
  b->angle = NormalizeAngle(a + b->angle);
}

// ConvertToAxisAlignedBoundingBox<Box<float>> (PointsToAxisAlignedBox over the four corners).
void ToAxisAlignedF(BoxF* b) {
  if (b->angle == 0.0f) return;
  const SinCos r = SinCosf(b->angle * kDegToRad);
  const float p1x = b->x + (r.c * b->w), p1y = b->y + (r.s * b->w);
  const float ns = -r.s;
  const float px[4] = {b->x, p1x, p1x + (ns * b->h), b->x + (ns * b->h)};
  const float py[4] = {b->y, p1y, p1y + (r.c * b->h), b->y + (r.c * b->h)};
  float mnx = px[0], mny = py[0], mxx = px[0], mxy = py[0];
  for (int k = 1; k < 4; k++) {
    mnx = mnx > px[k] ? px[k] : mnx, mny = mny > py[k] ? py[k] : mny;
    mxx = px[k] > mxx ? px[k] : mxx, mxy = py[k] > mxy ? py[k] : mxy;
  }
  *b = {mnx, mny, mxx - mnx, mxy - mny, 0.0f};
}

// GetCenter<Box<float>>.
std::pair<float, float> CenterF(const BoxF& b) {
  const SinCos r = SinCosf(NormalizeAngle(b.angle) * kDegToRad);
  const float hw = b.w * 0.5f, hh = b.h * 0.5f;
  return {(b.x - (r.s * hh)) + (r.c * hw), (b.y + (r.c * hh)) + (r.s * hw)};
}

// ExpandBoxToContainBox<Box<float>, Box<float>> (0x1008915d8).
void ExpandBoxF(const BoxF& src, BoxF* dst) {
  if (dst->w <= 0.0f || dst->h <= 0.0f) {
    *dst = src;
    dst->angle = NormalizeAngle(src.angle);
    return;
  }
  const float sa = NormalizeAngle(src.angle), da = NormalizeAngle(dst->angle);
  if (sa == 0.0f && da == 0.0f) {
    float x = dst->x, y = dst->y;
    const float dx = x - src.x;
    if (dx > 0.0f) dst->x = src.x, dst->w = dx + dst->w, x = src.x;
    const float dy = y - src.y;
    if (dy > 0.0f) dst->y = src.y, dst->h = dy + dst->h, y = src.y;
    const float r = src.x + src.w;
    if (r > x + dst->w) dst->w = r - x;
    const float b = src.y + src.h;
    if (b > y + dst->h) dst->h = b - y;
    return;
  }
  BoxF s = src;
  RotateBoxF(&s, dst->x, dst->y, -da);
  float px[4], py[4];
  px[0] = s.x, py[0] = s.y;
  if (s.angle == 0.0f) {
    px[1] = s.x + s.w, py[1] = s.y;
    px[2] = s.x + s.w, py[2] = s.y + s.h;
    px[3] = s.x, py[3] = s.y + s.h;
  } else {
    const SinCos r = SinCosf(s.angle * kDegToRad);
    px[1] = s.x + (r.c * s.w), py[1] = s.y + (r.s * s.w);
    px[2] = px[1] - (r.s * s.h), py[2] = py[1] + (r.c * s.h);
    px[3] = s.x - (r.s * s.h), py[3] = s.y + (r.c * s.h);
  }
  const float qx = dst->x + dst->w, qy = dst->y + dst->h;
  float mnx = dst->x > px[0] ? px[0] : dst->x, mny = dst->y > py[0] ? py[0] : dst->y;
  float mxx = px[0] > qx ? px[0] : qx, mxy = py[0] > qy ? py[0] : qy;
  for (int k = 1; k < 4; k++) {
    mnx = mnx > px[k] ? px[k] : mnx, mny = mny > py[k] ? py[k] : mny;
    mxx = px[k] > mxx ? px[k] : mxx, mxy = py[k] > mxy ? py[k] : mxy;
  }
  const SinCos r = SinCosf(da * kDegToRad);
  const float ex = mnx - dst->x, ey = mny - dst->y;
  const float nx = (dst->x + (r.c * ex)) - (r.s * ey);
  const float ny = (dst->y + (r.s * ex)) + (r.c * ey);
  dst->x = nx, dst->y = ny, dst->w = mxx - mnx, dst->h = mxy - mny;
}

// Average angle of FindContainingBoxWithAverageAngle<Box<float>> / ComputeNodeBoxFromChildren.
float AverageAngleF(const std::vector<float>& a) {
  if (a.empty()) return 0.0f;
  if (a.size() == 1) return NormalizeAngle(a[0]);
  float s = 0.0f, c = 0.0f;
  for (float v : a) {
    const SinCos r = SinCosf(v * kDegToRad);
    s = s + r.s;
    c = c + r.c;
  }
  const float as = fabsf(s);
  if (fabsf(c) <= 1e-5f && as <= 1e-5f) {
    float best = 180.0f, out = 180.0f;
    for (float v : a) {
      const float j = NormalizeAngle(v);
      if (j == -out) out = fabsf(out);
      if (fabsf(j) < best) out = j, best = fabsf(j);
    }
    return out;
  }
  if (as > 1e-5f || c > static_cast<float>(a.size()) * -0.99999f) return atan2f(s, c) * kRadToDeg;
  return 180.0f;
}

// FindContainingBoxWithAverageAngle<Box<float>>.
BoxF ContainingBoxF(const std::vector<BoxF>& boxes) {
  std::vector<float> angles;
  for (const BoxF& b : boxes) angles.push_back(b.angle);
  BoxF out{0, 0, 0, 0, AverageAngleF(angles)};
  for (const BoxF& b : boxes) {
    if (out.w <= 0.0f || out.h <= 0.0f) out.x = b.x, out.y = b.y, out.w = 1.0f, out.h = 1.0f;
    ExpandBoxF(b, &out);
  }
  return out;
}

overlap::Box ToOverlapBox(const BoxF& b) { return {b.x, b.y, b.w, b.h, b.angle}; }

// SafeBoxOverlapArea<Box<float>> (0x10084d5d4) for boxes without curve / vertices (box_overlap.h).
float OverlapAreaF(const BoxF& a, const BoxF& b) { return overlap::BoxOverlapArea(ToOverlapBox(a), ToOverlapBox(b)); }

// R2BoundingBox of AxisAlignedR2BoundingBox: (xmin, ymin, xmax, ymax).
struct R2Box {
  double x0, y0, x1, y1;
};
R2Box AxisAlignedR2(const BoxF& b) {
  BoxF a = b;
  ToAxisAlignedF(&a);
  return {a.x, a.y, a.x + a.w, a.y + a.h};
}

// ---- google_ocr::box_util::RobustBoxEstimation (0x1008d8680) ----

// ---- RemoveOverlapsWordPruningStep phase 2: overlapping lines (see notes/layout.md) ----

using overlap::OverlapInfo;
using overlap::OverlapMap;

// box_util::OverlapComputer (box_overlap.h). Every caller's boxes come from BoundingPolygonToBoundingBox
// (RotatedBoundingBox via AppendRotatedBoundingBoxes, or RotatedBox<float>), which normalizes the angle like ToBoxF.
OverlapMap ComputeOverlaps(const std::vector<BoxF>& boxes, int split) {
  std::vector<overlap::Box> ob;
  for (const BoxF& b : boxes) ob.push_back(ToOverlapBox(b));
  return overlap::ComputeOverlaps(ob, split);
}

// Spec values of the rpn_gro_2024_q4 config.
constexpr double kWpIouThreshold = 0.6f, kWpOverlapThreshold = 0.6f;
constexpr float kWpCompleteWordOverlap = 0.1f, kWpCompleteConfDiff = 0.2f;
constexpr double kWpWordOverlap = 0.5;
constexpr float kWpWordConfDiff = 0.05f, kWpCommonChars = 0.6f, kWpExactMatch = 0.5f;

// CommonCharsPortion(a, b): (exact match, share of b's characters (bytes, with multiplicity) also in a).
std::pair<bool, float> CommonCharsPortion(const std::string& a, const std::string& b) {
  if (a == b) return {true, 1.0f};
  std::map<char, int> cnt;
  for (char c : b) cnt[c]++;
  for (char c : a) cnt[c]--;
  unsigned missing = 0;
  for (const auto& [c, k] : cnt) missing += k > 0 ? k : 0;
  const int len = std::max(static_cast<int>(b.size()), 1);
  return {false, 1.0f - static_cast<float>(missing) / static_cast<float>(static_cast<unsigned>(len))};
}

using PrunedWords = std::set<std::pair<int, int>>;  // (line index, word index)

float SupportWithPrunedWords(const PageLayout& l, const LineWords& lw, int li, const PrunedWords& pruned) {
  float sup = 0.0f;
  for (size_t k = 0; k < lw.words.size(); k++)
    if (!pruned.count({li, static_cast<int>(k)}))
      sup = sup + l.entities[lw.words[k]].confidence.value_or(0.0f) * static_cast<float>(lw.counts[k]);
  return sup;
}

void RecomputeLineConfidenceFromWords(PageLayout* l, const LineWords& lw, int li, const PrunedWords& pruned) {
  double sum = 0.0;
  int count = 0;
  for (size_t k = 0; k < lw.words.size(); k++) {
    if (pruned.count({li, static_cast<int>(k)})) continue;
    sum = sum + static_cast<double>(l->entities[lw.words[k]].confidence.value_or(0.0f) * static_cast<float>(lw.counts[k]));
    count += lw.counts[k];
  }
  l->entities[lw.line].confidence = count > 0 ? static_cast<float>(sum / static_cast<double>(static_cast<unsigned>(count))) : 0.0f;
}

// DoPruneOverlap: prunes words of line `pi` covered by words of line `oi`.
void DoPruneOverlap(const PageLayout& l, const std::vector<LineWords>& lines, int oi, int pi, double word_overlap,
                    double conf_diff, PrunedWords* pruned) {
  const LineWords &P = lines[pi], &O = lines[oi];
  std::vector<BoxF> boxes;
  for (int w : P.words) boxes.push_back(ToBoxF(l.entities[w].box.value_or(RotatedBox())));
  for (int w : O.words) boxes.push_back(ToBoxF(l.entities[w].box.value_or(RotatedBox())));
  const int n = static_cast<int>(P.words.size());
  const OverlapMap m = ComputeOverlaps(boxes, n);
  for (int a = 0; a < n; a++) {
    if (pruned->count({pi, a})) continue;
    const Entity& w = l.entities[P.words[a]];
    for (auto it = m.lower_bound({a, INT32_MIN}); it != m.end() && it->first.first == a; ++it) {
      const int b = it->first.second - n;
      if (pruned->count({oi, b})) continue;
      const Entity& o = l.entities[O.words[b]];
      const float ov = it->second.over_first;
      if (static_cast<double>(kWpExactMatch) >= ov && word_overlap >= ov) continue;
      const auto [exact, portion] = CommonCharsPortion(o.text.value_or(""), w.text.value_or(""));
      const bool exact_test = static_cast<double>(kWpExactMatch) < ov && exact;
      bool prune;
      if (word_overlap >= ov) {
        prune = exact_test;
      } else {
        const bool keep = conf_diff <= (w.confidence.value_or(0.0f) - o.confidence.value_or(0.0f)) &&
                          kWpCommonChars > portion;
        prune = !keep || exact_test;
      }
      if (prune) pruned->insert({pi, a});
    }
  }
}

// DeleteTextEntitiesWithNoSymbols(layout, retain_empty_paragraphs = false, false): bottom-up over the level order,
// non-symbol TEXT entities whose children are all deleted (or that have none) are deleted. Returns the old -> new
// index map.
std::vector<int> DeleteTextEntitiesWithNoSymbols(PageLayout* l) {
  const auto kids = ChildLists(*l);
  const auto order = AllDescendants(*l, kids);
  std::vector<bool> del(l->entities.size(), false);
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const Entity& e = l->entities[*it];
    const int t = static_cast<int>(e.type);
    if (e.content_type.value_or(0) > 1 || t > 4 || t == 1) continue;
    bool all = true;
    for (int c : kids[*it]) all &= del[c];
    if (all) del[*it] = true;
  }
  return DeleteWithDescendants(l, del);
}

// Polygon of type curved_box.
bool IsCurved(const Entity& e) { return e.box && e.box->curve; }

void PruneOverlappingLines(PageLayout* l, const std::vector<LineWords>& lines,
                           const std::vector<std::pair<float, int>>& order) {
  std::vector<BoxF> boxes;
  for (const LineWords& lw : lines) boxes.push_back(ToBoxF(l->entities[lw.line].box.value_or(RotatedBox())));
  const OverlapMap overlaps = ComputeOverlaps(boxes, -1);
  std::set<int> removed;
  PrunedWords pruned;
  std::set<std::pair<int, int>> processed;
  auto conf = [&](int li) { return l->entities[lines[li].line].confidence.value_or(0.0f); };
  // skip_curved (spec default true): curved lines neither prune nor get pruned.
  auto curved = [&](int li) { return IsCurved(l->entities[lines[li].line]); };
  for (const auto& [total, i] : order) {
    if (removed.count(i) || curved(i)) continue;
    // FindCoveredBoxesType
    const Entity& ei = l->entities[lines[i].line];
    std::vector<std::pair<int, int>> covered;
    for (auto it = overlaps.lower_bound({i, INT32_MIN}); it != overlaps.end() && it->first.first == i; ++it) {
      const int j = it->first.second;
      if (removed.count(j) || curved(j)) continue;
      const Entity& ej = l->entities[lines[j].line];
      const bool same_orientation = ei.orientation.value_or(0) == ej.orientation.value_or(0);
      const bool same_direction = ei.writing_direction.value_or(0) == ej.writing_direction.value_or(0);
      const bool same = (ei.writing_direction.value_or(0) == 2 || ej.writing_direction.value_or(0) == 2)
                            ? same_orientation && same_direction
                            : same_orientation;
      const OverlapInfo& o = it->second;
      int type = 0;
      if (same) {
        if (o.iou > 0.95f) type = 2;
        else if (o.over_first > 0.95f) type = 6;
        else if (o.over_second > 0.95f) type = 9;
        else if (kWpIouThreshold < o.iou) type = 1;
        else if (kWpOverlapThreshold < o.over_first) type = 5;
        else if (kWpOverlapThreshold < o.over_second) type = 8;
      } else {
        if (o.iou > 0.95f) type = 3;
        else if (o.over_first > 0.95f) type = 7;
        else if (o.over_second > 0.95f) type = 10;
        else if (kWpIouThreshold < o.iou) type = 4;
        else if (kWpOverlapThreshold < o.over_second) type = 8;
      }
      if (type) covered.emplace_back(j, type);
    }
    if (covered.empty()) continue;
    const float sup_i = SupportWithPrunedWords(*l, lines[i], i, pruned);
    std::map<int, float> sup;
    for (const auto& [j, type] : covered) {
      if (removed.count(j) || removed.count(i)) continue;
      if (!processed.insert({std::min(i, j), std::max(i, j)}).second) continue;
      sup[j] = SupportWithPrunedWords(*l, lines[j], j, pruned);
    }
    for (const auto& [j, type] : covered) {
      const auto s = sup.find(j);
      if (s == sup.end()) continue;
      const float sup_j = s->second;
      if (type == 7 && sup_j > sup_i) {
        removed.insert(i);
        continue;
      }
      if ((type & ~1) == 2) {
        const Entity &a = l->entities[lines[i].line], &b = l->entities[lines[j].line];
        if (sup_i == sup_j)
          removed.insert(a.direction_confidence.value_or(0.0f) >= b.direction_confidence.value_or(0.0f) ? j : i);
        else
          removed.insert(sup_j <= sup_i ? j : i);
        continue;
      }
      const bool complete = (0x6D0u >> type) & 1;
      const double word_overlap = complete ? static_cast<double>(kWpCompleteWordOverlap) : kWpWordOverlap;
      const double conf_diff = complete ? kWpCompleteConfDiff : kWpWordConfDiff;
      const int prune = sup_i <= sup_j ? i : j, other = prune == i ? j : i;
      DoPruneOverlap(*l, lines, other, prune, word_overlap, conf_diff, &pruned);
      RecomputeLineConfidenceFromWords(l, lines[prune], prune, pruned);
      if (conf(prune) == 0.0f) removed.insert(prune);
    }
  }
  if (removed.empty() && pruned.empty()) return;
  std::vector<bool> del(l->entities.size(), false);
  for (int i : removed) del[lines[i].line] = true;
  for (const auto& [li, k] : pruned) del[lines[li].words[k]] = true;
  DeleteWithDescendants(l, del);
  DeleteTextEntitiesWithNoSymbols(l);
}

// ---- RemoveOverlappingWithCurvedBoxesStep (0x100869b1c) ----

// Spec threshold (field 2); the config carries the spec without fields.
constexpr double kCurvedOverlapThreshold = 0.3;

// LineSymbolsArea (0x10086b600): sum of the symbol box areas (Box<float>).
double LineSymbolsArea(const PageLayout& l, int line, const std::vector<std::vector<int>>& kids) {
  double area = 0.0;
  for (int s : FilterBy(l, DescendantsOf(line, kids), Type::kSymbol)) {
    const BoxF b = ToBoxF(l.entities[s].box.value_or(RotatedBox()));
    area = area + static_cast<double>(b.w * b.h);
  }
  return area;
}

// Overlap area of the symbols of word `wa` with those of word `wb`: sum of I/area(a) * area(a) over the symbol
// pairs, area(a) from the integer box.
double SymbolsOverlapArea(const PageLayout& l, int wa, int wb, const std::vector<std::vector<int>>& kids) {
  const auto sa = FilterBy(l, DescendantsOf(wa, kids), Type::kSymbol);
  const auto sb = FilterBy(l, DescendantsOf(wb, kids), Type::kSymbol);
  std::vector<BoxF> boxes;
  for (int s : sa) boxes.push_back(ToBoxF(l.entities[s].box.value_or(RotatedBox())));
  for (int s : sb) boxes.push_back(ToBoxF(l.entities[s].box.value_or(RotatedBox())));
  const int n = static_cast<int>(sa.size());
  const OverlapMap m = ComputeOverlaps(boxes, n);
  double area = 0.0;
  for (int k = 0; k < n; k++) {
    const RotatedBox b = l.entities[sa[k]].box.value_or(RotatedBox());
    const float ak = static_cast<float>(b.height * b.width);
    for (auto it = m.lower_bound({k, INT32_MIN}); it != m.end() && it->first.first == k; ++it)
      area = area + static_cast<double>(it->second.over_first * ak);
  }
  return area;
}

// Curved lines, best (confidence + direction_confidence / 100) * symbols first, against every later line in that
// order: words whose symbols overlap are marked, and a line whose symbol overlap exceeds the threshold share of the
// smaller line's symbol area is removed (a straight one only by its marked words, unless all are marked).
void RemoveOverlappingWithCurvedBoxes(PageLayout* l) {
  const auto kids = ChildLists(*l);
  const auto lines = FilterBy(*l, AllDescendants(*l, kids), Type::kLine);
  const int n = static_cast<int>(lines.size());
  std::vector<BoxF> boxes;
  std::vector<std::pair<float, int>> order;
  int num_curved = 0;
  for (int i = 0; i < n; i++) {
    const Entity& e = l->entities[lines[i]];
    boxes.push_back(ToBoxF(e.box.value_or(RotatedBox())));
    float score = 0.0f;
    if (IsCurved(e)) {
      const int symbols = static_cast<int>(FilterBy(*l, DescendantsOf(lines[i], kids), Type::kSymbol).size());
      score = static_cast<float>(
          (static_cast<double>(e.direction_confidence.value_or(0.0f)) * 0.01 +
           static_cast<double>(e.confidence.value_or(0.0f))) *
          static_cast<double>(symbols));
      num_curved++;
    }
    order.emplace_back(score, i);
  }
  if (num_curved == 0) return;
  std::sort(order.begin(), order.end(), std::greater<std::pair<float, int>>());
  const OverlapMap overlaps = ComputeOverlaps(boxes, -1);
  std::vector<bool> removed(n, false);
  std::vector<std::vector<bool>> marks(n);
  int num_removed = 0;
  for (int a = 0; a < num_curved; a++) {
    const int ia = order[a].second;
    if (removed[ia] || a + 1 >= n) continue;
    for (int b = a + 1; b < n; b++) {
      const int ib = order[b].second;
      const auto o = overlaps.find({ia, ib});
      if (o == overlaps.end() || o->second.over_first == 0.0f) continue;
      const auto wa = FilterBy(*l, DescendantsOf(lines[ia], kids), Type::kWord);
      const auto wb = FilterBy(*l, DescendantsOf(lines[ib], kids), Type::kWord);
      if (marks[ib].empty()) marks[ib].resize(wb.size(), false);
      std::vector<BoxF> wboxes;
      for (int w : wa) wboxes.push_back(ToBoxF(l->entities[w].box.value_or(RotatedBox())));
      for (int w : wb) wboxes.push_back(ToBoxF(l->entities[w].box.value_or(RotatedBox())));
      const int na = static_cast<int>(wa.size());
      const OverlapMap wm = ComputeOverlaps(wboxes, na);
      double total = 0.0;
      for (int k = 0; k < na; k++) {
        for (auto it = wm.lower_bound({k, INT32_MIN}); it != wm.end() && it->first.first == k; ++it) {
          const int j = it->first.second - na;
          const double area = SymbolsOverlapArea(*l, wa[k], wb[j], kids);
          total = total + area;
          if (area > 0.0) marks[ib][j] = true;
        }
      }
      const double area_a = LineSymbolsArea(*l, lines[ia], kids), area_b = LineSymbolsArea(*l, lines[ib], kids);
      double smaller = area_b >= area_a ? area_a : area_b;
      if (smaller < 0.0001) smaller = 0.0001;
      if (kCurvedOverlapThreshold < total / smaller) removed[ib] = true, num_removed++;
    }
  }
  if (num_removed == 0) return;
  std::vector<bool> del(l->entities.size(), false);
  bool recompute = false;
  for (int i = 0; i < n; i++) {
    if (!removed[i]) continue;
    const bool all = std::all_of(marks[i].begin(), marks[i].end(), [](bool m) { return m; });
    if (IsCurved(l->entities[lines[i]]) || all) {
      del[lines[i]] = true;
      continue;
    }
    const auto words = FilterBy(*l, DescendantsOf(lines[i], kids), Type::kWord);
    for (size_t k = 0; k < words.size(); k++)
      if (marks[i][k]) del[words[k]] = true;
    recompute = true;
  }
  DeleteWithDescendants(l, del);
  DeleteTextEntitiesWithNoSymbols(l);
  if (recompute) {
    RecomputeBoxesFromSymbols(l);
    RecomputeWordTextContent(l);
    RecomputeConfidencesFromSymbols(l);
  }
}

// rotation_util::RobustAngleEstimation<float>: votes the dominant axis direction, then the median of the angles
// relative to it.
float RobustAngleEstimation(const std::vector<float>& angles) {
  if (angles.empty()) return 0.0f;
  static constexpr float kDirs[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  int votes[4] = {0, 0, 0, 0};
  std::vector<std::pair<float, float>> dirs;
  for (float a : angles) {
    const SinCos r = SinCosf(a * kDegToRad);
    int best = 0;
    for (int k = 1; k < 4; k++) {
      const float cur = ((r.c * kDirs[best][0]) + 0.0f) + (r.s * kDirs[best][1]);
      const float cand = ((r.c * kDirs[k][0]) + 0.0f) + (r.s * kDirs[k][1]);
      if (cur < cand) best = k;
    }
    votes[best]++;
    dirs.emplace_back(r.c, r.s);
  }
  int dom = 0;
  for (int k = 1; k < 4; k++)
    if (votes[dom] < votes[k]) dom = k;
  const float dx = kDirs[dom][0], dy = kDirs[dom][1];
  struct Keyed {
    float c, s, key;
  };
  std::vector<Keyed> keyed;
  for (const auto& [c, s] : dirs) keyed.push_back({c, s, atan2f((dx * s) - (dy * c), ((dx * c) + 0.0f) + (dy * s))});
  std::sort(keyed.begin(), keyed.end(), [](const Keyed& a, const Keyed& b) { return a.key < b.key; });
  const size_t n = keyed.size();
  float mc = keyed[n / 2].c, ms = keyed[n / 2].s;
  if (n % 2 == 0) mc = (mc + keyed[n / 2 - 1].c) * 0.5f, ms = (ms + keyed[n / 2 - 1].s) * 0.5f;
  if (ms == 0.0f && mc == 0.0f) return 0.0f;
  return atan2f(ms, mc) * kRadToDeg;
}

// Box of an entity from its descendant symbols' boxes and word break boxes (descendant level order), with the
// robust angle. Lines/words whose part angles spread (cos < 0.99) become curved in the original (not supported).
// google_ocr::CreateCurveRepresentation (0x1008daf60) with use_max_height = true and a moving-average window of 1
// (the call in RobustBoxEstimation), followed by the mid line as RobustBoxEstimation stores it: the midpoints of the
// top / bottom polylines without consecutive duplicates. Boxes (and the frame) are turned clockwise first for
// top-to-bottom writing. Mirrors the detector's version (detector.cc), which uses a window of 6.
Curve CurveFromBoxes(const std::vector<RotatedBox>& boxes, const RotatedBox& robust, bool top_to_bottom) {
  RotatedBox frame = robust;
  if (top_to_bottom) TurnClockwise(&frame);
  std::vector<std::pair<float, float>> pts;
  float t = 0.0f;
  for (RotatedBox b : boxes) {
    if (top_to_bottom) TurnClockwise(&b);
    const float x = static_cast<float>(b.left), y = static_cast<float>(b.top);
    const float w = static_cast<float>(b.width), h = static_cast<float>(b.height);
    float p[8];
    p[0] = x, p[1] = y;
    if (b.angle == 0.0f) {
      p[2] = x + w, p[3] = y, p[4] = x + w, p[5] = h + y, p[6] = x, p[7] = h + y;
    } else {
      const SinCos r = SinCosf(b.angle * kDegToRad);
      p[2] = (r.c * w) + x, p[3] = (r.s * w) + y;
      p[4] = p[2] - (r.s * h), p[5] = (r.c * h) + p[3];
      p[6] = x - (r.s * h), p[7] = (r.c * h) + y;
    }
    pts.push_back({(p[0] + p[6]) * 0.5f, (p[1] + p[7]) * 0.5f});
    pts.push_back({(p[2] + p[4]) * 0.5f, (p[3] + p[5]) * 0.5f});
    if (t < h) t = h;
  }
  const SinCos dir = SinCosf(frame.angle * kDegToRad);
  const float ox = static_cast<float>(frame.left), oy = static_cast<float>(frame.top);
  auto key = [&](const std::pair<float, float>& q) { return (dir.c * (q.first - ox) + 0.0f) + dir.s * (q.second - oy); };
  std::sort(pts.begin(), pts.end(),
            [&](const std::pair<float, float>& a, const std::pair<float, float>& b) { return key(a) < key(b); });
  std::vector<std::pair<float, float>> mid;
  const size_t n = pts.size();
  if (n > 1) {
    mid.push_back(pts[0]);  // window 1: the moving average is the point itself
    for (size_t i = 1; i + 1 < n; i++) {
      const float dx = pts[i].first - mid.back().first, dy = pts[i].second - mid.back().second;
      if (dx * dx + dy * dy > t * (t * 0.25f)) mid.push_back(pts[i]);
    }
  }
  if (n > 0) mid.push_back(pts[n - 1]);
  std::vector<std::pair<float, float>> top, bottom;
  const float half = t * 0.5f;
  for (size_t i = 1; i < mid.size(); i++) {
    const auto &a = mid[i - 1], &b = mid[i];
    const SinCos r = SinCosf(atan2f(b.first - a.first, a.second - b.second));
    const float cx = r.c * half, cy = r.s * half;
    top.push_back({a.first - cx, a.second - cy});
    bottom.push_back({cx + a.first, cy + a.second});
    if (i + 1 == mid.size()) {
      top.push_back({b.first - cx, b.second - cy});
      bottom.push_back({cx + b.first, cy + b.second});
    }
  }
  Curve c;
  c.thickness = static_cast<double>(t);
  c.top_to_bottom = top_to_bottom;
  for (size_t k = 0; k < top.size() && k < bottom.size(); k++) {
    const float x = (top[k].first + bottom[k].first) * 0.5f, y = (top[k].second + bottom[k].second) * 0.5f;
    if (k > 0 && static_cast<float>(c.points.back().first) == x && static_cast<float>(c.points.back().second) == y)
      continue;
    c.points.emplace_back(x, y);
  }
  return c;
}

void RobustBoxEstimation(PageLayout* l, int e, const std::vector<std::vector<int>>& kids) {
  std::vector<RotatedBox> boxes;
  for (int d : DescendantsOf(e, kids)) {
    const Entity& ent = l->entities[d];
    if (ent.type == Type::kSymbol) boxes.push_back(ent.box.value_or(RotatedBox()));
    if (ent.type == Type::kWord && ent.word_break && ent.word_break->box) boxes.push_back(*ent.word_break->box);
  }
  if (boxes.empty()) return;
  std::vector<float> angles;
  for (const auto& b : boxes) angles.push_back(b.angle);
  const float angle = RobustAngleEstimation(angles);
  RotatedBox out;
  out.angle = angle;
  for (const auto& b : boxes) {
    if (out.width < 1 || out.height <= 0) out.left = b.left, out.top = b.top, out.width = 1, out.height = 1;
    ExpandBoxToContainBox(b, &out);
  }
  Entity& ent = l->entities[e];
  if (ent.type == Type::kWord || ent.type == Type::kLine) {
    const SinCos r = SinCosf(angle * kDegToRad);
    float min_cos = 1.0f;
    for (float a : angles) {
      const SinCos ra = SinCosf(a * kDegToRad);
      const float d = (r.s * ra.s) + ((r.c * ra.c) + 0.0f);
      if (d < min_cos) min_cos = d;
    }
    if (static_cast<double>(min_cos) < 0.99) out.curve = CurveFromBoxes(boxes, out, ent.writing_direction.value_or(0) == 2);
  }
  ent.box = out;
}

// ---- SortChildrenOfEntity (step_utils.cc, 0x100ab62c0) with CompareByBox + RerankByPriority ----

// CanonicalizeWithMinimumAbsoluteAngle<goodoc::RotatedBoundingBox>: turns the box by 90 degree steps until its
// angle is in [-45, 45].
void CanonicalizeMinAbsAngle(RotatedBox* b) {
  float i = b->angle;
  const float a = NormalizeAngle(i);
  float v4 = a <= 45.0f ? a : a + -360.0f;
  while (v4 < -45.0f) {
    const SinCos r = SinCosf(i * kDegToRad);
    const float w = static_cast<float>(b->width);
    const int l = RoundToInt((r.c * w) + static_cast<float>(b->left));
    const int t = RoundToInt((r.s * w) + static_cast<float>(b->top));
    b->left = l, b->top = t;
    std::swap(b->width, b->height);
    i = NormalizeAngle(i + 90.0f);
    v4 = v4 + 90.0f;
  }
  b->angle = v4;
}

struct CompareByBox {
  int textline_order = 0, writing_direction = 0;
  struct Cache {
    bool defined = false;
    RotatedBox box;
  };
  std::vector<Cache> cache;

  // a before b
  bool Less(int ia, int ib) const {
    const Cache& A = cache[ia];
    if (!A.defined) return false;
    const Cache& B = cache[ib];
    if (!B.defined) return true;
    const int la = A.box.left, ta = A.box.top, wa = A.box.width, ha = A.box.height;
    const int lb = B.box.left, tb = B.box.top, wb = B.box.width, hb = B.box.height;
    if ((textline_order & ~1) == 2) {
      const int cb = static_cast<int>(static_cast<double>(hb) * 0.5 + static_cast<double>(tb));
      const int ca = static_cast<int>(static_cast<double>(ha) * 0.5 + static_cast<double>(ta));
      if (ta + ha <= cb || ta > cb || tb + hb <= ca || tb > ca)  // not on the same row
        return textline_order == 2 ? ca < cb : ca > cb;
      return writing_direction != 0 ? la + wa > lb + wb : la < lb;
    }
    const int ca = static_cast<int>(static_cast<double>(wa) * 0.5 + static_cast<double>(la));
    const int cb = static_cast<int>(static_cast<double>(wb) * 0.5 + static_cast<double>(lb));
    if (la + wa < cb || la > cb || lb + wb < ca || lb > ca) return textline_order != 0 ? ca > cb : ca < cb;
    return writing_direction == 2 ? ta < tb : ta > tb;
  }
};

// Reorders the children of e: each child is ranked by how many siblings precede it (stable).
void SortChildrenOfEntity(PageLayout* l, int e, std::vector<int>* track) {
  const auto kids = ChildLists(*l);
  const std::vector<int>& ch = kids[e];
  const Entity& p = l->entities[e];
  if (!p.box) return;  // BoundingPolygonToBoundingBox fails
  const float neg_angle = -ToBoxF(*p.box).angle;
  CompareByBox cmp;
  cmp.textline_order = p.textline_order.value_or(0);
  cmp.writing_direction = p.writing_direction.value_or(0);
  for (int c : ch) {
    CompareByBox::Cache k;
    const Entity& ce = l->entities[c];
    k.defined = ce.box.has_value();
    if (k.defined) {
      k.box = *ce.box;
      const SinCos r = SinCosf(neg_angle * kDegToRad);
      const float x = static_cast<float>(k.box.left), y = static_cast<float>(k.box.top);
      const float ny = ((r.s * x) + 0.0f) + (r.c * y);
      k.box.left = RoundToInt(((r.c * x) + 0.0f) - (r.s * y));
      k.box.top = RoundToInt(ny);
      k.box.angle = NormalizeAngle(neg_angle + k.box.angle);
      CanonicalizeMinAbsAngle(&k.box);
    }
    cmp.cache.push_back(k);
  }
  const int n = static_cast<int>(ch.size());
  std::vector<int> prio(n, 0);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i != j && cmp.Less(i, j)) prio[j]++;
  std::vector<int> idx(n);
  for (int i = 0; i < n; i++) idx[i] = i;
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return prio[a] < prio[b]; });
  // PageLayout::ReorderChildren: the sorted children take the children's slots.
  std::vector<int> order(l->entities.size());
  for (size_t i = 0; i < order.size(); i++) order[i] = static_cast<int>(i);
  for (int k = 0; k < n; k++) order[ch[k]] = ch[idx[k]];
  const auto remap = Reorder(l, order);
  if (track)
    for (int& t : *track) t = t >= 0 ? remap[t] : t;
}

// ---- ClusterSortGcnStep::ClusterAndSortParagraphs (0x100889440) ----

struct OrderNode {
  int paragraph = -1;  // leaf: paragraph entity
  int parent = -1;
  bool column = false;  // children ordered by columns (sort prediction)
  std::vector<int> children;
  bool sorted = false;
  BoxF box;
  float area = 0;
  float sum_line_height = 0;
  int num_lines = 0;
  bool rtl = false;
};

void ComputeNodeBoxFromChildren(std::vector<OrderNode>* nodes, int i, const float* ref_angle) {
  OrderNode& n = (*nodes)[i];
  n.sum_line_height = 0;
  n.num_lines = 0;
  std::vector<float> angles;
  for (int c : n.children) {
    float a = (*nodes)[c].box.angle;
    if (ref_angle) {
      while (true) {
        float d = fmodf(fabsf(a - *ref_angle), 360.0f);
        if (d > 180.0f) d = 360.0f - d;
        if (d <= 45.0f) break;
        a = a + 90.0f;
      }
    }
    angles.push_back(a);
    n.sum_line_height = (*nodes)[c].sum_line_height + n.sum_line_height;
    n.num_lines += (*nodes)[c].num_lines;
  }
  BoxF box{0, 0, 0, 0, AverageAngleF(angles)};
  for (int c : n.children) {
    const BoxF& cb = (*nodes)[c].box;
    if (box.w <= 0.0f || box.h <= 0.0f) box.x = cb.x, box.y = cb.y, box.w = 1.0f, box.h = 1.0f;
    ExpandBoxF(cb, &box);
  }
  n.box = box;
  n.area = box.w * box.h;
}

// (anonymous)::SortChildren (0x10088e534): children boxes in the node's frame, sorted by row (column) center;
// a skyline over their (shrunk) extents gives "is above" edges; topological order, leftmost (topmost) first.
void SortOrderChildren(std::vector<OrderNode>* nodes, int i) {
  OrderNode& n = (*nodes)[i];
  if (n.sorted || n.children.size() < 2) return;
  const auto [cx, cy] = CenterF(n.box);
  struct SortBox {
    int node;
    float key1, key2, lo, hi;
    int indegree = 0;
    std::vector<int> next;
  };
  std::vector<SortBox> sb;
  for (int c : n.children) {
    BoxF b = (*nodes)[c].box;
    RotateBoxF(&b, cx, cy, -n.box.angle);
    ToAxisAlignedF(&b);
    float start, end, key1, key2;
    if (n.column) {
      start = b.y, end = b.y + b.h;
      key2 = b.y + (b.h * 0.5f);
      const float xc = b.x + (b.w * 0.5f);
      key1 = n.rtl ? -xc : xc;
    } else {
      start = b.x, end = b.x + (b.w * 1.0f);
      key1 = b.y + (b.h * 0.5f);
      const float xc = b.x + (0.5f * b.w);
      key2 = n.rtl ? -xc : xc;
    }
    double m = static_cast<double>(n.sum_line_height) * 0.25 / static_cast<double>(n.num_lines);
    const double len = static_cast<double>(end - start) * 0.25;
    if (len < m) m = len;
    const float mf = static_cast<float>(m);
    sb.push_back({c, key1, key2, start + mf, end - mf, 0, {}});
  }
  std::stable_sort(sb.begin(), sb.end(), [](const SortBox& a, const SortBox& b) { return a.key1 < b.key1; });
  // skyline: key -> box owning [key, next key) (-1: none)
  std::map<float, int> sky;
  for (int k = 0; k < static_cast<int>(sb.size()); k++) {
    const float lo = sb[k].lo, hi = sb[k].hi;
    auto it = sky.lower_bound(lo);
    if (it != sky.begin() && it != sky.end() && it->first > lo) --it;
    const auto first = it;
    int last = -1;
    for (; it != sky.end() && !(it->first > hi); ++it) {
      const int owner = it->second;
      if (owner >= 0) sb[owner].next.push_back(k), sb[k].indegree++;
      last = owner;
    }
    auto erase_from = first;
    if (erase_from != sky.end() && erase_from->first < lo) ++erase_from;
    sky.erase(erase_from, it);
    sky[lo] = k;
    sky[hi] = last;
  }
  using Item = std::pair<float, int>;
  auto greater = [](const Item& a, const Item& b) { return a.first > b.first; };
  std::vector<Item> heap;
  for (int k = 0; k < static_cast<int>(sb.size()); k++)
    if (sb[k].indegree == 0) heap.emplace_back(sb[k].key2, k), std::push_heap(heap.begin(), heap.end(), greater);
  n.children.clear();
  while (!heap.empty()) {
    std::pop_heap(heap.begin(), heap.end(), greater);
    const int k = heap.back().second;
    heap.pop_back();
    n.children.push_back(sb[k].node);
    for (int s : sb[k].next)
      if (--sb[s].indegree == 0) heap.emplace_back(sb[s].key2, s), std::push_heap(heap.begin(), heap.end(), greater);
  }
  n.sorted = true;
}

void TraverseOrderTree(const std::vector<OrderNode>& nodes, int i, std::vector<int>* paragraphs) {
  if (nodes[i].children.empty()) {
    paragraphs->push_back(nodes[i].paragraph);
    return;
  }
  for (int c : nodes[i].children) TraverseOrderTree(nodes, c, paragraphs);
}

// box_util::RotateOrientationClockwise<Box<float>>: the same rectangle with its origin corner moved q quadrants
// clockwise.
void RotateOrientationClockwise(BoxF* b, int q) {
  q = q <= 0 ? -(-q & 3) : (q & 3);
  if (q == 0) return;
  const float w = b->w, h = b->h;
  if (q & 1) b->w = h, b->h = w;
  const SinCos r = SinCosf(b->angle * kDegToRad);
  float x, y;
  if (q == 3) {
    x = b->x - (h * r.s);
    y = b->y + (h * r.c);
  } else if (q == 2) {
    x = (b->x + (w * r.c)) - (h * r.s);
    y = (h * r.c) + (b->y + (w * r.s));
  } else {
    x = b->x + (w * r.c);
    y = b->y + (w * r.s);
  }
  b->x = x, b->y = y;
  b->angle = NormalizeAngle(b->angle + static_cast<float>(static_cast<unsigned>(90 * q)));
}

// GetTextlineOrderBox (cluster_sort_gcn_step.cc): the paragraph box turned so that text lines run top to bottom:
// textline order RIGHT_TO_LEFT -> 1 quadrant, LEFT_TO_RIGHT -> 3.
BoxF TextlineOrderBox(const Entity& p) {
  BoxF b = ToBoxF(p.box.value_or(RotatedBox()));
  const int order = p.textline_order.value_or(0);
  if (order == 1) RotateOrientationClockwise(&b, 1);
  if (order == 0) RotateOrientationClockwise(&b, 3);
  return b;
}

// Returns the paragraphs in reading order. `lines`: the GCN's lines (level order), `column` their predictions.
// BoxArea<Box<float>> of a curved box (0x10088d310): CurvedBoxToPoints (0x100202230) offsets the deduplicated
// float mid-line by +-thickness / 2 along the normals (averaged segment angles inside, the end segments' at the
// ends); AppendBoxPolygons cuts that outline into one quad per segment and PolygonSetArea sums their areas.
float CurvedBoxAreaF(const Curve& curve) {
  std::vector<std::pair<float, float>> p;
  for (const auto& [x, y] : curve.points) {
    const std::pair<float, float> q{static_cast<float>(x), static_cast<float>(y)};
    if (p.empty() || q.first != p.back().first || q.second != p.back().second) p.push_back(q);
  }
  const int n = static_cast<int>(p.size());
  if (n < 2) return 0.0f;
  std::vector<float> angles;
  for (int i = 1; i < n; i++) angles.push_back(atan2f(p[i].second - p[i - 1].second, p[i].first - p[i - 1].first));
  constexpr double kMinusHalfPi = -1.5707963267948966;
  const float h = static_cast<float>(curve.thickness) * 0.5f;
  std::vector<std::pair<float, float>> top, bottom;
  for (int i = 0; i < n; i++) {
    float a;
    if (i == 0) a = angles[0];
    else if (i == n - 1) a = angles.back();
    else {
      const float a0 = angles[i - 1], a1 = angles[i];
      const float sum = a0 + a1;
      if (static_cast<double>(fabsf(a0 - a1)) <= M_PI) a = sum * 0.5f;
      else a = static_cast<float>((static_cast<double>(sum) + 2.0 * M_PI) * 0.5);
    }
    const SinCos2 r = SinCosD(static_cast<double>(a) + kMinusHalfPi);
    const float dx = h * static_cast<float>(r.c), dy = h * static_cast<float>(r.s);
    top.push_back({p[i].first + dx, p[i].second + dy});
    bottom.push_back({p[i].first - dx, p[i].second - dy});
  }
  // PolygonArea: |sum of cross(prev, cur) * 0.5| around the quad, accumulated in double and rounded per term.
  auto quad_area = [](const std::pair<float, float> (&q)[4]) {
    float acc = 0.0f;
    for (int k = 0; k < 4; k++) {
      const auto& a = q[k];
      const auto& b = q[(k + 1) % 4];
      const float c = (a.first * b.second) - (a.second * b.first);
      acc = static_cast<float>((static_cast<double>(c) * 0.5) + static_cast<double>(acc));
    }
    return fabsf(acc);
  };
  float area = 0.0f;
  for (int k = 0; k + 1 < n; k++) {
    const std::pair<float, float> q[4] = {top[k], top[k + 1], bottom[k + 1], bottom[k]};
    area = area + quad_area(q);
  }
  return area;
}

std::vector<int> OrderParagraphs(const PageLayout& l, const std::vector<int>& lines, const std::vector<GraphEdge>& edges,
                                 const std::vector<bool>& pred) {
  constexpr float kChildAreaOverlappingThreshold = 0.25f;
  const int n = static_cast<int>(lines.size());
  // Per paragraph: line area predicted column-wise vs not; a line's bit is its paragraph's majority.
  std::vector<std::pair<float, float>> para_area(l.entities.size(), {0.0f, 0.0f});
  for (int i = 0; i < n; i++) {
    const Entity& e = l.entities[lines[i]];
    if (e.parent < 0 || !e.box) continue;
    const BoxF b = ToBoxF(*e.box);
    const float a = e.box->curve && e.box->curve->points.size() >= 2 ? CurvedBoxAreaF(*e.box->curve) : b.w * b.h;
    auto& pa = para_area[e.parent];
    if (pred[i]) pa.first = a + pa.first;
    else pa.second = a + pa.second;
  }
  std::vector<bool> bit(n);
  for (int i = 0; i < n; i++) {
    const int p = l.entities[lines[i]].parent;
    bit[i] = p >= 0 && !(para_area[p].first <= para_area[p].second);
  }
  struct EdgeMeta {
    int to, dir;
    bool close, near;  // union height < 2 * min height; < 3 * min height + both heights
  };
  std::vector<std::vector<EdgeMeta>> adj(n);
  for (const GraphEdge& ge : edges) {
    const Entity& A = l.entities[lines[ge.a]];
    const Entity& B = l.entities[lines[ge.b]];
    if (A.writing_direction.value_or(0) != B.writing_direction.value_or(0)) continue;
    const RotatedBox ra = A.box.value_or(RotatedBox()), rb = B.box.value_or(RotatedBox());
    float d = fmodf(fabsf(ra.angle - rb.angle), 360.0f);
    if (d > 180.0f) d = 360.0f - d;
    if (d > 30.0f) continue;
    const int minh = rb.height >= ra.height ? ra.height : rb.height;
    const BoxF fa = ToBoxF(ra), fb = ToBoxF(rb);
    const BoxF u = ContainingBoxF({fa, fb});
    int dir = 0;
    if (u.w <= fa.w + fb.w) {
      const SinCos r = SinCosf((u.angle + 90.0f) * kDegToRad);
      const auto [ax, ay] = CenterF(fa);
      const auto [bx, by] = CenterF(fb);
      dir = ((r.s * (ay - by)) + ((r.c * (ax - bx)) + 0.0f)) <= 0.0f ? -1 : 1;
    }
    BoxF ra2 = fa, rb2 = fb;
    RotateBoxF(&ra2, 0.0f, 0.0f, -u.angle);
    ToAxisAlignedF(&ra2);
    RotateBoxF(&rb2, 0.0f, 0.0f, -u.angle);
    ToAxisAlignedF(&rb2);
    const float mh = static_cast<float>(minh);
    const bool near = u.h < (mh * 3.0f) + (ra2.h + rb2.h);
    const bool close = u.h < mh + mh;
    adj[ge.a].push_back({ge.b, dir, close, near});
    adj[ge.b].push_back({ge.a, -dir, close, near});
  }

  std::vector<OrderNode> nodes;
  std::vector<int> node_of(l.entities.size(), -1);
  std::vector<int> roots;
  std::vector<std::pair<R2Box, int>> rtree;  // every inserted (box, node); stale entries stay
  std::vector<bool> unvisited(n, true);
  for (int i = 0; i < n; i++) {
    if (!unvisited[i]) continue;
    const bool b = bit[i];
    std::deque<int> q{i};
    unvisited[i] = false;
    std::vector<int> comp;
    while (!q.empty()) {
      const int j = q.front();
      q.pop_front();
      comp.push_back(j);
      for (const EdgeMeta& e : adj[j]) {
        if (unvisited[e.to] && b == bit[e.to] && e.near) {
          q.push_back(e.to);
          unvisited[e.to] = false;
        } else if (b && e.dir != 0) {
          for (const EdgeMeta& e2 : adj[e.to]) {
            if (unvisited[e2.to] && e2.dir + e.dir == 0 && bit[e2.to] && e2.close) {
              q.push_back(e2.to);
              unvisited[e2.to] = false;
            }
          }
        }
      }
    }
    const int before = static_cast<int>(nodes.size());
    for (int j : comp) {
      const Entity& line = l.entities[lines[j]];
      const int p = line.parent;
      if (p < 0) continue;
      // rotated_box: height; curved_box: thickness (double)
      const float h = !line.box ? 0.0f
                      : line.box->curve ? static_cast<float>(line.box->curve->thickness)
                                        : static_cast<float>(line.box->height);
      if (node_of[p] >= 0) {
        OrderNode& pn = nodes[node_of[p]];
        pn.sum_line_height = pn.sum_line_height + h;
        pn.num_lines++;
        continue;
      }
      node_of[p] = static_cast<int>(nodes.size());
      OrderNode pn;
      pn.paragraph = p;
      pn.column = b;
      pn.box = TextlineOrderBox(l.entities[p]);
      pn.sum_line_height = h;
      pn.num_lines = 1;
      nodes.push_back(pn);
    }
    if (static_cast<int>(nodes.size()) == before) continue;
    OrderNode blk;
    blk.column = b;
    blk.rtl = l.entities[lines[i]].writing_direction.value_or(0) == 1;
    nodes.push_back(blk);
    const int bi = static_cast<int>(nodes.size()) - 1;
    for (int k = before; k < bi; k++) {
      nodes[bi].children.push_back(k);
      nodes[k].parent = bi;
      ComputeNodeBoxFromChildren(&nodes, bi, nullptr);
    }
    roots.push_back(bi);
    rtree.emplace_back(AxisAlignedR2(nodes[bi].box), bi);
  }

  // Nest roots that overlap a bigger root by more than the threshold of their own area (smallest first).
  struct Item {
    float area;
    int num_children, node;
  };
  auto greater = [](const Item& a, const Item& b) { return a.area > b.area; };
  for (bool changed = true; changed;) {
    changed = false;
    std::vector<Item> heap;
    for (int r : roots) {
      heap.push_back({nodes[r].area, static_cast<int>(nodes[r].children.size()), r});
      std::push_heap(heap.begin(), heap.end(), greater);
    }
    roots.clear();
    while (!heap.empty()) {
      std::pop_heap(heap.begin(), heap.end(), greater);
      const Item top = heap.back();
      heap.pop_back();
      const int k = top.node;
      if (nodes[k].area != top.area || static_cast<int>(nodes[k].children.size()) != top.num_children) continue;
      SortOrderChildren(&nodes, k);
      const R2Box q = AxisAlignedR2(nodes[k].box);
      std::set<int> cand;
      for (const auto& [r, id] : rtree)
        if (r.x0 <= q.x1 && !(r.x1 < q.x0 || r.y0 > q.y1) && r.y1 >= q.y0) cand.insert(id);
      for (int m : cand) {
        if (m == k || nodes[m].parent != -1) continue;
        const float ov = OverlapAreaF(nodes[k].box, nodes[m].box);
        if (ov > nodes[k].area * kChildAreaOverlappingThreshold) {
          nodes[m].children.push_back(k);
          nodes[k].parent = m;
          nodes[m].sorted = false;
          const float ref = nodes[m].box.angle;
          ComputeNodeBoxFromChildren(&nodes, m, &ref);
          heap.push_back({nodes[m].area, static_cast<int>(nodes[m].children.size()), m});
          std::push_heap(heap.begin(), heap.end(), greater);
          rtree.emplace_back(AxisAlignedR2(nodes[m].box), m);
          changed = true;
          break;
        }
      }
      if (nodes[k].parent == -1) roots.push_back(k);
    }
  }

  OrderNode root;
  root.column = false;  // row_wise_at_top_level
  nodes.push_back(root);
  const int ri = static_cast<int>(nodes.size()) - 1;
  int rtl = 0;
  for (int k = 0; k < ri; k++) {
    if (nodes[k].parent != -1) continue;
    nodes[k].parent = ri;
    nodes[ri].children.push_back(k);
    rtl += nodes[k].rtl;
  }
  if (nodes[ri].children.empty()) return {};  // no lines (the original CHECK-fails in TraverseOrderTree)
  nodes[ri].rtl = 2 * rtl > static_cast<int>(nodes[ri].children.size());
  ComputeNodeBoxFromChildren(&nodes, ri, nullptr);
  SortOrderChildren(&nodes, ri);
  std::vector<int> order;
  TraverseOrderTree(nodes, ri, &order);
  return order;
}

// ---- EstimateLinesStep (0x100849320) with a MergeLinesSpec: MergeLinesStep::AnalyzeDescendants per paragraph ----

// MergeLinesSpec of the config (EstimateLinesSpec field 4 {5: 2.5, 7: 1}); the rest are proto defaults. Field 7
// (adjacent lines only) also skips ReorderUsingDFS + RecomputeBoxesFromSymbols after merging. No overlap remover
// spec is set, so Merge's RemoveOverlappingInSubtree detects nothing.
constexpr double kMlMinBreadthRatio = 0.6;
constexpr double kMlMaxAngleDifference = 3.0;
constexpr double kMlMinBreadthOverlap = 0.6;
constexpr double kMlMaxDepthGap = 2.5;
constexpr bool kMlAdjacentOnly = true;
constexpr bool kMlDeleteEmptyEntities = true;  // field 8: DeleteTextEntitiesWithNoSymbols after each direction

constexpr double kDegToRadD = 0.017453292519943295;  // M_PI / 180

// PageLayout::Orientation(e): the nearest entity (e or an ancestor) with an orientation label, -1 if none.
int OrientationSource(const PageLayout& l, int e) {
  for (int i = e, guard = 0; i >= 0 && i < static_cast<int>(l.entities.size()) && guard < 64; i = l.entities[i].parent, guard++) {
    const Entity& x = l.entities[i];
    if (x.orientation || x.writing_direction || x.textline_order) return i;
  }
  return -1;
}

struct OrientationLabel {
  int orientation = 0, writing_direction = 0, textline_order = 0;
  bool operator<(const OrientationLabel& o) const {
    return std::tie(orientation, writing_direction, textline_order) <
           std::tie(o.orientation, o.writing_direction, o.textline_order);
  }
};
OrientationLabel OrientationOf(const PageLayout& l, int e) {
  const int s = OrientationSource(l, e);
  if (s < 0) return {};
  const Entity& x = l.entities[s];
  return {x.orientation.value_or(0), x.writing_direction.value_or(0), x.textline_order.value_or(0)};
}

// page_layout_utils::Thickness: extent across the writing direction.
int Thickness(const RotatedBox& b, int writing_direction) { return writing_direction == 2 ? b.width : b.height; }

// PageLayout::Thickness / LineLength(e): the entity's box (PolygonContainingDescendants; every entity here has one)
// measured in its orientation.
int EntityThickness(const PageLayout& l, int e) {
  return Thickness(l.entities[e].box.value_or(RotatedBox()), OrientationOf(l, e).writing_direction);
}
int EntityLineLength(const PageLayout& l, int e) {
  return LineLength(l.entities[e].box.value_or(RotatedBox()), OrientationOf(l, e).writing_direction);
}

// page_layout_utils::StartCoordinate / StartTextlineCoordinate (0x100c1f1c4 / 0x100c1f7d8): where the box starts
// along the writing direction / textline order d, for page orientation o.
int StartCoordinate(const RotatedBox& b, int o, int d) {
  const double h2 = b.height / 2.0, w2 = b.width / 2.0, left = b.left, top = b.top;
  const float a = b.angle;
  auto sc = [&] { return SinCosD(static_cast<double>(a) * kDegToRadD); };
  auto from_left = [&] {  // LABEL_16
    if (a == 0.0f) return static_cast<int>(left);
    const double t = h2 == 0.0 ? 0.0 : sc().s * h2;
    return static_cast<int>(round(left - t));
  };
  auto from_right = [&] {  // LABEL_9
    if (a == 0.0f) return static_cast<int>(b.width + left);
    const SinCos2 r = sc();
    const double v = (b.width == 0 ? -0.0 : r.c * b.width) + left;
    return static_cast<int>(round(h2 == 0.0 ? v : v - r.s * h2));
  };
  auto from_top = [&] {  // LABEL_19
    if (a == 0.0f) return static_cast<int>(top);
    const double t = w2 == 0.0 ? -0.0 : sc().s * w2;
    return static_cast<int>(round(t + top));
  };
  if (o == 0) return d == 0 ? from_left() : d == 1 ? from_right() : from_top();
  if (d == 1) {
    if (o == 2) return from_right();
    if (a == 0.0f) return static_cast<int>(h2 + top);
    const SinCos2 r = sc();
    double v = (b.width == 0 ? -0.0 : r.s * b.width) + top;
    if (h2 != 0.0) v = h2 * r.c + v;
    return static_cast<int>(round(v));
  }
  if (o == 2) return d == 0 ? from_left() : from_top();
  const double v4 = d != 0 ? w2 : h2, v6 = d != 0 ? left : top;
  if (a == 0.0f) return static_cast<int>(v4 + v6);
  const double t = v4 == 0.0 ? -0.0 : v4 * sc().c;
  return static_cast<int>(round(t + v6));
}

// page_layout_utils::EndCoordinate / EndTextlineCoordinate (0x100c1f498 / 0x100c1faac).
int EndCoordinate(const RotatedBox& b, int o, int d) {
  const double h2 = b.height / 2.0, w2 = b.width / 2.0, left = b.left, top = b.top;
  const float a = b.angle;
  auto sc = [&] { return SinCosD(static_cast<double>(a) * kDegToRadD); };
  auto at_left = [&] {  // LABEL_9
    if (a == 0.0f) return static_cast<int>(left);
    const double t = h2 == 0.0 ? 0.0 : sc().s * h2;
    return static_cast<int>(round(left - t));
  };
  auto at_right = [&] {  // LABEL_12
    if (a == 0.0f) return static_cast<int>(b.width + left);
    const SinCos2 r = sc();
    const double v = (b.width == 0 ? -0.0 : r.c * b.width) + left;
    return static_cast<int>(round(h2 == 0.0 ? v : v - r.s * h2));
  };
  auto at_bottom = [&] {  // LABEL_19
    if (a == 0.0f) return static_cast<int>(b.height + top);
    const SinCos2 r = sc();
    double v = (w2 == 0.0 ? -0.0 : r.s * w2) + top;
    if (b.height != 0) v = r.c * b.height + v;
    return static_cast<int>(round(v));
  };
  if (o == 0) return d == 0 ? at_right() : d == 1 ? at_left() : at_bottom();
  if (d == 1) {
    if (o == 2) return at_left();
    if (a == 0.0f) return static_cast<int>(h2 + top);
    const double t = h2 == 0.0 ? -0.0 : h2 * sc().c;
    return static_cast<int>(round(t + top));
  }
  if (d != 0) {
    if (o == 2) return at_bottom();
    if (a == 0.0f) return static_cast<int>(w2 + left);
    const SinCos2 r = sc();
    const double v = (w2 == 0.0 ? -0.0 : w2 * r.c) + left;
    return static_cast<int>(round(b.height == 0 ? v : v - r.s * b.height));
  }
  if (o == 2) return at_right();
  if (a == 0.0f) return static_cast<int>(h2 + top);
  const SinCos2 r = sc();
  double v = (b.width == 0 ? -0.0 : r.s * b.width) + top;
  if (h2 != 0.0) v = h2 * r.c + v;
  return static_cast<int>(round(v));
}

// CompareByReadingOrder (0x100ab5480): a before b along the writing direction, then the textline order.
struct ReadingOrderKey {
  int start, end, tl_start, tl_end;
};
ReadingOrderKey ReadingOrderKeyOf(const PageLayout& l, int e) {
  const RotatedBox b = l.entities[e].box.value_or(RotatedBox());
  const OrientationLabel o = OrientationOf(l, e);
  return {StartCoordinate(b, o.orientation, o.writing_direction), EndCoordinate(b, o.orientation, o.writing_direction),
          StartCoordinate(b, o.orientation, o.textline_order), EndCoordinate(b, o.orientation, o.textline_order)};
}
bool CompareByReadingOrder(bool rtl, const ReadingOrderKey& a, const ReadingOrderKey& b) {
  // Both boxes run against the expected direction.
  const bool reversed = rtl ? a.start < a.end && b.start < b.end : a.start > a.end && b.start > b.end;
  int x = a.start, y = b.start;
  bool flip = reversed;
  if (x == y) {
    flip = rtl ? a.tl_start < a.tl_end && b.tl_start < b.tl_end : a.tl_start > a.tl_end && b.tl_start > b.tl_end;
    x = a.tl_start, y = b.tl_start;
  }
  const bool lt = flip ? x < y : y < x, gt = flip ? y < x : x < y;
  return rtl ? lt : gt;
}

// ComputeSymbolMetrics(layout, e).first (0x100ab4080 -> ComputeSymbolMetricsFromWords 0x100ab4168): mean symbol
// length along the writing direction over the WORD descendants of e.
double MeanSymbolLength(const PageLayout& l, int e, const std::vector<std::vector<int>>& kids) {
  double sum = 0.0;
  int n = 0;
  for (int w : FilterBy(l, DescendantsOf(e, kids), Type::kWord)) {
    const std::vector<int> syms = FilterBy(l, DescendantsOf(w, kids), Type::kSymbol);
    n += static_cast<int>(syms.size());
    bool any_box = false;
    for (int s : syms) any_box |= l.entities[s].box.has_value();
    if (!any_box) {  // estimate from the word (then the symbols are still measured, as in the original)
      const int t = EntityThickness(l, w);
      const bool math = (l.entities[w].content_type.value_or(0) & ~1) == 6;
      sum = sum + (math ? static_cast<double>((static_cast<uint64_t>(syms.size()) * t) >> 1)
                        : static_cast<double>(EntityLineLength(l, w)));
      if (syms.empty()) continue;
    }
    for (int s : syms) sum = sum + static_cast<double>(EntityLineLength(l, s));
  }
  return n > 0 ? sum / static_cast<double>(static_cast<unsigned>(n)) : 0.0;
}

// Polygon as a rotated box: curved polygons (and missing ones) read as an empty box.
RotatedBox RotatedBoxOnly(const Entity& e) { return e.box && !e.box->curve ? *e.box : RotatedBox(); }

// MergeLinesStep::ShouldMerge(src, tgt) (0x100851a00).
bool ShouldMergeLines(const PageLayout& l, int src, int tgt, const std::vector<std::vector<int>>& kids) {
  const RotatedBox bs = RotatedBoxOnly(l.entities[src]), bt = RotatedBoxOnly(l.entities[tgt]);
  const int wd = OrientationOf(l, src).writing_direction;
  const int ts = Thickness(bs, wd), tt = Thickness(bt, wd);
  if (static_cast<double>(std::min(ts, tt)) / static_cast<double>(std::max(ts, tt)) < kMlMinBreadthRatio) return false;
  const double as = bs.angle, at = bt.angle;
  double d = fmod(fabs(as - at), 360.0);
  if (d > 180.0) d = 360.0 - d;
  if (!(d <= kMlMaxAngleDifference)) return false;
  RotatedBox m;
  m.left = bs.left, m.top = bs.top, m.width = 1, m.height = 1;
  m.angle = static_cast<float>((as + at) * 0.5);
  ExpandBoxToContainBox(bs, &m);
  ExpandBoxToContainBox(bt, &m);
  const int tm = Thickness(m, wd);
  if (!(static_cast<double>(tt + ts - tm) / static_cast<double>(tm) >= kMlMinBreadthOverlap)) return false;
  const int ds = EntityLineLength(l, src), dt = EntityLineLength(l, tgt), dm = LineLength(m, wd);
  double avg = (MeanSymbolLength(l, src, kids) + MeanSymbolLength(l, tgt, kids)) * 0.5;
  if (avg == 0.0) avg = static_cast<double>(EntityThickness(l, src) + EntityThickness(l, tgt)) / 4.0;
  return static_cast<double>(dm - (dt + ds)) / avg <= kMlMaxDepthGap;
}

// Entity indices that survive PageLayout reorders/deletes: every remap is applied to all registered references.
struct EntityRefs {
  std::vector<int> at;
  int Add(int e) {
    at.push_back(e);
    return static_cast<int>(at.size()) - 1;
  }
  void Apply(const std::vector<int>& remap) {
    for (int& t : at) t = t >= 0 ? remap[t] : -1;
  }
};

void SetWordTextFromSymbols(PageLayout* l, int w) {
  l->entities[w].text.reset();
  l->entities[w].text = EntityText(*l, w, ChildLists(*l));
}

// ConvertSoftHyphenToSymbol (0x100ab4ac0): a SOFT_HYPHEN break becomes a trailing "-" symbol (break NONE).
void ConvertSoftHyphenToSymbol(PageLayout* l, int w) {
  if (!l->entities[w].word_break || l->entities[w].word_break->type != 4) return;
  Entity s;
  s.type = Type::kSymbol;
  s.parent = w;
  s.text = "-";
  s.box = l->entities[w].word_break->box.value_or(RotatedBox());
  l->entities.push_back(std::move(s));
  l->entities[w].word_break = WordBreak{1, std::nullopt};
  SetWordTextFromSymbols(l, w);
  RecomputeBoxFromChildren(l, w, ChildLists(*l));
}

// ConvertEndingHyphenToSoftHyphen (0x100ab4d00): the word's break becomes NONE, or SOFT_HYPHEN (boxed by the
// removed symbol) when its last of several symbols is "-".
void ConvertEndingHyphenToSoftHyphen(PageLayout* l, int w, EntityRefs* refs) {
  if (l->entities[w].word_break && l->entities[w].word_break->type == 4) return;
  const auto kids = ChildLists(*l);
  l->entities[w].word_break = WordBreak{1, std::nullopt};
  if (kids[w].size() < 2) return;
  const int last = kids[w].back();
  if (l->entities[last].text.value_or("") != "-") return;
  l->entities[w].word_break = WordBreak{4, l->entities[last].box.value_or(RotatedBox())};
  std::vector<bool> del(l->entities.size(), false);
  del[last] = true;
  const int wr = refs->Add(w);
  refs->Apply(DeleteWithDescendants(l, del));
  SetWordTextFromSymbols(l, refs->at[wr]);
}

// MergeLinesStep::Merge(src, tgt) (0x1008523c0): the source's words move into the target, in reading order.
void MergeLine(PageLayout* l, int src, int tgt, EntityRefs* refs) {
  const bool rtl = OrientationOf(*l, tgt).writing_direction == 1;
  const int sr = refs->Add(src), tr = refs->Add(tgt);
  for (int r : {sr, tr}) {
    const auto kids = ChildLists(*l);
    const int e = refs->at[r];
    if (kids[e].empty()) continue;
    const int w = kids[e].back();
    ConvertSoftHyphenToSymbol(l, w);
    WordBreak& b = l->entities[w].word_break ? *l->entities[w].word_break : l->entities[w].word_break.emplace();
    if (b.type == 1) b.box.reset(), b.type = 2;  // NONE -> SPACE
  }
  src = refs->at[sr], tgt = refs->at[tr];
  const auto kids = ChildLists(*l);
  std::vector<int> ch = kids[tgt];  // SetParent appends the source's children
  ch.insert(ch.end(), kids[src].begin(), kids[src].end());
  for (int c : kids[src]) l->entities[c].parent = tgt;
  // SortChildren (stable) + ReorderChildren: the sorted children take the children's slots.
  std::vector<ReadingOrderKey> key(l->entities.size());
  for (int c : ch) key[c] = ReadingOrderKeyOf(*l, c);
  std::vector<int> sorted = ch, slots = ch;
  std::stable_sort(sorted.begin(), sorted.end(),
                   [&](int a, int b) { return CompareByReadingOrder(rtl, key[a], key[b]); });
  std::sort(slots.begin(), slots.end());
  std::vector<int> order(l->entities.size());
  for (size_t i = 0; i < order.size(); i++) order[i] = static_cast<int>(i);
  for (size_t k = 0; k < slots.size(); k++) order[slots[k]] = sorted[k];
  refs->Apply(Reorder(l, order));
}

// MergeLinesStep::AnalyzeForDirection (0x100850c40): repeated passes merging each line with the following lines
// (the next one only with kMlAdjacentOnly, widening after each merge) until nothing merges.
void MergeLinesForDirection(PageLayout* l, EntityRefs* refs, const std::vector<int>& lines) {
  std::vector<int> deleted;  // refs
  auto is_deleted = [&](int e) {
    for (int r : deleted)
      if (refs->at[r] == e) return true;
    return false;
  };
  const int n = static_cast<int>(lines.size());
  for (bool changed = true; changed;) {
    changed = false;
    for (int i = 0; i < n; i++) {
      const int a = refs->at[lines[i]];
      if (IsCurved(l->entities[a]) || is_deleted(a)) continue;
      int window = kMlAdjacentOnly ? 1 : n;
      for (int j = i + 1; j < n; j++) {
        const int tgt = refs->at[lines[i]], src = refs->at[lines[j]];
        if (!IsCurved(l->entities[src]) && !is_deleted(src) && ShouldMergeLines(*l, src, tgt, ChildLists(*l))) {
          MergeLine(l, src, tgt, refs);
          deleted.push_back(lines[j]);
          {
            const int t = refs->at[lines[i]];
            const auto kids = ChildLists(*l);
            if (!kids[t].empty() && !kids[kids[t].back()].empty()) ConvertEndingHyphenToSoftHyphen(l, kids[t].back(), refs);
          }
          const int t = refs->at[lines[i]];
          const auto kids = ChildLists(*l);
          RecomputeBoxFromChildren(l, t, kids);
          RecomputeConfidenceFromDescendants(l, t, kids, false);
          // Ancestors of the source left with it as their only child go too.
          for (int p = l->entities[refs->at[lines[j]]].parent; p >= 0 && kids[p].size() == 1; p = l->entities[p].parent)
            deleted.push_back(refs->Add(p));
          changed = true;
          window++;
        }
        if (window + i <= j) break;
      }
    }
  }
  if (deleted.empty()) return;  // nothing merged: no Delete, no DeleteTextEntitiesWithNoSymbols
  std::vector<bool> del(l->entities.size(), false);
  for (int r : deleted)
    if (refs->at[r] >= 0) del[refs->at[r]] = true;
  refs->Apply(DeleteWithDescendants(l, del));
  if (kMlDeleteEmptyEntities) refs->Apply(DeleteTextEntitiesWithNoSymbols(l));
}

// MergeLinesStep::AnalyzeDescendants (0x100850500): the paragraph's lines, merged per orientation.
void MergeLinesInParagraph(PageLayout* l, EntityRefs* refs, int para_ref) {
  const auto kids = ChildLists(*l);
  const int p = refs->at[para_ref];
  if (kids[p].empty()) return;
  const std::vector<int> lines = l->entities[kids[p][0]].type == Type::kLine
                                     ? FilterBy(*l, kids[p], Type::kLine)
                                     : FilterBy(*l, DescendantsOf(p, kids), Type::kLine);
  // flat_hash_map keyed by the serialized OrientationLabel; groups never interact, so their order does not matter.
  std::map<OrientationLabel, std::vector<int>> groups;
  for (int e : lines) groups[OrientationOf(*l, e)].push_back(refs->Add(e));
  for (const auto& [o, g] : groups) MergeLinesForDirection(l, refs, g);
}

void EstimateLines(PageLayout* l) {
  EntityRefs refs;
  std::vector<int> paras;
  for (size_t i = 0; i < l->entities.size(); i++)
    if (l->entities[i].type == Type::kParagraph) paras.push_back(refs.Add(static_cast<int>(i)));
  for (int r : paras)
    if (refs.at[r] >= 0) MergeLinesInParagraph(l, &refs, r);
}

}  // namespace

LayoutAnalyzer::~LayoutAnalyzer() = default;

std::unique_ptr<LayoutAnalyzer> LayoutAnalyzer::Create(const Resources& res) {
  std::string bytes;
  if (!res.Read(kModelPath, &bytes)) return nullptr;
  std::unique_ptr<LayoutAnalyzer> a(new LayoutAnalyzer());
  a->model_ = TfliteModel::Create(std::move(bytes), 1, nullptr, /*use_xnnpack=*/true);
  if (!a->model_) return nullptr;
  std::string config;
  if (!res.Read(kConfigPath, &config) || !FindFillSymbolGapsSpec(config, &a->fill_symbol_gaps_)) return nullptr;
  return a;
}

// ClusterSortGcnStep::GetPredictions (model version 2): node outputs > 0 (one per line). The edge outputs only feed
// line clustering, which this config skips.
bool LayoutAnalyzer::NodePredictions(const PageLayout& layout, std::vector<bool>* nodes, std::vector<float>* raw) {
  const LineGraph g = BuildLineGraph(layout);
  std::vector<float> out;
  if (!GcnNodeOutputs(model_.get(), g, static_cast<float>(layout.width), &out)) return false;
  nodes->assign(g.lines.size(), false);
  for (size_t k = 0; k < out.size(); k++) (*nodes)[k] = out[k] > 0.0f;
  if (raw) *raw = std::move(out);
  return true;
}

// ClusterSortGcnStep::AnalyzeInternal (0x100888284) with skip_line_clustering, recompute_entity_boxes_and_confidences
// and sort_lines_in_paragraphs; heuristic_blocks off: one BLOCK per paragraph, in reading order.
bool LayoutAnalyzer::ClusterSort(PageLayout* l) {
  const LineGraph g = BuildLineGraph(*l);
  std::vector<float> raw;
  if (!GcnNodeOutputs(model_.get(), g, static_cast<float>(l->width), &raw)) return false;
  std::vector<bool> pred(g.lines.size(), false);
  for (size_t k = 0; k < raw.size(); k++) pred[k] = raw[k] > 0.0f;
  std::vector<int> lines = g.lines;  // the original keeps entity pointers; follow them through the reorders
  auto track = [&lines](const std::vector<int>& remap) {
    for (int& t : lines) t = t >= 0 ? remap[t] : -1;
  };

  track(ReorderUsingDfs(l));
  {
    const auto kids = ChildLists(*l);
    const auto all = AllDescendants(*l, kids);
    for (auto it = all.rbegin(); it != all.rend(); ++it) RobustBoxEstimation(l, *it, kids);
  }
  RecomputeConfidencesFromSymbols(l);
  {
    const auto kids = ChildLists(*l);
    std::vector<bool> del(l->entities.size(), false);
    for (size_t i = 0; i < l->entities.size(); i++) {
      Entity& p = l->entities[i];
      if (p.type != Type::kParagraph) continue;
      if (kids[i].empty()) {
        del[i] = true;
        continue;
      }
      if (!p.orientation && !p.writing_direction && !p.textline_order) {
        const Entity& c = l->entities[kids[i][0]];
        p.orientation = c.orientation, p.writing_direction = c.writing_direction, p.textline_order = c.textline_order;
      }
    }
    track(DeleteWithDescendants(l, del));
  }
  for (size_t i = 0; i < l->entities.size(); i++)
    if (l->entities[i].type == Type::kParagraph) SortChildrenOfEntity(l, static_cast<int>(i), &lines);

  const std::vector<int> order = OrderParagraphs(*l, lines, g.edges, pred);
  std::vector<bool> old_block(l->entities.size(), false);
  for (size_t i = 0; i < l->entities.size(); i++) old_block[i] = l->entities[i].type == Type::kBlock;
  for (int p : order) {
    Entity blk;
    blk.type = Type::kBlock;
    const Entity& pe = l->entities[p];
    blk.orientation = pe.orientation, blk.writing_direction = pe.writing_direction;
    blk.textline_order = pe.textline_order;
    blk.box = pe.box;
    l->entities.push_back(std::move(blk));
    l->entities[p].parent = static_cast<int>(l->entities.size()) - 1;
  }
  old_block.resize(l->entities.size(), false);
  DeleteWithDescendants(l, old_block);
  ReorderUsingDfs(l);
  RecomputeConfidencesFromSymbols(l);
  return true;
}

void LayoutAnalyzer::ClusterSortInputs(const PageLayout& layout, std::vector<float>* features,
                                       std::vector<int64_t>* senders, std::vector<int64_t>* receivers) const {
  const LineGraph g = BuildLineGraph(layout);
  *features = NodeFeatures(g, static_cast<float>(layout.width));
  senders->clear(), receivers->clear();
  for (const GraphEdge& e : g.edges) senders->push_back(e.a), receivers->push_back(e.b);
}

bool LayoutAnalyzer::RunStep(const std::string& step, const GrayImage& gray, PageLayout* l) {
  if (step == "pre") SetDetectionConfidence(l);
  else if (step == "ConfidenceFilter") ConfidenceFilter(l);
  else if (step == "WordPruning") WordPruning(l);
  else if (step == "FillSymbolGaps") FillSymbolGaps(l, fill_symbol_gaps_);
  else if (step == "CurvedOverlaps") RemoveOverlappingWithCurvedBoxes(l);
  else if (step == "ClusterSort") return ClusterSort(l);
  else if (step == "EstimateLines") EstimateLines(l);
  else if (step == "AnalyzerEnd") RecomputeLineTextContent(l);
  else if (step == "Colors") EstimateWordColors(l, gray);
  else if (step == "PageLanguage") PopulatePageLanguage(l), l->creator_id = "";
  else return false;
  return true;
}

bool LayoutAnalyzer::Analyze(const GrayImage& gray, PageLayout* layout) {
  for (const char* s : {"pre", "ConfidenceFilter", "WordPruning", "FillSymbolGaps", "CurvedOverlaps", "ClusterSort",
                        "EstimateLines", "AnalyzerEnd", "Colors", "PageLanguage"})
    if (!RunStep(s, gray, layout)) return false;
  return true;
}

}  // namespace screenai
