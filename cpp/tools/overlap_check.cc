// overlap_check <dir>: replays the box pairs / sets trace/overlap_check.py ran through the original library's overlap
// routines and requires bit-identical results from cpp/src/box_overlap.h:
//   pairs.bin  Rectangle<float> OverlapArea and SafeBoxOverlapArea<Box<float>> vs overlap::RectOverlapArea /
//              overlap::BoxOverlapArea (also scored: the polygon-clipping approximation layout used before);
//   rbb.bin    SafeOverlapBoxes<goodoc::RotatedBoundingBox> (iou, I / area a, I / area b);
//   sets.bin   OverlapComputer<RotatedBoundingBox | RotatedBox<float>> maps vs overlap::ComputeOverlaps;
//   det_pairs.bin  the detector's overlap (AxisAlignedBoxOverlapArea<RotatedBox<float>> when both angles are 0, else
//              Rectangle<float>), both argument orders, vs overlap::BoxOverlapArea;
//   det_sets.bin   OverlapComputer<RotatedBox<float> const> (paragraph assignment) vs overlap::ComputeOverlaps.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "box_overlap.h"
#include "layout_graph.h"

namespace {

using screenai::overlap::Box;
namespace ov = screenai::overlap;

const char* kCategories[] = {"random",    "tiny_angle", "near_0",       "near_90",   "near_180",     "identical",
                             "nested",    "edge_touch", "corner_touch", "threshold", "det_threshold"};
constexpr int kNumCategories = 11;
const float kThresholds[] = {0.1f, 0.25f, 0.3f, 0.5f, 0.6f};

struct Reader {
  std::vector<char> d;
  size_t o = 0;
  explicit Reader(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    d.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  }
  template <class T>
  T get() {
    T v;
    std::memcpy(&v, d.data() + o, sizeof(T));
    o += sizeof(T);
    return v;
  }
};

uint32_t Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// The layout's OverlapAreaF before box_overlap.h: convex polygon clipping of the corner quads in double.
float OldOverlapArea(const Box& a, const Box& b) {
  if (a.angle == 0.0f && b.angle == 0.0f) return ov::AxisAlignedOverlapArea(a, b);
  auto corners = [](const Box& bx) {
    float p[8];
    screenai::BoxCorners({bx.x, bx.y, bx.w, bx.h, bx.angle}, p);
    std::vector<std::pair<double, double>> v;
    for (int k = 0; k < 4; k++) v.emplace_back(p[2 * k], p[2 * k + 1]);
    return v;
  };
  std::vector<std::pair<double, double>> poly = corners(a);
  const auto clip = corners(b);
  auto cross = [](const std::pair<double, double>& o, const std::pair<double, double>& p,
                  const std::pair<double, double>& q) {
    return (p.first - o.first) * (q.second - o.second) - (p.second - o.second) * (q.first - o.first);
  };
  for (size_t k = 0; k < 4 && !poly.empty(); k++) {
    const auto& e0 = clip[k];
    const auto& e1 = clip[(k + 1) % 4];
    std::vector<std::pair<double, double>> next;
    for (size_t i = 0; i < poly.size(); i++) {
      const auto& p = poly[i];
      const auto& q = poly[(i + 1) % poly.size()];
      const double cp = cross(e0, e1, p), cq = cross(e0, e1, q);
      if (cp >= 0) next.push_back(p);
      if ((cp >= 0) != (cq >= 0)) {
        const double t = cp / (cp - cq);
        next.emplace_back(p.first + t * (q.first - p.first), p.second + t * (q.second - p.second));
      }
    }
    poly = std::move(next);
  }
  double area = 0;
  for (size_t i = 0; i < poly.size(); i++) {
    const auto& p = poly[i];
    const auto& q = poly[(i + 1) % poly.size()];
    area += p.first * q.second - q.first * p.second;
  }
  return static_cast<float>(fabs(area) * 0.5);
}

struct Stats {
  int n = 0, rect_bad = 0, safe_bad = 0, old_bad = 0, old_flip = 0;
};

// Ratios the layout compares against thresholds: I / area(a), I / area(b), iou.
void Ratios(float inter, const Box& a, const Box& b, float r[3]) {
  const float aa = a.w * a.h, ab = b.w * b.h;
  r[0] = inter / aa, r[1] = inter / ab, r[2] = inter / ((aa + ab) - inter);
}

int CheckPairs(const std::string& dir) {
  Reader r(dir + "/pairs.bin");
  if (r.d.size() < 4) return fprintf(stderr, "pairs.bin: cannot read\n"), 1;
  const uint32_t n = r.get<uint32_t>();
  Stats st[kNumCategories];
  int flips[5][2] = {};  // per threshold: old says above / below while the original says the opposite
  for (uint32_t k = 0; k < n; k++) {
    const uint32_t cat = r.get<uint32_t>();
    float v[10];
    for (float& x : v) x = r.get<float>();
    const float ref_rect = r.get<float>(), ref_safe = r.get<float>();
    const Box a{v[0], v[1], v[2], v[3], v[4]}, b{v[5], v[6], v[7], v[8], v[9]};
    const float rect = ov::RectOverlapArea(ov::RectFromLTWHA(a.x, a.y, a.w, a.h, a.angle),
                                           ov::RectFromLTWHA(b.x, b.y, b.w, b.h, b.angle));
    const float safe = ov::BoxOverlapArea(a, b), old = OldOverlapArea(a, b);
    Stats& s = st[cat];
    s.n++;
    if (Bits(rect) != Bits(ref_rect)) {
      if (s.rect_bad++ < 3)
        fprintf(stderr, "rect %s: a=(%a %a %a %a %a) b=(%a %a %a %a %a) ref %a ours %a\n", kCategories[cat], a.x, a.y,
                a.w, a.h, a.angle, b.x, b.y, b.w, b.h, b.angle, ref_rect, rect);
    }
    if (Bits(safe) != Bits(ref_safe)) {
      if (s.safe_bad++ < 3)
        fprintf(stderr, "safe %s: a=(%a %a %a %a %a) b=(%a %a %a %a %a) ref %a ours %a\n", kCategories[cat], a.x, a.y,
                a.w, a.h, a.angle, b.x, b.y, b.w, b.h, b.angle, ref_safe, safe);
    }
    if (Bits(old) != Bits(ref_safe)) s.old_bad++;
    float rr[3], ro[3];
    Ratios(ref_safe, a, b, rr);
    Ratios(old, a, b, ro);
    bool flip = false;
    for (int t = 0; t < 5; t++)
      for (int j = 0; j < 3; j++) {
        const bool ref_above = rr[j] > kThresholds[t], old_above = ro[j] > kThresholds[t];
        if (ref_above != old_above) flips[t][old_above ? 0 : 1]++, flip = true;
      }
    s.old_flip += flip;
  }
  int rect_bad = 0, safe_bad = 0;
  printf("%-13s %7s %9s %9s %9s %11s\n", "pairs", "n", "rect_bad", "safe_bad", "old_bad", "old_flipped");
  for (int c = 0; c < kNumCategories; c++) {
    const Stats& s = st[c];
    printf("%-13s %7d %9d %9d %9d %11d\n", kCategories[c], s.n, s.rect_bad, s.safe_bad, s.old_bad, s.old_flip);
    rect_bad += s.rect_bad, safe_bad += s.safe_bad;
  }
  printf("old approximation threshold flips (ratio > t; pairs x {I/A, I/B, iou}):");
  for (int t = 0; t < 5; t++) printf("  %g: %d above / %d below", kThresholds[t], flips[t][0], flips[t][1]);
  printf("\n");
  return rect_bad + safe_bad;
}

int CheckRbb(const std::string& dir) {
  Reader r(dir + "/rbb.bin");
  if (r.d.size() < 4) return fprintf(stderr, "rbb.bin: cannot read\n"), 1;
  const uint32_t n = r.get<uint32_t>();
  int bad = 0;
  for (uint32_t k = 0; k < n; k++) {
    r.get<uint32_t>();
    Box bx[2];
    for (Box& b : bx) {
      b.x = static_cast<float>(r.get<int32_t>()), b.y = static_cast<float>(r.get<int32_t>());
      b.w = static_cast<float>(r.get<int32_t>()), b.h = static_cast<float>(r.get<int32_t>());
      b.angle = r.get<float>();
    }
    const uint32_t ok = r.get<uint32_t>();
    float ref[3];
    for (float& x : ref) x = r.get<float>();
    const float inter = ov::BoxOverlapArea(bx[0], bx[1]);
    float ours[3] = {0, 0, 0};  // iou, I / A, I / B
    if (inter != 0.0f) {
      float q[3];
      Ratios(inter, bx[0], bx[1], q);
      ours[0] = q[2], ours[1] = q[0], ours[2] = q[1];
    }
    const bool same = ok == (inter != 0.0f ? 1u : 0u) && Bits(ours[0]) == Bits(ref[0]) &&
                      Bits(ours[1]) == Bits(ref[1]) && Bits(ours[2]) == Bits(ref[2]);
    if (!same && bad++ < 3)
      fprintf(stderr, "rbb %u: ok %u (%a %a %a) ours (%a %a %a)\n", k, ok, ref[0], ref[1], ref[2], ours[0], ours[1],
              ours[2]);
  }
  printf("rbb pairs %u, mismatches %d\n", n, bad);
  return bad;
}

int CheckDetPairs(const std::string& dir) {
  Reader r(dir + "/det_pairs.bin");
  if (r.d.size() < 4) return fprintf(stderr, "det_pairs.bin: cannot read\n"), 1;
  const uint32_t n = r.get<uint32_t>();
  int count[kNumCategories] = {}, bad[kNumCategories] = {};
  for (uint32_t k = 0; k < n; k++) {
    const uint32_t cat = r.get<uint32_t>();
    float v[10];
    for (float& x : v) x = r.get<float>();
    const float ref_ab = r.get<float>(), ref_ba = r.get<float>();
    const Box a{v[0], v[1], v[2], v[3], v[4]}, b{v[5], v[6], v[7], v[8], v[9]};
    const float ab = ov::BoxOverlapArea(a, b), ba = ov::BoxOverlapArea(b, a);
    count[cat]++;
    if (Bits(ab) != Bits(ref_ab) || Bits(ba) != Bits(ref_ba)) {
      if (bad[cat]++ < 3)
        fprintf(stderr, "det %s: a=(%a %a %a %a %a) b=(%a %a %a %a %a) ref %a %a ours %a %a\n", kCategories[cat],
                a.x, a.y, a.w, a.h, a.angle, b.x, b.y, b.w, b.h, b.angle, ref_ab, ref_ba, ab, ba);
    }
  }
  int total = 0;
  printf("%-13s %7s %9s\n", "det_pairs", "n", "bad");
  for (int c = 0; c < kNumCategories; c++) {
    printf("%-13s %7d %9d\n", kCategories[c], count[c], bad[c]);
    total += bad[c];
  }
  return total;
}

int CheckSets(const std::string& dir, const char* name = "sets.bin") {
  Reader r(dir + "/" + name);
  if (r.d.size() < 4) return fprintf(stderr, "%s: cannot read\n", name), 1;
  const uint32_t n = r.get<uint32_t>();
  int bad = 0, entries = 0;
  for (uint32_t k = 0; k < n; k++) {
    const uint32_t kind = r.get<uint32_t>();
    const int32_t split = r.get<int32_t>();
    const uint32_t nb = r.get<uint32_t>();
    std::vector<Box> boxes(nb);
    for (Box& b : boxes) {
      if (kind == 0) {
        b.x = static_cast<float>(r.get<int32_t>()), b.y = static_cast<float>(r.get<int32_t>());
        b.w = static_cast<float>(r.get<int32_t>()), b.h = static_cast<float>(r.get<int32_t>());
      } else {
        b.x = r.get<float>(), b.y = r.get<float>(), b.w = r.get<float>(), b.h = r.get<float>();
      }
      b.angle = r.get<float>();
    }
    const uint32_t ne = r.get<uint32_t>();
    ov::OverlapMap ref;
    for (uint32_t e = 0; e < ne; e++) {
      const int32_t a = r.get<int32_t>(), b = r.get<int32_t>();
      ov::OverlapInfo info;
      info.iou = r.get<float>(), info.over_first = r.get<float>(), info.over_second = r.get<float>();
      ref[{a, b}] = info;
    }
    entries += ne;
    const ov::OverlapMap ours = ov::ComputeOverlaps(boxes, split);
    bool same = ours.size() == ref.size();
    auto jt = ref.cbegin();
    for (auto it = ours.cbegin(); same && it != ours.cend(); ++it, ++jt)
      same = it->first == jt->first && Bits(it->second.iou) == Bits(jt->second.iou) &&
             Bits(it->second.over_first) == Bits(jt->second.over_first) &&
             Bits(it->second.over_second) == Bits(jt->second.over_second);
    if (!same && bad++ < 3) fprintf(stderr, "set %u (kind %u, %u boxes): %zu entries vs ref %u\n", k, kind, nb,
                                    ours.size(), ne);
  }
  printf("%s: sets %u (%d map entries), mismatches %d\n", name, n, entries, bad);
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return fprintf(stderr, "usage: overlap_check <dir>\n"), 2;
  const std::string dir = argv[1];
  const int bad =
      CheckPairs(dir) + CheckRbb(dir) + CheckSets(dir) + CheckDetPairs(dir) + CheckSets(dir, "det_sets.bin");
  printf(bad ? "FAIL\n" : "OK\n");
  return bad ? 1 : 0;
}
