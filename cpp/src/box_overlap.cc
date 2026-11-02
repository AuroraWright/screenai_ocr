#include "box_overlap.h"

#include <cmath>

namespace screenai {
namespace overlap {
namespace {

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

constexpr float kDegToRad = __builtin_bit_cast(float, 0x3C8EFA35u);  // 0.017453292

float NormalizeAngle(float a) {  // (-180, 180]
  while (a <= -180.0f) a += 360.0f;
  while (a > 180.0f) a -= 360.0f;
  return a;
}

bool IsAxisAligned(const Rect& r) { return r.c == 1.0f && r.s == 0.0f; }

struct P {
  float x, y;
};

}  // namespace

Rect RectFromLTWHA(float l, float t, float w, float h, float angle) {
  const SinCos r = SinCosf(angle * kDegToRad);
  return {l, t, w, h, r.c, r.s};
}

// Sutherland-Hodgman clip of b's corners against a's top (keep y >= T), right (x <= R), bottom (y <= B) and left
// (x >= L) edges, at most 8 points per buffer; then the shoelace area. Entering edges interpolate from the outside
// point, leaving edges from the inside point (same products and quotient either way).
float IntersectAxisAlignedNonAxisAligned(const Rect& a, const Rect& b) {
  P in[8], out[8];
  if (IsAxisAligned(b)) {
    const float r = b.w + b.l, bt = b.t + b.h;
    in[0] = {b.l, b.t}, in[1] = {r, b.t}, in[2] = {r, bt}, in[3] = {b.l, bt};
  } else {
    const float x1 = b.l + (b.w * b.c), y1 = b.t + (b.w * b.s);
    const float ch = b.c * b.h, sh = b.s * b.h;
    in[0] = {b.l, b.t}, in[1] = {x1, y1}, in[2] = {x1 - sh, y1 + ch}, in[3] = {b.l - sh, b.t + ch};
  }
  // Horizontal edge y = v: keep (y >= v) when `ge`; the crossing x = p.x + ((v - p.y) * (q.x - p.x)) / (q.y - p.y).
  auto clip_h = [](const P* src, int n, P* dst, float v, bool ge) {
    int m = 0;
    for (int i = 0; i < n; i++) {
      const P p = src[i], q = src[i + 1 == n ? 0 : i + 1];
      const bool pin = ge ? p.y >= v : p.y <= v;
      const bool cross = pin ? (ge ? q.y < v : q.y > v) : (ge ? q.y > v : q.y < v);
      if (pin) dst[m++] = p;
      if (cross) dst[m++] = {p.x + (((v - p.y) * (q.x - p.x)) / (q.y - p.y)), v};
    }
    return m;
  };
  // Vertical edge x = v: y = p.y + ((v - p.x) * (q.y - p.y)) / (q.x - p.x).
  auto clip_v = [](const P* src, int n, P* dst, float v, bool ge) {
    int m = 0;
    for (int i = 0; i < n; i++) {
      const P p = src[i], q = src[i + 1 == n ? 0 : i + 1];
      const bool pin = ge ? p.x >= v : p.x <= v;
      const bool cross = pin ? (ge ? q.x < v : q.x > v) : (ge ? q.x > v : q.x < v);
      if (pin) dst[m++] = p;
      if (cross) dst[m++] = {v, p.y + (((v - p.x) * (q.y - p.y)) / (q.x - p.x))};
    }
    return m;
  };
  int n = clip_h(in, 4, out, a.t, true);
  if (n < 3) return 0.0f;
  n = clip_v(out, n, in, a.l + a.w, false);
  if (n < 3) return 0.0f;
  n = clip_h(in, n, out, a.t + a.h, false);
  if (n < 3) return 0.0f;
  n = clip_v(out, n, in, a.l, true);
  if (n < 3) return 0.0f;
  float sum = 0.0f;
  for (int i = 0; i + 1 < n; i++) sum = sum + ((in[i].x * in[i + 1].y) - (in[i].y * in[i + 1].x));
  return fabsf((sum + ((in[n - 1].x * in[0].y) - (in[n - 1].y * in[0].x))) * 0.5f);
}

float RectOverlapArea(const Rect& a, const Rect& b) {
  if (IsAxisAligned(a) && IsAxisAligned(b)) {
    const float l = b.l > a.l ? b.l : a.l, t = b.t > a.t ? b.t : a.t;
    float r = a.l + a.w, bt = a.t + a.h;
    const float br = b.l + b.w, bb = b.t + b.h;
    if (r > br) r = br;
    if (bt > bb) bt = bb;
    float w = r - l, h = bt - t;
    if (w < 0.0f) w = 0.0f;
    if (h < 0.0f) h = 0.0f;
    return w * h;
  }
  if (IsAxisAligned(a)) return IntersectAxisAlignedNonAxisAligned(a, b);
  if (IsAxisAligned(b)) return IntersectAxisAlignedNonAxisAligned(b, a);
  // Both rotated: rotate both into a's frame (a becomes axis aligned).
  const float c = a.c, s = a.s;
  const Rect ra{(c * a.l) + (s * a.t), (c * a.t) - (s * a.l), a.w, a.h, 1.0f, 0.0f};
  const Rect rb{(c * b.l) + (s * b.t), (c * b.t) - (s * b.l), b.w, b.h, (c * b.c) + (s * b.s), (c * b.s) - (s * b.c)};
  return IntersectAxisAlignedNonAxisAligned(ra, rb);
}

float AxisAlignedOverlapArea(const Box& a, const Box& b) {
  const float l = a.x >= b.x ? a.x : b.x;
  float r = a.x + a.w;
  if (b.x + b.w < r) r = b.x + b.w;
  const float w = r - l;
  if (!(w >= 0.0f)) return 0.0f;
  const float t = a.y >= b.y ? a.y : b.y;
  float bt = a.y + a.h;
  if (b.y + b.h < bt) bt = b.y + b.h;
  const float h = bt - t;
  return h < 0.0f ? 0.0f : w * h;
}

float BoxOverlapArea(const Box& a, const Box& b) {
  if (a.angle == 0.0f && b.angle == 0.0f) return AxisAlignedOverlapArea(a, b);
  return RectOverlapArea(RectFromLTWHA(a.x, a.y, a.w, a.h, a.angle), RectFromLTWHA(b.x, b.y, b.w, b.h, b.angle));
}

Box AlignedBox(const Box& b) {
  if (NormalizeAngle(b.angle) == 0.0f) return {b.x, b.y, b.w, b.h, 0.0f};
  const SinCos r = SinCosf(b.angle * kDegToRad);
  const float sh = r.s * b.h, ch = r.c * b.h;
  const float x1 = b.x + (r.c * b.w), y1 = b.y + (r.s * b.w);
  const P p[4] = {{b.x, b.y}, {x1, y1}, {x1 - sh, y1 + ch}, {b.x + -sh, b.y + ch}};
  P mn = p[0], mx = p[0];
  for (int k = 1; k < 4; k++) {
    if (mn.x > p[k].x) mn.x = p[k].x;
    if (mn.y > p[k].y) mn.y = p[k].y;
    if (p[k].x > mx.x) mx.x = p[k].x;
    if (p[k].y > mx.y) mx.y = p[k].y;
  }
  return {mn.x, mn.y, mx.x - mn.x, mx.y - mn.y, 0.0f};
}

OverlapMap ComputeOverlaps(const std::vector<Box>& boxes, int split) {
  OverlapMap m;
  const int n = static_cast<int>(boxes.size());
  std::vector<Box> aligned;
  for (const Box& b : boxes) aligned.push_back(AlignedBox(b));
  // FindOverlappingAlignedBoxesFromSweep's test on the aligned boxes.
  auto candidate = [&](int i, int j) {
    const Box &p = aligned[i], &q = aligned[j];
    if (!(p.w > 0.0f && p.h > 0.0f && q.w > 0.0f && q.h > 0.0f)) return false;
    const float l = p.x >= q.x ? p.x : q.x;
    float r = p.x + p.w;
    if (q.x + q.w < r) r = q.x + q.w;
    const float w = r - l;
    if (!(w >= 0.0f)) return false;
    const float t = p.y >= q.y ? p.y : q.y;
    float bt = p.y + p.h;
    if (q.y + q.h < bt) bt = q.y + q.h;
    const float h = bt - t;
    return h >= 0.0f && w * h != 0.0f;
  };
  for (int a = 0; a < n; a++) {
    if (split >= 0 && a >= split) break;
    for (int b = split >= 0 ? (a + 1 > split ? a + 1 : split) : a + 1; b < n; b++) {
      if (!candidate(a, b)) continue;
      const Box &A = boxes[a], &B = boxes[b];
      const float inter = BoxOverlapArea(A, B);
      if (inter == 0.0f) continue;
      const float aa = A.w * A.h, ab = B.w * B.h;
      const float iou = inter / ((aa + ab) - inter);
      m[{a, b}] = {iou, inter / aa, inter / ab};
      m[{b, a}] = {iou, inter / ab, inter / aa};
    }
  }
  return m;
}

}  // namespace overlap
}  // namespace screenai
