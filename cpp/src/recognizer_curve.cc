// SPDX-License-Identifier: MPL-2.0
#include "recognizer_curve.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "recognizer_words.h"

// The thin-plate-spline system is solved with the same Eigen version and float mode as the original (vendored in
// third_party/eigen; other Eigen versions round differently). The namespace is renamed so these template
// instantiations never merge with the (older) Eigen inside the TensorFlow Lite libraries.
#define Eigen screenai_eigen
#include "../../third_party/eigen/Eigen/Core"
#include "../../third_party/eigen/Eigen/LU"
#undef Eigen

namespace screenai::rec {

namespace {

void SinCosD(double x, double* s, double* c) {
#if defined(__APPLE__)
  const __double2 r = __sincos_stret(x);
  *s = r.__sinval, *c = r.__cosval;
#else
  *s = sin(x), *c = cos(x);
#endif
}

struct Correspondence {
  float tx, ty;  // line image (normalized, later sample grid) coordinates
  float sx, sy;  // page coordinates
};

// box_util::CurvedBoxCentersAndAngles: n + 1 (center, normal angle) pairs: the first point, the midpoints of the
// segments and the last point, each with the angle of its segment turned to point "down" across the line.
bool CentersAndAngles(const std::vector<std::pair<double, double>>& p, std::vector<float>* out) {
  const int n = static_cast<int>(p.size());
  if (n < 2) return false;
  auto push = [&](const std::pair<double, double>& a, const std::pair<double, double>& b, int j) {
    const double ang = atan2(p[j].first - p[j - 1].first, p[j - 1].second - p[j].second);
    out->push_back(static_cast<float>((a.first + b.first) * 0.5));
    out->push_back(static_cast<float>((a.second + b.second) * 0.5));
    out->push_back(static_cast<float>(ang));
  };
  push(p[0], p[0], 1);
  for (int i = 1; i <= n; i++) {
    const int j = std::min(i, n - 1);
    push(p[i - 1], p[j], j);
  }
  return true;
}

float BitsToFloat(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// Halide's float log (range reduction to [0.75, 1.5) + odd/even polynomial), times r2: the TPS kernel r2 log r2.
float TpsKernel(float r2) {
  static const float c0 = BitsToFloat(0x3D9C7946), c2 = BitsToFloat(0x3E5333C6), c4 = BitsToFloat(0x3EAA99CD);
  static const float c1 = BitsToFloat(0xBE266E2A), c3 = BitsToFloat(0xBE809085), c5 = BitsToFloat(0xBEFFFCBE);
  static const float ln2 = BitsToFloat(0x3F317218);
  if (!(r2 > 0.0f)) return 0.0f;
  int32_t b;
  memcpy(&b, &r2, 4);
  const int32_t m = b & static_cast<int32_t>(0x807FFFFF);
  const int32_t t = m >> 22;
  const int32_t rb = m - (t << 23) + 0x3F800000;
  float reduced;
  memcpy(&reduced, &rb, 4);
  const float x1 = reduced + -1.0f;
  const float e = static_cast<float>((b >> 23) + t + -127);
  const float x2 = x1 * x1;
  const float odd = x1 * (x2 * (x2 * (x2 * c0 + c2) + c4) + 1.0f);
  const float even = x2 * (x2 * (x2 * c1 + c3) + c5);
  return r2 * (e * ln2 + (even + odd));
}

// BilinearInterpolationFloatHalide: samples (pixel centers at +0.5, edges clamped), along y first.
float SampleF(const float* src, int w, int h, float x, float y) {
  const float px = x + -0.5f, py = y + -0.5f;
  const float fx = floorf(px), fy = floorf(py);
  const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
  const int x0 = std::max(std::min(ix, w - 1), 0), x1 = std::max(std::min(ix + 2, w) - 1, 0);
  const int y0 = std::max(std::min(iy, h - 1), 0), y1 = std::max(std::min(iy + 2, h) - 1, 0);
  const float wx = px - fx, wy = py - fy;
  auto at = [&](int xx, int yy) { return src[static_cast<size_t>(yy) * w + xx]; };
  const float a = at(x0, y0) * (1.0f - wy) + wy * at(x0, y1);
  const float b = (1.0f - wy) * at(x1, y0) + wy * at(x1, y1);
  return (1.0f - wx) * a + wx * b;
}

int Div255(int v) { return (v + ((v + 128) >> 8) + 128) >> 8; }

// BilinearInterpolationU8Halide: 8-bit weights, exact /255 rounding, along y first.
uint8_t SampleU8(const GrayImage& img, float x, float y) {
  const float px = x + -0.5f, py = y + -0.5f;
  const float fx = floorf(px), fy = floorf(py);
  const int ix = static_cast<int>(fx), iy = static_cast<int>(fy);
  const int wx = static_cast<int>((px - fx) * 255.0f), wy = static_cast<int>((py - fy) * 255.0f);
  const int w = img.width, h = img.height;
  const int x0 = std::max(std::min(ix, w - 1), 0), x1 = std::max(std::min(ix + 2, w) - 1, 0);
  const int y0 = std::max(std::min(iy, h - 1), 0), y1 = std::max(std::min(iy + 2, h) - 1, 0);
  auto at = [&](int xx, int yy) { return static_cast<int>(img.data[static_cast<size_t>(yy) * w + xx]); };
  const int c0 = Div255(at(x0, y0) * (255 - wy) + at(x0, y1) * wy);
  const int c1 = Div255(at(x1, y0) * (255 - wy) + at(x1, y1) * wy);
  return static_cast<uint8_t>(Div255(c0 * (255 - wx) + c1 * wx));
}

}  // namespace

GrayImage CurvedLineImage(const GrayImage& page, const Curve& curve, bool float_thickness) {
  GrayImage out;
  std::vector<float> ca;
  if (page.width <= 0 || page.height <= 0 || !CentersAndAngles(curve.points, &ca)) return out;
  // A goodoc polygon's thickness reaches the ocr::photo::CurvedBoundingBox as (double)(float); the detector's own
  // box (pyramid crops in CropDetections) keeps the double.
  const float thick_f = static_cast<float>(curve.thickness);
  const double thick = float_thickness ? static_cast<double>(thick_f) : curve.thickness;
  std::vector<Correspondence> corr;
  float len = 0.0f;
  for (size_t i = 0; i < ca.size(); i += 3) {
    const float cx = ca[i], cy = ca[i + 1];
    float s, c;
    SinCos(ca[i + 2], &s, &c);
    if (i > 0) len = len + hypotf(cx - ca[i - 3], cy - ca[i - 2]);
    const double half = thick * 0.5;
    const float dx = static_cast<float>(half * static_cast<double>(c));
    const float dy = static_cast<float>(half * static_cast<double>(s));
    corr.push_back({len, 0.0f, cx - dx, cy - dy});
    corr.push_back({len, thick_f, cx + dx, cy + dy});
  }
  const int W = static_cast<int>(len + 1.5f), H = static_cast<int>(thick_f);
  if (W < 1 || H < 1) return out;
  for (Correspondence& k : corr) k.tx = k.tx / static_cast<float>(W), k.ty = k.ty / static_cast<float>(H);

  // ApplyThinPlateSplineWarp: the spline lives on a (W/4 + 2) x (H/4 + 2) grid (plus a one-sample border).
  const int sw = W / 4 + 2, sh = H / 4 + 2;
  if (sw > W || sh > H) return out;
  for (Correspondence& k : corr) {
    k.tx = k.tx * static_cast<float>(sw) + 1.0f;
    k.ty = k.ty * static_cast<float>(sh) + 1.0f;
  }
  // ComputeSplineCoefficients: [K P; P^T 0] [w; a] = [src; 0], K_ij = r2 log r2, solved by PartialPivLU<MatrixXf>.
  const int n = static_cast<int>(corr.size()), N = n + 3;
  screenai_eigen::MatrixXf A = screenai_eigen::MatrixXf::Zero(N, N);
  screenai_eigen::MatrixXf rhs = screenai_eigen::MatrixXf::Zero(N, 2);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float v = 0.0f;
      if (i != j) {
        const float dx = corr[i].tx - corr[j].tx, dy = corr[i].ty - corr[j].ty;
        const float r2 = dx * dx + dy * dy;
        if (r2 > 0.0f) v = r2 * logf(r2);
      }
      A(i, j) = v;
    }
    A(n, i) = A(i, n) = 1.0f;
    A(n + 1, i) = A(i, n + 1) = corr[i].tx;
    A(n + 2, i) = A(i, n + 2) = corr[i].ty;
    rhs(i, 0) = corr[i].sx;
    rhs(i, 1) = corr[i].sy;
  }
  const screenai_eigen::MatrixXf coef = screenai_eigen::PartialPivLU<screenai_eigen::MatrixXf>(A).solve(rhs);

  // ThinPlateSplineInterpolationHalide: page coordinates at the grid samples (centers at +0.5).
  const int gw = sw + 2, gh = sh + 2;
  std::vector<float> grid[2];
  for (int ch = 0; ch < 2; ch++) grid[ch].resize(static_cast<size_t>(gw) * gh);
  std::vector<float> acc0(gw), acc1(gw);
  for (int gy = 0; gy < gh; gy++) {
    const float y = static_cast<float>(gy) + 0.5f;
    std::fill(acc0.begin(), acc0.end(), 0.0f);
    std::fill(acc1.begin(), acc1.end(), 0.0f);
    for (int i = 0; i < n; i++) {
      const float dy = corr[i].ty - y;
      const float dy2 = dy * dy;
      const float w0 = coef(i, 0), w1 = coef(i, 1);
      for (int gx = 0; gx < gw; gx++) {
        const float dx = corr[i].tx - (static_cast<float>(gx) + 0.5f);
        const float u = TpsKernel(dx * dx + dy2);
        acc0[gx] = acc0[gx] + w0 * u;
        acc1[gx] = acc1[gx] + w1 * u;
      }
    }
    const float ya0 = y * coef(n + 2, 0), ya1 = y * coef(n + 2, 1);
    for (int gx = 0; gx < gw; gx++) {
      const float x = static_cast<float>(gx) + 0.5f;
      grid[0][static_cast<size_t>(gy) * gw + gx] = ((acc0[gx] + coef(n, 0)) + x * coef(n + 1, 0)) + ya0;
      grid[1][static_cast<size_t>(gy) * gw + gx] = ((acc1[gx] + coef(n, 1)) + x * coef(n + 1, 1)) + ya1;
    }
  }

  // Each output pixel: its grid position, the interpolated page position there, the page pixel at that position.
  const float fx = static_cast<float>(sw) / static_cast<float>(W), fy = static_cast<float>(sh) / static_cast<float>(H);
  out.width = W, out.height = H;
  out.data.resize(static_cast<size_t>(W) * H);
  for (int v = 0; v < H; v++) {
    const float gy = fy * (static_cast<float>(v) + 0.5f) + 1.0f;
    for (int u = 0; u < W; u++) {
      const float gx = fx * (static_cast<float>(u) + 0.5f) + 1.0f;
      const float px = SampleF(grid[0].data(), gw, gh, gx, gy);
      const float py = SampleF(grid[1].data(), gw, gh, gx, gy);
      out.data[static_cast<size_t>(v) * W + u] = SampleU8(page, px, py);
    }
  }
  return out;
}

CurveMapper::CurveMapper(const Curve& curve) {
  const std::vector<std::pair<double, double>>& p = curve.points;
  top_to_bottom_ = curve.top_to_bottom.value_or(false);
  const int n = static_cast<int>(p.size());
  if (n < 2) return;  // never mapped: the original leaves the mapper without samples
  const double offset = 0.0;  // BoundingBoxCoordinateMapper(Box<double>, 0)
  std::vector<double> ang;
  for (int i = 0; i + 1 < n; i++) ang.push_back(atan2(p[i + 1].second - p[i].second, p[i + 1].first - p[i].first));
  first_cos_ = cos(ang.front()), first_sin_ = sin(ang.front());
  last_cos_ = cos(ang.back()), last_sin_ = sin(ang.back());
  // GetAverageAngle: linear blend, taking the short way around.
  auto average = [](double a2, double a3, double t) {
    if (std::fabs(a3 - a2) > M_PI) {
      if (a2 >= a3) a3 += 2 * M_PI;
      else a2 += 2 * M_PI;
    }
    return (1.0 - t) * a3 + t * a2;
  };
  const int last = static_cast<int>(ang.size()) - 1;
  double cum = 0.0;
  int k = 0;
  for (int i = 0; i + 1 < n; i++) {
    const double sx = p[i + 1].first - p[i].first, sy = p[i + 1].second - p[i].second;
    const double seg = sqrt(sx * sx + sy * sy);
    const double prev = cum;
    cum = cum + seg;
    if (!(cum > static_cast<double>(k))) continue;
    const double bx = p[i].first + offset * (sx / seg), by = p[i].second + offset * (sy / seg);
    do {
      const double t = (static_cast<double>(k) - prev) / seg;
      const double a = t >= 0.5 ? average(i != last ? ang[i + 1] : ang[i], ang[i], t + -0.5)
                                : average(ang[i], i != 0 ? ang[i - 1] : ang[i], t + 0.5);
      double s, c;
      SinCosD(a + M_PI_2, &s, &c);
      dx_down_.push_back(c);
      dy_down_.push_back(s);
      mid_.emplace_back(bx + sx * t + c * offset, by + sy * t + offset * s);
      k++;
    } while (cum > static_cast<double>(k));
  }
  along_ = k;
  across_ = static_cast<int>(curve.thickness);
}

std::pair<double, double> CurveMapper::MapToSource(int x, int y) const {
  if (along_ < 1 || across_ < 1) return {0.0, 0.0};
  const int a = top_to_bottom_ ? y : x;
  const int across = top_to_bottom_ ? across_ + ~x : y;
  const double d = static_cast<double>(across) + static_cast<double>(static_cast<unsigned>(across_)) * -0.5;
  if (a < 0) {
    return {(mid_.front().first + first_cos_ * static_cast<double>(a)) + d * dx_down_.front(),
            (mid_.front().second + first_sin_ * static_cast<double>(a)) + d * dy_down_.front()};
  }
  if (a >= along_) {
    const double e = static_cast<double>(static_cast<unsigned>(a - along_ + 1));
    return {(mid_.back().first + last_cos_ * e) + d * dx_down_.back(),
            (mid_.back().second + last_sin_ * e) + d * dy_down_.back()};
  }
  return {mid_[a].first + d * dx_down_[a], mid_[a].second + d * dy_down_[a]};
}

RotatedBox CurveToPage(const CurveMapper& mapper, const RotatedBox& rel) {
  // aksara::box_utils::Point::Rotate of the top-right corner about the top-left one.
  const double ox = static_cast<double>(rel.left), oy = static_cast<double>(rel.top);
  const double dx = static_cast<double>(rel.left + rel.width) - ox, dy = oy - oy;
  double s, c;
  SinCosD(static_cast<double>(rel.angle) * M_PI / 180.0, &s, &c);
  const double rx = ox + (c * dx - s * dy), ry = oy + (s * dx + c * dy);
  const auto p0 = mapper.MapToSource(static_cast<int>(ox), static_cast<int>(oy));
  const auto p1 = mapper.MapToSource(static_cast<int>(rx), static_cast<int>(ry));
  RotatedBox out = rel;
  out.left = static_cast<int>(p0.first);
  out.top = static_cast<int>(p0.second);
  out.angle = static_cast<float>(atan2(p1.second - p0.second, p1.first - p0.first) * 57.29577951308232);
  return out;
}

}  // namespace screenai::rec
