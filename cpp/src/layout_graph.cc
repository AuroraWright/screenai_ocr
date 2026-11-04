// SPDX-License-Identifier: MPL-2.0
#include "layout_graph.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>

#include "boost/polygon/voronoi_builder.hpp"
#include "boost/polygon/voronoi_diagram.hpp"

namespace screenai {
namespace {

constexpr float kDegToRad = __builtin_bit_cast(float, 0x3C8EFA35u);

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

struct BoxPoint {
  int x = 0, y = 0;
  std::vector<int> boxes;  // boxes this (scaled, rounded) outline point belongs to
};

using PairSet = std::set<std::pair<int, int>>;

std::pair<int, int> Ordered(int a, int b) { return a < b ? std::make_pair(a, b) : std::make_pair(b, a); }

// GetMaxCoordinateValue: largest |coordinate| over all box corners.
float MaxCoordinate(const std::vector<BoxF>& boxes) {
  float m = 0.0f;
  for (const BoxF& b : boxes) {
    float p[8];
    BoxCorners(b, p);
    for (int k = 0; k < 4; k++) {
      float v = fabsf(p[2 * k]);
      const float v2 = fabsf(p[2 * k + 1]);
      if (v < v2) v = v2;
      if (m < v) m = v;
    }
  }
  return m;
}

// GetScaledPointsFromBoxes (0x100abac5c).
void ScaledPoints(const std::vector<BoxF>& boxes, int nh, int nv, std::vector<BoxPoint>* points,
                  std::vector<std::vector<int>>* box_points, float* scale) {
  *scale = 1073741824.0f / (MaxCoordinate(boxes) + 1.0f);
  box_points->assign(boxes.size(), {});
  std::map<std::pair<int, int>, int> index;
  const float vdiv = static_cast<float>(nv + 1);
  for (size_t bi = 0; bi < boxes.size(); bi++) {
    const BoxF& b = boxes[bi];
    const float w = b.w, h = b.h;
    if (!(w >= 1e-4f) || !(h >= 1e-4f)) continue;
    std::vector<std::pair<float, float>> pts = {{b.x, b.y}, {b.x + w, b.y}, {b.x, b.y + h}, {b.x + w, b.y + h}};
    const float step_v = h / vdiv;
    for (int k = 1; k <= nv; k++) {
      const float d = step_v * static_cast<float>(k);
      pts.emplace_back(b.x, d + b.y);
      pts.emplace_back(b.x + w, d + b.y);
      if (w <= h) pts.emplace_back(b.x + (w * 0.5f), d + b.y);
    }
    int n = nh;
    if (nh < 0) n = static_cast<int>(ceilf(w / step_v)) - 1;
    if (n >= 1) {
      const float step_h = w / static_cast<float>(n + 1);
      for (int k = 1; k <= n; k++) {
        const float d = step_h * static_cast<float>(k);
        pts.emplace_back(d + b.x, b.y);
        pts.emplace_back(d + b.x, b.y + h);
        if (w > h) pts.emplace_back(d + b.x, b.y + (h * 0.5f));
      }
    }
    if (b.angle != 0.0f && pts.size() >= 2) {
      for (size_t k = 1; k < pts.size(); k++) {
        const SinCos r = SinCosf(b.angle * kDegToRad);
        const float dx = pts[k].first - b.x, dy = pts[k].second - b.y;
        pts[k].first = (b.x + (dx * r.c)) - (r.s * dy);
        pts[k].second = (r.c * dy) + (b.y + (dx * r.s));
      }
    }
    for (const auto& [px, py] : pts) {
      const int x = static_cast<int>(std::llround(static_cast<double>(px) * static_cast<double>(*scale)));
      const int y = static_cast<int>(std::llround(static_cast<double>(*scale) * static_cast<double>(py)));
      auto [it, inserted] = index.emplace(std::make_pair(x, y), static_cast<int>(points->size()));
      if (inserted) points->push_back({x, y, {}});
      (*points)[it->second].boxes.push_back(static_cast<int>(bi));
      (*box_points)[bi].push_back(it->second);
    }
  }
}

// GetDelaunay (0x100abb61c): Delaunay neighbors from boost::polygon's Voronoi diagram. Like the original, only the
// first three cells around each Voronoi vertex are connected (co-circular sites share one merged vertex).
std::vector<std::vector<int>> Delaunay(const std::vector<BoxPoint>& points) {
  boost::polygon::voronoi_builder<int> builder;
  for (const BoxPoint& p : points) builder.insert_point(p.x, p.y);
  boost::polygon::voronoi_diagram<double> vd;
  builder.construct(&vd);
  std::vector<std::vector<int>> adj(points.size());
  PairSet seen;
  for (const auto& v : vd.vertices()) {
    const auto* e0 = v.incident_edge();
    int c[3];
    c[0] = static_cast<int>(e0->cell()->source_index());
    const auto* e1 = e0->rot_next();
    if (e1 == e0) continue;
    c[1] = static_cast<int>(e1->cell()->source_index());
    const auto* e2 = e1->rot_next();
    if (e2 == e0) continue;
    c[2] = static_cast<int>(e2->cell()->source_index());
    for (int k = 0; k < 3; k++) {
      const auto pr = Ordered(c[k], c[(k + 1) % 3]);
      if (!seen.insert(pr).second) continue;
      adj[pr.first].push_back(pr.second);
      adj[pr.second].push_back(pr.first);
    }
  }
  return adj;
}

// box_util::BoxContainsPoint<Box<float>, Point> for an axis-aligned/rotated (non-curved) box.
bool BoxContainsPoint(const BoxF& b, float px, float py) {
  const SinCos r = SinCosf(b.angle * -kDegToRad);
  const float dx = px - b.x, dy = py - b.y;
  const float x = (b.x + (dx * r.c)) - (r.s * dy);
  const float y = (r.c * dy) + (b.y + (dx * r.s));
  return !(x < b.x || y < b.y) && x < b.x + b.w && y < b.y + b.h;
}

float Norm2(int dx, int dy) {
  const float fx = static_cast<float>(dx), fy = static_cast<float>(dy);
  return fx * fx + fy * fy;
}

}  // namespace

void BoxCorners(const BoxF& b, float p[8]) {
  p[0] = b.x, p[1] = b.y;
  if (b.angle == 0.0f) {
    p[2] = b.x + b.w, p[3] = b.y;
    p[4] = b.x + b.w, p[5] = b.y + b.h;
    p[6] = b.x, p[7] = b.y + b.h;
    return;
  }
  const SinCos r = SinCosf(b.angle * kDegToRad);
  const float x1 = b.x + (r.c * b.w), y1 = b.y + (r.s * b.w);
  p[2] = x1, p[3] = y1;
  p[4] = x1 - (r.s * b.h), p[5] = y1 + (r.c * b.h);
  p[6] = b.x - (r.s * b.h), p[7] = b.y + (r.c * b.h);
}

std::vector<GraphEdge> GenerateBoxNeighborEdges(const std::vector<BoxF>& boxes, int nh, int nv) {
  std::vector<BoxPoint> points;
  std::vector<std::vector<int>> box_points;
  float scale = 0;
  ScaledPoints(boxes, nh, nv, &points, &box_points, &scale);
  const auto adj = Delaunay(points);
  std::vector<GraphEdge> edges;
  PairSet seen;
  auto add = [&](int a, int b) {
    const auto pr = Ordered(a, b);
    if (seen.insert(pr).second) edges.push_back({pr.first, pr.second, 0.0f});
  };
  // ProcessOverlappingPoints: boxes sharing an outline point.
  for (const BoxPoint& p : points)
    for (size_t i = 0; i < p.boxes.size(); i++)
      for (size_t j = i + 1; j < p.boxes.size(); j++) add(p.boxes[i], p.boxes[j]);
  // MarkInsidePoints: points of other boxes reachable through Delaunay edges inside a box.
  std::vector<bool> inside(points.size(), false);
  std::vector<int> found;
  for (size_t bi = 0; bi < boxes.size(); bi++) {
    BoxF sb = boxes[bi];
    sb.x *= scale, sb.y *= scale, sb.w *= scale, sb.h *= scale;
    found = box_points[bi];
    for (size_t k = 0; k < found.size(); k++) {
      for (int q : adj[found[k]]) {
        if (std::find(found.begin(), found.end(), q) != found.end()) continue;
        if (BoxContainsPoint(sb, static_cast<float>(points[q].x), static_cast<float>(points[q].y))) found.push_back(q);
      }
    }
    for (size_t k = box_points[bi].size(); k < found.size(); k++) {
      const int p = found[k];
      inside[p] = true;
      for (int ob : points[p].boxes) add(static_cast<int>(bi), ob);
    }
  }
  // AddBetaSkeletonEdges (beta = 1): Delaunay edges whose diametral circle holds no neighboring point.
  const float inv_scale2 = 1.0f / (scale * scale);
  std::vector<GraphEdge> cand;
  for (size_t p = 0; p < points.size(); p++) {
    if (inside[p] || adj[p].empty()) continue;
    const BoxPoint& P = points[p];
    for (int q : adj[p]) {
      if (static_cast<int>(p) > q || inside[q]) continue;
      const BoxPoint& Q = points[q];
      const float r2 = Norm2(P.x - Q.x, P.y - Q.y) * 0.25f;
      const int mx = (Q.x + P.x) / 2, my = (Q.y + P.y) / 2;
      bool blocked = false;
      for (int r : adj[p])
        if (r != q && Norm2(points[r].x - mx, points[r].y - my) < r2) blocked = true;
      if (!blocked)
        for (int r : adj[q])
          if (r != static_cast<int>(p) && Norm2(points[r].x - mx, points[r].y - my) < r2) blocked = true;
      if (blocked) continue;
      for (int a : P.boxes)
        for (int b : Q.boxes)
          if (a != b) cand.push_back({a, b, inv_scale2 * Norm2(P.x - Q.x, P.y - Q.y)});
    }
  }
  std::sort(cand.begin(), cand.end(), [](const GraphEdge& x, const GraphEdge& y) { return x.weight < y.weight; });
  for (const GraphEdge& e : cand)
    if (seen.insert(Ordered(e.a, e.b)).second) edges.push_back(e);
  return edges;
}

}  // namespace screenai
