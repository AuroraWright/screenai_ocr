// SPDX-License-Identifier: MPL-2.0
// Line graph construction for the layout GCN (google_ocr::box_beta_skeleton): neighbor edges between boxes from the
// beta-skeleton (beta = 1, Gabriel graph) of points sampled on the box outlines, plus edges between overlapping and
// contained boxes. Mirrors the original's float/int arithmetic and edge order.
#pragma once

#include <vector>

namespace screenai {

// google_ocr::box_util::Box<float>: (left, top) is the top-left corner before rotating by angle (degrees) about it.
struct BoxF {
  float x = 0, y = 0, w = 0, h = 0, angle = 0;
};

struct GraphEdge {
  int a = 0, b = 0;
  float weight = 0;  // squared distance of the witnessing points (beta-skeleton edges only)
};

// box_util::AppendBoxPoints<Box<float>>: corners in order top-left, top-right, bottom-right, bottom-left.
void BoxCorners(const BoxF& b, float xy[8]);

// GenerateBoxNeighborEdges(boxes, points_per_side_horizontal, points_per_side_vertical, &edges, false, 1, 1).
std::vector<GraphEdge> GenerateBoxNeighborEdges(const std::vector<BoxF>& boxes, int points_horizontal,
                                                int points_vertical);

}  // namespace screenai
