// SPDX-License-Identifier: MPL-2.0
// Overlap areas of rotated boxes as google_ocr::box_util / google_ocr::Rectangle<float> compute them (layout
// analysis: OverlapComputer, SafeOverlapBoxes, SafeBoxOverlapArea). Same float operations in the same order as the
// original (no fused multiply-adds there); checked bit-exact against the library by trace/overlap_check.py +
// tools/overlap_check.cc. See notes/layout.md "Overlap areas".
#pragma once

#include <map>
#include <utility>
#include <vector>

namespace screenai {
namespace overlap {

// box_util Box<float> / RotatedBox<float> (no curve or vertices): (x, y) is the top-left corner before rotation
// about it, angle in degrees clockwise.
struct Box {
  float x = 0, y = 0, w = 0, h = 0, angle = 0;
};

// google_ocr::Rectangle<float>: left, top, width, height, unit direction (cos, sin) of the angle.
struct Rect {
  float l, t, w, h, c, s;
};

// Rectangle<float>::CreateFromLTWHA (0x100c4aec4): sincosf(angle * 0.017453292f).
Rect RectFromLTWHA(float l, float t, float w, float h, float angle);
// Rectangle<float>::IntersectAxisAlignedNonAxisAligned (0x100c4b288); `a` must be axis aligned.
float IntersectAxisAlignedNonAxisAligned(const Rect& a, const Rect& b);
// Rectangle<float>::OverlapArea (0x100c4b134).
float RectOverlapArea(const Rect& a, const Rect& b);

// AxisAlignedBoxOverlapArea<Box<float>> (0x10084e0fc); the RotatedBox<float> / RotatedBoundingBox (int)
// instantiations give the same values for these inputs.
float AxisAlignedOverlapArea(const Box& a, const Box& b);
// SafeBoxOverlapArea<Box<float>> (0x10084d5d4) / SafeOverlapBoxes area for boxes without curve or vertices: both
// angles 0 -> axis aligned, else Rectangle<float> OverlapArea. Angles as given (not normalized).
float BoxOverlapArea(const Box& a, const Box& b);

// CopyBoundingBox<RotatedBox<float> | RotatedBoundingBox, AlignedBox<float>> (0x100812634 / 0x100868e9c): the box
// itself when its normalized angle is 0, else the min/max of its four corners (sincosf of the given angle).
Box AlignedBox(const Box& b);

struct OverlapInfo {
  float iou = 0, over_first = 0, over_second = 0;  // I / union, I / area(first), I / area(second)
};
using OverlapMap = std::map<std::pair<int, int>, OverlapInfo>;

// box_util::OverlapComputer (RotatedBox<float> 0x10086929c, RotatedBoundingBox 0x1008689b4 via SafeOverlapBoxes
// 0x100842280): candidate pairs (lower index first) whose aligned boxes both have w, h > 0 and overlap with a nonzero
// area (FindOverlappingAlignedBoxes); pairs with nonzero BoxOverlapArea are stored as (a, b) and (b, a). With
// split >= 0 only pairs between [0, split) and [split, n).
OverlapMap ComputeOverlaps(const std::vector<Box>& boxes, int split);

}  // namespace overlap
}  // namespace screenai
