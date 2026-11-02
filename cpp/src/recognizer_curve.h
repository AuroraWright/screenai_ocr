// Curved text lines (LINE polygons that are a goodoc.CurvedBoundingBox):
//   ocr::photo::ImageUtil::ImageCropCurvedPatch (box_util::CurvedBoxCentersAndAngles + ApplyThinPlateSplineWarp:
//   Eigen thin-plate-spline solve, then the ThinPlateSplineInterpolation / BilinearInterpolationFloat /
//   BilinearInterpolationU8 Halide pipelines) straightens the line into the recognizer's line image, and
//   aksara::box_utils::BoundingBoxCoordinateMapper (InitFromMidLineCurve, MapToSourceCoordinates) maps boxes found
//   in that image back onto the page (BoxRelativeToPageRelativeVec, curved branch).
#pragma once

#include <utility>
#include <vector>

#include "image.h"
#include "page_layout.h"

namespace screenai::rec {

// Line image of a curved line: (int)(center-line length + 1.5) x (int)thickness pixels. Empty when the original
// fails (degenerate curve, or less than 3 pixels thick).
GrayImage CurvedLineImage(const GrayImage& page, const Curve& curve, bool float_thickness = true);

// Line-image -> page coordinates along the mid-line curve, one sample per pixel of arc length.
class CurveMapper {
 public:
  explicit CurveMapper(const Curve& curve);
  std::pair<double, double> MapToSource(int x, int y) const;

 private:
  int along_ = 0, across_ = 0;  // samples along the curve, thickness
  bool top_to_bottom_ = false;
  std::vector<std::pair<double, double>> mid_;  // source_mid_points_
  std::vector<double> dx_down_, dy_down_;
  double first_cos_ = 0, first_sin_ = 0, last_cos_ = 0, last_sin_ = 0;
};

// BoxRelativeToPageRelativeVec for a curved line: the box's top-left and (rotated) top-right corners are mapped;
// size is kept, the angle follows the mapped top edge.
RotatedBox CurveToPage(const CurveMapper& mapper, const RotatedBox& rel);

}  // namespace screenai::rec
