// In-memory page layout passed between the OCR stages (detection -> recognition -> layout analysis -> output).
// Field meanings follow the original engine's goodoc.PageLayoutEntity; only fields the pipeline uses are kept.
// WriteGoodocProto() serializes into goodoc.PageLayoutProto wire format so stages can be diffed against the
// layouts dumped from the original library (trace/stage_dump.py).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace screenai {

// Curved line geometry (goodoc.CurvedBoundingBox / ocr.photo.CurvedBoundingBox): a polyline along the middle of
// the text plus its thickness. The detector emits it for lines whose raw boxes bend.
struct Curve {
  std::vector<std::pair<double, double>> points;  // mid_line_curve, page coordinates
  double thickness = 0;
  std::optional<bool> top_to_bottom;  // set on goodoc polygons, unset in the detector's ocr.photo copy
};

// Rotated rectangle: (left, top) is the top-left corner *before* rotation about it; angle in degrees, clockwise.
// When `curve` is set the box is the curve's enclosing rotated_bounding_box, and the polygon is serialized as a
// curved_box instead of a rotated_box.
struct RotatedBox {
  int left = 0, top = 0, width = 0, height = 0;
  float angle = 0;
  std::optional<Curve> curve;
};

enum class EntityType { kWord = 0, kLine = 2, kParagraph = 3, kBlock = 4, kSymbol = 5 };

// google_ocr.PhotoOcrTextDetectionInfo (extension on LINE entities produced by the detector).
struct DetectionInfo {
  RotatedBox detection_box;
  int detection_image_scale = 1000;
  float average_raw_detections_height = 0;
  float variance_raw_detections_height = 0;
  float detection_nms_score = 0;
  int num_raw_boxes = 0;
  float sum_raw_boxes_scores = 0;
  float left_padding = 0, right_padding = 0;
  std::optional<float> garbage_prob;
  bool fallback_upscaling = false;
};

struct Colors {
  int foreground_gray = 0, background_gray = 0;
  std::optional<int> foreground_rgb, background_rgb;
  std::optional<int> foreground_gray_threshold;
  std::vector<std::pair<int, int>> foreground_samples, background_samples;
};

struct WordBreak {
  int type = 0;  // goodoc.Break.BreakType: 1 NONE, 2 SPACE
  std::optional<RotatedBox> box;
};

// aksara.TextLineResult carried on LINE entities (aksara.api_internal.InternalEntityExt.result).
struct AksaraResult {
  std::optional<std::string> text, language, predicted_script;
  std::optional<float> confidence;
  std::optional<bool> right_to_left;
};

struct Entity {
  EntityType type = EntityType::kLine;
  int parent = -1;  // index into PageLayout::entities
  std::optional<RotatedBox> box;
  std::optional<int> content_type;  // goodoc ContentType (0 = TEXT)
  // orientation (goodoc.OrientationLabel): page orientation, writing direction, textline order
  std::optional<int> orientation, writing_direction, textline_order;
  std::optional<float> confidence, detection_confidence, type_confidence, direction_confidence;
  std::optional<int> predicted_direction;
  std::vector<std::string> languages;  // language_mix bcp47 tags
  std::optional<std::string> text;
  std::optional<std::string> recognizer_script_id;
  std::optional<WordBreak> word_break;
  std::optional<Colors> colors;
  std::optional<DetectionInfo> detection;
  std::optional<AksaraResult> aksara;
};

struct PageLayout {
  int width = 0, height = 0;
  std::vector<Entity> entities;
  // Page-level goodoc.LanguageCombination (bcp47 tag, weight), set by layout analysis.
  std::optional<std::vector<std::pair<std::string, float>>> language_mix;
  std::optional<std::string> creator_id;
};

// goodoc.PageLayoutProto wire format (subset).
std::string WriteGoodocProto(const PageLayout& layout);
// Parses the same subset from a goodoc.PageLayoutProto (e.g. a stage dump of the original library).
bool ReadGoodocProto(std::string_view bytes, PageLayout* layout);

}  // namespace screenai
