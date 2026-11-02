// Text line detection (google_ocr GocrGroupRpnTextDetectionMutator with GroupRpnDetectorV2): gray page image ->
// LINE entities with rotated boxes and PhotoOcrTextDetectionInfo, in the order the original emits them.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "image.h"
#include "page_layout.h"
#include "resources.h"

namespace screenai {

class TfliteModel;

class TextDetector {
 public:
  static std::unique_ptr<TextDetector> Create(const Resources& res);
  ~TextDetector();

  // Appends one LINE entity (followed by its empty PARAGRAPH parent) per detected text line.
  bool Detect(const GrayImage& gray, PageLayout* out);

  struct Config;  // detector settings read from the .binarypb (see detector.cc)

 private:
  TextDetector() = default;
  std::unique_ptr<Config> config_;
  std::unique_ptr<TfliteModel> model_;
};

// One level of the detector's image pyramid (GroupRpnDetectorV2): its scale relative to the page and its image. The
// reference level (index 0) and a level whose image was reused by the next one have no image.
struct DetectorPyramidLevel {
  float scale = 0;
  std::optional<GrayImage> image;
};

// Rebuilds the pyramid as the detector's CropDetections sees it: lines with detection_image_scale < 1000 are cropped
// from the level of that scale (the line recognizer's first-pass images). Empty when the config cannot be read.
std::vector<DetectorPyramidLevel> DetectorPyramid(const Resources& res, const GrayImage& gray);

}  // namespace screenai
