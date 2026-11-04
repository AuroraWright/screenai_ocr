// SPDX-License-Identifier: MPL-2.0
// The full OCR pipeline as the original runs it: (pass-through coarse classifier) -> gray conversion -> text
// detection -> multi-pass line recognition -> page layout analysis -> VisualAnnotation.
#pragma once

#include <memory>
#include <mutex>
#include <string>

#include "detector.h"
#include "image.h"
#include "layout_analyzer.h"
#include "recognizer.h"
#include "resources.h"

namespace screenai {

class OcrEngine {
 public:
  static std::unique_ptr<OcrEngine> Create(std::shared_ptr<Resources> resources);

  // Serialized chrome_screen_ai.VisualAnnotation; false if a stage failed.
  bool Run(const RgbaImage& rgba, std::string* visual_annotation);
  // Light pipeline: one line recognition pass (see LineRecognizer::SetLightMode).
  void SetLightMode(bool light);
  // Same, stopping before the VisualAnnotation conversion.
  bool RunLayout(const RgbaImage& rgba, PageLayout* layout);

 private:
  std::shared_ptr<Resources> resources_;
  std::unique_ptr<TextDetector> detector_;
  std::unique_ptr<LineRecognizer> recognizer_;
  std::unique_ptr<LayoutAnalyzer> layout_analyzer_;
  std::mutex mu_;
};

}  // namespace screenai
