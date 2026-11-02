#include "ocr_engine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "visual_annotation.h"

namespace screenai {

std::unique_ptr<OcrEngine> OcrEngine::Create(std::shared_ptr<Resources> resources) {
  auto engine = std::unique_ptr<OcrEngine>(new OcrEngine);
  engine->resources_ = std::move(resources);
  engine->detector_ = TextDetector::Create(*engine->resources_);
  engine->recognizer_ = LineRecognizer::Create(*engine->resources_);
  engine->layout_analyzer_ = LayoutAnalyzer::Create(*engine->resources_);
  if (!engine->detector_ || !engine->recognizer_ || !engine->layout_analyzer_) return nullptr;
  return engine;
}

namespace {

// SCREENAI_STAGE_DUMP=<dir>: writes each stage's output as goodoc.PageLayoutProto, named like trace/stage_dump.py
// names the original's (001_rec_in = detector output, 002_lay_in = recognizer output, 003_final).
void DumpStage(const PageLayout& layout, const char* name) {
  if (const char* dir = std::getenv("SCREENAI_STAGE_DUMP"))
    std::ofstream(std::string(dir) + "/" + name + ".pb", std::ios::binary) << WriteGoodocProto(layout);
}

// SCREENAI_TIMING=1: per-stage wall time on stderr.
class StageTimer {
 public:
  StageTimer() : on_(std::getenv("SCREENAI_TIMING") != nullptr), t_(std::chrono::steady_clock::now()) {}
  void Mark(const char* stage) {
    if (!on_) return;
    const auto now = std::chrono::steady_clock::now();
    std::fprintf(stderr, "%s %.1f ms  ", stage, std::chrono::duration<double, std::milli>(now - t_).count());
    t_ = now;
  }
  ~StageTimer() {
    if (on_) std::fprintf(stderr, "\n");
  }

 private:
  bool on_;
  std::chrono::steady_clock::time_point t_;
};

}  // namespace

bool OcrEngine::RunLayout(const RgbaImage& rgba, PageLayout* layout) {
  std::lock_guard<std::mutex> lock(mu_);
  StageTimer timer;
  const GrayImage gray = RgbaToGray(rgba);
  timer.Mark("gray");
  *layout = PageLayout();
  // The stages read the page size from the layout (the original keeps it in the mutator context).
  layout->width = gray.width;
  layout->height = gray.height;
  if (!detector_->Detect(gray, layout)) return false;
  timer.Mark("detect");
  DumpStage(*layout, "001_rec_in");
  if (!recognizer_->Recognize(rgba, gray, layout)) return false;
  timer.Mark("recognize");
  DumpStage(*layout, "002_lay_in");
  if (!layout_analyzer_->Analyze(gray, layout)) return false;
  timer.Mark("layout");
  DumpStage(*layout, "003_final");
  return true;
}

bool OcrEngine::Run(const RgbaImage& rgba, std::string* visual_annotation) {
  PageLayout layout;
  if (!RunLayout(rgba, &layout)) return false;
  *visual_annotation = ConvertToVisualAnnotation(layout);
  return true;
}

}  // namespace screenai
