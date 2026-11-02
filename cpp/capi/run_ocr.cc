// run_ocr <model_dir> <image> <out_visual_annotation.pb> [out_layout.pb]: the whole pipeline on one image file.
#include <cstdio>
#include <fstream>
#include <string>

#include "ocr_engine.h"
#include "visual_annotation.h"

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) return fprintf(stderr, "usage: %s model_dir image out.pb [layout.pb]\n", argv[0]), 2;
  auto engine = screenai::OcrEngine::Create(screenai::Resources::FromDirectory(argv[1]));
  if (!engine) return fprintf(stderr, "failed to load models from %s\n", argv[1]), 1;
  screenai::RgbaImage rgba;
  if (!screenai::LoadImageFile(argv[2], &rgba)) return fprintf(stderr, "cannot read %s\n", argv[2]), 1;
  screenai::PageLayout layout;
  if (!engine->RunLayout(rgba, &layout)) return fprintf(stderr, "OCR failed\n"), 1;
  std::ofstream(argv[3], std::ios::binary) << screenai::ConvertToVisualAnnotation(layout);
  if (argc == 5) std::ofstream(argv[4], std::ios::binary) << screenai::WriteGoodocProto(layout);
}
