// run_detector <model_dir> <image.png> <out.pb>: text line detection only, written as goodoc.PageLayoutProto
// (comparable with the original's recognizer-input stage dump, testdata/stages/<img>/001_rec_in.pb).
#include <cstdio>
#include <fstream>

#include "detector.h"
#include "image.h"
#include "page_layout.h"
#include "resources.h"

int main(int argc, char** argv) {
  if (argc != 4) return fprintf(stderr, "usage: %s model_dir image.png out.pb\n", argv[0]), 2;
  auto res = screenai::Resources::FromDirectory(argv[1]);
  auto det = screenai::TextDetector::Create(*res);
  if (!det) return fprintf(stderr, "failed to load detector from %s\n", argv[1]), 1;
  screenai::RgbaImage rgba;
  if (!screenai::LoadImageFile(argv[2], &rgba)) return fprintf(stderr, "cannot read %s\n", argv[2]), 1;
  const screenai::GrayImage gray = screenai::RgbaToGray(rgba);
  screenai::PageLayout layout;
  layout.width = gray.width;
  layout.height = gray.height;
  if (!det->Detect(gray, &layout)) return fprintf(stderr, "detection failed\n"), 1;
  std::ofstream(argv[3], std::ios::binary) << screenai::WriteGoodocProto(layout);
}
