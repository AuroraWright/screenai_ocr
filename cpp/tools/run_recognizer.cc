// SPDX-License-Identifier: MPL-2.0
// run_recognizer <model_dir> <image.png> <in_layout.pb> <out.pb>: line recognition only, from a detector layout
// (e.g. testdata/stages/<img>/001_rec_in.pb) to the layout-analysis input (comparable with 002_lay_in.pb).
#include <cstdio>
#include <fstream>
#include <sstream>

#include "image.h"
#include "page_layout.h"
#include "recognizer.h"
#include "resources.h"

int main(int argc, char** argv) {
  if (argc != 5) return fprintf(stderr, "usage: %s model_dir image.png in_layout.pb out.pb\n", argv[0]), 2;
  auto res = screenai::Resources::FromDirectory(argv[1]);
  auto rec = screenai::LineRecognizer::Create(*res);
  if (!rec) return fprintf(stderr, "failed to load recognizer from %s\n", argv[1]), 1;
  screenai::RgbaImage rgba;
  if (!screenai::LoadImageFile(argv[2], &rgba)) return fprintf(stderr, "cannot read %s\n", argv[2]), 1;
  const screenai::GrayImage gray = screenai::RgbaToGray(rgba);
  std::ifstream in(argv[3], std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  screenai::PageLayout layout;
  if (!screenai::ReadGoodocProto(ss.str(), &layout)) return fprintf(stderr, "cannot parse %s\n", argv[3]), 1;
  if (!rec->Recognize(rgba, gray, &layout)) return fprintf(stderr, "recognition failed\n"), 1;
  std::ofstream(argv[4], std::ios::binary) << screenai::WriteGoodocProto(layout);
}
