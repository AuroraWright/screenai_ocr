// SPDX-License-Identifier: MPL-2.0
// run_layout <model_dir> <image> <in.pb> <out.pb> [--step NAME] [--features out.bin]
// Runs the layout analysis (PageLayoutAnalysisMutator) on a goodoc.PageLayoutProto (e.g. a 002_lay_in.pb stage
// dump) and writes the resulting layout. --step runs a single stage (see LayoutAnalyzer::RunStep). --features
// writes the ClusterSortGcn model inputs of the input layout (int32 n, int32 e, n*30 floats, 2*e int64, then the
// n raw node outputs when the model ran).
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include "image.h"
#include "layout_analyzer.h"
#include "page_layout.h"

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: %s <model_dir> <image> <in.pb> <out.pb> [--step NAME] [--features]\n", argv[0]);
    return 2;
  }
  std::string step, features;
  for (int i = 5; i + 1 < argc; i++) {
    if (!std::strcmp(argv[i], "--step")) step = argv[++i];
    else if (!std::strcmp(argv[i], "--features")) features = argv[++i];
  }
  auto res = screenai::Resources::FromDirectory(argv[1]);
  auto analyzer = screenai::LayoutAnalyzer::Create(*res);
  if (!analyzer) return std::fprintf(stderr, "failed to load %s\n", screenai::LayoutAnalyzer::kModelPath), 1;
  screenai::RgbaImage rgba;
  if (!screenai::LoadImageFile(argv[2], &rgba)) return std::fprintf(stderr, "cannot read %s\n", argv[2]), 1;
  const screenai::GrayImage gray = screenai::RgbaToGray(rgba);
  std::ifstream f(argv[3], std::ios::binary);
  const std::string in((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  screenai::PageLayout layout;
  if (!screenai::ReadGoodocProto(in, &layout)) return std::fprintf(stderr, "cannot parse %s\n", argv[3]), 1;
  if (layout.width == 0) layout.width = gray.width, layout.height = gray.height;
  if (!features.empty()) {
    std::vector<float> feat;
    std::vector<int64_t> s, r;
    analyzer->ClusterSortInputs(layout, &feat, &s, &r);
    std::ofstream o(features, std::ios::binary);
    const int32_t n = static_cast<int32_t>(feat.size() / 30), e = static_cast<int32_t>(s.size());
    o.write(reinterpret_cast<const char*>(&n), 4).write(reinterpret_cast<const char*>(&e), 4);
    o.write(reinterpret_cast<const char*>(feat.data()), feat.size() * 4);
    o.write(reinterpret_cast<const char*>(s.data()), s.size() * 8).write(reinterpret_cast<const char*>(r.data()), r.size() * 8);
    std::vector<bool> pred;
    std::vector<float> raw;
    if (analyzer->NodePredictions(layout, &pred, &raw)) o.write(reinterpret_cast<const char*>(raw.data()), raw.size() * 4);
  }
  const bool ok = step.empty() ? analyzer->Analyze(gray, &layout) : analyzer->RunStep(step, gray, &layout);
  if (!ok) return std::fprintf(stderr, "layout analysis failed\n"), 1;
  std::ofstream(argv[4], std::ios::binary) << screenai::WriteGoodocProto(layout);
  std::printf("%zu entities\n", layout.entities.size());
  return 0;
}
