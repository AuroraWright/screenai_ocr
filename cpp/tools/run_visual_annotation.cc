// SPDX-License-Identifier: MPL-2.0
// run_visual_annotation <layout.pb> <out.pb>: goodoc.PageLayoutProto -> chrome_screen_ai.VisualAnnotation.
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "page_layout.h"
#include "visual_annotation.h"

int main(int argc, char** argv) {
  if (argc != 3) return fprintf(stderr, "usage: %s layout.pb out.pb\n", argv[0]), 2;
  std::ifstream f(argv[1], std::ios::binary);
  std::string in((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  screenai::PageLayout layout;
  if (!screenai::ReadGoodocProto(in, &layout)) return fprintf(stderr, "parse failed\n"), 1;
  std::ofstream(argv[2], std::ios::binary) << screenai::ConvertToVisualAnnotation(layout);
}
