// Reads a goodoc.PageLayoutProto into screenai::PageLayout and writes it back (tests reader/writer coverage).
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "page_layout.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream f(argv[1], std::ios::binary);
  std::string in((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  screenai::PageLayout layout;
  if (!screenai::ReadGoodocProto(in, &layout)) return fprintf(stderr, "parse failed\n"), 1;
  std::ofstream(argv[2], std::ios::binary) << screenai::WriteGoodocProto(layout);
  printf("%zu entities\n", layout.entities.size());
}
