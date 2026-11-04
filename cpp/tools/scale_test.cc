// SPDX-License-Identifier: MPL-2.0
// Reads cases from stdin: "sw sh dw dh mode path_in path_out" per line; writes ScalePlane output for each.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "libyuv_scale.h"

int main() {
  int sw, sh, dw, dh, mode;
  std::string in, out;
  while (std::cin >> sw >> sh >> dw >> dh >> mode >> in >> out) {
    std::ifstream f(in, std::ios::binary);
    std::vector<uint8_t> src(static_cast<size_t>(sw) * sh), dst(static_cast<size_t>(dw) * dh);
    f.read(reinterpret_cast<char*>(src.data()), src.size());
    screenai::ScalePlane(src.data(), sw, sw, sh, dst.data(), dw, dw, dh, static_cast<screenai::FilterMode>(mode));
    std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(dst.data()), dst.size());
  }
}
