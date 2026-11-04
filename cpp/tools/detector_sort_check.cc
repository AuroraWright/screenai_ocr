// SPDX-License-Identifier: MPL-2.0
// detector_sort_check <dir>: replays the arrays trace/det_sort_check.py sorted with the original library's own
// std::sort instantiations (one file per detector sort site) through detsort::Sort (cpp/src/detector_sort.h) and
// counts cases whose resulting element order differs.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "detector_sort.h"

namespace {

template <size_t N>
struct Elem {
  unsigned char b[N];
};

template <size_t N, class C>
int Check(const std::string& dir, const char* site, C comp) {
  std::ifstream f(dir + "/" + site + ".bin", std::ios::binary);
  std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (d.size() < 8) return fprintf(stderr, "%s: cannot read\n", site), 1;
  size_t o = 0;
  auto u32 = [&] {
    uint32_t v;
    std::memcpy(&v, d.data() + o, 4);
    o += 4;
    return v;
  };
  if (u32() != N) return fprintf(stderr, "%s: element size mismatch\n", site), 1;
  const uint32_t ncases = u32();
  int bad = 0, ties = 0;
  for (uint32_t k = 0; k < ncases; k++) {
    const uint32_t n = u32();
    std::vector<Elem<N>> in(n), ref(n);
    std::memcpy(in.data(), d.data() + o, n * N), o += n * N;
    std::memcpy(ref.data(), d.data() + o, n * N), o += n * N;
    screenai::detsort::Sort(in.begin(), in.end(), comp);
    if (std::memcmp(in.data(), ref.data(), n * N)) bad++;
    for (uint32_t i = 0; i + 1 < n; i++) ties += !comp(in[i], in[i + 1]) && !comp(in[i + 1], in[i]);
  }
  printf("%-16s %u cases (%d tied adjacent pairs): %d mismatches\n", site, ncases, ties, bad);
  return bad != 0;
}

template <size_t N>
float F(const Elem<N>& e, size_t off) {
  float v;
  std::memcpy(&v, e.b + off, 4);
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) return fprintf(stderr, "usage: %s dir\n", argv[0]), 2;
  const std::string dir = argv[1];
  int r = 0;
  r |= Check<24>(dir, "line_links", [](const Elem<24>& a, const Elem<24>& b) { return F(a, 20) < F(b, 20); });
  r |= Check<8>(dir, "curve_points", [](const Elem<8>& a, const Elem<8>& b) {
    // key = (dir.x * (p.x - o.x) + 0) + dir.y * (p.y - o.y) with dir (1, 0) and origin (0, 0)
    return ((1.0f * (F(a, 0) - 0.0f)) + 0.0f) + 0.0f * (F(a, 4) - 0.0f) <
           ((1.0f * (F(b, 0) - 0.0f)) + 0.0f) + 0.0f * (F(b, 4) - 0.0f);
  });
  r |= Check<8>(dir, "line_order", [](const Elem<8>& a, const Elem<8>& b) { return F(a, 0) < F(b, 0); });
  r |= Check<56>(dir, "paragraph_nms", [](const Elem<56>& a, const Elem<56>& b) { return F(a, 48) > F(b, 48); });
  r |= Check<12>(dir, "angle_estimation", [](const Elem<12>& a, const Elem<12>& b) { return F(a, 8) < F(b, 8); });
  r |= Check<288>(dir, "line_paragraph", [](const Elem<288>& a, const Elem<288>& b) {
    int32_t x, y;
    std::memcpy(&x, a.b + 276, 4), std::memcpy(&y, b.b + 276, 4);
    return x < y;
  });
  return r;
}
