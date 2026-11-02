// bench_ocr <model_dir> <runs> <images...>: median wall time of OcrEngine::Run per image after one warm-up run.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "ocr_engine.h"

int main(int argc, char** argv) {
  if (argc < 4) return fprintf(stderr, "usage: %s model_dir runs images...\n", argv[0]), 2;
  auto engine = screenai::OcrEngine::Create(screenai::Resources::FromDirectory(argv[1]));
  if (!engine) return 1;
  const int runs = std::atoi(argv[2]);
  for (int i = 3; i < argc; i++) {
    screenai::RgbaImage rgba;
    if (!screenai::LoadImageFile(argv[i], &rgba)) return 1;
    std::string out;
    engine->Run(rgba, &out);  // warm-up: lazily loaded script recognizers
    std::vector<double> ms;
    for (int r = 0; r < runs; r++) {
      const auto t = std::chrono::steady_clock::now();
      engine->Run(rgba, &out);
      ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
    }
    std::sort(ms.begin(), ms.end());
    printf("%s: median %.1f ms (min %.1f)\n", argv[i], ms[ms.size() / 2], ms[0]);
  }
}
