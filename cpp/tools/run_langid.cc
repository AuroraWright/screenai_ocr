// run_langid <model_dir> [--hex] [--topk K]
// Reads one text per stdin line (hex-encoded UTF-8 with --hex) and prints, per line:
//   <language> <confidence> <confidence bits> <softmax bits, comma-separated> <logit bits, comma-separated>
// ("- -" instead of the bits when the model invoke fails, as it does in the original for invalid UTF-8)
// or, with --topk, the K best raw labels and scores. Used by tests/test_langid.py.
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "language_id.h"

using screenai::LanguageIdentifier;

static uint32_t Bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

static std::string Unhex(const std::string& h) {
  std::string out;
  for (size_t i = 0; i + 1 < h.size(); i += 2) out += static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16));
  return out;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <model_dir> [--hex] [--topk K]\n", argv[0]);
    return 2;
  }
  bool hex = false;
  int topk = 0;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--hex")) hex = true;
    if (!std::strcmp(argv[i], "--topk") && i + 1 < argc) topk = std::atoi(argv[++i]);
  }
  auto res = screenai::Resources::FromDirectory(argv[1]);
  auto lid = LanguageIdentifier::Create(*res);
  if (!lid) {
    std::fprintf(stderr, "failed to load %s\n", LanguageIdentifier::kModelPath);
    return 1;
  }
  std::string line;
  while (std::getline(std::cin, line)) {
    const std::string text = hex ? Unhex(line) : line;
    if (topk > 0) {
      for (const auto& p : lid->TopK(text, topk)) std::printf("%s:%.6f ", p.language.c_str(), p.score);
      std::printf("\n");
      continue;
    }
    std::vector<float> sm, lg;
    const bool ok = lid->Scores(text, &sm, &lg);
    const auto r = lid->IdentifyLanguage(text);
    std::printf("%s %.9g %08x ", r.language.c_str(), r.confidence, Bits(r.confidence));
    if (!ok) {  // invoke failed (no tokens: invalid UTF-8)
      std::printf("- -\n");
      continue;
    }
    for (size_t i = 0; i < sm.size(); i++) std::printf("%s%08x", i ? "," : "", Bits(sm[i]));
    std::printf(" ");
    for (size_t i = 0; i < lg.size(); i++) std::printf("%s%08x", i ? "," : "", Bits(lg[i]));
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
