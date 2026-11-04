// SPDX-License-Identifier: MPL-2.0
// select_check <model_dir> <dir> [<likely_dump_dir>]: replays <dir>/grid.txt (trace/select_grid.py) through
// LineRecognizer::SelectKey and compares with the keys the original's LanguageBasedLineRecognizerConfigSelector::Select
// returned (<dir>/orig.txt, trace/select_check.py). With a trace/likely_script_dump.py dump, also compares
// rec::MostLikelyScript with the original's on every dumped tag. Prints mismatches and a summary.
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "recognizer.h"
#include "recognizer_langcode.h"

namespace {

std::string Unhex(const std::string& h) {
  if (h == "=") return "";
  std::string out;
  for (size_t i = 0; i + 1 < h.size(); i += 2) out += static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return fprintf(stderr, "usage: %s <model_dir> <dir>\n", argv[0]), 2;
  auto res = screenai::Resources::FromDirectory(argv[1]);
  auto rec = res ? screenai::LineRecognizer::Create(*res) : nullptr;
  if (!rec) return fprintf(stderr, "cannot load the recognizer\n"), 1;
  std::ifstream grid(std::string(argv[2]) + "/grid.txt"), orig(std::string(argv[2]) + "/orig.txt");
  std::string line, want;
  int n = 0, bad = 0;
  while (std::getline(grid, line)) {
    if (!std::getline(orig, want)) return fprintf(stderr, "orig.txt is short\n"), 1;
    std::istringstream f(line);
    std::string ct, wd, sid, langs;
    f >> ct >> wd >> sid >> langs;
    screenai::Entity e;
    if (ct != "-") e.content_type = std::stoi(ct);
    if (wd != "-") e.writing_direction = std::stoi(wd);
    if (sid != "-") e.recognizer_script_id = Unhex(sid);
    if (langs != "-" && langs != "=none") {
      std::istringstream l(langs);
      std::string tag;
      while (std::getline(l, tag, ',')) e.languages.push_back(Unhex(tag));
    }
    const std::string got = rec->SelectKey(e).value_or("!ERR");
    n++;
    if (got != want) {
      bad++;
      std::string shown;
      for (const auto& t : e.languages) shown += (shown.empty() ? "" : ",") + ("'" + t + "'");
      printf("ct=%s wd=%s sid=%s langs=[%s]: original %s, ours %s\n", ct.c_str(), wd.c_str(),
             sid == "-" ? "-" : ("'" + Unhex(sid) + "'").c_str(), shown.c_str(), want.c_str(), got.c_str());
    }
  }
  printf("%d cases, %d mismatches\n", n, bad);
  if (argc > 3) {
    const std::string d = argv[3];
    std::map<int, std::string> names;
    std::ifstream nf(d + "/names.txt");
    int code;
    std::string name;
    while (nf >> code >> name) names[code] = name;
    for (const char* phase : {"lang", "arab", "region", "region0", "script", "variants", "variants2", "extlang"}) {
      std::ifstream f(d + "/" + phase + ".txt");
      int cases = 0, wrong = 0;
      while (std::getline(f, line)) {
        const size_t sp = line.rfind(' ');
        const std::string tag = line.substr(0, sp), want_s = names[std::stoi(line.substr(sp + 1))];
        const std::string got = screenai::rec::MostLikelyScript(tag);
        cases++;
        if (got != want_s && wrong++ < 20) printf("  %s '%s': original %s, ours %s\n", phase, tag.c_str(), want_s.c_str(), got.c_str());
      }
      printf("MostLikelyScript %s: %d tags, %d mismatches\n", phase, cases, wrong);
      bad += wrong;
    }
  }
  return bad != 0;
}
