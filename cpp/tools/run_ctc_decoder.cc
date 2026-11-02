// Runs the LM CTC beam decoder on decoder inputs dumped from the original library (trace/lm_dump.py) and
// compares with the original outputs bit for bit.
//
//   run_ctc_decoder <resources_root> <dump_dir> [script ...]
//
// For every dec_<addr>.bin the matching script is the one whose class -> LM symbol table, label count and prior
// equal the dump (scripts default to all LM scripts). Every call_<k>.bin is decoded and its labels (class, frame,
// costs) compared with call_<k>_out.bin. Env PRINT=1 prints the decoded labels.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ctc_decoder.h"
#include "resources.h"

using screenai::CtcBeamDecoder;
using screenai::CtcDecoderInput;
using screenai::CtcDecoderOutput;

namespace {

const char* kModelDir = "gocr/gocr_models/line_recognition_mobile_convnext320_omni";

std::string ReadAll(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

template <typename T>
T Get(const std::string& b, size_t* o) {
  T v;
  memcpy(&v, b.data() + *o, sizeof(T));
  *o += sizeof(T);
  return v;
}

struct DecoderDump {
  int num_classes = 0;
  std::vector<std::vector<int>> table;
  std::vector<float> prior;
};

bool LoadDecoderDump(const std::string& path, DecoderDump* d) {
  const std::string b = ReadAll(path);
  if (b.size() < 4 || b.compare(0, 4, "LMDC")) return false;
  size_t o = 4 + 8;
  const uint32_t nl = Get<uint32_t>(b, &o);
  o += nl;
  d->num_classes = Get<int32_t>(b, &o);
  const int nlm = Get<int32_t>(b, &o);
  for (int i = 0; i < nlm; i++) {
    const uint32_t n = Get<uint32_t>(b, &o);
    o += n + 4 + 28 + 16 + 12;
    const int rows = Get<int32_t>(b, &o);
    for (int r = 0; r < rows; r++) {
      const int m = Get<int32_t>(b, &o);
      std::vector<int> ids(m);
      memcpy(ids.data(), b.data() + o, 4 * m);
      o += 4 * m;
      if (i == 0) d->table.push_back(std::move(ids));
    }
  }
  const std::string pr = ReadAll(path.substr(0, path.size() - 4) + "_prior.bin");
  d->prior.resize(pr.size() / 4);
  memcpy(d->prior.data(), pr.data(), d->prior.size() * 4);
  return true;
}

bool Matches(const CtcBeamDecoder& dec, const DecoderDump& d) {
  if (dec.num_classes() != d.num_classes || dec.prior().size() != d.prior.size()) return false;
  if (memcmp(dec.prior().data(), d.prior.data(), 4 * d.prior.size())) return false;
  for (size_t c = 0; c < d.table.size(); c++)
    if (dec.language_model().ClassSymbols(static_cast<int>(c)) != d.table[c]) return false;
  return static_cast<int>(d.table.size()) == dec.language_model().num_classes();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <resources_root> <dump_dir> [script ...]\n", argv[0]);
    return 2;
  }
  auto res = screenai::Resources::FromDirectory(argv[1]);
  const std::string dir = argv[2];
  std::vector<std::string> scripts;
  for (int i = 3; i < argc; i++) scripts.push_back(argv[i]);
  if (scripts.empty())
    scripts = {"arab", "cyrl", "deva", "hani", "jpan", "knda", "kore", "mlym", "taml", "telu", "thai"};
  std::map<std::string, std::unique_ptr<CtcBeamDecoder>> decoders;
  for (const std::string& s : scripts) {
    std::string err;
    auto d = CtcBeamDecoder::Create(*res, kModelDir, s, &err);
    if (!d) {
      fprintf(stderr, "%s: %s\n", s.c_str(), err.c_str());
      continue;
    }
    decoders[s] = std::move(d);
  }
  const bool print = getenv("PRINT") != nullptr;

  std::vector<std::string> files;
  if (DIR* dp = opendir(dir.c_str())) {
    while (dirent* e = readdir(dp)) files.push_back(e->d_name);
    closedir(dp);
  }
  std::sort(files.begin(), files.end());
  std::map<uint64_t, std::string> dec_script;  // decoder address -> script
  for (const std::string& f : files) {
    if (f.rfind("dec_", 0) != 0 || f.find("_prior") != std::string::npos) continue;
    DecoderDump d;
    if (!LoadDecoderDump(dir + "/" + f, &d)) continue;
    const uint64_t addr = strtoull(f.c_str() + 4, nullptr, 16);
    std::string match;
    for (const auto& [s, dec] : decoders)
      if (Matches(*dec, d)) {
        match = s;
        break;
      }
    printf("%s: %s\n", f.c_str(), match.empty() ? "NO MATCHING SCRIPT (label map / LM table / prior differ)" :
                                                  ("script " + match).c_str());
    if (match.empty()) {
      // report the first table difference against scripts with the same class count
      for (const auto& [s, dec] : decoders) {
        if (dec->num_classes() != d.num_classes) continue;
        for (size_t c = 0; c < d.table.size(); c++)
          if (dec->language_model().ClassSymbols(static_cast<int>(c)) != d.table[c]) {
            printf("  %s: first table diff at class %zu (%s)\n", s.c_str(), c, dec->label(static_cast<int>(c)).c_str());
            break;
          }
      }
    }
    dec_script[addr] = match;
  }

  int calls = 0, exact = 0, labels_total = 0;
  for (const std::string& f : files) {
    if (f.rfind("call_", 0) != 0 || f.find("_out") != std::string::npos) continue;
    const std::string b = ReadAll(dir + "/" + f);
    size_t o = 4;
    const uint64_t addr = Get<uint64_t>(b, &o);
    const int n = Get<int32_t>(b, &o), B = Get<int32_t>(b, &o), T = Get<int32_t>(b, &o), K = Get<int32_t>(b, &o);
    const int length = Get<int32_t>(b, &o), has_idx = Get<int32_t>(b, &o);
    (void)n, (void)B;
    std::vector<float> values(T * K), ids(has_idx ? T * K : 0);
    memcpy(values.data(), b.data() + o, 4 * T * K);
    o += 4 * T * K;
    if (has_idx) memcpy(ids.data(), b.data() + o, 4 * T * K);
    const std::string script = dec_script[addr];
    calls++;
    if (script.empty()) {
      printf("%s: skipped (unknown decoder)\n", f.c_str());
      continue;
    }
    CtcDecoderInput in;
    in.num_frames = length;
    in.k = K;
    in.values = values.data();
    in.class_ids = has_idx ? ids.data() : nullptr;
    CtcDecoderOutput out;
    decoders[script]->Decode(in, &out);

    const std::string ob = ReadAll(dir + "/" + f.substr(0, f.size() - 4) + "_out.bin");
    std::vector<CtcDecoderOutput::Label> ref((ob.size() - 8) / 40);
    for (size_t i = 0; i < ref.size(); i++) {
      size_t q = 8 + 40 * i;
      ref[i].label = Get<int32_t>(ob, &q);
      ref[i].frame = Get<int32_t>(ob, &q);
      ref[i].optical_cost = Get<double>(ob, &q);
      ref[i].lm_cost = Get<double>(ob, &q);
      ref[i].prior_cost = Get<double>(ob, &q);
      ref[i].confidence = Get<double>(ob, &q);
    }
    bool same = ref.size() == out.labels.size();
    size_t first_diff = std::min(ref.size(), out.labels.size());
    for (size_t i = 0; i < std::min(ref.size(), out.labels.size()); i++) {
      const auto &a = ref[i], &c = out.labels[i];
      if (a.label != c.label || a.frame != c.frame || memcmp(&a.optical_cost, &c.optical_cost, 8) ||
          memcmp(&a.lm_cost, &c.lm_cost, 8) || memcmp(&a.prior_cost, &c.prior_cost, 8) ||
          memcmp(&a.confidence, &c.confidence, 8)) {
        same = false;
        first_diff = i;
        break;
      }
    }
    exact += same;
    labels_total += static_cast<int>(ref.size());
    std::string text;
    for (const auto& l : out.labels) text += decoders[script]->label(l.label);
    printf("%s %s T=%d labels=%zu/%zu %s '%s'\n", f.c_str(), script.c_str(), length, out.labels.size(), ref.size(),
           same ? "EXACT" : "DIFF", text.c_str());
    if (!same) {
      for (size_t i = first_diff; i < std::min(first_diff + 3, std::max(ref.size(), out.labels.size())); i++) {
        if (i < ref.size())
          printf("   ref[%zu] %d @%d %.9g %.9g %.9g\n", i, ref[i].label, ref[i].frame, ref[i].optical_cost,
                 ref[i].lm_cost, ref[i].prior_cost);
        if (i < out.labels.size())
          printf("   our[%zu] %d @%d %.9g %.9g %.9g\n", i, out.labels[i].label, out.labels[i].frame,
                 out.labels[i].optical_cost, out.labels[i].lm_cost, out.labels[i].prior_cost);
      }
    }
    if (print)
      for (const auto& l : out.labels)
        printf("   %d @%d '%s' %.9g %.9g %.9g\n", l.label, l.frame, decoders[script]->label(l.label).c_str(),
               l.optical_cost, l.lm_cost, l.prior_cost);
  }
  printf("%d/%d calls exact (%d labels)\n", exact, calls, labels_total);
  return exact == calls ? 0 : 1;
}
