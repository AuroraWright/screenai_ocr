// bench_model <model.tflite> <dump_dir> <prefix> <threads> <interpreters> <invokes> [use_xnnpack=1]
// Times Invoke on the dumped inputs (<prefix>_in.json shapes). With interpreters > 1, `invokes` invokes are spread
// over that many interpreters on as many std::threads (the original's TfliteModelPooledRunner layout).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <thread>
#include <vector>

#include "tflite_model.h"

using namespace screenai;

static std::string ReadFile(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

int main(int argc, char** argv) {
  if (argc < 7) return fprintf(stderr, "usage: see header\n"), 2;
  const std::string model = ReadFile(argv[1]), dir = argv[2], prefix = argv[3];
  const int threads = atoi(argv[4]), interps = atoi(argv[5]), invokes = atoi(argv[6]);
  const bool xnn = argc < 8 || atoi(argv[7]);
  const std::string json = ReadFile(dir + "/" + prefix + "_in.json");
  std::vector<std::vector<int>> shapes;
  std::regex shape_re("\"shape\": \\[([0-9, ]+)\\]");
  for (std::sregex_iterator it(json.begin(), json.end(), shape_re), end; it != end; ++it) {
    std::vector<int> s;
    std::stringstream ss((*it)[1].str());
    std::string tok;
    while (std::getline(ss, tok, ',')) s.push_back(atoi(tok.c_str()));
    shapes.push_back(s);
  }
  std::vector<std::unique_ptr<TfliteModel>> ms;
  for (int i = 0; i < interps; i++) {
    auto m = TfliteModel::Create(model, threads, nullptr, xnn);
    for (size_t k = 0; k < shapes.size(); k++) m->Resize(static_cast<int>(k), shapes[k]);
    m->AllocateTensors();
    for (size_t k = 0; k < shapes.size(); k++) {
      const std::string b = ReadFile(dir + "/" + prefix + "_in_" + std::to_string(k) + ".bin");
      memcpy(m->input(static_cast<int>(k))->data.raw, b.data(), std::min(b.size(), m->input(static_cast<int>(k))->bytes));
    }
    m->Invoke();  // warm-up
    ms.push_back(std::move(m));
  }
  double best = 1e9;
  for (int rep = 0; rep < 7; rep++) {
    const auto t = std::chrono::steady_clock::now();
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int i = 0; i < interps; i++)
      pool.emplace_back([&, i] {
        while (next.fetch_add(1) < invokes) ms[i]->Invoke();
      });
    for (auto& th : pool) th.join();
    best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count());
  }
  printf("threads %d interpreters %d invokes %d: %.1f ms\n", threads, interps, invokes, best);
}
