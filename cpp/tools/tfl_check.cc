// Runs a .tflite model on tensors dumped by trace/tfl_dump.py and compares outputs bit-for-bit.
// usage: tfl_check model.tflite dump_dir call_prefix [threads] [xnnpack 0/1]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter_builder.h"

namespace tflite::ops::builtin {
TfLiteRegistration* Register_HARD_SWISH_REF();
TfLiteRegistration* Register_QUANTIZE_REF();
}

static std::string Slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int main(int argc, char** argv) {
  if (argc < 4) return 2;
  std::string dir = argv[2], pre = argv[3];
  int threads = argc > 4 ? atoi(argv[4]) : 1;
  bool xnn = argc > 5 ? atoi(argv[5]) : 0;
  auto model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  if (getenv("HS_REF")) resolver.AddBuiltin(tflite::BuiltinOperator_HARD_SWISH, tflite::ops::builtin::Register_HARD_SWISH_REF());
  if (getenv("Q_REF")) resolver.AddBuiltin(tflite::BuiltinOperator_QUANTIZE, tflite::ops::builtin::Register_QUANTIZE_REF(), 1, 3);
  std::unique_ptr<tflite::Interpreter> interp;
  tflite::InterpreterBuilder(*model, resolver)(&interp, threads);
  // input shapes from <pre>_in.json: every "shape": [..] in order
  std::string js = Slurp(dir + "/" + pre + "_in.json");
  size_t pos = 0;
  for (size_t i = 0; i < interp->inputs().size(); i++) {
    pos = js.find("\"shape\": [", pos);
    if (pos == std::string::npos) break;
    pos += 10;
    std::vector<int> dims;
    while (js[pos] != ']') {
      dims.push_back(strtol(js.c_str() + pos, nullptr, 10));
      pos = js.find_first_of(",]", pos);
      if (js[pos] == ',') pos += 2;
    }
    interp->ResizeInputTensor(interp->inputs()[i], dims);
  }
  TfLiteDelegate* d = nullptr;
  if (xnn) {
    auto o = TfLiteXNNPackDelegateOptionsDefault();
    o.num_threads = threads;
    d = TfLiteXNNPackDelegateCreate(&o);
    interp->ModifyGraphWithDelegate(d);
  }
  if (interp->AllocateTensors() != kTfLiteOk) { fprintf(stderr, "alloc failed\n"); return 1; }
  if (getenv("PRINT_PLAN")) {
    printf("PLAN nodes=%zu:", interp->execution_plan().size());
    for (int n : interp->execution_plan()) {
      const auto* nr = interp->node_and_registration(n);
      printf(" %d%s%s", nr->second.builtin_code, nr->second.custom_name ? ":" : "", nr->second.custom_name ? nr->second.custom_name : "");
    }
    printf("\n");
  }
  // optional warm-up: PRE_BATCHES="1,3" invokes at those batch sizes first, then restores the dumped shape
  if (const char* pb = getenv("PRE_BATCHES")) {
    std::vector<int> want(interp->input_tensor(0)->dims->data, interp->input_tensor(0)->dims->data + interp->input_tensor(0)->dims->size);
    for (const char* q = pb; *q;) {
      int b = strtol(q, (char**)&q, 10);
      if (*q == ',') q++;
      std::vector<int> d = want; d[0] = b;
      interp->ResizeInputTensor(interp->inputs()[0], d); interp->AllocateTensors();
      memset(interp->input_tensor(0)->data.raw, 255, interp->input_tensor(0)->bytes);
      interp->Invoke();
    }
    interp->ResizeInputTensor(interp->inputs()[0], want); interp->AllocateTensors();
  }
  for (size_t i = 0; i < interp->inputs().size(); i++) {
    std::string b = Slurp(dir + "/" + pre + "_in_" + std::to_string(i) + ".bin");
    TfLiteTensor* t = interp->input_tensor(i);
    if (b.size() != t->bytes) { fprintf(stderr, "in %zu size %zu vs %zu\n", i, b.size(), t->bytes); return 1; }
    memcpy(t->data.raw, b.data(), b.size());
  }
  if (interp->Invoke() != kTfLiteOk) { fprintf(stderr, "invoke failed\n"); return 1; }
  int bad = 0;
  for (size_t i = 0; i < interp->outputs().size(); i++) {
    std::string b = Slurp(dir + "/" + pre + "_out_" + std::to_string(i) + ".bin");
    const TfLiteTensor* t = interp->output_tensor(i);
    size_t n = std::min(b.size(), t->bytes), diff = 0, maxd = 0;
    for (size_t k = 0; k < n; k++) if (b[k] != t->data.raw[k]) {
      diff++;
      if (t->type == kTfLiteUInt8) maxd = std::max<size_t>(maxd, abs((uint8_t)b[k] - (uint8_t)t->data.raw[k]));
    }
    if (getenv("PER_ROW") && t->dims->size > 1) {
      size_t per = t->bytes / t->dims->data[0];
      printf("   rows:");
      for (int r = 0; r < t->dims->data[0]; r++) { size_t c = 0; for (size_t k = r * per; k < (r + 1) * per; k++) c += b[k] != t->data.raw[k]; printf(" %zu", c); }
      printf("\n");
    }
    printf("out %zu %-40s bytes %zu/%zu  differing %zu  maxdiff %zu\n", i, t->name, t->bytes, b.size(), diff, maxd);
    bad += diff != 0 || b.size() != t->bytes;
  }
  interp.reset();
  if (d) TfLiteXNNPackDelegateDelete(d);
  return bad ? 1 : 0;
}
