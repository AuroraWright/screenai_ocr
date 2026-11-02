// Thin wrapper over a TFLite interpreter running with the XNNPACK delegate (the configuration whose outputs
// match the original library bit-for-bit). Not thread-safe: use one TfliteModel per thread.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace screenai {

class TfliteModel {
 public:
  // `custom_ops` may add registrations (e.g. the language-ID ops) to the builtin resolver.
  using OpRegistrar = void (*)(tflite::MutableOpResolver*);
  static std::unique_ptr<TfliteModel> Create(std::string model_bytes, int num_threads = 1,
                                             OpRegistrar custom_ops = nullptr, bool use_xnnpack = true);
  ~TfliteModel();

  tflite::Interpreter* interpreter() { return interpreter_.get(); }
  int InputIndex(const std::string& name) const;   // position in inputs(), -1 if absent
  int OutputIndex(const std::string& name) const;  // position in outputs(), -1 if absent
  TfLiteTensor* input(int i) { return interpreter_->input_tensor(i); }
  const TfLiteTensor* output(int i) const { return interpreter_->output_tensor(i); }
  const TfLiteTensor* output(const std::string& name) const;

  // Resizes input i (no-op when unchanged); call AllocateTensors() afterwards.
  bool Resize(int i, const std::vector<int>& dims);
  bool AllocateTensors();
  bool Invoke();

 private:
  TfliteModel() = default;
  std::string bytes_;
  std::unique_ptr<tflite::FlatBufferModel> model_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  TfLiteDelegate* delegate_ = nullptr;
  bool dirty_ = true;
};

}  // namespace screenai
