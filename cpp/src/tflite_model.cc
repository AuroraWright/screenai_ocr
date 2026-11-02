#include "tflite_model.h"

#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

namespace screenai {

std::unique_ptr<TfliteModel> TfliteModel::Create(std::string model_bytes, int num_threads, OpRegistrar custom_ops,
                                                 bool use_xnnpack) {
  std::unique_ptr<TfliteModel> m(new TfliteModel());
  m->bytes_ = std::move(model_bytes);
  m->model_ = tflite::FlatBufferModel::BuildFromBuffer(m->bytes_.data(), m->bytes_.size());
  if (!m->model_) return nullptr;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  if (custom_ops) custom_ops(&resolver);
  if (tflite::InterpreterBuilder(*m->model_, resolver)(&m->interpreter_, num_threads) != kTfLiteOk ||
      !m->interpreter_)
    return nullptr;
  if (use_xnnpack) {
    TfLiteXNNPackDelegateOptions opts = TfLiteXNNPackDelegateOptionsDefault();
    opts.num_threads = num_threads;
    m->delegate_ = TfLiteXNNPackDelegateCreate(&opts);
    if (m->interpreter_->ModifyGraphWithDelegate(m->delegate_) != kTfLiteOk) return nullptr;
  }
  return m;
}

TfliteModel::~TfliteModel() {
  interpreter_.reset();
  if (delegate_) TfLiteXNNPackDelegateDelete(delegate_);
}

int TfliteModel::InputIndex(const std::string& name) const {
  for (size_t i = 0; i < interpreter_->inputs().size(); i++)
    if (name == interpreter_->input_tensor(i)->name) return static_cast<int>(i);
  return -1;
}

int TfliteModel::OutputIndex(const std::string& name) const {
  for (size_t i = 0; i < interpreter_->outputs().size(); i++)
    if (name == interpreter_->output_tensor(i)->name) return static_cast<int>(i);
  return -1;
}

const TfLiteTensor* TfliteModel::output(const std::string& name) const {
  const int i = OutputIndex(name);
  return i < 0 ? nullptr : interpreter_->output_tensor(i);
}

bool TfliteModel::Resize(int i, const std::vector<int>& dims) {
  const TfLiteTensor* t = interpreter_->input_tensor(i);
  bool same = t->dims->size == static_cast<int>(dims.size());
  for (int d = 0; same && d < t->dims->size; d++) same = t->dims->data[d] == dims[d];
  if (same) return true;
  dirty_ = true;
  return interpreter_->ResizeInputTensor(interpreter_->inputs()[i], dims) == kTfLiteOk;
}

bool TfliteModel::AllocateTensors() {
  if (!dirty_) return true;
  dirty_ = false;
  return interpreter_->AllocateTensors() == kTfLiteOk;
}

bool TfliteModel::Invoke() { return AllocateTensors() && interpreter_->Invoke() == kTfLiteOk; }

}  // namespace screenai
