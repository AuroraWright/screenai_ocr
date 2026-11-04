// SPDX-License-Identifier: MPL-2.0
// Line language identification: MobileLangID V2 (google_ocr::MobileLangIdV2 over the expander "pod" classifier
// gocr/gocr_models/line_recognition_mobile_convnext320_omni/tflite_langid.tflite).
//
// The model takes the raw UTF-8 line as a string tensor. Its two custom ops are reimplemented here:
//   NGramHash              lowercases, splits into "^", one token per code point (non-letters -> " "), "$"
//                          (at most max_splits tokens), and hashes every 1..4-gram with MurmurHash64(seed) % vocab + 1.
//   KmeansEmbeddingLookup  product-quantized embedding: per id, sums codebook rows selected by a uint8 code
//                          table, averaged over the sequence.
// The hybrid int8 FULLY_CONNECTED layers run on the builtin TFLite kernels (no XNNPACK, like the original); SOFTMAX
// is replaced by a copy of the original's optimized_ops::SoftmaxImpl (its Eigen pexp/redux), so logits and softmax
// are bit-identical to the original library.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "resources.h"

namespace tflite {
class MutableOpResolver;
}

namespace screenai {

class TfliteModel;

class LanguageIdentifier {
 public:
  static constexpr const char* kModelPath = "gocr/gocr_models/line_recognition_mobile_convnext320_omni/tflite_langid.tflite";

  struct Prediction {
    std::string language;  // bcp47 label of the model ("unknown", "en", "ru-Latn", ...)
    float score = 0;       // softmax probability
  };
  struct Result {
    std::string language;  // after MobileLangIdV2 / LangId::SanitizeSaftLanguage post-processing
    float confidence = 0;  // softmax probability of the arg-max label (1.0 when the classifier returned nothing)
  };

  static std::unique_ptr<LanguageIdentifier> Create(const Resources& res);
  static std::unique_ptr<LanguageIdentifier> CreateFromModel(std::string model_bytes);
  ~LanguageIdentifier();

  // MobileLangIdV2::IdentifyLanguage: arg-max label, "unknown" -> "und", then SanitizeSaftLanguage ("und" when
  // confidence < 0.2, fewer than 5 characters or a "-Latn" label; "ru-PETR1708" for pre-1918 Russian).
  Result IdentifyLanguage(std::string_view utf8_text) const;
  std::string Identify(std::string_view utf8_text) const { return IdentifyLanguage(utf8_text).language; }

  // Raw model outputs: softmax (pod/Softmax) and logits (pod/batch_norm_2/batch_norm_core/add_1), label order.
  bool Scores(std::string_view utf8_text, std::vector<float>* softmax, std::vector<float>* logits = nullptr) const;
  // The k highest-scoring labels, best first (ties keep model order). Labels are unsanitized.
  std::vector<Prediction> TopK(std::string_view utf8_text, int k) const;
  const std::vector<std::string>& labels() const { return labels_; }

  // Adds NGramHash and KmeansEmbeddingLookup to a resolver and replaces its float SOFTMAX with the original
  // library's Eigen-based kernel (needed for bit-exact probabilities).
  static void RegisterCustomOps(tflite::MutableOpResolver* resolver);

  // NGramHash tokenization: byte (offset, length) of each token in the returned token string. Empty on invalid
  // UTF-8 (the original op then produces no tokens and the model sees an all-zero embedding).
  struct Tokens {
    std::string text;
    std::vector<std::pair<size_t, size_t>> spans;
  };
  static Tokens Tokenize(std::string_view utf8_text, int max_splits, bool lowercase);

 private:
  LanguageIdentifier() = default;
  std::unique_ptr<TfliteModel> model_;
  std::vector<std::string> labels_;
  mutable std::mutex mu_;
};

}  // namespace screenai
