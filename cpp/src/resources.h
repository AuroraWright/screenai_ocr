// Access to the user-provided model files (the Screen AI component directory). Paths are relative to the
// component root, e.g. "gocr/gocr_models/detection/gocr_group_rpn_text_detection_model_2024_q4.tflite".
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace screenai {

class Resources {
 public:
  // Same callback shapes as the Chrome Screen AI library's SetFileContentFunctions.
  using SizeFn = uint32_t (*)(const char* relative_path);
  using ContentFn = void (*)(const char* relative_path, uint32_t size, char* buffer);

  static std::shared_ptr<Resources> FromDirectory(const std::string& root);
  static std::shared_ptr<Resources> FromCallbacks(SizeFn size_fn, ContentFn content_fn);

  // Returns false if the file is missing or empty.
  bool Read(const std::string& relative_path, std::string* out) const;

 private:
  std::function<bool(const std::string&, std::string*)> read_;
};

}  // namespace screenai
