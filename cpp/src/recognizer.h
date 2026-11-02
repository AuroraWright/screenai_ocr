// Line recognition (google_ocr MultiPassLineRecognitionMutator over GocrCTCDecoderRecognizer): for every LINE of the
// detector layout, crops/deskews the line, runs the "und" (multi-script) recognizer, and re-recognizes lines whose
// predicted script has a dedicated model (Jpan, Arab, Thai, ...) with that model (+ its character LM when it has
// one). Produces the recognized text, confidences, languages, direction, WORD and SYMBOL entities with boxes and
// word colors, in the order the original emits them (non-line entities first, then each line followed by its words,
// each word followed by its symbols).
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "image.h"
#include "page_layout.h"
#include "resources.h"

namespace screenai {

class LanguageIdentifier;

namespace rec {
class ScriptRecognizer;
}

class LineRecognizer {
 public:
  static std::unique_ptr<LineRecognizer> Create(const Resources& res);
  ~LineRecognizer();

  bool Recognize(const RgbaImage& rgba, const GrayImage& gray, PageLayout* layout);

  // Recognizer key a LINE goes to after the first pass (LanguageBasedLineRecognizerConfigSelector::Select);
  // nullopt where the original fails (malformed "-t-i0-handwrit" language tag).
  std::optional<std::string> SelectKey(const Entity& line) const;

 private:
  LineRecognizer() = default;
  rec::ScriptRecognizer* Get(const std::string& key);  // lazily loads the recognizer of a multi-pass key

  const Resources* res_ = nullptr;
  std::unique_ptr<LanguageIdentifier> langid_;
  std::map<std::string, std::string> config_files_;  // multi-pass key -> recognizer config file
  std::string default_key_ = "und";
  int maximum_iterations_ = 2;
  std::map<std::string, std::unique_ptr<rec::ScriptRecognizer>> recognizers_;
};

}  // namespace screenai
