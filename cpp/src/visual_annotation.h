// PageLayout -> chrome_screen_ai.VisualAnnotation, as ConvertOCRResultsToVisualAnnotationProto in
// chrome/accessibility/machine_intelligence/chrome_screen_ai/ocr.cc does it (the PerformOCR output format).
#pragma once

#include <string>

#include "page_layout.h"

namespace screenai {

// Returns the serialized VisualAnnotation (edition 2024, implicit field presence).
std::string ConvertToVisualAnnotation(const PageLayout& layout);

}  // namespace screenai
