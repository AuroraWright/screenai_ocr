// aksara::script_util::MostLikelyScript (0x100b2d148): the script a BCP 47 language tag is most likely written
// in, as LanguageBasedLineRecognizerConfigSelector::Select uses it to pick a recognizer. The tag is parsed and
// validated like i18n_identifiers::LanguageCodeConverter does; tags that do not parse give "Latn".
#pragma once

#include <string>
#include <string_view>

namespace screenai::rec {

std::string MostLikelyScript(std::string_view tag);

}  // namespace screenai::rec
