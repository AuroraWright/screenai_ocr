// SPDX-License-Identifier: MPL-2.0
// Visual -> logical reordering of recognized symbols: aksara::StandardTextReorderer::SymbolsToResult over
// i18n_bidi::BiDiConverter (ICU ubidi, UBIDI_REORDER_INVERSE_LIKE_DIRECT for display -> logical, UBIDI_REORDER_DEFAULT
// for logical -> display, UBIDI_KEEP_BASE_COMBINING, paragraph level 1 iff base_dir_rtl and the text has any RTL
// character, else 0). Port of ICU's ubidi level resolution (state tables of ubidi.cpp) for one paragraph without
// explicit embeddings/isolates (the recognizers never emit them).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace screenai::rec {

// StandardTextReorderer::SymbolsToResult: symbols (in recognition = visual order) -> logical line text and the
// byte offset of every symbol in it (-1 when the symbol's text is not found at its offset).
void SymbolsVisualToLogical(const std::vector<std::string>& symbols, bool base_dir_rtl, std::string* text,
                            std::vector<int>* offsets);

// aksara::MirroringSymbolManipulator::ActuallyTransform: if the text has any RTL character, every code point at an
// odd level of an RTL paragraph (UBIDI_REORDER_INVERSE_LIKE_DIRECT) is replaced by its Bidi_Mirroring_Glyph;
// the order is unchanged.
std::string MirrorRtlText(const std::string& text);

// u_charMirror: the Bidi_Mirroring_Glyph of `cp`, or `cp` itself.
uint32_t BidiMirrorGlyph(uint32_t cp);

}  // namespace screenai::rec
