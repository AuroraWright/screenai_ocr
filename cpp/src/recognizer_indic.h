// Visual -> logical joining of recognized Indic symbols: aksara::StandardTextReorderer::JoinAtomsToSymbols with the
// registered SymbolAtomManipulators (Bengali, Devanagari, Gujarati, Kannada, Mirroring, Tamil, Telugu;
// research/ocr/util/symbol_manipulators/*, applied last to first). The recognizers emit pre-base vowel signs, reph/arkavottu and split
// two-part vowels in visual order; each manipulator's JoinAndReorderStrings rewrites the label sequence into logical
// order and SplittingSymbolAtomManipulator::JoinSymbols groups the labels whose text was permuted into one symbol.
// Mirroring maps bracket glyphs of RTL text to their logical characters.
#pragma once

#include <vector>

#include "recognizer_words.h"

namespace screenai::rec {

// Joins `symbols` (one per decoded label, in recognition order) in place. A joined symbol's box is the union of its
// labels' boxes (CleanupSymbols) and its confidence the product of theirs (CreateTextLineResultFromCTCDecoderOutput).
// If any manipulator cannot align its output with the labels, the symbols stay one per label.
void JoinSymbolsLogical(std::vector<RecSymbol>* symbols);

}  // namespace screenai::rec
