// Recognized line -> WORD/SYMBOL geometry (line-relative coordinates until ToPage()):
//   aksara::api_internal::GenerateLineWordEntities (WordBuilder, SPACE word splitter),
//   layout_analyzer::RecomputeSpaceWordBreakBoxes, google_ocr::EstimateWordColorsSymbolAware,
//   aksara::box_utils::BoxRelativeToPageRelativeVec and goodoc::RecomputeBoxFromChildren
//   (box_util::FindContainingBoxWithAverageAngle / ExpandBoxToContainBox).
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "image.h"
#include "page_layout.h"

namespace screenai::rec {

struct RecSymbol {
  std::string text;
  float confidence = 0;
  RotatedBox box;
  bool mirrored = false;  // TextReorderer::IsMirrored: one label whose glyph was mirrored (symbol direction RTL)
};

struct RecWord {
  std::vector<int> symbols;  // indices into the line's symbols (whitespace symbols are not part of any word)
  std::string text;
  float confidence = 0;
  RotatedBox box;
  std::optional<WordBreak> brk;
  std::optional<int> writing_direction;  // goodoc WritingDirection set by the word builder (0 LTR, 1 RTL)
  std::optional<Colors> colors;
};

// WordBuilder over the line's symbols (in logical order); `line_text` is their concatenation.
std::vector<RecWord> BuildWords(const std::vector<RecSymbol>& symbols, const std::string& line_text);
// RecomputeSpaceWordBreakBoxes for horizontal (angle 0, line-relative) words.
void RecomputeSpaceWordBreakBoxes(const std::vector<RecSymbol>& symbols, std::vector<RecWord>* words);
// EstimateWordColorsSymbolAware on the (unscaled) gray line image.
void EstimateWordColors(const GrayImage& line, const std::vector<RecSymbol>& symbols, std::vector<RecWord>* words);

// Line-relative box -> page box (BoxRelativeToPageRelativeVec with rounding).
RotatedBox ToPage(const RotatedBox& line_box, const RotatedBox& rel);
// goodoc::RecomputeBoxFromChildren for rotated children.
RotatedBox ContainingBoxWithAverageAngle(const std::vector<RotatedBox>& boxes);
// box_util::ExpandBoxToContainBox(src, dst).
void ExpandBoxToContainBox(const RotatedBox& src, RotatedBox* dst);

// sinf/cosf as the original computes them (Apple __sincosf_stret where available).
void SinCos(float x, float* s, float* c);

// Unicode helpers (ICU properties used by the word builder).
bool IsWhitespaceText(const std::string& utf8);   // every code point is a separator (Z*)
bool IsPunctuationText(const std::string& utf8);  // every code point is punctuation (P*)
bool HasAnyRtl(const std::string& utf8);          // bidi R, AL, RLE, RLO
bool HasAnyLtr(const std::string& utf8);          // bidi L, EN, AN, LRE, LRO
std::vector<uint32_t> DecodeUtf8(const std::string& utf8);

}  // namespace screenai::rec
