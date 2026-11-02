// Page layout analysis: google_ocr::PageLayoutAnalysisMutator (aksara PageLayoutAnalyzer with the
// aksara_page_layout_analysis_rpn_gro_2024_q4 config) applied to the recognized layout. Turns recognized
// paragraphs/lines/words/symbols into BLOCK > PARAGRAPH > LINE > WORD > SYMBOL in reading order, filters
// low-confidence and overlapping lines, recomputes boxes, confidences, line texts, word colors and the page
// language mix. See notes/layout.md.
#pragma once

#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "image.h"
#include "page_layout.h"
#include "resources.h"

namespace screenai {

class TfliteModel;

// PageLayoutAnalyzerSpec.FillSymbolGapsSpec (proto defaults).
struct FillSymbolGapsOptions {
  bool use_nuggets_refinement = false;
  bool has_min_single_symbol_length = false;
  double min_single_symbol_length = 0.3;  // minimum_single_symbol_length_as_fraction_of_thickness
  bool vertically_expand = false;
  std::set<std::string> ignored_scripts;
};
void ParseFillSymbolGapsSpec(std::string_view bytes, FillSymbolGapsOptions* options);

class LayoutAnalyzer {
 public:
  static constexpr const char* kModelPath = "gocr/layout/cluster_sort/model_v2.tflite";
  static constexpr const char* kConfigPath = "aksara/aksara_page_layout_analysis_rpn_gro_2024_q4.binarypb";

  static std::unique_ptr<LayoutAnalyzer> Create(const Resources& res);
  ~LayoutAnalyzer();

  // MutateSub: everything between the recognizer output and the final layout.
  bool Analyze(const GrayImage& gray, PageLayout* layout);

  // Runs one stage of Analyze (for testing against per-step dumps of the original): "pre" (detection confidence),
  // "ConfidenceFilter", "WordPruning", "FillSymbolGaps", "CurvedOverlaps", "ClusterSort", "EstimateLines" (merges
  // lines), "AnalyzerEnd" (line texts),
  // "Colors" (word colors), "PageLanguage" (+ creator_id).
  bool RunStep(const std::string& step, const GrayImage& gray, PageLayout* layout);

  // ClusterSortGcn model inputs for the current layout: features [n, 30] and the two edge endpoint vectors.
  void ClusterSortInputs(const PageLayout& layout, std::vector<float>* features, std::vector<int64_t>* senders,
                         std::vector<int64_t>* receivers) const;
  // Per-line "sort" predictions of the GCN (raw node outputs optional).
  bool NodePredictions(const PageLayout& layout, std::vector<bool>* nodes, std::vector<float>* raw);

 private:
  LayoutAnalyzer() = default;
  bool ClusterSort(PageLayout* layout);
  std::unique_ptr<TfliteModel> model_;
  FillSymbolGapsOptions fill_symbol_gaps_;
};

}  // namespace screenai
