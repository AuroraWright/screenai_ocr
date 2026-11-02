# Layout analysis (PageLayoutAnalysisMutator) — reverse-engineering notes

Stage: `002_lay_in.pb` -> `003_final.pb`. Entry PageLayoutAnalysisMutator::MutateSub (0x100170da0).
Config `aksara/aksara_page_layout_analysis_rpn_gro_2024_q4.binarypb` (Any -> google_ocr.PageLayoutAnalysisMutatorConfig
{analyzer_spec}); model `gocr/layout/cluster_sort/model_v2.tflite`.
Per-step dumps (trace/lay_steps.py, taken at each step's *entry*): 000_Mutator_in .. 011_final.

All arithmetic is float unless noted; mirror operation order (build is -ffp-contract=off). rintf = FRINTX (ties even),
roundf = FRINTA.

## Tree semantics
- Children(e): entities with parent == e in index order (Reload builds them).
- Descendants(root): BFS / level order. ReorderUsingDFS: preorder over Children, then ReorderEntities (renumbers,
  remaps parent ids).
- FilterBy(type) without a span: index order.
- Delete(set): removes entities + all descendants, keeps order of the rest, remaps parents. Emptied paragraphs stay.

## Mutator flow
1. If analyzer EstimateWordColors enabled: disabled in the analyzer, run after it as EstimateEntitiesColors.
2. Page image/width/height set (output always carries image{width,height,channels=1}).
3. Pre-step per LINE (index order): with PhotoOcrTextDetectionInfo ext, `detection_confidence =
   sum_raw_boxes_scores / (float)num_raw_boxes` (set even when num == 0 -> NaN/inf); without ext: cleared.
4. Analyzer steps: ConfidenceFilter -> RemoveOverlapsWordPruning -> FillSymbolGaps -> RemoveOverlappingWithCurvedBoxes
   -> ClusterSortGcn -> EstimateLines (MergeLines per paragraph) -> EstimateWordColors (disabled).
5. Analyzer end: LINE text_content = PageLayout::Text of the line. Word texts are NOT recomputed here (fuzz: a word
   whose symbols differ from its text keeps its text), so this is not RecomputeLineTextContentFromSymbols 0x100c06054.
6. EstimateEntitiesColors (WORD only), PopulatePageLanguage, creator_id = "" (ctx2pl 0x10013283c).

Final output differences vs input: BLOCK (no parent) + PARAGRAPH get confidence, orientation(3), box; LINE text_content;
WORD colors gain fg/bg rgb + threshold; page creator_id "" and weighted language_mix.

## ConfidenceFilter (0x10083f7dc, Init 0x10083f6ec)
Per LINE (index order), rec = entity confidence:
- rec == 0 or no WORD children -> delete line. No detection ext -> keep.
- score = bias + (((rec*w_rec + garbage_prob*w_gar) + det_conf*w_det) + direction_conf*w_dir); keep if score >= thr,
  thr = -logf(1/t - 1) (t<=0: -FLT_MAX, t>=1: FLT_MAX).
- w_rec 3.1155417, w_det 7.5824437, w_gar -8.527104, w_dir 3.6975038, bias -8.170908, t 0.45.
- word_confidence_threshold 0 -> word phase skipped.
Observed: en_orient drops a Thai line; multiscript drops an Arabic line.

## RemoveOverlapsWordPruning (0x1008713c0)
Spec: iou 0.6, overlap 0.6, skip_curved (spec byte +33, default true; config sends the spec empty), complete_overlap_word_overlap 0.1, complete_conf_diff 0.2,
word_overlap_threshold 0.5 (double), word_conf_diff 0.05, common_chars 0.6, exact_match 0.5, retain_empty_paragraphs
false, apply_accumulated false (multiplier 4, accum 10).
- Phase 1: lines = FilterBy(LINE, Descendants(root)) (BFS). Word conf = (float)(double sum of symbol confidences /
  count) or 0; line conf = (float)(total/count) or 0; pairs ((float)total, idx) sorted with std::greater.
- Phase 2: OverlapComputer, FindCoveredBoxesType 0x100873850, ComputeSupportWithPrunedWords 0x100874040,
  DoPruneOverlap, RecomputeLineConfidenceFromWords; type 7 with support > own -> remove own; types 2/3 -> remove lower
  support (tie: direction_confidence); mask 0x6D0 types use the complete thresholds; lines with conf 0 removed;
  then Delete + DeleteTextEntitiesWithNoSymbols 0x1008768a8.
Observed deletions: en_fonts (rotated "{" line), multiscript (a Thai line).
Phase 2 detail (implemented: PruneOverlappingLines; 0 mismatches on all 12 per-step dumps 003 -> 004):
- Line boxes = BoundingPolygonToBoundingBox<RotatedBox<float>>(line polygon). OverlapComputer<RotatedBox<float>>
  (0x10086929c): candidate pairs from FindOverlappingAlignedBoxes on the axis-aligned hulls; area =
  AxisAlignedBoxOverlapArea if both angles 0 else Rectangle<float>::OverlapArea; if area != 0, map[(i,j)] =
  {iou = I/(Ai+Aj-I), I/Ai, I/Aj} and map[(j,i)] mirrored (Ai = w*h floats). GetOverlapInfoRange(i) = entries (i,*)
  in ascending j.
- Loop over lines in (float total symbol conf, idx) descending order; skip removed lines (and curved if skip_curved).
  FindCoveredBoxesType(i) (0x100873850) for each j not removed (not curved): same = orientation equal and (if either
  writing_direction == TOP_TO_BOTTOM) writing_direction equal. same: iou>0.95 -> 2, I/Ai>0.95 -> 6, I/Aj>0.95 -> 9,
  iou>iou_thr -> 1, I/Ai>ovl_thr -> 5, I/Aj>ovl_thr -> 8. different: iou>0.95 -> 3, I/Ai>0.95 -> 7, I/Aj>0.95 -> 10,
  iou>iou_thr -> 4, I/Aj>ovl_thr -> 8.
- sup_i = ComputeSupportWithPrunedWords(i) = float sum over non-pruned words of conf * (#symbols with conf). For each
  (j,type): skip if i or j removed or pair (min,max) already processed; mark processed; sup_j -> map; type 10 sums
  (accumulated rule only with apply_accumulated). Then per (j,type) with sup_j in the map: type 7 and sup_j > sup_i
  -> remove i; types 2/3 -> remove lower support (tie: remove j if conf_i(+216 field) >= conf_j else i); else
  thresholds (word_ovl, conf_diff) = types in mask 0x6D0 {4,6,7,9,10}: (complete_overlap_word_overlap +144 float,
  complete_conf_diff +160) else (word_overlap_threshold +152 double, word_conf_diff +164); sup_i <= sup_j ->
  DoPruneOverlap(other=j, prune=i), RecomputeLineConfidenceFromWords(i), conf_i == 0 -> remove i; else the mirror.
- DoPruneOverlap (0x1008749cc): word boxes (prune-line words first, then other-line words) -> OverlapComputer with
  cross pairs only; for each non-pruned word w of the prune line and each overlapping non-pruned word o of the other
  line: ov = I/A_w; skip if ov <= exact_match(+172) && ov <= word_ovl; (exact, portion) = CommonCharsPortion(o.text,
  w.text) (equal -> (true, 1); else 1 - (#chars of w not matched in o, byte multiset)/max(len w,1)); ov <= word_ovl:
  prune iff ov > exact_match && exact; ov > word_ovl: prune unless (conf_w - conf_o >= conf_diff && portion <
  common_chars(+168)) && !(ov > exact_match && exact). Word texts are empty at this point, so exact is true.
- RecomputeLineConfidenceFromWords: double sum of (float)(conf*count) over non-pruned words / total count, or 0.
- Afterwards: Delete(removed lines + pruned words), DeleteTextEntitiesWithNoSymbols(retain_empty_paragraphs).


## FillSymbolGaps (0x10085a318)

RefineLinesByNuggets (implemented without nuggets; 0 mismatches on all 19 per-step dumps 004 -> 005): per LINE (index
order), words = WORD descendants. Config: ignored_scripts {Arab Beng Cyrl Deva Geor Grek Gujr Hani Hebr Jpan Knda Kore
Mlym Taml Telu Thai und} (line recognizer_script_id), minimum_single_symbol_length_as_fraction_of_thickness 0.3.
RefineLinesByNuggets' first argument = spec has-bit 0x20 = has(minimum_single_symbol_length_as_fraction_of_thickness)
(field 3, from FillSymbolGapsSpec::_InternalSerialize; runtime has_bits 0x27 at step+120 via lldb on the shipped
config). Spec layout (step+104): +40 use_nuggets_refinement, +41 vertically_expand, +42 loose_lines, +43 enabled, +48
min length. Parsed from the config in LayoutAnalyzer::Create (Any.value -> analyzer_spec(1) -> spec(1) ->
fill_symbol_boxes(1)). Padding: PhotoOcrTextDetectionInfo +128/+132 = left_padding (14) / right_padding (15) (tags 0x75
/ 0x7d in its _InternalSerialize); the start bound uses left_padding, or right_padding when the (turned) line box angle
is 180 deg off detection_box.angle (+88 -> BoundingBox +48 = angle, field 5). No direction-dependent choice otherwise. Nuggets come from PhotoOcrTextDetectionInfo.nuggets (field 3), which our
inputs never carry, so the clustering / LocateWordBoundaries path never runs.
- Non-ignored line: frame = line box (TTB writing: turned clockwise: swap w/h, origin += round(w*(cos, sin)), angle
  += 90), start bound = (int)(left_padding + (float)frame.left); left/right padding swapped when
  | |frame.angle - detection_box.angle| - 180 | < 0.01. Per word, DeltaSymbolExpansionWithConstraints on its symbols
  (children): boxes turned (TTB) and RotateBoundingBox<int>(about frame origin, -frame.angle); sequentially for
  k = 0..n-1 (left neighbour = k-1, right = k+1; swapped for RTL writing): n == 1: delta = (int)(0.3 * h) - w if
  positive; else delta = (k == n-1 ? 0.8 : 0.4) * gap(left nb) + (k == 0 ? 0.8 : 0.4) * gap(right nb), gap = max(0,
  max(left) - min(left) - width of the leftmost) using already-updated neighbours. new_left = rint((double)left +
  delta * -0.5) (ties even); if below the bound: delta -= bound - new_left, new_left = bound. Unless the box's left
  is already < bound: left = new_left, width = max((int)(delta + w), 1). Then rotate back (+angle) and turn back
  (TTB: swap, origin -= ..., angle += 270). No-op for contiguous symbols (all Latin dev images); it fires for Indic /
  SE-Asian scripts with gaps between (merged) symbols (gujr_guru: Guru lines, mlym_sinh: Sinh, khmr_laoo) and for
  single-symbol words narrower than 0.3 * height.
- Every line then: RecomputeSpaceWordBreakBoxes(words, false) (0x100ab2c08): for consecutive words (k, k+1) with break
  type SPACE/SURE_SPACE, both with children and a break box: gap = DepthGapBetweenEntities(last symbol of k, first
  symbol of k+1) = max(0, LineLength(union) - LineLength(a) - LineLength(b)) (LineLength = width, height for
  TOP_TO_BOTTOM writing; union with ExpandBoxToContainBox); ResizeBox by (int)(gap - LineLength(break box)) along the
  writing direction (centered: angle 0 -> left = (int)(left + d * -0.5f)), then that extent = max(extent, 1). This
  fixed the RTL word_break boxes (rtl, rtl_mixed).
use_nuggets -> RefineLinesByNuggets (0x10085b0c0, not RE'd; apparently no-op on test data), then
RecomputeBoxesFromSymbols: RecomputeBoxFromChildren for every entity in reverse Descendants order:
- collect boxes of children having one (curved child: its rotated_bounding_box, computed from the polygon if
  missing), plus each child's word_break box; if any: box = FindContainingBoxWithAverageAngle(boxes). A curved
  entity keeps its curve and only gets a new rotated_bounding_box (0x... goodoc::RecomputeBoxFromChildren).
- FindContainingBoxWithAverageAngle (0x100896340): n==1 -> normalized angle; else s = Σ sinf(a*0.017453f),
  c = Σ cosf(...); if |c|>1e-5 || |s|>1e-5: angle = (|s|>1e-5 || c > n*-0.99999f) ? atan2f(s,c)*57.29578f : 180;
  else min-|a| normalized angle. Start {0,0,0,0,angle}; per box: if w<1 or h<=0 use {left,top,1,1};
  ExpandBoxToContainBox(child, result).
- ExpandBoxToContainBox (0x10081ccbc): angles normalized to (-180,180]; both 0 -> integer union; else rotate src
  corners into dst frame, min/max (`a > b ? b : a`), rotate min back; roundf, clamp ±2147500000.0f.
Line confidence also changes in this step (recomputed from symbols).

## RemoveOverlappingWithCurvedBoxes (0x100869b1c)
Spec (config field 16, empty): enabled (field 1, default true), threshold (field 2, double, default 0.3; step+128).
- lines = FilterBy(LINE, Descendants(root)). Per line: box = BoundingPolygonToBoundingBox (curved: its
  rotated_bounding_box, written back unchanged). Score (pair<float,int>): straight 0; curved
  (float)((dir_conf(+216) * 0.01 + conf(+188)) * (double)#symbols). No curved line -> no-op.
- Pairs sorted descending (reverse-iterator introsort). OverlapComputer over all line boxes (no split).
- For a < #curved (sorted position), ia = order[a] not removed; for every b > a (removed not checked), ib = order[b],
  skip if GetOverlapInfo(ia, ib) I/area(ia) == 0. Words of both lines (FilterBy WORD), marks[ib] resized to #words
  when empty; OverlapComputer(words a ++ words b, split = #a); per (k, j) pair: symbols of both words, OverlapComputer
  split, area = sum (double) of (float)(I/area(sym a) * (float)(int)(w*h of sym a)); total += area; area > 0 marks
  word j of ib. LineSymbolsArea (0x10086b600) = double sum of float w*h of the line's symbols; if threshold <
  total / max(min(area_b, area_a), 1e-4): removed[ib].
- Removed curved line, or straight line with every word marked (or no words): line deleted; else the marked words are
  deleted. Then DeleteTextEntitiesWithNoSymbols; after a partial deletion RecomputeBoxesFromSymbols,
  RecomputeWordTextContentFromSymbols (0x100c06260: every WORD's text = concatenation of its descendant symbols'
  texts; content types 6/7 join through LatexTextReorderer::JoinSymbolsRangeToText instead, never produced by the
  recognizer, not mirrored), RecomputeConfidencesFromSymbols.
- testdata/extra_curve/arc_stamp: a gray straight "ROUND STAMP" across the arc; the straight 2-word line is deleted.
  testdata/extra_paths/curved_partial: straight 'ab OVERLAPPED cd' just above the apex -> only the middle word is
  deleted ('ab cd'; boxes/confidences recomputed). extra_paths/curved_vs_curved: the lower-scored of two curved lines
  is deleted whole.

## ClusterSortGcn (0x100888284)
Config: heuristic_blocks false, skip_line_clustering true, sort_lines_in_paragraphs true, recompute true,
multiple_inputs, version 2 (GcnLayoutModelSpec: points_per_side_horizontal 10, vertical 2 defaults).
1. LineClusteringFeatures (0x100986b80): per LINE (BFS): BoundingPolygonToBoundingBox -> Box<float>{l,t,w,h,angle}.
   If orientation.writing_direction == 2: left += roundf(cos(a)*h) ... (rotate 90: swap w/h, angle += 90, normalize
   into range with ±360 loops). Extra feature per line: first word's first char CJK -> -1, else first word box width /
   (float)page.image.? (int at image+40), no words -> 0.
   Edges: box_beta_skeleton::GenerateBoxNeighborEdges(boxes, ppsH, ppsV, edges, ..., 1.0, 1.0): scaled points on box
   perimeters (GetScaledPointsFromBoxes) -> boost::polygon voronoi -> Delaunay triangle edges (deduped pairs) ->
   ProcessOverlappingPoints -> MarkInsidePoints -> AddBetaSkeletonEdges (sorted).
   GetBoxEdgeTensors -> features [N,30] float, inputs_1/inputs_2 [E] int64 (edge endpoints).
   Example en_simple (3 lines): inputs_1 = [0,0], inputs_2 = [1,2]; outputs [4] and [3].
2. GetPredictions (0x100887478): version 2: node output (> 0.0) -> vector<bool> sort predictions (count = first
   tensor dim), edge output (> 0.0) -> cluster predictions. No edges -> predictions = N falses.
3. Line clustering skipped.
4. Recompute: ReorderUsingDFS; reverse-Descendants RobustBoxEstimation (0x1008d8680); RecomputeConfidencesFromSymbols
   (only_if_missing false); delete PARAGRAPHs with no children; paragraph without orientation copies first child's.
   - RobustBoxEstimation(e): boxes = descendant (level order) SYMBOL boxes + WORD word_break boxes; angle =
     rotation_util::RobustAngleEstimation (votes dir (1,0),(0,1),(-1,0),(0,-1) by max dot; dominant = first max;
     key = atan2(dx*s - dy*c, (dx*c + 0) + dy*s); std::sort by key; median (avg of the two middle (c,s) for even n);
     atan2(s,c)*kRadToDeg, 0 if both 0); box grown with int ExpandBoxToContainBox from (l,t,1,1).
     WORD/LINE with min over boxes of (sin e sin a + (cos e cos a + 0)) < 0.99 (double compare) become curved:
     CreateCurveRepresentation(raw boxes, dir = (cos, sin) of the box angle, origin = box (left, top),
     use_max_height = true, window = 1) on the same boxes (and the robust box as frame), all turned clockwise
     first when the entity's own writing_direction is TOP_TO_BOTTOM. Edge midpoints of every box (left: corners
     0/3, right: 1/2), std::sort by (c (x - ox) + 0) + s (y - oy), mid line keeps a point once it moved more than
     t/2 (t = max box height) and always the last; top/bottom = mid -+ (t/2)(cos, sin)(atan2(dx, -dy)). Stored:
     curved_box {rotated_bounding_box = robust box (unturned), thickness = (double)t, top_to_bottom = TTB,
     mid_line_curve = (top[k] + bottom[k]) * 0.5f without consecutive duplicates}. (CurveFromBoxes; the detector's
     copy in detector.cc uses window 6.)
5. Per PARAGRAPH (index order): SortChildrenOfEntity (0x100ab62c0): CompareByBox(textline_order, writing_direction)
   on child boxes with (l,t) rotated by -paragraph angle (roundf), angle normalized, then
   CanonicalizeWithMinimumAbsoluteAngle. RerankByPriority: prio[j] = #i preceding j; stable_sort by prio;
   ReorderChildren (children keep their slots).
6. ClusterAndSortParagraphs (0x100889440). Spec offsets: +40 sort_paragraphs_by_line_reading_order, +45
   heuristic_blocks, +48 child_area_overlapping_threshold (0.25 default), step+139 = row_wise_at_top_level.
   - line bit = paragraph's line area with prediction true > false (all false in the test images). Area =
     BoxArea<Box<float>>(BoundingPolygonToBoundingBox(bp, 0, 1)) (0x10088d310): w * h, but for a curved line
     (CopyCurvedBoxToBox: enclosing box + InitBoundingBoxCurvedBox float curve) the area of its convex polygon
     set: CurvedBoxToPoints (0x100202230; float mid-line points, consecutive duplicates dropped; seg angle atan2f;
     inner points the average of the two segment angles (sum * 0.5f, or (sum + 2 pi) * 0.5 in double when |a - b|
     > pi), ends the end segments'; n = sincos(double(a) - pi/2) rounded to float, offset = (0.5f * thickness) * n
     in float; top = p + offset, bottom = p - offset), one quad per segment (top k, top k+1, bottom k+1, bottom
     k), PolygonArea = |sum_k (float)(x_k y_k+1 - y_k x_k+1) * 0.5 + acc| in double rounded to float per term,
     float sum (PolygonSetArea). Node line height: rotated_box height, curved_box thickness (double -> float).
     (Fuzz: wave with a duplicated curved line moved into another paragraph flipped the vote.)
   - per GCN edge with equal writing directions and angle diff <= 30: union = FindContainingBoxWithAverageAngle<Box<float>>;
     dir = 0 if union.w > wA + wB else sign of (sin,cos)(union.angle+90).(cA - cB) (<= 0 -> -1); near = union.h <
     3*minh + hA' + hB' (heights after rotating by -union.angle + axis-align); close = union.h < 2*minh.
   - BFS over lines (line order = the GCN's level order): neighbour joins if same bit and near; for bit 1 also
     two-hop opposite-dir close edges. Each component: OrderTreeNode per new paragraph (box = GetTextlineOrderBox,
     line height sums), one parent node with them (ComputeNodeBoxFromChildren), inserted into an RTree.
   - Nesting rounds: min-heap by node area over roots; stale entries skipped; SortChildren(node); RTree candidates
     (std::set, ascending) that are roots: SafeBoxOverlapArea > area * 0.25 -> becomes child; repeat while changes.
   - Page root (all parentless nodes, index order) -> SortChildren -> TraverseOrderTree (leaves = paragraphs).
   - SortChildren (0x10088e534): child boxes rotated by -node.angle about node center, axis aligned; key1 = row
     center (y + h/2) [column nodes: x center], key2 = x center (negated if rtl) [column: y center], extent
     [x + m, x + w - m], m = min(len/4, avg line height/4) in double. stable_sort by key1; skyline map<float,int>
     gives edges from overlapped previous boxes; topological order with a min-heap on key2.
   - heuristic_blocks false: one BLOCK per paragraph in traversal order (orientation + box copied), old BLOCKs
     deleted.
7. ReorderUsingDFS; RecomputeConfidencesFromSymbols.
Implemented in layout_analyzer.cc; 0 mismatches on all 12 per-step dumps (006 -> 007). Overlap areas: see
"Overlap areas" (box_overlap.cc, bit-exact).
GetTextlineOrderBox: Box<float> of the paragraph, then RotateOrientationClockwise by 1 quadrant for textline_order
RIGHT_TO_LEFT (vertical CJK columns, e.g. ja_vertical) and 3 for LEFT_TO_RIGHT; TOP_TO_BOTTOM unchanged.
RotateOrientationClockwise(q): odd q swaps w/h; origin q=1 (x + w c, y + w s), q=2 (x + w c - h s, h c + (y + w s)),
q=3 (x - h s, y + h c) with the original w, h; angle += 90 q (normalized). That makes the column boxes "rows" so the
skyline sort orders right column first. OrientationLabel in memory: +24 orientation, +28 writing_direction, +40
textline_order.

## EstimateLines (0x100849320) = MergeLinesStep (0x100850500)
The config's EstimateLinesSpec has only field 4 (MergeLinesSpec {5: 2.5, 7: 1}), has-bit 0x2, so AnalyzeInternal skips
EstimateLines/PartitionWords (0 lldb hits) and runs MergeLinesStep::AnalyzeDescendants on every PARAGRAPH (FilterBy at
entry, skipped once no longer Contains). Runtime spec (step+280, MergeLinesStep at step+176): adjacent_only (f7, +136)
= 1, f8 (+138) = 1 (DeleteTextEntitiesWithNoSymbols after each direction), min_breadth_ratio 0.6, max_angle_difference
3.0, min_breadth_overlap 0.6, max_depth_gap 2.5; no overlap-remover spec (its OverlapSpec has no thresholds, so
RemoveOverlappingInSubtree finds nothing).
- AnalyzeDescendants: LINE children (first child a LINE) else LINE descendants, grouped by serialized
  Orientation(line) (flat_hash_map); AnalyzeForDirection per group. f7 also skips ReorderUsingDFS +
  RecomputeBoxesFromSymbols after merging.
- AnalyzeForDirection (0x100850c40): passes until one merges nothing; `deleted` set kept across passes. For i: A =
  lines[i] unless curved (polygon type 3) / deleted; window = f7 ? 1 : n; j = i+1..: B = lines[j]; if B not curved /
  deleted and ShouldMerge(B, A): Merge(B, A), deleted += B, ConvertEndingHyphenToSoftHyphen(last word of A) if it has
  symbols, RecomputeBoxFromChildren(A), RecomputeConfidenceFromDescendants(A), deleted += ancestors of B (nearest
  first) while they have exactly one child, ++window; stop when i + window <= j. End: if anything was deleted,
  Delete(deleted), then DeleteTextEntitiesWithNoSymbols (no merge -> no-op: a symbol-less word survives; fuzz
  telu2).
- ShouldMerge(src B, tgt A) (0x100851a00), rotated_box polygons (else empty box), Orientation(B): t = Thickness
  (height; width for TTB); min/max (double) < 0.6 -> no. d = fmod(|(double)aB - (double)aA|, 360), > 180 -> 360 - d;
  !(d <= 3) -> no. m = {B.left, B.top, 1, 1, (float)((aB + aA) * 0.5)}, ExpandBoxToContainBox(B), (A);
  (double)(tA + tB - tm) / tm < 0.6 -> no. avg = (ComputeSymbolMetrics(B).first + (A).first) * 0.5 (mean symbol
  LineLength over WORD descendants, 0x100ab4168), 0 -> (Thickness(B) + Thickness(A)) / 4.0; merge iff
  (double)(LineLength(m) - (LineLength(A) + LineLength(B))) / avg <= 2.5 (entity LineLength with its own orientation).
- Merge (0x1008523c0): for B's then A's last word: ConvertSoftHyphenToSymbol (SOFT_HYPHEN break -> "-" symbol, break
  NONE), then break NONE -> SPACE (box cleared). B's words SetParent(A) (appended), SortChildren(A) stable with
  CompareByReadingOrder(IsRTL(A)) (0x100ab5480: Start/EndCoordinate along the writing direction, ties on the
  textline coordinates; page_layout_utils Start/End*Coordinate use double sincos(angle * pi/180)), lattice merge (none).
- ConvertEndingHyphenToSoftHyphen (0x100ab4d00): unless SOFT_HYPHEN: break cleared to NONE; with >= 2 symbols and a
  last "-" symbol: break SOFT_HYPHEN boxed by it, symbol deleted, word text from symbols.
Observed: strip_text (two detector lines overlapping on 'with' -> one line '... strip with with readable ...',
A's last word break SPACE); en_columns ShouldMerge 6x, no merge.

## EstimateEntitiesColors (0x1008298c0, lambda 0x10082a5f4), WORD only
Image is 1-channel, so GetRGBValue fails and EstimateColorsOfWord (0x100220340) is always used.
- padding = (float)vertical_box_padding (0.1). Box (l,t,w,h,angle); w <= 2 -> error.
- fractions [-pad, 0.33f, 0.67f, 1+pad]; rad = angle*0.017453292f; __sincosf_stret.
- per f: d = f*(float)h; sx = (int)rintf((float)l - sin*d) clamp [0, min(w_img,gw)-1];
  sy = (int64)rintf(cos*d + (float)t) clamp [0, min(h_img,gh)-1];
  k in 0..w-1: x = clamp(rintf(cos*k + (float)sx)), y = clamp(rintf(sin*k + (float)sy)); hist[gray]++;
  pos[gray] = (x + y*W)*ch. total = w*4.
- ComputeOtsuThresholdOfVector (0x10021f198), bugged formula: cum/mom prefix sums; N=cum[255], M=mom[255],
  S0=(float)max(total,1); for t with c=cum[t] not 0/N: m0 = mom[t]/c, m1 = (M-mom[t])/(N-c) (integer div);
  d=(float)m0-(float)m1; v = d*(((float)c/S0 * (float)(M-mom[t])/S0)*d); strictly greater wins (init -1).
- EstimateColor (0x10021fec0): num_low = cum[t]; 0 or total -> error, Colors cleared entirely. Else medians of
  low/high halves (first i with cumulative > n/2); fg = darker-count side rule (num_low <= num_high: fg=low);
  threshold = t; rgb = gray * 0x01010100 (via pos of median). Sampling points kept.

## PopulatePageLanguage
For each WORD (index order), each tag: map[tag] += (double)(float)(word.confidence * (float)num_children(word)).
Stable sort by weight desc (ties: absl hash order — nondeterministic; we use first-seen). total = sequential double
sum in sorted order; weight = (float)(v/total), or 1.0 if total == 0. Page language_mix always set (cleared first).

## Reachability of the remaining branches (shipped configs + libchromescreenai.so)
Runtime values read with lldb at the step entries (scratchpad xp/spec_flags.py, fsg_flag.py); the specs come from
the config only, so they are the same for every image.
- Detection nuggets (RefineLinesByNuggets clustering: FindNuggetsAfterSpace, CreateWordClusters, LocateWordBoundaries):
  UNREACHABLE. ExtractAndRotateNuggets reads PhotoOcrTextDetectionInfo.nuggets (field 3) only; with none,
  AxisAlignedNuggets is empty and the `nuggets empty || ...` test takes the DeltaSymbolExpansion path. That field is
  written only by proto_converter::DetectionBoxToDetectionInfo (0x100825bcc) from DetectionBox.nuggets, which only
  ExtractNuggetsFromCroppedImage fills (TextDetector::BuildNuggets has no callers). Its callers
  (region_proposal_text_detector_util::CropMultiScaleDetection / CropDetections <- google_ocr::CropDetections) are the
  V1 path of GocrGroupRpnTextDetectionMutator::MutateSub; the detector config has options.version = 2 (field 3.17),
  so MutateSub calls GroupRpnDetectorV2 (lldb backtrace), whose static call graph (2557 functions) reaches neither.
  lldb on arc_stamp / en_simple: google_ocr::CropDetections, CropMultiScaleDetection, ExtractNuggetsFromCroppedImage,
  FindNuggetsAfterSpace, CreateWordClusters 0 hits; DetectionBoxToDetectionInfo called per line with 0 nuggets
  (DetectionBox+64). None of the 128 001/002 stage dumps carries nuggets. (build_nuggets = 1 in the config is unused
  by V2.)
- Non-nugget FillSymbolGaps path: UNREACHABLE, use_nuggets_refinement (field 2) = 1 (runtime step+144 = 1).
- VerticallyExpandWordAndSymbolBoxes: UNREACHABLE. Called only from FillSymbolGapsStep::AnalyzeInternal: 0x10085a3bc
  gated by vertically_expand (spec field 4 = 0 in the config, runtime step+145 = 0) and 0x10085aeb8 in the
  non-nugget branch.
- heuristic_blocks: UNREACHABLE. ClusterSortGcnSpec field 4 = 0 in the config (runtime spec+45 = 0); casp.c:2080 is
  its only read.
- apply_accumulated (type-10 accumulated support rule): UNREACHABLE. The word-pruning spec is present but empty
  (field 6 ""), Clear() zeroes it; runtime step+129 = 0 (multiplier 4.0, count 10 unused); wp2.c:870 its only read.
- skip_curved: REACHABLE (runtime step+137 = 1). arc_stamp exercises it: lldb on the original hits the outer
  `polygon type == 3 -> skip line` branch (0x100871f00) once and FindCoveredBoxesType's curved-candidate skip
  (0x100873abc) once; without the skip our word-pruning step gives 579 mismatches on arc_stamp (also c4/c5/d0/d2
  candidates).
- RemoveOverlappingWithCurvedBoxes partial word deletion: REACHABLE (no config gate); extra_paths/curved_partial
  (per-step dump: straight line 'ab OVERIMBRED cd' -> 'ab cd').

## Decompiles
Scratchpad lay_dc/*.c (not in repo): mutate.c, analyzer.c, conf_filter.c, word_pruning*.c, wp*.c, fill_gaps.c,
refine_nuggets.c, cluster_sort.c, lcf.c, getpred.c, casp.c, should_merge.c, robust_box.c, sort_children.c, rcfs.c,
reorder_dfs.c, gen_edges.c, delaunay.c, add_beta*.c, mark_inside.c, scaled_pts.c, ecow.c, estcolor.c, ...

## Status (2026-11-01)
run_layout on 002_lay_in vs 003_final: 0 cmp_layout mismatches and equal language_mix/creator_id on all 68 cases
(testdata/stages, stages_e2e, stages_extra_lm, stages_extra_indic, stages_extra_rot, stages_extra_curve,
stages_extra_paths, stages_fuzz); tests/test_layout.py checks this. Fuzz: 1618/1621 injected cases match (+ 8 curve probes), the three
left are boxless SOFT_HYPHEN breaks (see Differential testing). RTL needed no special ordering rules
beyond the existing rtl flag in SortChildren (negated x key) - the RTL diffs were all word_break boxes
(RecomputeSpaceWordBreakBoxes). Curved boxes (RobustBoxEstimation, RecomputeBoxFromChildren) are
implemented (stages_extra_curve), as are RemoveOverlappingWithCurvedBoxes (arc_stamp, extra_paths) and skip_curved
in word pruning (arc_stamp) and EstimateLines/MergeLines (strip_text). Not implemented, all unreachable with the shipped configs (see Reachability): detection
nuggets / word clustering in RefineLinesByNuggets, the non-nugget FillSymbolGaps path, heuristic_blocks,
apply_accumulated, VerticallyExpandWordAndSymbolBoxes, the LaTeX word-text join (content types 6/7).

## Overlap areas (cpp/src/box_overlap.{h,cc})
Every overlap area the layout stage computes ends in google_ocr::Rectangle<float>:
- SafeBoxOverlapArea<Box<float>> (0x10084d5d4; ClusterAndSortParagraphs nesting, also ClusterSortGcnStep::ShouldMerge /
  BoxDistanceOverThicknessRatio of the skipped line clustering): a curved box (>= 2 curve points) -> convex polygon
  sets + IntersectPolygonSets (never reached: node boxes have no curve); vertices -> polygon path (never); both
  angles == 0 -> AxisAlignedBoxOverlapArea<Box<float>> (0x10084e0fc); else CreateFromLTWHA x2 + OverlapArea.
- OverlapComputer<goodoc::RotatedBoundingBox> (0x1008689b4; DoPruneOverlap, GetWordOverlap,
  RemoveOverlappingWithCurvedBoxes lines / words / symbols) -> SafeOverlapBoxes<RotatedBoundingBox> (0x100842280):
  angle != 0 on either -> Rectangle path on (float)ints, else the int AxisAlignedBoxOverlapArea (0x100846a80,
  (float)(unsigned)(w * h)); areas (float)(h * w) ints; iou = I / ((A + B) - I), I / A, I / B; area 0 -> no entry.
  Boxes from AppendRotatedBoundingBoxes / BoundingPolygonToBoundingBox<RotatedBoundingBox> (angle normalized).
- OverlapComputer<RotatedBox<float>> (0x10086929c; RemoveOverlapsWordPruningStep lines, boxes from
  BoundingPolygonToBoundingBox<RotatedBox<float>>, normalized): same, AxisAlignedBoxOverlapArea<RotatedBox<float>>.
- Both computers: candidates from FindOverlappingAlignedBoxes(sweep) on CopyBoundingBox<..., AlignedBox<float>>
  (box itself when the normalized angle is 0, else min/max of the 4 corners from sincosf(raw angle)): boxes with
  aligned w or h <= 0 are skipped, pairs need aligned overlap w >= 0, h >= 0 and w * h != 0; pairs come out as
  (lower, higher) index and SafeOverlapBoxes(boxes[lower], boxes[higher]) (OverlapArea is not symmetric bitwise).
- Rectangle<float>::CreateFromLTWHA (0x100c4aec4): (c, s) = sincosf(angle * 0.017453292f). OverlapArea (0x100c4b134):
  both axis aligned (c == 1 && s == 0) -> max/min box, clamped at 0, w * h; one axis aligned -> it is `this` of
  IntersectAxisAlignedNonAxisAligned; both rotated -> both rotated into a's frame: a' = (c l + s t, c t - s l, w, h,
  (1, 0)), b' = (c l + s t, c t - s l, w, h, (c cb + s sb, c sb - s cb)).
- IntersectAxisAlignedNonAxisAligned (0x100c4b288): b's corners (axis aligned: (l, t), (w + l, t), (w + l, t + h),
  (l, t + h); else (l, t), (l + w c, t + w s), that - (s h, -c h), (l - s h, t + c h)); Sutherland-Hodgman against
  top (keep y >= T), right (x <= R), bottom (y <= B), left (x >= L); inside point kept, crossing when the next point
  is strictly on the other side; crossing = start + ((v - start_c) * (d_other)) / (d_c) (operand order differs per
  branch, commutative only); < 3 points after any edge -> 0; shoelace float sum from 0, last edge added, * 0.5f,
  fabsf. No fused multiply-adds anywhere (FMUL/FADD/FSUB/FDIV only).
- Check: trace/overlap_check.py (lldb; calls the library's own functions on generated inputs) +
  cpp/tools/overlap_check: 125500 pairs (random 30000, tiny angles 1e-3..0.5 deg 30000, near 0 / +-90 / 180 20000,
  identical 4000, nested 6000, edge-touching 6000, corner-touching 4000, 1500 threshold sweeps x 17 ulp steps of
  I/A or iou around 0.1 / 0.25 / 0.3 / 0.5 / 0.6), 41834 SafeOverlapBoxes<RotatedBoundingBox> int pairs, 3000
  OverlapComputer sets (both instantiations, 37424 map entries, degenerate / touching boxes, splits): 0 mismatches.
  The polygon-clipping approximation used before differed bitwise on 77740 / 125500 pairs and put a ratio on the
  other side of a threshold for 4940 pairs (4939 of them threshold sweeps, 1 tiny-angle pair).
- The detector (paragraph NMS, level-1 grouping, paragraph assignment) now calls box_overlap.cc too: the original
  uses the same Rectangle<float> routines, AxisAlignedBoxOverlapArea<RotatedBox<float>> 0x10080de18 (same float ops
  as the <Box<float>> instantiation) and OverlapComputer<RotatedBox<float> const> 0x100812194. Replayed bit-exact by
  trace/overlap_check.py det_pairs.bin (176,500 pairs x both orders) and det_sets.bin (3,000 sets); the deleted
  private copy also matched on all of them.

## Differential testing: trace/lay_inject.py + tools/lay_fuzz.py
- trace/lay_inject.py (lldb): at PageLayoutAnalysisMutator::MutateSub entry (x1 = ctx; PageLayout = ctx+176, its
  PageLayoutProto* at +16) replaces the layout with a file parsed by the library's own
  proto2::MessageLite::ParseFromArray (0x101bbb1fc) + goodoc::PageLayout::Reload (0x100c08ddc), then dumps every
  step like lay_steps.py. Everything else layout reads comes from the real run of the image: the gray image
  (ImageCache, word colors) and the page metadata/size, so a case runs on its base dump's own image. Manifest LI_CASES
  = one "<layout.pb> <out_dir>" per image passed to ref/run_ref.py ("-" = no injection).
  Validation: injecting each of the 80 stored 002_lay_in reproduces its 003_final byte-exact (80/80); empty layout
  -> empty output; en_simple's layout on a blank image of the same size differs only in colors.
- tools/lay_fuzz.py: `gen` (seeded random edits of real 002_lay_in: jitter, angle, dup_line, split_line, dup_word,
  move_line, del_word, del_symbol, break, hyphen_symbol, script, ctype, conf, curve, orient, overlap), `targeted`
  (merge thresholds and hyphen conversions on strip_text), `orig` (runs the original through lay_inject), `cmp`
  (run_layout vs the original's final; failures bucketed by the first step whose output differs when that step of
  ours is run on the original's input of the step).
- Results (2026-11-01): targeted 71 + random 1550 (seeds 1-5, 1-7 ops; 250 on curve/rotated bases). Bugs found and
  fixed: MergeLines ran Delete/DeleteTextEntitiesWithNoSymbols with nothing merged; ClusterAndSortParagraphs area vote
  and line height for curved lines; RemoveOverlappingWithCurvedBoxes word text recompute; the AnalyzerEnd line text
  does NOT recompute word texts (a wrong guess caught by the fuzz). Remaining differences are inputs the original
  cannot produce: a SOFT_HYPHEN break without a box (ConvertSoftHyphenToSymbol then writes an empty rotated_box
  message, which RotatedBox cannot represent; the recognizer never emits SOFT_HYPHEN).
- Thresholds probed against the original (all matched): gap (strip_text shift_x 76 merges, 78 not), overlap (shift_y
  8 / 9), thickness ratio (subsumed by the overlap test for nested boxes), angle (1-word lines: d = 3 merges, 3.01
  not; 357.5 wraps), ending "-" -> SOFT_HYPHEN, SOFT_HYPHEN -> "-" symbol, RTL, 3 -> 1 lines.
- Regression subset: testdata/stages_fuzz/<case>/{002_lay_in = injected layout, 003_final = original's final,
  ops.txt}, images testdata/fuzz/<case>.png (copies of the base images); in tests/test_layout.py SETS.
