# Line recognition (MultiPassLineRecognitionMutator) — RE notes

Decompiles: `ida/decomp/rec/` (function address prefixes), scratch decompiles in the session scratchpad
(`rec/merge.c reduce.c colors.c ctlr.c dirutils.c`).

## Config
- Multi-pass config `gocr/gocr_models/gocr_line_recognition_omni_mobile_chrome_multiscript_2024_q4.binarypb` =
  google.protobuf.Any(MultiPassLineRecognitionMutatorConfig). Key -> config file (in
  `gocr/gocr_models/line_recognition_mobile_convnext320_omni/`):
  Arab arab_fst_config.pb, Beng beng_config.pb, Cyrl cyrl_fst_config.pb, Deva deva_fst_config.pb, Geor geor_config.pb,
  Grek grek_config.pb, Gujr gujr_config.pb, Guru guru_2024_q3_config.pb, Hani hani_fst_config.pb, Hebr hebr_config.pb,
  Jpan jpan_fst_config.pb, Khmr khmr_2024_q3_config.pb, Knda knda_fst_2024_q3_config.pb, Kore kore_fst_config.pb,
  Laoo laoo_2024_q3_config.pb, Mlym mlym_fst_config.pb, Sinh sinh_2024_q3_config.pb, Taml taml_fst_config.pb,
  Telu telu_fst_2024_q3_config.pb, Thai thai_fst_2024_q3_config.pb, und gocr_mobile_und_config.pb (specificity -1).
  Selector LanguageBasedLineRecognizerConfigSelector default "und"; maximum_iterations 2.
- und config: model gocr_mobile_und.tflite, dequantize_topk 5, output names [decode_logits, symbol_left_offset,
  symbol_right_offset, symbol_top, symbol_height, script_activations, script_gates, content_type_activations,
  content_type_gates, direction_activations, direction_gates], only_dequantize_first_output, label map
  gocr_mobile_und_label_map.pb, langid MobileLangIdV2, scaled height 32, chunk_width 136, frame_width 4, pads 16/16,
  kFilterBox, has_direction_id, gated_activation_start_pos 5, num_chunk_per_batch 8. CTC cfg: word_splitter SPACE,
  keep_all_hyphens, 31 script_names (Zzzz,Zyyy,Latn,Hani,Jpan,Kore,Cyrl,Arab,Thai,Deva,Grek,Hebr,Taml,Armn,Beng,Mymr,
  Geor,Sinh,Khmr,Laoo,Telu,Mlym,Ethi,Knda,Gujr,Guru,Orya,Tibt,Syrc,Cher,Thaa), supported_scripts Zzzz,Zyyy,Latn,
  content_type_names TEXT,HANDWRITTEN_TEXT, compute_word_colors, clamp_symbol_box, symbol_box_index 2, script_index 6,
  content_type_index 7, direction_index 8.
- Script configs (e.g. jpan): hanijpan.tflite, hanijpan_label_map.pb, recognizer_script_id "Jpan", beam 5/5.0,
  sel 5/5.0, lm_weight .6, ins -1, prior .1, blank .3, self_loop -.6, confidence_scorer AvgLogits, prior_path.
  hebr: base_dir_rtl, no LM (greedy). arab: arab_fst LM, base_dir_rtl.
  Script models have 5 outputs (decode_logits, symbol_height, symbol_left_offset, symbol_right_offset, symbol_top),
  no output_name in config (names likely inherited from und by PrepareOneRecognizer 100151978 — open question).
- Decode configs: `.venv/bin/python -I tools/d2.py tools protos_extracted google_ocr.LineRecognizerConfig <file>`.
- Flags `use_neon_optimized_sparse_dequantization` (TfliteModelPooledRunnerConfig 17) and
  `use_aarch64_reduce_activations_gates_and_softmax` unset in all shipped configs -> generic paths.

## Line image
(prototype verified bit-exact vs tfl input dumps: scratchpad rec/proto_line.py)
- angle==0: clip box to image, crop. Else aligned bounding box of rotated rect (float corners l + (w-1)*c ...,
  i.e. pixel corners, with sinf/cosf of a*0.0174532924f; roundf; W = R-L+1, H = B-T+1), clip, warp with inverse affine [c,-s,s,c,l-cl,t-ct] (rad=(double)a*pi/180 -> float
  sin/cos), fixed point *1024 (vcvtns), +16 rounding, 5-bit fractional bilinear; interior: float weights sum,
  rint; border: clamped coords, Q15 weights.
- scale to height 32: sc=32.0f/h, dw=max(1,rint(sc*w)), dh=max(1,rint(sc*h)), libyuv ScalePlane kFilterBox.
- chunks: offsets 0,136,...; each chunk 168 wide (16 left pad, 136, 16 right pad) filled from image incl.
  neighbouring context; zero elsewhere. Batches of 8 chunks, zero-padded.

## Lines of large text: detector pyramid crops (detection_image_scale < 1000)
- GroupRpnTextDetectionMutator's CropDetections (anon $_1 0x10081c244, trace "CropMultiScaleDetections") cuts
  every line with ImageCropRotateAndScaleGray 0x1008a7830 (angle 0: clip + copy; rotated: WarpAffine; curved:
  ImageCropCurvedPatch with the detector's double thickness) from the pyramid level it picked (box scaled by s,
  integer truncation, then scaled back by 1/s) and these crops land in the ImageCache under the line's key
  (ImageCacheKey = polygon + property). ImageCache::ExtractFromPL 0x100b05eec only crops lines from the page
  whose key is not cached yet, and RotateLine 0x100af3404 re-keys the cached image (turned) when the direction
  head flips the box, so every pass of such a line reads the pyramid crop.
- Ours: DetectorPyramid (detector.cc, the detector's own BuildPyramid, reference level dropped) is rebuilt on
  the first such line; the level whose (int)(scale*1000) equals detection_image_scale is cropped with the box
  whose scale-back gives the page box (PyramidCropBox inverts the truncation; curve points/thickness from the
  double-precision detection_box).
- GenerateLineWordsMaybeSetColors 0x1001ff2c4: f = 1000/detection_image_scale (lines without the detection extension use another ratio; unused);
  symbol and word-break boxes become floor(l*f), floor(t*f), ceil((l+w)*f)-l', ceil((t+h)*f)-t' and the color
  sampling points (int)(p*f) before they are mapped onto the page. Colors are estimated on the pyramid crop.
- Verified: extra_det big_text_huge / big_text_mid / drop_cap2, extra_ties sym_arc (250), and big_text_huge as
  owocr feeds it (LANCZOS to 2048x1280: a curved line at 625, scratchpad stage dump).

## Model output handling (RunBatch 0x1001a7650)
- Dense dequant of non-logit outputs: bias=(float)((double)scale*(double)(-zp)); i in blocks of 8 while i<=n-8:
  fma((float)q,scale,bias); tail (float)((double)scale*(double)(q-zp)).
- Logits: SparseDequantizeBatch (0x10098a0c0) threshold (int)(0.01f/scale+zp) clamped [0,255], k=5 -> [B,T,5,2]
  {class,value}. Equivalent to SparseDequantizeFrame in ctc_decoder.cc (scan order identical: classes ascending).
- MergeChunks: per chunk frames ceil(w/4) (w=min(136,W-offset)) copied from source frames starting at 16/4=4.
- ReduceActivationsGatesAndSoftMax (generic): acc[c]+=(float)(gate[t]*act[t][c]); gatesum+=gate[t];
  v=acc[c]/(gatesum+1.0f); softmax with max start -FLT_MAX strict >, expf, float sum sequential, divide.

## UpdateLineResult (1001fc580)
- direction argmax over first 4 (strict >) -> predicted_direction, direction_confidence; garbage_prob=dir[4].
- greedy CTC: per frame argmax (first max) over 5 entries, label=rintf(class), not in map -> blank;
  prob=1/sum expf(v-max); collapse runs; emit frame=run start, prob of last frame of the run.
- script argmax -> name; content type argmax; language = LangID(text) if script in supported else script name
  (und pass). Script models: language = LangID.

## Symbol boxes: BuildDefaultBoundingBox
scale=(double)(float)(W/(float)T). If any prediction missing: x=min(llround(scale*f),(int)(W-1)),
w=max(1,llround(scale)), y=0,h=(int)H. If text==" ": same x,w; y,h from first non-space label's top/height.
Else center=(float)(((double)f+0.5)*scale); l=left[f]*H; x=clamp(llroundf(center-l),0,(int)(W-1)) (float
subtraction: 279.49999-6e-6 -> 280, extra_ties/sym_arc);
w=max(1,llroundf(l+H*right[f])) clamp min(w,(int)(W-(float)x)); y=clamp(llroundf(H*top[f]),0,(int)(H-1));
h=max(1,llroundf(H*height[f])) clamp min(h,(int)(H-(float)y)).
RTL: predictions = [rev(right), rev(left), (1-rev(top))-rev(height), rev(height)] reversed within valid width;
logits reversed too.

## Reorderer / words
- StandardTextReorderer::JoinAtomsToSymbols (0x100229160): one symbol per label for LTR; symbol box = atom box.
  Line text = concat; symbol conf = product of atom probs; line conf = (float)(sum double / nsym) incl. spaces.
- WordBuilder: whitespace symbol -> break type SPACE (2) (or 3 if already 2), break box expands to contain symbol
  box; non-ws -> start word if none; finalize at SplitBySpace token ends: break NONE(1) default, writing dir,
  text concat, box union, conf=(float)(sum/n).
- RecomputeSpaceWordBreakBoxes: gap=max(0,unionW(last_i,first_i+1)-wA-wB); angle 0: break.left=word.left+word.width,
  width=max(1,ceil(gap)).
- Colors: EstimateWordColorsSymbolAware (0x100a79ac0) — Otsu over bg regions (above/below/right/left of word,
  margin 0.3*h) + fg symbol boxes; full algorithm in the session summary / colors.c.
- Page transform: P=(line.x+box.left, line.y+box.top) rotated about line origin by line angle (double),
  round; angle=(float)NormalizedAngle(line.angle+box.angle); word box = FindContainingBoxWithAverageAngle
  (0x100896340) of children (+break boxes); ExpandBoxToContainBox (0x10081ccbc).

## Output order (002_lay_in)
non-LINE entities first (input order), then per LINE: line, then word, its symbols, next word, ...
LINE gets content_type, orientation, confidence, language, recognizer_script_id, direction fields, garbage_prob,
aksara result {text, language, confidence, predicted_script (und pass only), right_to_left (rtl)}.

## Direction head (every pass)
- The direction head is applied in the script passes too (script models have direction_index, e.g. thai 6).
  d in {1,2}: line box = RotateClockwise(box, 2); logits AND symbol-box predictions are reversed (frame t <- T-1-t),
  so label frames index the reversed sequence. Script-pass input image is cut from the *original* (pass-1 input) box
  and turned by 180 degrees when the box was flipped (verified against tfl input dumps: khmr_laoo; cutting the
  flipped box instead is off by one pixel in x and y).
- The pixel-corner aligned box only matters for |angle| near 180 (detector boxes of upside-down pages,
  testdata/extra_rot): with the old (w, h) corners the clip region ended one row/column short and the border
  samples were clamped to ink.
- aksara right_to_left: set by base_dir_rtl recognizers only, = the paragraph direction (HasAnyRTL(text)); a Hebr
  pass on a Latin garbage line gives false.
- Debug: SCREENAI_REC_DUMP=<dir> writes every model input batch (8x32x168 uint8) as <seq>_in.bin.

## LM decoding (configs with ctc_decoder_config.lm_spec)
- CtcBeamDecoder (ctc_decoder.h) on {values, class_ids}[T][5] from the sparse logits; labels -> symbols as greedy.
- Confidence: CTCDecoderConfidenceScorer::ScoreNth (0x100a72020) -> FeatureExtractor::Extract (0x100a71200):
  row = values[label.frame]; top = min(k-1, logits_top_k); nth_element(greater) at top, resize top+1, sort desc;
  logit = values[frame][first j with rintf(class_ids[frame][j]) == label]; feature_i = topk_i - logit (cost
  features disabled). AvgLogitsConfidenceScorer (0x100a78118): conf = 1.0f / Eigen sum_i exp(feature_i)
  (inner-most reducer: one NEON pexp packet of 4 + scalar expf tail, sum = tail + faddp(packet)). Stored as double.

## Bidi reordering (StandardTextReorderer::SymbolsToResult 0x100227f40)
- display = concat(LogicalToDisplay(symbol)) (BiDiConverter mode 0 = UBIDI_REORDER_DEFAULT), text =
  DisplayToLogical(display) (mode 5 = UBIDI_REORDER_INVERSE_LIKE_DIRECT); both converters get options
  KEEP_BASE_COMBINING only (the BiDiConverter default is DO_MIRRORING); paragraph level = base_dir_rtl &&
  HasAnyRTL(text). ConvertUtf8Text returns UBIDI_LTR text unchanged, else ubidi_writeReordered + getLogicalMap.
  Symbol offset = byte offset of min(logical map) over its code points (-1 if the substring check fails); symbols
  are then in logical (offset) order for the word builder. RTL lines: line writing_direction RTL, no
  RecomputeSpaceWordBreakBoxes / word colors.
- recognizer_bidi.cc ports ICU's ubidi.cpp level resolution (impTabProps, impTabL/R_DEFAULT,
  impTabR_INVERSE_LIKE_DIRECT + impAct1, paired brackets N0 in DEFAULT mode only, adjustWSLevels) for one paragraph
  without explicit embeddings. INVERSE_LIKE_DIRECT differs from the plain UBA on visual text, e.g. visual
  'test 123 (םלוע) םולש' -> 'שלום )עולם( 123 test' (verified against libicucore). Tables from Unicode 15.0
  (tools/gen_bidi_tables.py: classes, BidiMirroring.txt, BidiBrackets.txt).

## Symbol manipulators (StandardTextReorderer::JoinAtomsToSymbols 0x100229160; recognizer_indic.cc)
- Every registered manipulator (registry order = alphabetical: Bengali, Devanagari, Gujarati, Kannada, Mirroring,
  Tamil, Telugu) runs on every line, last to first, before SymbolsToResult. One symbol per CTC label (atom) first;
  if a manipulator fails, symbols stay one per atom.
- Splitting manipulators (Indic): JoinAndReorderStrings(symbol texts) -> logical string; JoinSymbols groups
  consecutive symbols greedily until AlignTextToPieces succeeds and no normalization decomposed sequence
  straddles the group end; the group's text = output substring (normalized to precomposed). Joined symbol box =
  union of atom boxes, confidence = (float) product (double) of atom confidences, line confidence recomputed as
  the mean. Per-script reorder rules: see the comments in recognizer_indic.cc (Devanagari/Gujarati/Bengali pre-base
  vowels, Kannada arkavottu regexes, Tamil left vowels swap, Telugu vowel marks after virama pieces; Tamil and
  Bengali two-part vowel normalization maps).
- MirroringSymbolManipulator (mapping): if the line text has any RTL char, BiDiConverter mode 5 with DO_MIRRORING
  only, paragraph RTL, and output[k] = converted[logical_map[k]] -> each char at an odd level is replaced by its
  Bidi_Mirroring_Glyph, order unchanged; texts handed back to the symbols code point by code point.
- WordBuilder::BuildWords: a symbol with one atom whose single-cp text is the mirror of the atom's text
  (TextReorderer::IsMirrored 0x100c944a8) gets orientation.writing_direction RTL.

## Config-gated heads / misc
- GocrCTCDecoderRecognizerConfig symbol_box_index (19), script_index (20), content_type_index (21),
  direction_index (22): a head is only used when its index is > 0. knda/telu configs have no symbol_box_index
  -> default symbol boxes (full line height); taml/deva have 2.
- Script passes without a content-type head keep the und pass's content type (arab handwriting line). Only the
  und config has one, so handwritten lines re-recognized by a script model (extra_lm/arab, extra_curve/arc_bottom_ja,
  extra_rot/latin_cjk_180) all keep HANDWRITTEN_TEXT.
- A line that decodes to no symbols has no aksara result text / confidence (extra_paths_rec/curve_flip).
- Trailing whitespace symbols after the last word give it no word break (taml 'தலைநகரம் ').

## Vertical lines / content type (e2e ja_vertical, cjk_signs)
- Top-to-bottom lines (predicted_direction 1 or 3, writing_direction TTB): after the words are built, the page
  boxes of the line, words, symbols and break boxes are made upright with RotateOrientationClockwise(box, 3)
  (e.g. 370,32,390x46 @89.94 -> 324,32,46x390 @-0.0589). Words take the line's writing direction (TTB).
  d == 1 takes the same path after the 180-degree flip (testdata/extra_paths_rec/vert_cols_180 '12').
- The symbol/break/color boxes of a vertical straight line are mapped with the upright line box turned back by one
  quadrant (RotateClockwise(RotateClockwise(box, 3), 1)), not the horizontal box itself: the angle round trip
  moves every symbol's page angle by an ulp (lldb on FindContainingBoxWithAverageAngle 0x100896340 in
  vert_ja_180: symbols -90.0830994 for line -90.0830917), which only shows in the averaged word angle near +-180.
- WORD and SYMBOL `type` = the line's content type (HANDWRITTEN_TEXT when the und content head says so).

## Curved lines (recognizer_curve.cc; polygon.curved_box)
Verified bit-exact against lldb dumps of the Halide buffers / Eigen LU (scratchpad rec/halide_dump.py, lu_dump.py,
proto_curve.py) and model-input chunks; extra_curve (arc_top, wave, arc_bottom_ja, arc_gentle, arc_stamp) and
extra_rot/scripts_175 match 002_lay_in.
- Image: GocrLineRecognizer::CreateLineImage -> ImageCache -> google_ocr::ImageCrop (image_utils.cc), polygon case
  3 -> CopyCurvedBoxToBox (thickness -> float -> double) -> ImageUtil::ImageCropCurvedPatch 0x100d512c0:
  - box_util::CurvedBoxCentersAndAngles 0x100d917c0: n+1 entries: p0, midpoints of segments, p[n-1] (float of
    double (a+b)*0.5); angle = float(atan2(b.x-a.x, a.y-b.y)) of the segment (first/last reuse the end segment).
  - per entry: len += hypotf (float); half = thick_d*0.5; d = float(half*(double)cosf), float(half*sinf);
    correspondences target (len, 0) -> src c-d, target (len, thick_f) -> src c+d. W = (int)(len+1.5f), H = (int)thick_f;
    targets /= (W, H).
  - ApplyThinPlateSplineWarp 0x100d541c0 with sample grid sw = W/4+2, sh = H/4+2 (error if > W, H): targets
    t*s+1; ComputeSplineCoefficients: K_ij = r2*logf(r2) (r2 = dx*dx+dy*dy, 0 if <= 0), P = [1 x y], rhs = src;
    Eigen::PartialPivLU<MatrixXf>.solve. Needs Eigen 1da510b8 with -ffp-contract=off to be exact (vendored subset
    in third_party/eigen; Eigen 3.4 / the TFLite build's Eigen give different LU rounding).
  - ThinPlateSplineInterpolationHalide: grid (sw+2)x(sh+2), sample (i+0.5, j+0.5): acc += w_i * U (sequential),
    U = r2 * halide_log(r2) (bit-level range reduction + odd/even polynomial, constants in the .cc), then
    ((acc + a0) + x*a1) + y*a2.
  - output pixel (u,v): grid pos (sw/W)*(u+0.5)+1, (sh/H)*(v+0.5)+1; BilinearInterpolationFloatHalide (p-0.5,
    floor, clamped, interpolate along y first); BilinearInterpolationU8Halide (weights (int)(frac*255), exact
    /255 rounding (v + ((v+128)>>8) + 128)>>8, along y first) from the gray page.
- Boxes: GenerateLineWordsMaybeSetColors -> BoundingPolygonToBoundingBox<Box<double>> (MakeCurvedBox: x=y=0,
  w = curve length, h = thickness, points, thickness (double), top_to_bottom) -> BoxRelativeToPageRelativeVec
  0x100203a04 curved branch: BoundingBoxCoordinateMapper(box, offset 0) InitFromMidLineCurve<CurvedBox> 0x100204720:
  one sample per integer arc length k (while cum > k): t = (k-prev)/seg, angle blended with the neighbour segment
  (GetAverageAngle, t +- 0.5), down = sincos(angle + pi/2), mid = (p + d*t) + down*offset. MapToSourceCoordinates
  0x100c41780: mid[x] + (y - thick*0.5)*down[x], extrapolated along the first/last segment outside [0, count).
  Symbol/break/color boxes: map (int) top-left and (int) Rotate()d top-right; left/top = (int) of the mapped
  top-left, angle = float(atan2(dy, dx) * 180/pi), size kept. Line polygon stays the curved_box.
- Turned curved boxes (direction flip d=1/2, upright of vertical lines): RotateOrientationClockwise<Box<float>>
  0x100212304 keeps the curve; per quadrant (only with >= 2 points) top_to_bottom is cleared if set, else the
  mid-line points are reversed and it is set; the rotated_bounding_box turns like a straight box.
  testdata/extra_paths_rec/curve_flip (d=2), vert_cols / vert_cols_180 (top_to_bottom curved columns).

## Second-pass recognizer selection (LanguageBasedLineRecognizerConfigSelector::Select 0x100148160)
- recognizer_script_id is used as the key when set and content type is CUSTOM (101); never the case here.
- Else languages of language_mix (stable-sorted by weight; one per line here). Key suffixes, most specific
  first: "-t-i0-handwrit:vert", ":vert", "-t-i0-handwrit", "" (handwrit for HANDWRITTEN_TEXT lines or a tag ending
  in -t-i0-handwrit, which is stripped; vert for top-to-bottom lines). Per suffix: tag+s, language subtag+s (not
  und/mul), MostLikelyScript(tag)+s, and "Hani"+s for Hans/Hant. Fallback default key; the low-confidence key
  (threshold > 0) is not in the shipped config. Shipped keys are bare scripts and "und", so only "" matches.
- A "-t-i0-handwrit" that is not at the end of the tag (last occurrence) is an error status ("needs to come at
  the very end"); the whole OCR call fails (CalculatorGraph::Run, lldb injection). Ours: SelectKey -> nullopt,
  Recognize fails. Languages after a matching one are not looked at, so their errors do not count.
- MostLikelyScript (aksara::script_util 0x100b2d148 + ScriptCodeConverter names): LanguageCodeConverter parse
  (case-insensitive, '_' or '-'; grandfathered tags; language must be a known 2/3-letter code, extlang only with
  its registered prefix (zh-yue -> yue), script a known ISO 15924 code (Zzzz = none), region a known one,
  registered variants only, extensions t/u/x) -> explicit script, else special cases (bh Deva, grc Grek, mey/prs
  Arab, alg/ath/... Latn), else i18n_identifiers LanguageInfo::GetLikelyScriptCode (language + region). Anything
  that does not parse is Latn (25); no likely script is Zzzz. Implemented in recognizer_langcode.cc with tables
  generated from lldb dumps of the original (trace/likely_script_dump.py -> trace/gen_likely_scripts.py ->
  recognizer_langcode.inc): ~3.07M tags (all 2/3-letter codes, x 286 regions, x Arab, x 212 scripts, extlang
  prefixes x 3-letter codes, variants, grandfathered) identical (cpp/tools/select_check with the dump dir).
  Script names from the und script head (Armn, Mymr, ...) arrive as tags that do not parse -> Latn; they select
  their recognizer through the tag itself.
- Verified against the original's Select called from lldb on 44445 synthetic LINE entities
  (trace/select_grid.py -> trace/select_check.py -> cpp/tools/select_check): every langid label, region/script/
  malformed/handwrit tags (zh-Hant, zh-TW, zh-HK, sr-Latn, ja-Latn, und, mul, '', xx, ...) and script names, alone
  and paired with each first-pass script result, x content type (unset/0/1/2/101) x writing direction
  (unset/0..3), and CUSTOM recognizer_script_id cases: identical. End to end, trace/select_inject.py overwrites
  the language at Select under lldb (SCREENAI_DEBUG_SELECT_LANG does the same in ours): en_simple and
  extra_lm/hani with zh-Hant / zh-Hans (Latin lines go to Hani) match 002_lay_in and 003_final.
- Select is also called once per line before the first pass (no language_mix yet -> default key).
- Reached when the und script head says Zzzz/Zyyy/Latn but langid names a non-Latin language: all-caps Cyrillic
  homoglyph words (testdata/extra_paths_rec/select_mls: 'МОСКВА' read 'MOСКBА', langid sr -> Cyrl; also ru).
  zh -> Hans -> Hani is not reached by images: the und model knows 6 hanzi, so a line langid calls zh is a Hani
  line for the script head too (covered by the Select grid and the injection runs above).
