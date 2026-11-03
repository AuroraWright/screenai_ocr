# screenai_ocr: C++ reimplementation of the Chrome Screen AI OCR library

A clean-room-style C++ reimplementation of the OCR part of Chrome's `libchromescreenai.so`, built from reverse
engineering (IDA, lldb traces). It runs the same TFLite models, which **are not included**: point it at your own
Screen AI component directory (default location used by owocr: `~/.config/screen_ai/resources`).

It builds a drop-in `libchromescreenai.so` exporting the same C API (`SetFileContentFunctions`,
`InitOCRUsingCallback`, `SetOCRLightMode`, `GetMaxImageDimension`, `PerformOCR`, `FreeLibraryAllocatedCharArray`,
`UninitializeOCR`, ...). `PerformOCR` returns a serialized `chrome_screen_ai.VisualAnnotation`.

## Pipeline

| Stage | Original | Here |
|---|---|---|
| gray conversion | visionkit ImageConversion | `src/image.cc` |
| text detection | GocrGroupRpnTextDetectionMutator | `src/detector.cc` |
| line recognition (multi-pass, per-script models, CTC + LM beam, bidi/text reordering, words, colors) | MultiPassLineRecognitionMutator | `src/recognizer*.cc`, `src/ctc_decoder.cc`, `src/lm_*.cc` |
| language ID (MobileLangID V2, custom NGramHash / KmeansEmbeddingLookup ops) | | `src/language_id.cc` |
| page layout analysis (pruning, ClusterSort GCN reading order, blocks/paragraphs, boxes) | PageLayoutAnalysisMutator | `src/layout_analyzer.cc` |
| VisualAnnotation conversion | ConvertOCRResultsToVisualAnnotationProto (ocr.cc) | `src/visual_annotation.cc` |
| C API | chrome_screen_ai exports | `capi/` |

Reverse-engineering notes: `notes/recognizer.md`, `notes/layout.md`, `notes/language_id.md`, and the header comments
of each stage.

## Building

Bit-exact results need TensorFlow Lite built the way the original is: **no floating-point contraction** (the
original never fuses multiply-adds; clang's default `fmadd` changes rounding in XNNPACK's quantized MEAN and flips
recognizer outputs), with XNNPACK at the version the original ships.

1. TFLite (TensorFlow 2.21.0 `tensorflow/lite`), XNNPACK `28d1fc6d908bc4d80920be989177b7eb1a4aebc8` and the
   dependency pins from that XNNPACK revision (cpuinfo `66ee79c0`, pthreadpool `15a6644b`, kleidiai `5147983e`):

   ```sh
   cmake -S tensorflow-2.21.0/tensorflow/lite -B tflite_nofma_build -G Ninja -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_C_FLAGS=-ffp-contract=off -DCMAKE_CXX_FLAGS=-ffp-contract=off -DTFLITE_ENABLE_XNNPACK=ON \
     -DFETCHCONTENT_SOURCE_DIR_XNNPACK=<XNNPACK-28d1fc6d...> -DCPUINFO_SOURCE_DIR=<cpuinfo-66ee79c0...> \
     -DPTHREADPOOL_SOURCE_DIR=<pthreadpool-15a6644b...> -DKLEIDIAI_SOURCE_DIR=<kleidiai-5147983e...>
   ninja -C tflite_nofma_build tensorflow-lite
   ```

2. Eigen: `../third_party/eigen/` is a vendored header subset of Eigen commit `1da510b8` (MPL-2.0), used only for
   the thin-plate-spline solve that straightens curved text lines; that exact version (with `-ffp-contract=off`)
   reproduces the original's coefficients bit for bit. Its namespace is renamed to keep it apart from TFLite's copy.

3. This project (the TFLite paths default to `../third_party/`):

   ```sh
   cmake -S cpp -B cpp/build -G Ninja -DTFLITE_SOURCE_DIR=<tensorflow-2.21.0> -DTFLITE_BUILD_DIR=<tflite_nofma_build>
   ninja -C cpp/build chromescreenai run_ocr
   ```

   Every `src/*.cc` goes into `screenai_core`, and every `tools/*.cc` becomes a stage tool (`run_detector`,
   `run_recognizer`, `run_layout`, `run_visual_annotation`, `run_ctc_decoder`, `tfl_check`, ...).

## Using it

- Command line: `run_ocr <model_dir> <image> <out_visual_annotation.pb> [out_layout.pb]`.
  Set `SCREENAI_STAGE_DUMP=<dir>` to write each stage's layout as `goodoc.PageLayoutProto`, and `SCREENAI_TIMING=1`
  for per-stage wall times on stderr. `bench_ocr <model_dir> <runs> <images...>` reports median end-to-end times;
  `bench_model` times one model on tensors dumped by `trace/tfl_dump.py` with a given thread/interpreter layout.
- Threading follows the original's runner configs: the detector uses one 4-thread interpreter, and each line
  recognizer keeps a pool of 4 single-threaded interpreters that run the 8-line batches in parallel
  (`TfliteModelPooledRunner`). On the 64 test images total time is within a few percent of the original.
- As a drop-in library: put `cpp/build/libchromescreenai.so` in place of the original in a model directory. owocr
  loads `~/.config/screen_ai/resources/libchromescreenai.so`. To avoid touching your model directory, make a
  directory of symlinks to its contents and replace only the `.so` link (what `tools/compare_owocr.py` does).
  Without file callbacks, `InitOCRUsingCallback` also accepts `SCREEN_AI_MODEL_DIR=<dir>`.
- `SetOCRLightMode(true)` is accepted but ignored: only the default (non-light) pipeline is reimplemented, which is
  what owocr uses (it calls `SetOCRLightMode(false)`). Set before init, the original's light mode changes results on
  pages with non-Latin text (43 of the 92 test images); Latin-only pages are unchanged. Main-content extraction is
  not implemented.

## Verification

The Python tools decode protos with descriptors pulled from your copy of the original library (not tracked in git):
`python -I tools/extract_protos.py <path to original libchromescreenai.so> protos_extracted`.

- `tools/compare_owocr.py <orig model dir> <our .so> <images...>`: runs owocr's own `ChromeScreenAI` class against
  both libraries and compares the raw VisualAnnotation and owocr's parsed result.
- `tools/api_check.py <lib> <model dir> <images...>`: C API timing, output hashes and edge cases.
- Per stage: `tools/cmp_layout.py` against the original's stage dumps (`trace/stage_dump.py` under lldb) in
  `testdata/stages*/`; `tests/test_layout.py` for layout regression.
- Test images (all generated; originals' stage dumps in `testdata/stages*/`): `make_tests.py` (12, development),
  `make_e2e.py` (`e2e/`, 10), `make_holdout2.sh` (`holdout2/`, 12, held out from stage development), `extra_lm/`
  (10 LM scripts, CoreText), `extra_indic/` (9), `extra_rot/` (6, 175-185 degrees), `make_extra_curve.py`
  (`extra_curve/`, 5 curved-text pages), and targeted path coverage: `make_extra_paths.py` (`extra_paths/`, 2,
  curved-line overlap pruning), `extra_paths_rec/gen.sh` (6: flipped curves, language-to-script recognizer selection,
  RTL bidi mixes, upside-down vertical text), `make_extra_ties.py` (`extra_ties/`, 8, tie-heavy layouts),
  `make_extra_det.py` (`extra_det/`, 12: drop caps, tiny/huge/strip pages, large text cropped from pyramid levels),
  `extra_fuzz/` (4 fuzz pages that differed from the original before their fixes; provenance in its README).
- Rarely taken paths were checked for reachability against the shipped configs before writing tests; paths the
  configs make unreachable (detection nuggets, vertical word expansion, `heuristic_blocks`, accumulated support,
  `fallback_upscaling`, the legacy pyramid path) are documented with evidence in `notes/` instead.

- Differential injection tests (same perturbation applied to the original under lldb and to ours via an opt-in
  debug env var, which has no effect when unset):
  - recognizer selection: `trace/select_check.py` calls the original's selector on a 44,445-case grid of language
    tags x first-pass scripts x content types/directions, `tools/select_check` replays it through ours; language
    overrides mid-run with `trace/select_inject.py` / `SCREENAI_DEBUG_SELECT_LANG`.
  - layout: `trace/lay_inject.py` replaces the original's layout input (its own protobuf parser) and `tools/lay_fuzz.py`
    generates targeted and random layout edits; regression subset in `testdata/stages_fuzz/`.
  - box overlap: `trace/overlap_check.py` calls the original's `Rectangle<float>` / `SafeBoxOverlapArea` /
    `OverlapComputer` on generated boxes, `tools/overlap_check` replays them through `src/box_overlap.cc`.
  - detector: `SCREENAI_DET_INJECT` / `trace/det_dump.py DD_INJECT` edit the model outputs to force exact ties in all
    sorts; cases in `testdata/det_inject/`, run with `trace/det_inject_check.py`.

## Robustness

`tools/api_robust.py <lib> <model_dir> <check>` runs the same C API checks on the original and on ours (release, or
the AddressSanitizer + UBSan build: `cmake -S cpp -B cpp/build_asan -DCMAKE_BUILD_TYPE=None
-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1" ...` with the same linker flags, and
`API_ROBUST_INSERT=$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib)`). Checks: `fuzz` (random bitmaps;
`text` for distorted real pages), `lifecycle`, `faults` (missing / truncated / garbage / bit-flipped model files),
`threads`, `soak`. Results (2026-11-02, sanitizer build): no sanitizer reports or crashes on the 92 test images,
1,200 fuzz cases or 160 model-file faults; 8 concurrent threads give identical output; memory is flat over 800 runs.
Intentional differences from the original:
- Negative bitmap sizes return an empty result (the original aborts); initializing twice is a no-op (the original
  aborts); a missing or corrupt layout model fails init (the original crashes).
- After a failed `PerformOCR` (e.g. a side of exactly 32768), the original returns NULL for every later call until
  it is re-initialized; ours keeps working.
- Script recognizers load on first use, so a broken script config only fails pages that need that script; the
  original reads all script configs at init and fails init. The engine config file is not read.
- Matched on purpose: a truncated/corrupt model or label map or LM symbol table fails like the original (init, or
  the OCR call that needs it), and a missing language-ID model runs without language ID, byte-identical to the
  original.
The text fuzz found four recognizer bugs (geometry only: vertical-line space boxes, flipped curved lines, the
angles of vertical and curved vertical lines re-recognized in the second pass); they are fixed and kept as
`testdata/extra_fuzz/`. The one remaining difference is not a bug: the original runs its Indic symbol manipulators
in hash-map order seeded per process, so on lines mixing Bengali and Devanagari marks separate runs of the original
disagree with each other (`testdata/extra_fuzz/README`); ours gives the result of the original with ASLR off.

Status (2026-11-01): all 92 test images give byte-identical `PerformOCR` output at full resolution and identical
owocr results vs the original library. Injection tests: recognizer selection 44,445/44,445 (plus `MostLikelyScript`
on ~3.07M tags), layout 1618/1621 (the 3 differences need a soft-hyphen word break without a box, which the recognizer
never produces), box overlap 125,500 pairs + 3,000 box sets bit-identical, detector forced ties 13/13 with every sort tying. Total time on the test images is within a few
percent of the original. Known unmodelled inputs: weighted multi-language mixes in recognizer selection, math
content types (6/7) in word text joins; none occur in the pipeline. Light mode and main-content extraction are not
implemented.
