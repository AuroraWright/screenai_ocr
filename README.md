# screenai_ocr

A C++ reimplementation of the OCR engine inside Chrome's Screen AI library (`libchromescreenai.so`).

Chrome ships Screen AI as a closed binary with a set of TensorFlow Lite models. Tools such as
[owocr](https://github.com/AuroraWright/owocr) load that binary to get its high-quality OCR. This project rebuilds the
OCR part of the library from reverse engineering. It produces a drop-in `libchromescreenai.so` that runs the same
models and gives byte-for-byte the same results.

**The models are not included.** You need your own copy of the Screen AI component, for example the one owocr
downloads to `~/.config/screen_ai/resources`. This project never ships or modifies it.

## Status

- **Same output as the original.** All 96 test images give byte-identical `PerformOCR` output to the original
  library, and owocr parses the same results from both.
- **Same speed.** Total time on the test images is within a few percent of the original.
- **Same C API.** The library exports the same functions, so callers load it unchanged: `SetFileContentFunctions`,
  `InitOCRUsingCallback`, `SetOCRLightMode`, `GetMaxImageDimension`, `PerformOCR`,
  `FreeLibraryAllocatedCharArray`, `UninitializeOCR`, and the rest.
- **Light mode** is supported and matches the original's light mode (92 test images and 600 fuzzed pages).
- **Not implemented:** main-content extraction, the other half of Screen AI. owocr doesn't use it.

## Building

You need CMake, Ninja and clang, and a TensorFlow Lite build made the same way as the original's. The details matter:
a TFLite built with default settings gives slightly different numbers, and that is enough to change some recognized
text.

### 1. Build TensorFlow Lite

Use TensorFlow 2.21.0's `tensorflow/lite`, with the exact XNNPACK revision the original ships and that revision's
dependency pins:

| Component   | Revision                                   |
|-------------|--------------------------------------------|
| XNNPACK     | `28d1fc6d908bc4d80920be989177b7eb1a4aebc8` |
| cpuinfo     | `66ee79c0`                                 |
| pthreadpool | `15a6644b`                                 |
| kleidiai    | `5147983e`                                 |

Build it with **floating-point contraction turned off** (`-ffp-contract=off`). The original never fuses
multiply-adds; clang does by default, and the different rounding flips some recognizer outputs.

```sh
cmake -S tensorflow-2.21.0/tensorflow/lite -B tflite_nofma_build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS=-ffp-contract=off -DCMAKE_CXX_FLAGS=-ffp-contract=off -DTFLITE_ENABLE_XNNPACK=ON \
  -DFETCHCONTENT_SOURCE_DIR_XNNPACK=<XNNPACK-28d1fc6d...> -DCPUINFO_SOURCE_DIR=<cpuinfo-66ee79c0...> \
  -DPTHREADPOOL_SOURCE_DIR=<pthreadpool-15a6644b...> -DKLEIDIAI_SOURCE_DIR=<kleidiai-5147983e...>
ninja -C tflite_nofma_build tensorflow-lite
```

### 2. Build this project

```sh
cmake -S cpp -B cpp/build -G Ninja -DTFLITE_SOURCE_DIR=<tensorflow-2.21.0> -DTFLITE_BUILD_DIR=<tflite_nofma_build>
ninja -C cpp/build chromescreenai run_ocr
```

If you put the TensorFlow source and build in `third_party/tensorflow-2.21.0` and `third_party/tflite_nofma_build`,
you can leave out the two `-D` paths.

This builds `cpp/build/libchromescreenai.so` and the `run_ocr` command-line tool. Building everything (`ninja -C
cpp/build`) also gives a tool per pipeline stage (`run_detector`, `run_recognizer`, `run_layout`, ...) from
`cpp/tools/`.

Everything else is already in the repo:
- `third_party/eigen/` holds part of Eigen's headers at commit `1da510b8` (MPL-2.0). It is used only to straighten
  curved text lines, and that exact version reproduces the original's numbers bit for bit. Its namespace is renamed so
  it doesn't clash with TFLite's copy of Eigen.
- `third_party/stb/` holds the stb image loader.

## Using it

### With owocr, or any other program that loads the original

Put `cpp/build/libchromescreenai.so` in place of the original library in a Screen AI model directory. owocr loads
`~/.config/screen_ai/resources/libchromescreenai.so`.

To try it without touching your model directory, make a new directory of symlinks to its contents and replace only
the `.so` link. `tools/compare_owocr.py` does this.

If the caller doesn't provide file callbacks, `InitOCRUsingCallback` reads the models from the directory in
`SCREEN_AI_MODEL_DIR`.

### From the command line

```sh
cpp/build/run_ocr <model_dir> <image> <out_visual_annotation.pb> [out_layout.pb]
```

The output is a serialized `chrome_screen_ai.VisualAnnotation`, the same message `PerformOCR` returns. Two environment
variables help with debugging:
- `SCREENAI_STAGE_DUMP=<dir>` writes each stage's layout as a `goodoc.PageLayoutProto`.
- `SCREENAI_TIMING=1` prints per-stage times to stderr.

For benchmarks, `bench_ocr <model_dir> <runs> <images...>` reports median end-to-end times. `bench_model` times one
model on tensors dumped by `trace/tfl_dump.py`.

### Light mode

`SetOCRLightMode(true)` runs a single recognition pass and skips the script-specific second pass. It is faster and
gives the same results on Latin-only pages, but can differ on pages with other scripts (43 of 92 test images).
As in the original, the setting:
- takes effect from the next `PerformOCR` call,
- can be made before or after init,
- survives `UninitializeOCR`.

owocr turns light mode off.

### Threading

Threading follows the original's configuration. The text detector uses one interpreter with 4 threads. Each line
recognizer keeps a pool of 4 single-threaded interpreters that recognize batches of 8 lines in parallel.

## Windows and other x86-64 machines

The original's Windows build (`chrome_screen_ai.dll`) doesn't give exactly the same results as its Mac and Linux arm64
builds. Its x86 code paths round differently in two image-resizing steps:
- libyuv's column filter (`ScaleFilterCols_SSSE3`) uses a 7-bit fraction.
- The warp that straightens rotated lines uses integer weights.

When you build for an x86-64 target, `cpp/src/x86_compat.h` switches both on automatically. Arm builds are unchanged.
You can force the choice with `-DSCREENAI_X86_COMPAT=0` or `=1`.

With these switched on, the images fed to the models match the Windows DLL (version 153.05). The detector's input
matches exactly, and 94.5% of the line images fed to the recognizer match byte for byte.

The final output still differs on most images. The Windows DLL runs the detector model with different XNNPACK float
kernels, and those depend on the CPU. Running the DLL on the same machine with and without AVX2 changes its output on
85 of 96 images, so there is no single Windows result to match. This project doesn't reproduce those kernels.

## How it works

The library is a pipeline of stages. Each one reimplements a component of the original:

| Stage | What it does | Original component | Code |
|---|---|---|---|
| Image conversion | Converts the bitmap to grayscale | visionkit `ImageConversion` | `cpp/src/image.cc` |
| Text detection | Finds text lines on the page | `GocrGroupRpnTextDetectionMutator` | `cpp/src/detector.cc` |
| Line recognition | Reads each line with per-script models over multiple passes: CTC decoding with a language-model beam search, bidirectional text, words, text colors | `MultiPassLineRecognitionMutator` | `cpp/src/recognizer*.cc`, `ctc_decoder.cc`, `lm_*.cc` |
| Language ID | Identifies each line's language (MobileLangID V2) | | `cpp/src/language_id.cc` |
| Layout analysis | Prunes overlapping lines, works out reading order (a graph network), and groups lines into blocks and paragraphs | `PageLayoutAnalysisMutator` | `cpp/src/layout_analyzer.cc` |
| Output | Builds the `VisualAnnotation` result | `ConvertOCRResultsToVisualAnnotationProto` | `cpp/src/visual_annotation.cc` |
| C API | The exported functions | `chrome_screen_ai` exports | `cpp/capi/` |

The reverse-engineering notes are in `cpp/notes/` (`recognizer.md`, `layout.md`, `language_id.md`) and in the header
comments of each stage.

## Testing

### Setup

The Python tools decode protobufs using descriptors taken from your copy of the original library. Extract them once
(the result isn't tracked in git):

```sh
python -I tools/extract_protos.py <path to original libchromescreenai.so> protos_extracted
```

### Comparing with the original

- `tools/compare_owocr.py <orig model dir> <our .so> <images...>` runs owocr's own `ChromeScreenAI` class against both
  libraries. It compares the raw output and owocr's parsed results.
- `tools/api_check.py <lib> <model dir> <images...>` checks C API timing, output hashes and edge cases.
- `tools/cmp_layout.py` compares each stage's output with dumps from the original in `testdata/stages*/`. The dumps
  are made with `trace/stage_dump.py` under lldb.
- `tests/test_layout.py` is the layout regression test.

### Test images

All test images are generated, and the original's outputs for them are stored next to them in `testdata/`.

| Set | Images | Purpose | Generator |
|---|---|---|---|
| `testdata/` | 12 | development | `make_tests.py` |
| `e2e/` | 10 | end-to-end | `make_e2e.py` |
| `holdout2/` | 12 | held out while the stages were written | `make_holdout2.sh` |
| `extra_lm/` | 10 | scripts with language models | `gen.sh` (CoreText) |
| `extra_indic/` | 9 | Indic scripts | `gen.sh` |
| `extra_rot/` | 6 | pages rotated 175–185° | `gen.sh` |
| `extra_curve/` | 5 | curved text | `make_extra_curve.py` |
| `extra_paths/` | 2 | pruning of overlapping curved lines | `make_extra_paths.py` |
| `extra_paths_rec/` | 6 | flipped curves, recognizer choice by language, mixed right-to-left text, upside-down vertical text | `gen.sh` |
| `extra_ties/` | 8 | layouts full of ties | `make_extra_ties.py` |
| `extra_det/` | 12 | drop caps, tiny, huge and strip-shaped pages, large text | `make_extra_det.py` |
| `extra_fuzz/` | 4 | pages where fuzzing found bugs (since fixed; see its README) | |

Before a test was written for a rarely used code path, it was checked whether the shipped configs can reach that path
at all. For the paths they can't reach, `cpp/notes/` documents the evidence instead of a test.

### Injection tests

Some cases are too rare to hit with real images. For these, the same change is made to the original (under lldb) and
to ours (through an opt-in debug environment variable that does nothing when unset), and the results are compared.

| Area | Cases | Result | Tools |
|---|---|---|---|
| Recognizer choice: language tags × first-pass scripts × content types and directions | 44,445 | all identical, plus `MostLikelyScript` on about 3.07 million tags | `trace/select_check.py`, `cpp/tools/select_check.cc`, `trace/select_inject.py`, `SCREENAI_DEBUG_SELECT_LANG` |
| Layout: targeted and random edits to the layout input | 1,621 | 1,618 identical (below) | `trace/lay_inject.py`, `tools/lay_fuzz.py`, `testdata/stages_fuzz/` |
| Box overlap computations | 125,500 box pairs and 3,000 box sets | bit-identical | `trace/overlap_check.py`, `cpp/tools/overlap_check.cc` |
| Detector: model outputs edited to force exact ties in every sort | 13 | all identical | `SCREENAI_DET_INJECT`, `trace/det_dump.py`, `trace/det_inject_check.py`, `testdata/det_inject/` |

The 3 layout differences need a soft-hyphen word break without a box, which the recognizer never produces.

Two kinds of input aren't modelled because the pipeline never produces them: weighted mixes of several languages in
recognizer selection, and math content types in word text joins.

## Robustness

`tools/api_robust.py <lib> <model_dir> <check>` runs the same stress tests on the original and on ours. The checks are:
- `fuzz`: random bitmaps, or `text` for distorted real pages.
- `lifecycle`: init, uninit and mode-switching sequences.
- `faults`: missing, truncated, garbage and bit-flipped model files.
- `threads` and `soak`.

To run it with AddressSanitizer and UBSan, build a separate copy and point the tool at the sanitizer runtime. In place
of `...`, pass the TFLite paths as in the normal build and the same `-fsanitize` flags as linker flags:

```sh
cmake -S cpp -B cpp/build_asan -DCMAKE_BUILD_TYPE=None \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1" ...
API_ROBUST_INSERT=$(clang -print-file-name=libclang_rt.asan_osx_dynamic.dylib) python -I tools/api_robust.py ...
```

Last run (2026-11-02, sanitizer build): no sanitizer reports or crashes on the test images, 1,200 fuzz cases or 160
model-file faults. 8 threads running at once give identical output, and memory stays flat over 800 runs.

Fuzzing found four recognizer bugs, all affecting only box geometry, never text: spaces on vertical lines, flipped
curved lines, and the angles of vertical and curved vertical lines re-read in the second pass. They are fixed, and
the pages that found them are kept in `testdata/extra_fuzz/`.

### Where ours behaves differently on purpose

In a few error cases ours keeps going where the original crashes or stops working:

| Situation | Original | Ours |
|---|---|---|
| Negative bitmap size | aborts | returns an empty result |
| Initializing twice | aborts | does nothing the second time |
| Missing or corrupt layout model | crashes | init fails |
| After a failed `PerformOCR` (for example an image side of exactly 32768) | returns NULL until re-initialized | keeps working |
| A broken config for one script | init fails (it reads all script configs at init) | only pages that need that script fail (recognizers load on first use) |

Ours also doesn't read the engine config file.

Some error cases match on purpose. A truncated or corrupt model, label map or language-model symbol table fails the
same way as in the original, either at init or in the OCR call that needs it. A missing language-ID model makes both
run without language ID, with byte-identical output.

One known difference is in the original itself, not a bug in ours. The original applies its Indic symbol rules in
hash-map order, which changes from process to process. On lines that mix Bengali and Devanagari marks, two runs of the
original can disagree with each other. Ours gives the result the original gives with address randomization (ASLR) off.
See `testdata/extra_fuzz/README.md`.

## Repository layout

| Path | Contents |
|---|---|
| `cpp/src/` | the pipeline stages |
| `cpp/capi/` | the C API, `run_ocr` and `bench_ocr` |
| `cpp/tools/` | per-stage tools and checkers |
| `cpp/notes/` | reverse-engineering notes |
| `tools/` | Python comparison, fuzzing and robustness tools |
| `trace/` | lldb scripts that trace and inject into the original |
| `tests/` | regression tests |
| `testdata/` | generated test images and the original's outputs |
| `screenai/` | a Python port of libyuv's scaler, used by tests |
| `ref/` | runs the original library to make reference outputs |
| `third_party/` | Eigen subset, stb, and the generated `VisualAnnotation` Python bindings |

## License

This project is licensed under the [Mozilla Public License 2.0](https://mozilla.org/MPL/2.0/); see `LICENSE`.

Bundled third-party code keeps its own licence:
- `third_party/eigen/`: Eigen, MPL-2.0.
- `cpp/src/boost/`: Boost.Polygon, Boost Software License 1.0 (except `cstdint.hpp`, which is ours).
- `third_party/stb/`: stb_image, public domain or MIT, at your choice.

The license covers only this project's code, not the Screen AI library or its models, which you supply yourself.
