# Language ID (cpp/src/language_id.{h,cc})

Reimplements google_ocr::MobileLangIdV2 over
`gocr/gocr_models/line_recognition_mobile_convnext320_omni/tflite_langid.tflite` (string-tensor input).

## Validation status

`tests/test_langid.py cpp/build_lay/run_langid <model_dir> <dirs>` against lldb dumps of the original
(`trace/lid_dump.py`: every model invoke's softmax/logit tensors + every IdentifyLanguage result):

- 12 stage images (all recognized line texts) + `testdata/extra_lid` (54 multilingual sentences, 25 edge cases,
  3 invalid-UTF-8 byte strings): **380 cases (190 invokes, 190 results), 0 mismatches**.
- Softmax and logits are bit-identical for every invoke; every (language, confidence-bits) result is identical.
- Covered paths: `und` via confidence < 0.2, via < 5 chars, via `-Latn` labels; `ru-PETR1708`; invalid UTF-8
  (original's op yields no tokens -> invoke fails -> result {"und", 1.0}).
- Known divergence (not testable): a *truncated trailing* UTF-8 sequence makes the original's tokenizer loop forever;
  ours stops at the truncated sequence.

How exactness was reached:
- NGramHash / KmeansEmbeddingLookup custom ops reimplemented (MurmurHash64, PQ codebooks, mean over sequence).
- Hybrid int8 FULLY_CONNECTED on builtin kernels, **no XNNPACK delegate** (the original doesn't use it for this model).
- SOFTMAX replaced by a copy of the original's optimized_ops::SoftmaxImpl (Eigen pexp + redux order), registered by
  `LanguageIdentifier::RegisterCustomOps`.

## Post-processing (what the recognizer stream needs)

`IdentifyLanguage(text)`:
1. If invoke fails or no scores: return {"und", 1.0}.
2. Arg-max = first index with strictly greater score, scanning from 0 (initial best score 0.0).
3. Label "unknown" -> "und".
4. `SanitizeSaftLanguage(text, lang, conf)`:
   - `conf < 0.2` (float compared to double 0.2) or `Utf8StrLen(text) < 5` -> "und"
   - label ending in "-Latn" (length >= 5) -> "und"
   - else if lang == "ru" and `(double)len * 0.005 < count` -> "ru-PETR1708", where count = runes in
     {I, i, U+0462/3 ѣ, U+0472/3 ѳ, U+0474/5 ѵ} + word-final hard signs (alpha, ъ/Ъ, non-alpha triples)
   - otherwise lang unchanged.
5. Returned confidence is the raw arg-max softmax probability (unsanitized).

## Mapping to entities (observed in 002_lay_in / 003_final dumps)

- The recognizer writes the sanitized tag as the LINE's `language_mix.languages[0].bcp47_tag` and the same tag on
  every WORD of that line (tags only, no weight). "und" is kept as a tag.
- Page-level `language_mix` (with weights) is computed later by the layout mutator (PopulatePageLanguage, see
  layout.md): per WORD, weight += word.confidence * num_children(word); normalized; "und" lines that survive
  contribute like any other tag.
