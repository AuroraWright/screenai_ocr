# Fuzz regression pages

These pages come from the text fuzzer in `tools/api_robust.py` (`<lib> <model_dir> fuzz 600 <seed> text`), which
distorts real test pages and compares ours with the original. On each of them, ours gave different output from the
original. All four differences were bugs in box geometry, not text. They are fixed, and the pages are kept so they stay
fixed. The original's stage dumps for them are in `../stages_extra_fuzz/`.

## The pages

| Page | Bug it found |
|---|---|
| `vert_space.png` | A recognized space on a vertical line got the wrong box. The recognizer's `RecomputeSpaceWordBreakBoxes` measures gaps along the writing direction, so for top-to-bottom lines it uses heights, not widths. |
| `curve_flip_float.png` | Curved lines that the direction model flips upside down. `RotateOrientationClockwise<Box<float>>` rounds the curve to float, and we didn't. |
| `vert_tile_180.png` | Vertical lines kept from the first pass. They map their symbols without the angle round trip that re-recognized lines go through. |
| `curve_2pass.png` | Curved vertical lines re-recognized in the second pass. Their box round trip rounds the curve to float. |

## How to regenerate them

Each page is one fuzz case. Rerun the fuzz command with the given seed and take the given case:

| Page | Command seed | Case | Source image | Distortion | Size | Color type | Case seed |
|---|---|---|---|---|---|---|---|
| `vert_space.png` | 11 | 38 | `testdata/extra_paths_rec/vert_cols.png` | resize | 1457 × 363 | 14 | 895829541 |
| `curve_flip_float.png` | 11 | 388 | `testdata/extra_rot/latin_cjk_175.png` | lowcontrast | 2137 × 2409 | 4 | 307010521 |
| `vert_tile_180.png` | 23 | 209 | `testdata/extra_paths_rec/vert_cols_180.png` | tile | 2634 × 1960 | 4 | 236994653 |
| `curve_2pass.png` | 37 | 128 | `testdata/extra_paths_rec/vert_cols_180.png` | resize | 1028 × 1652 | 4 | 441497745 |

This folder is excluded from the fuzzer's source images, so adding these pages didn't change which cases the seeds
produce.

## A difference that isn't kept

Case 3 with seed 37 (`extra_indic/deva2.png`, lowcontrast) also differed, but it has no single right answer. The
original applies its Indic symbol rules in the order of an `absl::flat_hash_map` (`FunctionRegistry::GetAll`), and that
order is seeded randomly in each process. On this line, a mix of Bengali and Devanagari marks makes the order matter,
so separate runs of the original give one of two outputs. Ours always gives the one the original gives with address
randomization (ASLR) off, for example under lldb.
