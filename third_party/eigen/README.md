# Eigen (subset)

This folder holds part of [Eigen](https://gitlab.com/libeigen/eigen) at commit
`1da510b8155f8fa81924616dcc517ecc4202e4d3`, the version TensorFlow master pins. Eigen is under the MPL-2.0 (see
`COPYING.MPL2` and `LICENSES`).

Only the headers needed for `Eigen/Core` and `Eigen/LU` on arm64 are kept.

## Why it's here

Curved text lines are straightened with a thin-plate spline, which means solving a small linear system. The original
library solves it with `Eigen::PartialPivLU<MatrixXf>`, built with floating-point contraction off
(`-ffp-contract=off`). This exact Eigen version, built the same way, reproduces the original's coefficients bit for
bit. Older Eigen releases round differently.

It is used only by `cpp/src/recognizer_curve.cc`. That file renames the namespace to `screenai_eigen` while including
it, so it never mixes with the older Eigen copy inside TensorFlow Lite.
