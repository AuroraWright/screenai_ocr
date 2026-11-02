Subset of Eigen (https://gitlab.com/libeigen/eigen) at commit 1da510b8155f8fa81924616dcc517ecc4202e4d3
(the version TensorFlow master pins), MPL-2.0 (see COPYING.MPL2 / LICENSES).

Only the headers reached from `Eigen/Core` + `Eigen/LU` on arm64 are kept. It is used by
cpp/src/recognizer_curve.cc for the thin-plate-spline solve of curved text lines: the original library solves that
system with Eigen::PartialPivLU<MatrixXf> built with -ffp-contract=off, and this exact Eigen version reproduces its
coefficients bit for bit (older Eigen releases do not).
