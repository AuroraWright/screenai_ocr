# Boost.Polygon (Voronoi subset)

This folder holds the Voronoi builder and diagram headers from Boost.Polygon, Boost 1.92, unmodified. Boost is under
the Boost Software License 1.0 (see `LICENSE_1_0.txt`).

## Why it's here

Layout analysis groups text lines using a Delaunay graph of their positions. The original library builds that graph
(`google_ocr::box_beta_skeleton::GetDelaunay`) on top of `boost::polygon::voronoi_builder`. Using the same Boost code
gives exactly the same triangulation, including how it handles points that lie on a common circle, where other
algorithms could pick different triangles. It is used by `cpp/src/layout_graph.cc`.

## Local changes

`cstdint.hpp` is our own small shim that stands in for Boost's version, so the rest of Boost isn't needed.
