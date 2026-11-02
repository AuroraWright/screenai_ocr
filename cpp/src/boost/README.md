Vendored subset of Boost.Polygon (Boost 1.92, Boost Software License 1.0, see LICENSE_1_0.txt): the voronoi
builder/diagram headers, unmodified. The original library builds its Delaunay graph for layout line clustering
(google_ocr::box_beta_skeleton::GetDelaunay) on boost::polygon::voronoi_builder, so the same code is used here to
reproduce its triangulation (including its handling of co-circular points). cstdint.hpp is a local shim.
