cmake_minimum_required(VERSION 3.20)
project(screenai_ocr CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# The original library is built without floating-point contraction; fused multiply-adds change rounding.
add_compile_options(-ffp-contract=off)

# Prebuilt TensorFlow Lite (cmake build of tensorflow/lite, also built with -ffp-contract=off):
# TFLITE_SOURCE_DIR = TF source root, TFLITE_BUILD_DIR = its build directory.
set(TFLITE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tensorflow-2.21.0" CACHE PATH "")
set(TFLITE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tflite_nofma_build" CACHE PATH "")
file(GLOB_RECURSE TFLITE_LIBS "${TFLITE_BUILD_DIR}/*.a")
list(FILTER TFLITE_LIBS EXCLUDE REGEX "(gtest|gmock|benchmark|tensorflow-src|testdata)")
add_library(tflite_prebuilt INTERFACE)
target_include_directories(tflite_prebuilt INTERFACE
  "${TFLITE_SOURCE_DIR}"
  "${TFLITE_BUILD_DIR}/flatbuffers/include"
  "${TFLITE_BUILD_DIR}/abseil-cpp"
  "${TFLITE_BUILD_DIR}/xnnpack/include"
  "${TFLITE_BUILD_DIR}/pthreadpool-source/include")
target_link_libraries(tflite_prebuilt INTERFACE ${TFLITE_LIBS})
if(APPLE)
  target_link_libraries(tflite_prebuilt INTERFACE "-framework Foundation")
endif()

# Core library: every src/*.cc
file(GLOB SCREENAI_SRCS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
add_library(screenai_core STATIC ${SCREENAI_SRCS})
target_include_directories(screenai_core PUBLIC src "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/stb")
target_link_libraries(screenai_core PUBLIC tflite_prebuilt)
set_target_properties(screenai_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Every tools/*.cc is a standalone executable linked against the core library.
file(GLOB SCREENAI_TOOLS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tools/*.cc")
foreach(t ${SCREENAI_TOOLS})
  get_filename_component(name ${t} NAME_WE)
  add_executable(${name} ${t})
  target_link_libraries(${name} screenai_core)
endforeach()

# Drop-in libchromescreenai.so (C API) once all pipeline stages exist.
set(STAGE_HEADERS src/detector.h src/recognizer.h src/layout_analyzer.h)
set(HAVE_ALL_STAGES ON)
foreach(h ${STAGE_HEADERS})
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${h}")
    set(HAVE_ALL_STAGES OFF)
  endif()
endforeach()
if(HAVE_ALL_STAGES)
  add_library(chromescreenai SHARED capi/c_api.cc capi/ocr_engine.cc)
  target_include_directories(chromescreenai PRIVATE capi)
  target_link_libraries(chromescreenai PRIVATE screenai_core)
  # Same file name as the original so callers can load it unchanged.
  set_target_properties(chromescreenai PROPERTIES PREFIX "" OUTPUT_NAME "libchromescreenai" SUFFIX ".so")
  if(APPLE)
    target_link_options(chromescreenai PRIVATE "-Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/capi/exports.txt")
  else()
    target_link_options(chromescreenai PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/capi/exports.map")
  endif()
  add_executable(run_ocr capi/run_ocr.cc capi/ocr_engine.cc)
  target_include_directories(run_ocr PRIVATE capi)
  target_link_libraries(run_ocr screenai_core)
  add_executable(bench_ocr capi/bench_ocr.cc capi/ocr_engine.cc)
  target_include_directories(bench_ocr PRIVATE capi)
  target_link_libraries(bench_ocr screenai_core)
endif()
