cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wetext_processor_c VERSION 1.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR ${CMAKE_BINARY_DIR})
set(FETCHCONTENT_BASE_DIR ${fc_base})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Build all dependencies (openfst/glog/gflags) as static libraries so the only
# artifact is a single self-contained wetext_processor_c shared library:
# - glog would otherwise cache BUILD_SHARED_LIBS=ON, turning openfst into a DLL
#   whose FST_FLAGS_* data symbols are not exported on Windows (LNK2019);
# - a dynamic libfst.so would point into the build tree and break deployment.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
if(MSVC)
  # For windows, please use unicode(3 bytes per chinese char) instead of gbk(2 bytes per chinese char).
  # https://github.com/wenet-e2e/wenet/issues/882#issuecomment-1101246299
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
  # Keep windows.h from defining min/max macros, which break
  # std::numeric_limits<>::max() inside OpenFst headers.
  add_compile_definitions(NOMINMAX)
endif()

include(openfst)
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(utils)
add_subdirectory(processor)
