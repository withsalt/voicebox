set(GFLAGS_NAMESPACE "gflags")

set(gflags_URL https://github.com/gflags/gflags/archive/refs/tags/v2.3.0.tar.gz)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gflags-2.3.0.tar.gz)
  set(gflags_URL ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gflags-2.3.0.tar.gz)
endif()

FetchContent_Declare(gflags
  URL      ${gflags_URL}
  URL_HASH SHA256=f619a51371f41c0ad6837b2a98af9d4643b3371015d873887f7e8d3237320b2f
)
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)
