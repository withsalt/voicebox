set(glog_URL https://github.com/google/glog/archive/refs/tags/v0.7.1.zip)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glog-0.7.1.zip)
  set(glog_URL ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glog-0.7.1.zip)
endif()

FetchContent_Declare(glog
  URL      ${glog_URL}
  URL_HASH SHA256=c17d85c03ad9630006ef32c7be7c65656aba2e7e2fbfc82226b7e680c771fc88
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})
