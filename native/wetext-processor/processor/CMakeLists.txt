add_library(wetext_processor STATIC
  wetext_processor.cc
  wetext_token_parser.cc
)
target_link_libraries(wetext_processor PUBLIC fst wetext_utils)
if(UNIX)
  target_link_libraries(wetext_processor PUBLIC dl)
endif()

add_library(wetext_processor_c SHARED
  wetext_processor_c_api.cc
)
target_link_libraries(wetext_processor_c PRIVATE wetext_processor)
if(UNIX)
  # Self-contained .so: no libstdc++/libgcc runtime dependency on the target.
  target_link_options(wetext_processor_c PRIVATE -static-libstdc++ -static-libgcc)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Keep the statically linked C++ runtime private. Exporting its locale/facet
  # state can mix it with the host's libstdc++ and crash during error logging.
  set(WETEXT_EXPORTS "${CMAKE_CURRENT_SOURCE_DIR}/wetext_processor_c.map")
  target_link_options(wetext_processor_c PRIVATE "LINKER:--version-script=${WETEXT_EXPORTS}")
  set_property(TARGET wetext_processor_c APPEND PROPERTY LINK_DEPENDS "${WETEXT_EXPORTS}")
endif()
