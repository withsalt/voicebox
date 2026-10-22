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
