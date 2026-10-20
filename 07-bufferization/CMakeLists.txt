add_executable(annotate-address-spaces
  pass/AddressSpaceAnnotate.cpp
  ${CMAKE_SOURCE_DIR}/tools/mlir_pass_main.cpp
)
target_link_libraries(annotate-address-spaces PRIVATE ${MLIR_COMMON_LIBS})
