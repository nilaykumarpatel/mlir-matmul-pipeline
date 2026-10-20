add_executable(detect-matmul-loops
  pass/DetectMatMulLoops.cpp
  ${CMAKE_SOURCE_DIR}/tools/mlir_pass_main.cpp
)
target_link_libraries(detect-matmul-loops PRIVATE ${MLIR_COMMON_LIBS})
