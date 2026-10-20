add_executable(tile-matmul-loops
  pass/TileMatMulLoops.cpp
  ${CMAKE_SOURCE_DIR}/tools/mlir_pass_main.cpp
)
target_link_libraries(tile-matmul-loops PRIVATE ${MLIR_COMMON_LIBS})
