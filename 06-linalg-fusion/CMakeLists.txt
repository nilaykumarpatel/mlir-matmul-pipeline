add_executable(fuse-matmul-producers
  pass/FuseMatMulProducers.cpp
  ${CMAKE_SOURCE_DIR}/tools/mlir_pass_main.cpp
)
target_link_libraries(fuse-matmul-producers PRIVATE ${MLIR_COMMON_LIBS})
