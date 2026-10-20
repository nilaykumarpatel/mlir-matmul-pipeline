add_executable(raise-matmul-to-linalg
  pass/RaiseMatMulToLinalg.cpp
  ${CMAKE_SOURCE_DIR}/tools/mlir_pass_main.cpp
)
target_link_libraries(raise-matmul-to-linalg PRIVATE ${MLIR_COMMON_LIBS})
