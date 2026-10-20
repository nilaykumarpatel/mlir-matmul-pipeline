cmake_minimum_required(VERSION 3.20)
project(mlir-matmul-pipeline LANGUAGES C CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "In-source builds are not allowed. Use build.sh:\n"
    "  MLIR_PREFIX=/path/to/llvm-install ./build.sh")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# MLIR shared libs are built with -fno-rtti; match that to avoid typeinfo link errors.
add_compile_options(-fno-rtti)

# --------------------------------------------------------------------------
# Find MLIR/LLVM.
# Do not run cmake directly — use build.sh:
#   MLIR_PREFIX=/path/to/llvm-install ./build.sh
# --------------------------------------------------------------------------
if(NOT DEFINED MLIR_DIR)
  message(FATAL_ERROR
    "MLIR_DIR is not set. Do not invoke cmake directly.\n"
    "Use build.sh instead:\n"
    "  MLIR_PREFIX=/path/to/llvm-install ./build.sh")
endif()

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIR in: ${MLIR_DIR}")
message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# --------------------------------------------------------------------------
# Common MLIR libs used across passes
# --------------------------------------------------------------------------
get_property(mlir_libs GLOBAL PROPERTY MLIR_ALL_LIBS)
set(MLIR_COMMON_LIBS
  MLIRMlirOptMain
  MLIRPass
  MLIRTransforms
  MLIRRewrite
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRAffineDialect
  MLIRVectorDialect
  MLIRTensorDialect
  MLIRSupport
  MLIRIRDL
)

add_subdirectory(01-detect-matmul-loops)
add_subdirectory(02-linalg-generic)
add_subdirectory(04-loop-tiling)
add_subdirectory(05-raise-tiled-scf-to-linalg)
add_subdirectory(07-bufferization)
