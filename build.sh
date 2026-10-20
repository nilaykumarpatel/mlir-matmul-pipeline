#!/usr/bin/env bash
# Build all custom MLIR passes against a prebuilt MLIR installation.
#
# Usage:
#   MLIR_PREFIX=/path/to/llvm-install ./build.sh        # build
#   ./build.sh clean                                     # remove build dir
#
# MLIR_PREFIX must point to a directory that contains:
#   bin/clang++
#   lib/cmake/mlir/MLIRConfig.cmake
#   lib/cmake/llvm/LLVMConfig.cmake
#
# The build is configured in a 'build/' subdirectory on first run.
# Subsequent runs just call make.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$REPO/build"

if [ "${1:-}" = "clean" ]; then
  rm -rf "$BUILD_DIR"
  echo "Removed $BUILD_DIR"
  exit 0
fi

if [ -z "${MLIR_PREFIX:-}" ]; then
  echo "Error: MLIR_PREFIX is not set."
  echo "Usage: MLIR_PREFIX=/path/to/llvm-install ./build.sh"
  exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

if [ ! -f CMakeCache.txt ]; then
  cmake "$REPO" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER="$MLIR_PREFIX/bin/clang++" \
    -DCMAKE_C_COMPILER="$MLIR_PREFIX/bin/clang" \
    -DMLIR_DIR="$MLIR_PREFIX/lib/cmake/mlir" \
    -DLLVM_DIR="$MLIR_PREFIX/lib/cmake/llvm"
fi

make -j"$(nproc)"

echo ""
echo "Build complete. Binaries in $BUILD_DIR:"
find "$BUILD_DIR" -maxdepth 3 -type f \( \
  -name "detect-matmul-loops" -o \
  -name "raise-matmul-to-linalg" -o \
  -name "tile-matmul-loops" -o \
  -name "raise-tiled-scf-to-linalg" -o \
  -name "annotate-address-spaces" \
\)

echo ""
echo "Before running the tools, set:"
echo "  export LD_LIBRARY_PATH=\"\$MLIR_PREFIX/lib:\$LD_LIBRARY_PATH\""
