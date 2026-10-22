#!/usr/bin/env bash
# Builds libwetext_processor_c.so (Release) and copies it into the managed
# project's runtimes/linux-{x64,arm64}/native/ folder.
#
# Usage: ./build-linux.sh [x64|arm64] [config]
#   x64    - native build (default; requires an x86_64 host)
#   arm64  - native build on aarch64 hosts, cross compile elsewhere
#            (cross requires g++-aarch64-linux-gnu, see below)
#
# OpenFst/glog/gflags are built from third_party/ local caches (or fetched from
# GitHub by CMake FetchContent on first use) into a single self-contained .so.
# There is no separate prebuilt drop to download.
#
# Requires: cmake >= 3.14, g++, make (or ninja). Designed to run under WSL or
# any Linux. arm64 cross builds additionally need:
#   sudo apt install -y g++-aarch64-linux-gnu

set -euo pipefail

ARCH="${1:-x64}"
ARCH="${ARCH#linux-}"  # accept both "x64" and "linux-x64"
case "$ARCH" in
  x64|arm64) ;;
  *) echo "Unknown arch: $ARCH (expected x64 or arm64)" >&2; exit 1 ;;
esac
RID="linux-$ARCH"

NATIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${2:-Release}"
BUILD_DIR="$NATIVE_DIR/build/$RID"
OUT_DIR="$NATIVE_DIR/../../src/SDKs/WeTextProcessing.NET/runtimes/$RID/native"

# ---- Toolchain checks -------------------------------------------------------
command -v cmake >/dev/null || { echo "cmake not found. apt install cmake" >&2; exit 1; }

HOST_ARCH="$(uname -m)"
CMAKE_ARGS=(-S "$NATIVE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CONFIG")

if [[ "$RID" == "linux-arm64" && "$HOST_ARCH" != "aarch64" ]]; then
  # Cross compile with the aarch64 gcc toolchain (see toolchain file).
  command -v aarch64-linux-gnu-g++ >/dev/null || {
    echo "Cross build needs the aarch64 gcc toolchain:" >&2
    echo "  sudo apt install -y g++-aarch64-linux-gnu" >&2
    exit 1
  }
  CMAKE_ARGS+=(-DCMAKE_TOOLCHAIN_FILE="$NATIVE_DIR/cmake/linux-arm64.toolchain.cmake")
elif [[ "$RID" == "linux-x64" && "$HOST_ARCH" != "x86_64" ]]; then
  echo "linux-x64 build requires an x86_64 host (got $HOST_ARCH)" >&2
  exit 1
else
  command -v g++ >/dev/null || { echo "g++ not found. apt install g++" >&2; exit 1; }
fi

if command -v ninja >/dev/null; then
  CMAKE_ARGS+=(-G Ninja)
fi

# ---- Configure, build, deploy ----------------------------------------------
echo "==> Configuring ($RID, $CONFIG)"
cmake "${CMAKE_ARGS[@]}"

echo "==> Building"
cmake --build "$BUILD_DIR" --target wetext_processor_c --parallel "$(nproc)"

# Both the make and ninja generators drop the .so under processor/.
SO_PATH="$BUILD_DIR/processor/libwetext_processor_c.so"
[[ -f "$SO_PATH" ]] || { echo "Build artifact not found: $SO_PATH" >&2; exit 1; }

mkdir -p "$OUT_DIR"
cp "$SO_PATH" "$OUT_DIR/"
echo "==> Copied libwetext_processor_c.so -> $OUT_DIR"

echo "==> Exported wetext symbols:"
nm -D --defined-only "$OUT_DIR/libwetext_processor_c.so" 2>/dev/null | grep -i wetext || true
