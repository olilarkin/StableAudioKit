#!/usr/bin/env bash
# Compile MLX Metal shaders into mlx.metallib.
#
# Usage:
#   ./Scripts/compile-mlx-metallib.sh [output_dir]
#
# output_dir defaults to the debug build directory so the CLI finds the
# metallib automatically via the colocated search path.
#
# Run this once after `swift build --product StableAudioCLI`, and again
# whenever the mlx-swift dependency changes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"

# Allow override via argument; default to debug build output
OUTPUT_DIR="${1:-"$PACKAGE_DIR/.build/arm64-apple-macosx/debug"}"
OUTPUT_LIB="$OUTPUT_DIR/mlx.metallib"

# Target triple for AIR compilation. Override via METAL_TARGET to target
# iOS, visionOS, simulators, etc. The default targets macOS 14 arm64.
METAL_TARGET="${METAL_TARGET:-air64-apple-macosx14.0}"

# Search locations for the mlx-swift checkout, in order. The caller can prepend
# an extra path via MLX_SWIFT_DIR — useful when xcodebuild has resolved the
# package into its DerivedData SourcePackages directory.
SEARCH_DIRS=()
if [ -n "${MLX_SWIFT_DIR:-}" ]; then
    SEARCH_DIRS+=("$MLX_SWIFT_DIR")
fi
SEARCH_DIRS+=(
    "$PACKAGE_DIR/../mlx-swift"
    "$PACKAGE_DIR/.build/checkouts/mlx-swift"
)
# Plus any SourcePackages checkouts under build/xcframework-work derived dirs.
while IFS= read -r -d '' found; do
    SEARCH_DIRS+=("$found")
done < <(find "$PACKAGE_DIR/build" -type d -name "mlx-swift" -path "*/SourcePackages/checkouts/*" -print0 2>/dev/null)

METAL_BASE=""
for candidate in "${SEARCH_DIRS[@]}"; do
    if [ -d "$candidate/Source/Cmlx/mlx-generated/metal" ]; then
        METAL_BASE="$candidate/Source/Cmlx"
        break
    fi
done

if [ -z "$METAL_BASE" ]; then
    echo "error: Cannot find mlx-swift Metal sources. Tried:" >&2
    for d in "${SEARCH_DIRS[@]}"; do
        echo "  $d" >&2
    done
    exit 1
fi

# Only compile the pre-generated Metal sources — the files in
# mlx/mlx/backend/metal/kernels/ are template sources that require
# pre-processing and are not directly compilable.
GENERATED_DIR="$METAL_BASE/mlx-generated/metal"

IFS=$'\n' read -r -d '' -a METAL_FILES < <(
    find "$GENERATED_DIR" -name "*.metal" 2>/dev/null
    printf '\0'
) || true

if [ ${#METAL_FILES[@]} -eq 0 ]; then
    echo "error: No .metal files found in $GENERATED_DIR" >&2
    exit 1
fi

echo "Found ${#METAL_FILES[@]} Metal source files."

TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT

# Include paths needed by the Metal shaders
INCLUDE_ARGS=(
    -I "$GENERATED_DIR"
)

# Compile each .metal → .air
AIR_FILES=()
FAILED=0
for metal_file in "${METAL_FILES[@]}"; do
    base="$(basename "$metal_file" .metal)"
    air_file="$TMPDIR_WORK/$base.air"
    # Use a unique name if there are duplicate basenames across directories
    if [ -f "$air_file" ]; then
        air_file="$TMPDIR_WORK/$(echo "$metal_file" | md5 | cut -c1-8)_$base.air"
    fi
    echo "  Compiling $base.metal ..."
    if ! xcrun metal \
        -arch air64 \
        -target "$METAL_TARGET" \
        "${INCLUDE_ARGS[@]}" \
        -c "$metal_file" \
        -o "$air_file" 2>&1; then
        echo "warning: Failed to compile $metal_file (skipping)" >&2
        FAILED=$((FAILED + 1))
        continue
    fi
    AIR_FILES+=("$air_file")
done

if [ ${#AIR_FILES[@]} -eq 0 ]; then
    echo "error: All Metal compilations failed." >&2
    exit 1
fi

if [ $FAILED -gt 0 ]; then
    echo "warning: $FAILED shader(s) failed to compile and were skipped."
fi

# Link .air files → .metallib
echo "Linking ${#AIR_FILES[@]} shader(s) → $OUTPUT_LIB ..."
mkdir -p "$OUTPUT_DIR"
xcrun metallib -o "$OUTPUT_LIB" "${AIR_FILES[@]}"

echo "Done: $OUTPUT_LIB"

# Also copy into the SwiftPM test bundle so `swift test` can find the metallib.
TEST_BUNDLE_DIR="$PACKAGE_DIR/.build/arm64-apple-macosx/debug/StableAudioKitPackageTests.xctest/Contents/MacOS"
if [ -d "$TEST_BUNDLE_DIR" ]; then
    cp "$OUTPUT_LIB" "$TEST_BUNDLE_DIR/"
    echo "Copied to test bundle: $TEST_BUNDLE_DIR/mlx.metallib"
fi
