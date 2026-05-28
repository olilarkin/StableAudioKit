#!/usr/bin/env bash
# Builds StableAudioKit.xcframework.
#
# Default ("externalized") mode: only StableAudioKit's own object is merged into
# the static framework binary; MLX and swift-sentencepiece symbols are left
# UNDEFINED. A SwiftPM wrapper (Package.swift + helper target) is emitted next to
# the xcframework so consumers link a single shared mlx-swift at their final link
# step — avoiding a second copy of MLX in a process that also uses mlx-swift.
#
# Self-contained mode (STABLEAUDIO_SELFCONTAINED=1): the legacy behaviour — all
# transitive dependencies are statically bundled into one self-contained binary
# with no external dependencies. Use this only for hosts that never link MLX
# themselves.
#
# Usage:
#   ./Scripts/build-xcframework.sh [output_dir] [platform...]
#
# Platforms: macos  ios  ios-simulator  xros  xros-simulator  (default: all)
# Examples:
#   ./Scripts/build-xcframework.sh                         # all platforms
#   ./Scripts/build-xcframework.sh build/out macos         # macOS only
#   ./Scripts/build-xcframework.sh build/out macos ios ios-simulator
#   STABLEAUDIO_SELFCONTAINED=1 ./Scripts/build-xcframework.sh  # legacy bundle
#
# Output: <output_dir>/StableAudioKit.xcframework (default: build/xcframework)
#   plus, in externalized mode, <output_dir>/Package.swift and
#   <output_dir>/Sources/ for the SwiftPM wrapper.
#
# Requirements: macOS with Xcode 15+ command line tools. Package.swift declares
# the public mlx-swift fork by URL. By default this script rewrites it to the
# local ../mlx-swift checkout when present (faster iteration); set
# MLXSWIFT_USE_REMOTE=1 to force the remote fork URL instead.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "error: This script must run on macOS." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
RESOURCES_DIR="$SCRIPT_DIR/xcframework-resources"
COMPILE_METALLIB="$SCRIPT_DIR/compile-mlx-metallib.sh"

OUTPUT_DIR="${1:-"$PACKAGE_DIR/build/xcframework"}"
shift 2>/dev/null || true   # consume output_dir; remaining args are platform filters
PLATFORM_FILTER=("$@")      # empty = build all
WORK_DIR="$PACKAGE_DIR/build/xcframework-work"
SLICES_DIR="$WORK_DIR/slices"
SOURCE_PACKAGES_DIR="$WORK_DIR/SourcePackages"
XCFRAMEWORK_PATH="$OUTPUT_DIR/StableAudioKit.xcframework"

# Mlx-swift fork used in place of the ../mlx-swift path dependency.
MLXSWIFT_URL="https://github.com/olilarkin/mlx-swift"
MLXSWIFT_BRANCH="${MLXSWIFT_BRANCH:-main}"
MLXSWIFT_LOCAL_DIR="$PACKAGE_DIR/../mlx-swift"
MLXSWIFT_USE_REMOTE="${MLXSWIFT_USE_REMOTE:-}"

# swift-sentencepiece dependency, externalized into the wrapper alongside MLX.
SENTENCEPIECE_URL="https://github.com/jkrukowski/swift-sentencepiece"
SENTENCEPIECE_REVISION="b968826b1d3b76e37359abdbe2f4c0daaa96a50a"

# When set to 1, statically bundle every dependency into the framework binary
# (legacy, self-contained). Default: externalize MLX/sentencepiece + emit wrapper.
STABLEAUDIO_SELFCONTAINED="${STABLEAUDIO_SELFCONTAINED:-}"

# The package's Sources/StableAudioKit module name as exposed to Swift consumers.
MODULE_NAME="StableAudioKit"
FRAMEWORK_NAME="StableAudioKit"

# (slice_id, scheme_destination, sdk, min_os, swift_triple_prefix, metal_target_prefix)
# We build StableAudioKit (the SPM library product) per slice via xcodebuild.
SLICES=(
    "macos|generic/platform=macOS|macosx|14.0|arm64-apple-macosx|air64-apple-macosx"
    "ios|generic/platform=iOS|iphoneos|17.0|arm64-apple-ios|air64-apple-ios"
    "ios-simulator|generic/platform=iOS Simulator|iphonesimulator|17.0|arm64-apple-ios-simulator|air64-apple-ios"
    "xros|generic/platform=visionOS|xros|1.0|arm64-apple-xros|air64-apple-xros"
    "xros-simulator|generic/platform=visionOS Simulator|xrsimulator|1.0|arm64-apple-xros-simulator|air64-apple-xros"
)

if [[ ! -x "$COMPILE_METALLIB" ]]; then
    echo "error: $COMPILE_METALLIB not found or not executable." >&2
    exit 1
fi

mkdir -p "$WORK_DIR" "$SLICES_DIR" "$SOURCE_PACKAGES_DIR" "$OUTPUT_DIR"

# --- Prep: optionally rewrite Package.swift to use the public mlx-swift fork ---

PACKAGE_FILE="$PACKAGE_DIR/Package.swift"
PACKAGE_BACKUP="$WORK_DIR/Package.swift.bak"
RESOLVED_FILE="$PACKAGE_DIR/Package.resolved"
RESOLVED_BACKUP="$WORK_DIR/Package.resolved.bak"

restore_package() {
    if [[ -f "$PACKAGE_BACKUP" ]]; then
        mv "$PACKAGE_BACKUP" "$PACKAGE_FILE"
    fi
    if [[ -f "$RESOLVED_BACKUP" ]]; then
        mv "$RESOLVED_BACKUP" "$RESOLVED_FILE"
    fi
}
trap restore_package EXIT

if [[ -d "$MLXSWIFT_LOCAL_DIR" && "$MLXSWIFT_USE_REMOTE" != "1" ]]; then
    echo "Using local mlx-swift checkout: $MLXSWIFT_LOCAL_DIR"
    REWRITE_MODE="local"
else
    echo "Using remote mlx-swift package: $MLXSWIFT_URL ($MLXSWIFT_BRANCH)"
    REWRITE_MODE="remote"
fi

# Package.swift declares the remote fork by URL. Back it up and rewrite the
# mlx-swift dependency to match the selected mode; the EXIT trap restores it.
cp "$PACKAGE_FILE" "$PACKAGE_BACKUP"
if [[ -f "$RESOLVED_FILE" ]]; then
    cp "$RESOLVED_FILE" "$RESOLVED_BACKUP"
fi

python3 - "$PACKAGE_FILE" "$REWRITE_MODE" "$MLXSWIFT_URL" "$MLXSWIFT_BRANCH" <<'PY'
import re, sys, pathlib
path, mode, url, branch = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
src = pathlib.Path(path).read_text()
# Match either the local path form or the remote url+branch form.
dep_re = (
    r'\.package\(\s*(?:path:\s*"\.\./mlx-swift"'
    r'|url:\s*"[^"]*mlx-swift"\s*,\s*branch:\s*"[^"]*")\s*\)'
)
repl = '.package(path: "../mlx-swift")' if mode == "local" \
    else f'.package(url: "{url}", branch: "{branch}")'
new = re.sub(dep_re, repl, src)
if new == src:
    print("warn: no mlx-swift dependency found in Package.swift to rewrite",
          file=sys.stderr)
pathlib.Path(path).write_text(new)
PY

# --- Build each slice ---

build_slice() {
    local id="$1" dest="$2" sdk="$3" min_os="$4" triple_prefix="$5" metal_target_prefix="$6"

    echo ""
    echo "==> Building slice: $id ($dest)"

    local derived="$WORK_DIR/derived-$id"
    rm -rf "$derived"

    xcodebuild build \
        -scheme "$MODULE_NAME" \
        -destination "$dest" \
        -derivedDataPath "$derived" \
        -clonedSourcePackagesDirPath "$SOURCE_PACKAGES_DIR" \
        -configuration Release \
        -skipPackagePluginValidation \
        BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
        SKIP_INSTALL=NO \
        SWIFT_OPTIMIZATION_LEVEL="-O" \
        OTHER_SWIFT_FLAGS="-no-verify-emitted-module-interface" \
        ONLY_ACTIVE_ARCH=NO \
        ARCHS=arm64

    # Locate the build products directory for this slice.
    local products
    if [[ -d "$derived/Build/Products/Release-$sdk" ]]; then
        products="$derived/Build/Products/Release-$sdk"
    elif [[ -d "$derived/Build/Products/Release" ]]; then
        products="$derived/Build/Products/Release"
    else
        products=""
    fi
    if [[ -z "$products" ]]; then
        echo "error: Could not find Release products directory for slice $id." >&2
        exit 1
    fi
    echo "    products: $products"

    local merge_input
    merge_input="$WORK_DIR/merge-$id"
    rm -rf "$merge_input"
    mkdir -p "$merge_input"

    local link_inputs=()
    if [[ "$STABLEAUDIO_SELFCONTAINED" == "1" ]]; then
        # Self-contained: merge every linked target object/static archive the
        # build produced. Recent Xcode SwiftPM builds emit Swift/C/C++ targets as
        # <Target>.o files and C archives such as libSentencepiece.a.
        while IFS= read -r -d '' input; do
            link_inputs+=("$input")
        done < <(find "$products" -maxdepth 1 \( -name "*.o" -o -name "*.a" \) -print0)

        # Some SPM builds bury archives under PackageFrameworks; pick those up too.
        while IFS= read -r -d '' lib; do
            link_inputs+=("$lib")
        done < <(find "$derived/Build/Intermediates.noindex" -name "*.a" -print0 2>/dev/null)
    else
        # Externalized: merge ONLY StableAudioKit's own object so MLX and
        # sentencepiece symbols stay undefined, to be resolved by the consumer's
        # shared mlx-swift at their final link step (see the emitted wrapper).
        while IFS= read -r -d '' input; do
            link_inputs+=("$input")
        done < <(find "$products" -maxdepth 1 -name "$MODULE_NAME.o" -print0)
        if [[ ${#link_inputs[@]} -eq 0 ]]; then
            # Fallback: some toolchains place the target object under Intermediates.
            # Take only the first match — multiple would be per-arch duplicates and
            # merging them would produce duplicate-symbol errors.
            local fallback_obj
            fallback_obj="$(find "$derived/Build/Intermediates.noindex" \
                -name "$MODULE_NAME.o" -print -quit 2>/dev/null)"
            [[ -n "$fallback_obj" ]] && link_inputs+=("$fallback_obj")
        fi
    fi

    if [[ ${#link_inputs[@]} -eq 0 ]]; then
        echo "error: No link inputs found for slice $id." >&2
        if [[ "$STABLEAUDIO_SELFCONTAINED" != "1" ]]; then
            echo "       Looked for $MODULE_NAME.o under:" >&2
            echo "         $products" >&2
            echo "         $derived/Build/Intermediates.noindex" >&2
        fi
        exit 1
    fi

    echo "    merging ${#link_inputs[@]} link input(s)"
    local merged_binary="$merge_input/$FRAMEWORK_NAME"
    xcrun libtool -static -no_warning_for_no_symbols -o "$merged_binary" "${link_inputs[@]}"

    # Assemble the .framework bundle.
    local slice_root="$SLICES_DIR/$id"
    local fw="$slice_root/$FRAMEWORK_NAME.framework"
    rm -rf "$slice_root"
    mkdir -p "$fw/Headers" "$fw/Modules/$MODULE_NAME.swiftmodule"

    cp "$merged_binary" "$fw/$FRAMEWORK_NAME"
    cp "$RESOURCES_DIR/StableAudioKit.h" "$fw/Headers/StableAudioKit.h"
    cp "$RESOURCES_DIR/module.modulemap" "$fw/Modules/module.modulemap"

    # Find the StableAudioKit swiftmodule directory produced for this slice and
    # copy its arch-specific files into the framework. xcodebuild typically
    # writes it under Build/Intermediates.noindex/.../$MODULE_NAME.swiftmodule/.
    local found_modules=()
    while IFS= read -r -d '' f; do
        found_modules+=("$f")
    done < <(find "$derived" \
        -type d -name "$MODULE_NAME.swiftmodule" -print0 2>/dev/null)

    if [[ ${#found_modules[@]} -eq 0 ]]; then
        echo "error: No $MODULE_NAME.swiftmodule directory found for slice $id." >&2
        exit 1
    fi

    # Prefer a directory that contains a .swiftinterface (BUILD_LIBRARY_FOR_DISTRIBUTION output).
    local best=""
    for d in "${found_modules[@]}"; do
        if compgen -G "$d/*.swiftinterface" > /dev/null; then
            best="$d"
            break
        fi
    done
    if [[ -z "$best" ]]; then
        best="${found_modules[0]}"
    fi
    echo "    swiftmodule: $best"

    # Copy every per-arch artifact (swiftmodule/swiftdoc/swiftinterface/abi.json).
    find "$best" -maxdepth 1 -type f \
        \( -name "*.swiftmodule" -o -name "*.swiftdoc" \
           -o -name "*.swiftinterface" -o -name "*.private.swiftinterface" \
           -o -name "*.abi.json" -o -name "*.swiftsourceinfo" \) \
        -exec cp {} "$fw/Modules/$MODULE_NAME.swiftmodule/" \;

    # Write the per-slice Info.plist with the right CFBundleSupportedPlatforms /
    # MinimumOSVersion. We use a single hand-rolled file; macOS frameworks
    # normally use Versions/A but the flat layout works for create-xcframework
    # consumption on every Apple platform.
    local platform_value
    case "$sdk" in
        macosx)             platform_value="MacOSX" ;;
        iphoneos)           platform_value="iPhoneOS" ;;
        iphonesimulator)    platform_value="iPhoneSimulator" ;;
        xros)               platform_value="XROS" ;;
        xrsimulator)        platform_value="XRSimulator" ;;
        *) echo "error: unknown sdk $sdk" >&2; exit 1 ;;
    esac
    sed -e "s/__SUPPORTED_PLATFORM__/$platform_value/" \
        -e "s/__MIN_OS_VERSION__/$min_os/" \
        "$RESOURCES_DIR/Info.plist.tmpl" > "$fw/Info.plist"

    # Compile mlx.metallib for this slice and embed it in the framework.
    # The compile script targets macOS by default — invoke it with the
    # right Metal target so the resulting metallib loads at runtime.
    local metal_target
    case "$sdk" in
        macosx)             metal_target="${metal_target_prefix}${min_os}" ;;
        iphoneos)           metal_target="${metal_target_prefix}${min_os}" ;;
        iphonesimulator)    metal_target="${metal_target_prefix}${min_os}-simulator" ;;
        xros)               metal_target="${metal_target_prefix}${min_os}" ;;
        xrsimulator)        metal_target="${metal_target_prefix}${min_os}-simulator" ;;
    esac

    local mlx_checkout="$SOURCE_PACKAGES_DIR/checkouts/mlx-swift"
    METAL_TARGET="$metal_target" MLX_SWIFT_DIR="$mlx_checkout" \
        "$COMPILE_METALLIB" "$fw" >/dev/null

    # In self-contained mode, copy any *.bundle resources produced by SPM
    # dependencies (e.g. sentencepiece). In externalized mode the consumer's
    # SwiftPM build provides those bundles, so we ship only our own metallib.
    if [[ "$STABLEAUDIO_SELFCONTAINED" == "1" ]]; then
        while IFS= read -r -d '' bundle; do
            cp -R "$bundle" "$fw/"
        done < <(find "$products" -maxdepth 2 -type d -name "*.bundle" -print0)
    fi

    echo "    slice ready: $fw"
}

slice_enabled() {
    local id="$1"
    [[ ${#PLATFORM_FILTER[@]} -eq 0 ]] && return 0
    for p in "${PLATFORM_FILTER[@]}"; do
        [[ "$p" == "$id" ]] && return 0
    done
    return 1
}

# Validate any platform filters provided by the caller.
VALID_IDS=()
for entry in "${SLICES[@]}"; do
    IFS='|' read -r id _rest <<<"$entry"
    VALID_IDS+=("$id")
done
# Guard the expansion: under `set -u`, macOS's bash 3.2 treats "${arr[@]}"
# on an empty array as an unbound variable, which aborts an all-platforms run.
if [[ ${#PLATFORM_FILTER[@]} -gt 0 ]]; then
    for p in "${PLATFORM_FILTER[@]}"; do
        found=0
        for v in "${VALID_IDS[@]}"; do [[ "$p" == "$v" ]] && found=1 && break; done
        if [[ $found -eq 0 ]]; then
            echo "error: unknown platform '$p'. Valid values: ${VALID_IDS[*]}" >&2
            exit 1
        fi
    done
fi

for entry in "${SLICES[@]}"; do
    IFS='|' read -r id dest sdk min_os triple_prefix metal_target_prefix <<<"$entry"
    slice_enabled "$id" || continue
    build_slice "$id" "$dest" "$sdk" "$min_os" "$triple_prefix" "$metal_target_prefix"
done

# --- Combine slices into a single XCFramework ---

echo ""
echo "==> Creating XCFramework"
rm -rf "$XCFRAMEWORK_PATH"

CREATE_ARGS=()
for entry in "${SLICES[@]}"; do
    IFS='|' read -r id _rest <<<"$entry"
    slice_enabled "$id" || continue
    CREATE_ARGS+=(-framework "$SLICES_DIR/$id/$FRAMEWORK_NAME.framework")
done

xcodebuild -create-xcframework \
    "${CREATE_ARGS[@]}" \
    -output "$XCFRAMEWORK_PATH"

echo ""
echo "Done: $XCFRAMEWORK_PATH"

# --- Emit the SwiftPM wrapper (externalized mode only) ---
#
# The xcframework's binary deliberately leaves MLX / sentencepiece symbols
# undefined. A binaryTarget cannot declare package dependencies, so we ship a
# thin wrapper package: the public library product bundles the binaryTarget plus
# a helper target whose only job is to drag the shared mlx-swift / sentencepiece
# into the consumer's link graph. SwiftPM then resolves a single MLX across the
# whole dependency graph and the undefined symbols are satisfied at the final
# link step — one copy of MLX in the process.

if [[ "$STABLEAUDIO_SELFCONTAINED" != "1" ]]; then
    echo ""
    echo "==> Emitting SwiftPM wrapper"

    WRAPPER_LINK_DIR="$OUTPUT_DIR/Sources/StableAudioKitLink"
    mkdir -p "$WRAPPER_LINK_DIR"

    cat > "$OUTPUT_DIR/Package.swift" <<EOF
// swift-tools-version: 5.9
// Generated by Scripts/build-xcframework.sh — wrapper for the binary
// StableAudioKit.xcframework. Consumers add this package; SwiftPM links a single
// shared mlx-swift so MLX is not duplicated in the process.

import PackageDescription

let package = Package(
    name: "StableAudioKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "StableAudioKit", targets: ["StableAudioKitBinary", "StableAudioKitLink"]),
    ],
    dependencies: [
        .package(url: "$MLXSWIFT_URL", branch: "$MLXSWIFT_BRANCH"),
        .package(url: "$SENTENCEPIECE_URL", revision: "$SENTENCEPIECE_REVISION"),
    ],
    targets: [
        .binaryTarget(name: "StableAudioKitBinary", path: "StableAudioKit.xcframework"),
        // Source-less linker shim: pulls the shared dependencies into the link so
        // the binary framework's undefined MLX / sentencepiece symbols resolve.
        .target(
            name: "StableAudioKitLink",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "SentencepieceTokenizer", package: "swift-sentencepiece"),
            ]
        ),
    ]
)
EOF

    cat > "$WRAPPER_LINK_DIR/Link.swift" <<'EOF'
// Intentionally minimal. This target exists only so the wrapper's library
// product links mlx-swift and swift-sentencepiece, resolving the undefined
// symbols in StableAudioKit.xcframework. Consumers `import StableAudioKit`
// (the binary framework module), not this module.
EOF

    echo "    wrapper: $OUTPUT_DIR/Package.swift"
fi
