// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "StableAudioKit",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "StableAudioKit", targets: ["StableAudioKit"]),
        .executable(name: "StableAudioCLI", targets: ["StableAudioCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/olilarkin/mlx-swift", branch: "main"),
        .package(url: "https://github.com/jkrukowski/swift-sentencepiece", revision: "b968826b1d3b76e37359abdbe2f4c0daaa96a50a"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "StableAudioKit",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "SentencepieceTokenizer", package: "swift-sentencepiece"),
            ]
        ),
        .executableTarget(
            name: "StableAudioCLI",
            dependencies: [
                "StableAudioKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "StableAudioKitTests",
            dependencies: [
                "StableAudioKit",
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
    ]
)
