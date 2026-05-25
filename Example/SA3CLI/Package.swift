// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "SA3CLI",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(path: "../.."),
    ],
    targets: [
        .executableTarget(
            name: "SA3CLI",
            dependencies: [
                .product(name: "StableAudioKit", package: "StableAudioKit2"),
            ]
        ),
    ]
)
