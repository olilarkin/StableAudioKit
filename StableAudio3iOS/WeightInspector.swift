import Foundation

struct WeightInspector {
    func inspect() -> [WeightStatus] {
        let manifest = loadManifest() ?? Self.fallbackManifest
        return manifest.files.map { file in
            let url = Bundle.main.url(
                forResource: file.resourceName,
                withExtension: file.resourceExtension,
                subdirectory: "Weights"
            )

            return WeightStatus(
                role: file.role,
                fileName: file.fileName,
                expectedBytes: file.minimumBytes,
                actualBytes: url.flatMap(Self.fileSize)
            )
        }
    }

    private func loadManifest() -> WeightManifest? {
        guard let url = Bundle.main.url(
            forResource: "manifest",
            withExtension: "json",
            subdirectory: "Weights"
        ) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(WeightManifest.self, from: data)
        } catch {
            return nil
        }
    }

    private static func fileSize(_ url: URL) -> Int64? {
        guard let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
              let fileSize = values.fileSize
        else {
            return nil
        }
        return Int64(fileSize)
    }

    private static let fallbackManifest = WeightManifest(
        model: "stable-audio-3-small-music",
        format: "safetensors",
        files: [
            WeightFile(
                role: "T5Gemma text encoder",
                fileName: "t5gemma_f16.safetensors",
                minimumBytes: 500_000_000,
                sourceFileName: "t5gemma_f16.npz"
            ),
            WeightFile(
                role: "DiT small-music",
                fileName: "dit_sm-music_f16.safetensors",
                minimumBytes: 850_000_000,
                sourceFileName: "dit_sm-music_f16.npz"
            ),
            WeightFile(
                role: "same-s decoder",
                fileName: "same_s_decoder_f32.safetensors",
                minimumBytes: 200_000_000,
                sourceFileName: "same_s_decoder_f32.npz"
            ),
        ]
    )
}

private extension WeightFile {
    var resourceName: String {
        (fileName as NSString).deletingPathExtension
    }

    var resourceExtension: String {
        (fileName as NSString).pathExtension
    }
}

