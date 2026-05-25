import Foundation

public struct StableAudioWeights: Sendable {
    public let directory: URL
    public let manifest: WeightManifest?

    public init(directory: URL) throws {
        self.directory = directory
        let manifestURL = directory.appendingPathComponent("manifest.json")
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            let data = try Data(contentsOf: manifestURL)
            self.manifest = try JSONDecoder().decode(WeightManifest.self, from: data)
        } else {
            self.manifest = nil
        }
    }

    func url(for fileName: String) throws -> URL {
        let url = directory.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StableAudioWeightError.missing(fileName, directory)
        }
        return url
    }

    public func validate() throws -> [WeightStatus] {
        let expectedFiles = manifest?.files ?? Self.defaultFiles
        var statuses: [WeightStatus] = []
        for file in expectedFiles {
            let url = directory.appendingPathComponent(file.fileName)
            let size = try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64
            let status = WeightStatus(
                role: file.role,
                fileName: file.fileName,
                expectedBytes: file.minimumBytes,
                actualBytes: size
            )
            statuses.append(status)
        }
        return statuses
    }

    public func requireReady() throws {
        let missing = try validate().filter { !$0.isReady }
        if !missing.isEmpty {
            throw StableAudioWeightError.incomplete(missing)
        }
    }

    private static let defaultFiles = [
        WeightFile(role: "T5Gemma text encoder", fileName: "t5gemma_f16.safetensors", minimumBytes: 1, sourceFileName: "t5gemma_f16.npz"),
        WeightFile(role: "DiT small-music", fileName: "dit_sm-music_f16.safetensors", minimumBytes: 1, sourceFileName: "dit_sm-music_f16.npz"),
        WeightFile(role: "DiT small-sfx", fileName: "dit_sm-sfx_f16.safetensors", minimumBytes: 1, sourceFileName: "dit_sm-sfx_f16.npz"),
        WeightFile(role: "same-s decoder", fileName: "same_s_decoder_f32.safetensors", minimumBytes: 1, sourceFileName: "same_s_decoder_f32.npz"),
        WeightFile(role: "T5Gemma tokenizer", fileName: "t5gemma_tokenizer.model", minimumBytes: 1, sourceFileName: "t5gemma_f16.npz:TOKENIZER_MODEL"),
        WeightFile(role: "Conditioner small-music", fileName: "sa3_conditioner_sm-music.safetensors", minimumBytes: 1, sourceFileName: "dit_sm-music_f16.npz"),
        WeightFile(role: "Conditioner small-sfx", fileName: "sa3_conditioner_sm-sfx.safetensors", minimumBytes: 1, sourceFileName: "dit_sm-sfx_f16.npz"),
    ]
}

public enum StableAudioWeightError: LocalizedError {
    case missing(String, URL)
    case incomplete([WeightStatus])

    public var errorDescription: String? {
        switch self {
        case .missing(let fileName, let directory):
            return "Missing \(fileName) in \(directory.path)"
        case .incomplete(let statuses):
            let files = statuses.map { "\($0.fileName) (\($0.sizeSummary))" }.joined(separator: ", ")
            return "Model weights are incomplete: \(files)"
        }
    }
}
