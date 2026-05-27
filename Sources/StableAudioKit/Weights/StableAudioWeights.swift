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

    /// Verifies that the always-required files (T5Gemma + tokenizer) are
    /// present. Per-variant DiT, conditioner, and decoder files are checked
    /// lazily on first use; see `requireReady(for:)`.
    public func requireReady() throws {
        try requireFiles(Set(Self.alwaysRequiredFileNames))
    }

    /// Verifies that the files needed to run the given model variant are
    /// present (T5Gemma + tokenizer + the variant's DiT/conditioner +
    /// the matching autoencoder decoder).
    public func requireReady(for kind: StableAudioModelKind) throws {
        var required = Set(Self.alwaysRequiredFileNames)
        for file in Self.variantFiles(for: kind) {
            required.insert(file.fileName)
        }
        try requireFiles(required)
    }

    private func requireFiles(_ required: Set<String>) throws {
        let statuses = (try? validate()) ?? []
        var byName = [String: WeightStatus]()
        for status in statuses {
            byName[status.fileName] = status
        }
        var missing: [WeightStatus] = []
        for name in required {
            if let status = byName[name] {
                if !status.isReady { missing.append(status) }
            } else {
                let url = directory.appendingPathComponent(name)
                let size = try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64
                let placeholder = WeightStatus(role: name, fileName: name, expectedBytes: 1, actualBytes: size)
                if !placeholder.isReady { missing.append(placeholder) }
            }
        }
        if !missing.isEmpty {
            throw StableAudioWeightError.incomplete(missing)
        }
    }

    static func variantFiles(for kind: StableAudioModelKind) -> [WeightFile] {
        switch kind {
        case .smallMusic:
            return [
                WeightFile(role: "DiT small-music", fileName: "dit_sm-music_f16.safetensors", minimumBytes: 1, sourceFileName: "dit_sm-music_f16.npz"),
                WeightFile(role: "same-s decoder", fileName: "same_s_decoder_f32.safetensors", minimumBytes: 1, sourceFileName: "same_s_decoder_f32.npz"),
                WeightFile(role: "Conditioner small-music", fileName: "sa3_conditioner_sm-music.safetensors", minimumBytes: 1, sourceFileName: "dit_sm-music_f16.npz"),
            ]
        case .smallSFX:
            return [
                WeightFile(role: "DiT small-sfx", fileName: "dit_sm-sfx_f16.safetensors", minimumBytes: 1, sourceFileName: "dit_sm-sfx_f16.npz"),
                WeightFile(role: "same-s decoder", fileName: "same_s_decoder_f32.safetensors", minimumBytes: 1, sourceFileName: "same_s_decoder_f32.npz"),
                WeightFile(role: "Conditioner small-sfx", fileName: "sa3_conditioner_sm-sfx.safetensors", minimumBytes: 1, sourceFileName: "dit_sm-sfx_f16.npz"),
            ]
        case .medium:
            return [
                WeightFile(role: "DiT medium", fileName: "dit_medium_f16.safetensors", minimumBytes: 1, sourceFileName: "dit_medium_f16.npz"),
                WeightFile(role: "same-l decoder", fileName: "same_l_decoder_f32.safetensors", minimumBytes: 1, sourceFileName: "same_l_decoder_f32.npz"),
                WeightFile(role: "Conditioner medium", fileName: "sa3_conditioner_medium.safetensors", minimumBytes: 1, sourceFileName: "dit_medium_f16.npz"),
            ]
        }
    }

    /// Files needed in addition to `variantFiles(for:)` to run audio-to-audio
    /// generation (the SAME encoder for the variant's autoencoder family).
    static func encoderFiles(for kind: StableAudioModelKind) -> [WeightFile] {
        switch kind {
        case .smallMusic, .smallSFX:
            return [WeightFile(role: "same-s encoder", fileName: "same_s_encoder_f32.safetensors", minimumBytes: 1, sourceFileName: "same_s_encoder_f32.npz")]
        case .medium:
            return [WeightFile(role: "same-l encoder", fileName: "same_l_encoder_f32.safetensors", minimumBytes: 1, sourceFileName: "same_l_encoder_f32.npz")]
        }
    }

    /// Verifies that the encoder file required for audio-to-audio on the given
    /// model variant is present. Throws `StableAudioWeightError.incomplete`
    /// otherwise so callers can surface a clear error before sampling starts.
    public func requireEncoderReady(for kind: StableAudioModelKind) throws {
        var required = Set<String>()
        for file in Self.encoderFiles(for: kind) {
            required.insert(file.fileName)
        }
        try requireFiles(required)
    }

    private static let alwaysRequiredFileNames: [String] = [
        "t5gemma_f16.safetensors",
        "t5gemma_tokenizer.model",
    ]

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
    case unsupportedOnPlatform(StableAudioModelKind)

    public var errorDescription: String? {
        switch self {
        case .missing(let fileName, let directory):
            return "Missing \(fileName) in \(directory.path)"
        case .incomplete(let statuses):
            let files = statuses.map { "\($0.fileName) (\($0.sizeSummary))" }.joined(separator: ", ")
            return "Model weights are incomplete: \(files)"
        case .unsupportedOnPlatform(let kind):
            return "\(kind.displayName) is not available on this platform"
        }
    }
}
