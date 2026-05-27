import Foundation
import MLX
import MLXRandom
import SentencepieceTokenizer

public struct StableAudioGenerationRequest: Sendable {
    public var model: StableAudioModelKind
    public var prompt: String
    public var seconds: Float
    public var steps: Int
    public var seed: UInt64

    public init(
        model: StableAudioModelKind = .smallMusic,
        prompt: String,
        seconds: Float = 10,
        steps: Int = 8,
        seed: UInt64 = UInt64.random(in: 0 ..< UInt64(Int32.max))
    ) {
        self.model = model
        self.prompt = prompt
        self.seconds = seconds
        self.steps = steps
        self.seed = seed
    }
}

public struct StableAudioGenerationResult: Sendable {
    public let samples: [Float]
    public let channelCount: Int
    public let sampleRate: Int
    public let duration: Float
    public let latentLength: Int
    public let elapsedSeconds: TimeInterval

    public var frameCount: Int {
        samples.count / channelCount
    }
}

public enum StableAudioProgress: Sendable {
    case stage(String)
    case samplingStep(Int, Int)
}

protocol DiTModel {
    var latentLength: Int { get }
    var ioChannels: Int { get }
    func callAsFunction(
        _ x: MLXArray,
        timestep: MLXArray,
        crossAttention: MLXArray,
        globalCondition: MLXArray
    ) -> MLXArray
}

protocol AudioDecoder {
    var samplesPerStrideFrame: Int { get }
    func decodeChunked(latents: MLXArray) -> MLXArray
}

extension DiTSmallMusic: DiTModel {
    var ioChannels: Int { Self.ioChannels }
}

extension DiTMedium: DiTModel {
    var ioChannels: Int { Self.ioChannels }
}

extension SAMESDecoder: AudioDecoder {
    var samplesPerStrideFrame: Int { Self.outputChannels / 2 }

    func decodeChunked(latents: MLXArray) -> MLXArray {
        decodeChunked(latents: latents, chunkSize: 8, overlap: 2)
    }
}

extension SAMELDecoder: AudioDecoder {
    var samplesPerStrideFrame: Int { Self.outputChannels / 2 }

    func decodeChunked(latents: MLXArray) -> MLXArray {
        decodeChunked(latents: latents, chunkSize: 8, overlap: 2)
    }
}

public actor StableAudioPipeline {
    public static let sampleRate = 44_100
    public static let samplesPerLatent = 4_096

    private let weights: StableAudioWeights
    private var cachedTokenizer: SentencepieceTokenizer?
    private var cachedT5Encoder: T5GemmaEncoder?
    private var cachedConditioners: [StableAudioModelKind: SA3Conditioning] = [:]
    private var cachedDiTWeights: [StableAudioModelKind: [String: MLXArray]] = [:]
    private var cachedSAMESDecoder: SAMESDecoder?
    private var cachedSAMELDecoder: SAMELDecoder?

    public init(weightsDirectory: URL) throws {
        self.weights = try StableAudioWeights(directory: weightsDirectory)
        try self.weights.requireReady()
    }

    public static func load(from weightsDirectory: URL) throws -> StableAudioPipeline {
        try StableAudioPipeline(weightsDirectory: weightsDirectory)
    }

    public func generate(
        _ request: StableAudioGenerationRequest,
        progress: (@Sendable (StableAudioProgress) -> Void)? = nil
    ) throws -> StableAudioGenerationResult {
        let startedAt = Date()
        let seconds = max(0.1, request.seconds)
        let steps = max(1, request.steps)
        let latentLength = Self.latentLength(for: seconds)

        guard request.model.isAvailableOnThisPlatform else {
            throw StableAudioWeightError.unsupportedOnPlatform(request.model)
        }

        let conditioning = try Stream.withNewDefaultStream(device: .gpu) {
            progress?(.stage("T5"))
            let promptEncoding = try encodePrompt(prompt: request.prompt, maxLength: 256)

            progress?(.stage("Conditioning"))
            let conditioner = try loadConditioner(model: request.model)
            let conditioned = conditioner.makeConditioning(promptEncoding: promptEncoding, seconds: seconds)
            let crossAttention = conditioned.crossAttention.asType(.float16)
            let globalCondition = conditioned.globalCondition.asType(.float16)
            eval(crossAttention, globalCondition)
            return (crossAttention, globalCondition)
        }

        let latents = try Stream.withNewDefaultStream(device: .gpu) {
            progress?(.stage("DiT load"))
            let dit = try loadDiT(model: request.model, latentLength: latentLength)
            progress?(.stage("Sampling"))
            let latents = sample(
                dit: dit,
                latentLength: latentLength,
                steps: steps,
                seed: request.seed,
                crossAttention: conditioning.0,
                globalCondition: conditioning.1,
                progress: progress
            )
            eval(latents)
            return latents
        }

        let audio = try Stream.withNewDefaultStream(device: .gpu) {
            progress?(.stage("Decoder load"))
            let decoder = try loadDecoder(for: request.model)
            progress?(.stage("Decoder forward"))
            let patches = decoder.decodeChunked(latents: latents.asType(.float32))
            let audio = Self.patchedDecode(patches, samplesPerStrideFrame: decoder.samplesPerStrideFrame).asType(.float32)
            eval(audio)
            return audio
        }

        let requestedSamples = Int((seconds * Float(Self.sampleRate)).rounded())
        let trimmed = audio[0, 0..., 0 ..< requestedSamples]
        progress?(.stage("Done"))
        return StableAudioGenerationResult(
            samples: trimmed.asArray(Float.self),
            channelCount: 2,
            sampleRate: Self.sampleRate,
            duration: seconds,
            latentLength: latentLength,
            elapsedSeconds: Date().timeIntervalSince(startedAt)
        )
    }

    private func sample(
        dit: any DiTModel,
        latentLength: Int,
        steps: Int,
        seed: UInt64,
        crossAttention: MLXArray,
        globalCondition: MLXArray,
        progress: (@Sendable (StableAudioProgress) -> Void)?
    ) -> MLXArray {
        let schedule = Self.buildSchedule(steps: steps)
        var key = MLXRandom.key(seed)
        var x = MLXRandom.normal([1, dit.ioChannels, latentLength], dtype: .float16, key: key)
        eval(x)

        for index in 0 ..< steps {
            progress?(.samplingStep(index + 1, steps))
            let current = schedule[index]
            let next = schedule[index + 1]
            let t = MLXArray([current], [1]).asType(.float16)
            let velocity = dit.callAsFunction(x, timestep: t, crossAttention: crossAttention, globalCondition: globalCondition)
            let denoised = x - MLXArray(current, dtype: x.dtype) * velocity
            if index < steps - 1 && next > 0 {
                let split = MLXRandom.split(key: key)
                key = split.0
                let noise = MLXRandom.normal(x.shape, dtype: x.dtype, key: split.1)
                x = (1.0 - next) * denoised + next * noise
            } else {
                x = denoised
            }
            eval(x)
        }
        return x
    }

    private func encodePrompt(prompt: String, maxLength: Int) throws -> T5PromptEncoding {
        let batch = try tokenize(prompt: prompt, maxLength: maxLength)
        let encoder = try loadT5Encoder()
        let embeddings = encoder.encodeTokenIDs(batch.tokenIDs, attentionMask: batch.mask)
        eval(embeddings, batch.mask)
        return T5PromptEncoding(embeddings: embeddings, mask: batch.mask, tokenCount: batch.tokenCount)
    }

    private func tokenize(prompt: String, maxLength: Int) throws -> (tokenIDs: MLXArray, mask: MLXArray, tokenCount: Int) {
        let tokenizer = try loadTokenizer()
        let clipped = Array(try tokenizer.encode(prompt).prefix(maxLength)).map(Int32.init)
        let safeTokens = clipped.isEmpty ? [Int32(1)] : clipped
        let tokenCount = safeTokens.count

        var ids = Array(repeating: Int32(0), count: maxLength)
        var mask = Array(repeating: Int32(0), count: maxLength)
        for index in 0 ..< tokenCount {
            ids[index] = safeTokens[index]
            mask[index] = 1
        }

        return (
            MLXArray(ids, [1, maxLength]),
            MLXArray(mask, [1, maxLength]),
            tokenCount
        )
    }

    private func loadTokenizer() throws -> SentencepieceTokenizer {
        if let cachedTokenizer {
            return cachedTokenizer
        }
        let url = try weights.url(for: "t5gemma_tokenizer.model")
        let tokenizer = try SentencepieceTokenizer(modelPath: url.path(percentEncoded: false), tokenOffset: 0)
        cachedTokenizer = tokenizer
        return tokenizer
    }

    private func loadT5Encoder() throws -> T5GemmaEncoder {
        if let cachedT5Encoder {
            return cachedT5Encoder
        }
        let arrays = try loadArrays(url: weights.url(for: "t5gemma_f16.safetensors"), stream: .cpu)
        let encoder = try T5GemmaEncoder(weights: arrays)
        cachedT5Encoder = encoder
        return encoder
    }

    private func loadConditioner(model: StableAudioModelKind) throws -> SA3Conditioning {
        if let cachedConditioner = cachedConditioners[model] {
            return cachedConditioner
        }
        try weights.requireReady(for: model)
        let arrays = try loadArrays(url: weights.url(for: "\(model.conditionerResourceName).safetensors"), stream: .cpu)
        let conditioner = SA3Conditioning(weights: arrays)
        cachedConditioners[model] = conditioner
        return conditioner
    }

    private func loadDiT(model: StableAudioModelKind, latentLength: Int) throws -> any DiTModel {
        guard model.isAvailableOnThisPlatform else {
            throw StableAudioWeightError.unsupportedOnPlatform(model)
        }
        if let cached = cachedDiTWeights[model] {
            return makeDiT(model: model, weights: cached, latentLength: latentLength)
        }
        try weights.requireReady(for: model)
        let arrays = try loadArrays(url: weights.url(for: "\(model.ditResourceName).safetensors"), stream: .cpu)
        cachedDiTWeights[model] = arrays
        return makeDiT(model: model, weights: arrays, latentLength: latentLength)
    }

    private func makeDiT(model: StableAudioModelKind, weights: [String: MLXArray], latentLength: Int) -> any DiTModel {
        switch model {
        case .smallMusic, .smallSFX:
            return DiTSmallMusic(weights: weights, latentLength: latentLength)
        case .medium:
            return DiTMedium(weights: weights, latentLength: latentLength)
        }
    }

    private func loadDecoder(for model: StableAudioModelKind) throws -> any AudioDecoder {
        switch model.autoencoder {
        case .sameS:
            if let cached = cachedSAMESDecoder { return cached }
            try weights.requireReady(for: model)
            let arrays = try loadArrays(url: weights.url(for: "same_s_decoder_f32.safetensors"), stream: .cpu)
            let decoder = SAMESDecoder(weights: arrays)
            cachedSAMESDecoder = decoder
            return decoder
        case .sameL:
            if let cached = cachedSAMELDecoder { return cached }
            try weights.requireReady(for: model)
            let arrays = try loadArrays(url: weights.url(for: "same_l_decoder_f32.safetensors"), stream: .cpu)
            let decoder = SAMELDecoder(weights: arrays)
            cachedSAMELDecoder = decoder
            return decoder
        }
    }

    public static func latentLength(for seconds: Float) -> Int {
        var length = max(1, Int(ceil(seconds * Float(sampleRate) / Float(samplesPerLatent))))
        if length % 2 != 0 {
            length += 1
        }
        return length
    }

    static func buildSchedule(steps: Int) -> [Float] {
        var values: [Float] = []
        for index in 0 ... steps {
            let t = 1.0 - Float(index) / Float(steps)
            var shifted = logSNRShift(t)
            if index == 0 { shifted = 1.0 }
            if index == steps { shifted = 0.0 }
            values.append(shifted)
        }
        return values
    }

    private static func logSNRShift(_ t: Float, anchorLogSNR: Float = -6.2, logSNREnd: Float = 2.0) -> Float {
        if t <= 0 { return 0 }
        if t >= 1 { return 1 }
        let logSNR = logSNREnd - t * (logSNREnd - anchorLogSNR)
        return 1.0 / (1.0 + exp(logSNR))
    }

    static func patchedDecode(_ patches: MLXArray, samplesPerStrideFrame: Int) -> MLXArray {
        let batch = patches.dim(0)
        let length = patches.dim(2)
        var x = patches.reshaped(batch, 2, samplesPerStrideFrame, length)
        x = x.transposed(0, 1, 3, 2)
        return x.reshaped(batch, 2, length * samplesPerStrideFrame)
    }
}
