import Foundation
import MLX
import MLXRandom
import SentencepieceTokenizer

/// A time region within the requested output (in seconds) that should be
/// regenerated when inpainting. Audio outside the union of regions is kept
/// from `initAudio`. Continuation is just a region whose start equals the
/// source audio's length and whose end equals the requested total duration.
public struct InpaintRegion: Sendable, Equatable {
    public var startSeconds: Float
    public var endSeconds: Float

    public init(startSeconds: Float, endSeconds: Float) {
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
    }
}

public struct StableAudioGenerationRequest: Sendable {
    /// Source audio for audio-to-audio generation. Either a file URL (decoded
    /// via AVFoundation in `AudioReader`) or raw interleaved PCM the caller has
    /// already decoded. The pipeline resamples to 44.1 kHz and downmixes /
    /// upmixes to stereo automatically.
    public enum InitAudio: Sendable {
        case url(URL)
        case samples(values: [Float], sampleRate: Int, channelCount: Int)
    }

    public var model: StableAudioModelKind
    public var prompt: String
    public var seconds: Float
    public var steps: Int
    public var seed: UInt64
    /// Optional source audio. When set, the diffusion loop is initialized
    /// from this audio's encoded latents partially noised to
    /// `initNoiseLevel`, matching the upstream Python
    /// `model.generate(init_audio=..., init_noise_level=...)` flow.
    public var initAudio: InitAudio?
    /// 0 = identity (return roughly the input audio), 1 = full text-to-audio.
    /// Defaults to 0.9 to match the upstream Python example.
    public var initNoiseLevel: Float
    /// Optional inpaint/continuation regions over the output timeline (seconds).
    /// When non-nil, `initAudio` is required and `initNoiseLevel` is ignored —
    /// the diffusion loop runs the full schedule, regenerating frames inside
    /// the union of regions while preserving frames outside them via standard
    /// known-region renoising. Mirrors the upstream Python
    /// `model.generate(inpaint_audio=..., inpaint_mask_start_seconds=...,
    /// inpaint_mask_end_seconds=...)` flow. Regions may be unordered and
    /// overlap; they are sorted and merged before sampling.
    public var inpaintRegions: [InpaintRegion]?

    public init(
        model: StableAudioModelKind = .smallMusic,
        prompt: String,
        seconds: Float = 10,
        steps: Int = 8,
        seed: UInt64 = UInt64.random(in: 0 ..< UInt64(Int32.max)),
        initAudio: InitAudio? = nil,
        initNoiseLevel: Float = 0.9,
        inpaintRegions: [InpaintRegion]? = nil
    ) {
        self.model = model
        self.prompt = prompt
        self.seconds = seconds
        self.steps = steps
        self.seed = seed
        self.initAudio = initAudio
        self.initNoiseLevel = initNoiseLevel
        self.inpaintRegions = inpaintRegions
    }
}

public enum StableAudioRequestError: LocalizedError {
    case invalidInitNoiseLevel(Float)
    case inpaintRequiresInitAudio
    case emptyInpaintRegions
    case invalidInpaintRegion(start: Float, end: Float, totalSeconds: Float)

    public var errorDescription: String? {
        switch self {
        case .invalidInitNoiseLevel(let value):
            return "initNoiseLevel must be in [0, 1] (got \(value))"
        case .inpaintRequiresInitAudio:
            return "inpaintRegions requires initAudio to be set"
        case .emptyInpaintRegions:
            return "inpaintRegions must be non-empty when provided (use nil to disable inpainting)"
        case .invalidInpaintRegion(let start, let end, let total):
            return "Invalid inpaint region [\(start), \(end)] — must satisfy 0 <= start < end <= \(total)"
        }
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

protocol AudioEncoder {
    var samplesPerLatent: Int { get }
    func encodeChunked(audio: MLXArray) -> MLXArray
}

extension SAMESEncoder: AudioEncoder {
    var samplesPerLatent: Int { Self.samplesPerLatent }

    func encodeChunked(audio: MLXArray) -> MLXArray {
        encodeChunked(audio: audio, chunkSize: 8, overlap: 2)
    }
}

extension SAMELEncoder: AudioEncoder {
    var samplesPerLatent: Int { Self.samplesPerLatent }

    func encodeChunked(audio: MLXArray) -> MLXArray {
        encodeChunked(audio: audio, chunkSize: 8, overlap: 2)
    }
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
    private var cachedSAMESEncoder: SAMESEncoder?
    private var cachedSAMELEncoder: SAMELEncoder?

    public init(weightsDirectory: URL) throws {
        MLXRuntime.ensureConfigured()
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

        if request.initAudio != nil {
            guard request.initNoiseLevel >= 0, request.initNoiseLevel <= 1 else {
                throw StableAudioRequestError.invalidInitNoiseLevel(request.initNoiseLevel)
            }
        }

        let mergedInpaintRegions: [InpaintRegion]?
        if let regions = request.inpaintRegions {
            guard request.initAudio != nil else {
                throw StableAudioRequestError.inpaintRequiresInitAudio
            }
            guard !regions.isEmpty else {
                throw StableAudioRequestError.emptyInpaintRegions
            }
            for region in regions {
                guard region.startSeconds >= 0,
                      region.endSeconds > region.startSeconds,
                      region.endSeconds <= seconds
                else {
                    throw StableAudioRequestError.invalidInpaintRegion(
                        start: region.startSeconds,
                        end: region.endSeconds,
                        totalSeconds: seconds
                    )
                }
            }
            mergedInpaintRegions = Self.mergeRegions(regions)
        } else {
            mergedInpaintRegions = nil
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

        let initLatentsAndMask: (MLXArray, MLXArray?)? = try Stream.withNewDefaultStream(device: .gpu) { () -> (MLXArray, MLXArray?)? in
            guard let initAudio = request.initAudio else { return nil }
            progress?(.stage("Encoder load"))
            let encoder = try loadEncoder(for: request.model)
            progress?(.stage("Encoder forward"))
            let prepared = try Self.prepareInitAudio(initAudio, latentLength: latentLength)
            let latents = encoder.encodeChunked(audio: prepared).asType(.float16)
            let mask: MLXArray? = mergedInpaintRegions.map {
                Self.buildInpaintMask(regions: $0, totalSeconds: seconds, latentLength: latentLength).mask
            }
            if let mask {
                eval(latents, mask)
            } else {
                eval(latents)
            }
            return (latents, mask)
        }

        let initLatents = initLatentsAndMask?.0
        let inpaintMask = initLatentsAndMask?.1

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
                initLatents: initLatents,
                initNoiseLevel: request.initNoiseLevel,
                inpaintMask: inpaintMask,
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
        initLatents: MLXArray?,
        initNoiseLevel: Float,
        inpaintMask: MLXArray? = nil,
        progress: (@Sendable (StableAudioProgress) -> Void)?
    ) -> MLXArray {
        let schedule = Self.buildSchedule(steps: steps)
        var key = MLXRandom.key(seed)
        let split0 = MLXRandom.split(key: key)
        key = split0.0
        let noise = MLXRandom.normal([1, dit.ioChannels, latentLength], dtype: .float16, key: split0.1)
        eval(noise)

        // Inpaint always runs the full schedule starting from pure noise in the
        // masked region; the unmasked region is reseeded from clean init latents
        // at each step's noise level. Audio-to-audio (no mask) keeps its
        // existing partial-schedule behavior driven by initNoiseLevel.
        let startIndex: Int
        if inpaintMask != nil {
            startIndex = 0
        } else {
            startIndex = Self.startIndex(for: initLatents == nil ? 1.0 : initNoiseLevel, schedule: schedule)
        }

        let initRef: MLXArray? = initLatents.map { $0.asType(.float16) }

        var x: MLXArray
        if let initRef, inpaintMask == nil {
            let sigma = schedule[startIndex]
            x = (1.0 - sigma) * initRef + sigma * noise
        } else {
            // For inpaint, schedule[0] == 1.0 so the unmasked-side blend
            // collapses to noise as well; using `noise` directly keeps the
            // first-step path numerically identical.
            x = noise
        }
        eval(x)

        let total = steps - startIndex
        for index in startIndex ..< steps {
            progress?(.samplingStep(index - startIndex + 1, max(1, total)))
            let current = schedule[index]
            let next = schedule[index + 1]
            let t = MLXArray([current], [1]).asType(.float16)
            let velocity = dit.callAsFunction(x, timestep: t, crossAttention: crossAttention, globalCondition: globalCondition)
            let denoised = x - MLXArray(current, dtype: x.dtype) * velocity

            let stepNoise: MLXArray?
            if index < steps - 1 && next > 0 {
                let nextSplit = MLXRandom.split(key: key)
                key = nextSplit.0
                stepNoise = MLXRandom.normal(x.shape, dtype: x.dtype, key: nextSplit.1)
                x = (1.0 - next) * denoised + next * stepNoise!
            } else {
                stepNoise = nil
                x = denoised
            }

            if let mask = inpaintMask, let initRef {
                let xKnown: MLXArray
                if let stepNoise {
                    xKnown = (1.0 - next) * initRef + next * stepNoise
                } else {
                    xKnown = initRef
                }
                x = mask * x + (1.0 - mask) * xKnown
            }

            eval(x)
        }
        return x
    }

    static func mergeRegions(_ regions: [InpaintRegion]) -> [InpaintRegion] {
        guard !regions.isEmpty else { return [] }
        let sorted = regions.sorted { $0.startSeconds < $1.startSeconds }
        var merged: [InpaintRegion] = [sorted[0]]
        for region in sorted.dropFirst() {
            let last = merged[merged.count - 1]
            if region.startSeconds <= last.endSeconds {
                merged[merged.count - 1] = InpaintRegion(
                    startSeconds: last.startSeconds,
                    endSeconds: max(last.endSeconds, region.endSeconds)
                )
            } else {
                merged.append(region)
            }
        }
        return merged
    }

    /// Convert inpaint regions (in output-timeline seconds) into a
    /// latent-frame mask broadcastable against the DiT input shape
    /// `[1, ioChannels, latentLength]`. Frames inside any region are set to
    /// 1.0 (regenerate); frames outside are 0.0 (keep). Regions are assumed
    /// pre-validated and pre-merged (see `mergeRegions`).
    static func buildInpaintMask(
        regions: [InpaintRegion],
        totalSeconds: Float,
        latentLength: Int
    ) -> (frames: [(Int, Int)], mask: MLXArray) {
        var frames: [(Int, Int)] = []
        var values = [Float](repeating: 0, count: latentLength)
        let scale = Float(latentLength) / max(totalSeconds, .leastNormalMagnitude)
        for region in regions {
            let rawStart = Int((region.startSeconds * scale).rounded())
            let rawEnd = Int((region.endSeconds * scale).rounded())
            let startFrame = max(0, min(latentLength, rawStart))
            let endFrame = max(0, min(latentLength, rawEnd))
            guard endFrame > startFrame else { continue }
            for i in startFrame ..< endFrame {
                values[i] = 1.0
            }
            frames.append((startFrame, endFrame))
        }
        let mask = MLXArray(values, [1, 1, latentLength]).asType(.float16)
        return (frames, mask)
    }

    static func startIndex(for noiseLevel: Float, schedule: [Float]) -> Int {
        // Pick the schedule index whose sigma is closest to the requested level.
        // Ties resolve toward the noisier (lower index) end so that
        // noiseLevel == 1.0 always maps to index 0 (text-to-audio equivalent).
        var bestIndex = 0
        var bestDelta = Float.infinity
        for index in 0 ..< schedule.count - 1 {
            let delta = abs(schedule[index] - noiseLevel)
            if delta < bestDelta {
                bestDelta = delta
                bestIndex = index
            }
        }
        return bestIndex
    }

    static func prepareInitAudio(_ source: StableAudioGenerationRequest.InitAudio, latentLength: Int) throws -> MLXArray {
        let array: MLXArray
        switch source {
        case .url(let url):
            array = try AudioReader.loadStereo44k(url: url)
        case .samples(let values, let sampleRate, let channelCount):
            array = try AudioReader.loadStereo44k(samples: values, sampleRate: sampleRate, channelCount: channelCount)
        }
        let targetSamples = latentLength * samplesPerLatent
        let inputSamples = array.dim(2)
        if inputSamples == targetSamples {
            return array
        }
        if inputSamples > targetSamples {
            return array[0..., 0..., 0 ..< targetSamples]
        }
        let padding = MLXArray.zeros([1, 2, targetSamples - inputSamples], dtype: array.dtype)
        return concatenated([array, padding], axis: 2)
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

    private func loadEncoder(for model: StableAudioModelKind) throws -> any AudioEncoder {
        switch model.autoencoder {
        case .sameS:
            if let cached = cachedSAMESEncoder { return cached }
            try weights.requireReady(for: model)
            try weights.requireEncoderReady(for: model)
            let arrays = try loadArrays(url: weights.url(for: "same_s_encoder_f32.safetensors"), stream: .cpu)
            let encoder = SAMESEncoder(weights: arrays)
            cachedSAMESEncoder = encoder
            return encoder
        case .sameL:
            if let cached = cachedSAMELEncoder { return cached }
            try weights.requireReady(for: model)
            try weights.requireEncoderReady(for: model)
            let arrays = try loadArrays(url: weights.url(for: "same_l_encoder_f32.safetensors"), stream: .cpu)
            let encoder = SAMELEncoder(weights: arrays)
            cachedSAMELEncoder = encoder
            return encoder
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
