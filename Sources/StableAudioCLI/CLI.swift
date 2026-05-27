import ArgumentParser
import Foundation
import StableAudioKit

@main
struct StableAudioCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "stable-audio",
        abstract: "Generate audio using Stable Audio 3 MLX weights."
    )

    @Option(name: .long, help: "Text prompt for audio generation.")
    var prompt: String

    @Option(name: .long, help: "Model variant: smallMusic, smallSFX, or medium (macOS only).")
    var model: String = StableAudioModelKind.smallMusic.rawValue

    @Option(name: .long, help: "Audio duration in seconds.")
    var duration: Float = 10

    @Option(name: .long, help: "Number of sampling steps.")
    var steps: Int = 8

    @Option(name: .long, help: "Random seed.")
    var seed: UInt64?

    @Option(name: .long, help: "Path to prepared safetensors model directory.")
    var modelPath: String = "Resources/Weights"

    @Option(name: [.customShort("o"), .long], help: "Output WAV file path.")
    var output: String?

    @Option(name: .long, help: "Optional source audio file for audio-to-audio generation.")
    var initAudio: String?

    @Option(name: .long, help: "Audio-to-audio noise level in [0, 1] (1 = ignore input, 0 = identity).")
    var initNoiseLevel: Float = 0.9

    @Option(name: .long, parsing: .upToNextOption,
            help: "Inpaint mask region start times in seconds. Repeat per region; must pair 1:1 with --inpaint-mask-end. Requires --init-audio.")
    var inpaintMaskStart: [Float] = []

    @Option(name: .long, parsing: .upToNextOption,
            help: "Inpaint mask region end times in seconds. Length must match --inpaint-mask-start.")
    var inpaintMaskEnd: [Float] = []

    func run() async throws {
        let actualSeed = seed ?? UInt64.random(in: 0 ..< UInt64(Int32.max))
        guard let modelKind = StableAudioModelKind(rawValue: model) else {
            throw ValidationError("Unknown model '\(model)'. Use smallMusic, smallSFX, or medium.")
        }
        let weightsURL = URL(fileURLWithPath: modelPath)
        let pipeline = try StableAudioPipeline.load(from: weightsURL)
        let initAudioSource = initAudio.map { path in
            StableAudioGenerationRequest.InitAudio.url(URL(fileURLWithPath: path))
        }

        let inpaintRegions: [InpaintRegion]?
        if !inpaintMaskStart.isEmpty || !inpaintMaskEnd.isEmpty {
            guard inpaintMaskStart.count == inpaintMaskEnd.count else {
                throw ValidationError("--inpaint-mask-start and --inpaint-mask-end must have the same number of values (got \(inpaintMaskStart.count) vs \(inpaintMaskEnd.count)).")
            }
            guard initAudio != nil else {
                throw ValidationError("Inpaint regions require --init-audio.")
            }
            inpaintRegions = zip(inpaintMaskStart, inpaintMaskEnd).map {
                InpaintRegion(startSeconds: $0.0, endSeconds: $0.1)
            }
        } else {
            inpaintRegions = nil
        }

        let request = StableAudioGenerationRequest(
            model: modelKind,
            prompt: prompt,
            seconds: duration,
            steps: steps,
            seed: actualSeed,
            initAudio: initAudioSource,
            initNoiseLevel: initNoiseLevel,
            inpaintRegions: inpaintRegions
        )

        print("Model: \(modelKind.displayName)")
        print("Prompt: \(prompt)")
        print("Duration: \(duration)s, steps: \(steps), seed: \(actualSeed)")
        if let initAudio {
            if let inpaintRegions {
                let summary = inpaintRegions
                    .map { "[\($0.startSeconds)–\($0.endSeconds)]" }
                    .joined(separator: ", ")
                print("Inpaint source: \(initAudio); regions (s): \(summary)")
            } else {
                print("Init audio: \(initAudio) (noise level \(initNoiseLevel))")
            }
        }

        let result = try await pipeline.generate(request) { event in
            switch event {
            case .stage(let name):
                print(name)
            case .samplingStep(let step, let total):
                print("Step \(step)/\(total)", terminator: "\r")
                fflush(stdout)
            }
        }
        print()

        let outputPath = output ?? Self.defaultOutputName(prompt: prompt, seed: actualSeed)
        let outputURL = URL(fileURLWithPath: outputPath)
        try AudioWriter.write(result, to: outputURL)
        print("Wrote \(outputURL.path)")
    }

    private static func defaultOutputName(prompt: String, seed: UInt64) -> String {
        let stem = prompt
            .lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { !$0.isEmpty }
            .prefix(8)
            .joined(separator: "-")
        return "\(stem.isEmpty ? "stable-audio" : stem)-\(seed).wav"
    }
}
