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

    func run() async throws {
        let actualSeed = seed ?? UInt64.random(in: 0 ..< UInt64(Int32.max))
        guard let modelKind = StableAudioModelKind(rawValue: model) else {
            throw ValidationError("Unknown model '\(model)'. Use smallMusic, smallSFX, or medium.")
        }
        let weightsURL = URL(fileURLWithPath: modelPath)
        let pipeline = try StableAudioPipeline.load(from: weightsURL)
        let request = StableAudioGenerationRequest(
            model: modelKind,
            prompt: prompt,
            seconds: duration,
            steps: steps,
            seed: actualSeed
        )

        print("Model: \(modelKind.displayName)")
        print("Prompt: \(prompt)")
        print("Duration: \(duration)s, steps: \(steps), seed: \(actualSeed)")

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
