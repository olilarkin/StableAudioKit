import Foundation
import StableAudioKit

@main
struct SA3CLI {
    static func main() async throws {
        let weightsURL = URL(fileURLWithPath: CommandLine.arguments.dropFirst().first ?? "Resources/Weights")
        let pipeline = try StableAudioPipeline.load(from: weightsURL)
        let request = StableAudioGenerationRequest(
            model: .smallMusic,
            prompt: "lofi house loop, 120 BPM",
            seconds: 5,
            steps: 4,
            seed: 42
        )
        let result = try await pipeline.generate(request)
        let outputURL = URL(fileURLWithPath: "sa3-demo.wav")
        try AudioWriter.write(result, to: outputURL)
        print("Wrote \(outputURL.path)")
    }
}
