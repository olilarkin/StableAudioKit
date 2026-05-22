import Foundation
import AVFoundation

@MainActor
final class StableAudioViewModel: ObservableObject {
    @Published var isRunning = false
    @Published var runtimeDevice = "GPU"
    @Published var runtimeStatus = "Checking"
    @Published var runtimeResult = "-"
    @Published var pipelineStatus = "Checking weights"
    @Published var tensorLoadStatus = "Not loaded"
    @Published var t5Status = "Not run"
    @Published var prompt = "Tight acoustic drum kit groove, crisp snare, punchy kick, closed hi-hats, dry studio room, no melody"
    @Published var durationSeconds: Float = 1
    @Published var stepCount = 4
    @Published var timingStatus = "Not run"
    @Published var weightStatuses: [WeightStatus] = []

    private let inspector = WeightInspector()
    private let pipeline = StableAudioPipeline()
    private var audioPlayer: AVAudioPlayer?
    private var didBootstrap = false

    var canGenerate: Bool {
        !isRunning && allWeightsReady
    }

    var generateButtonTitle: String {
        isRunning ? "Generating..." : "Generate & Play"
    }

    var heroStatus: String {
        if isRunning {
            return "Generating locally. Keep the app open."
        }
        if allWeightsReady {
            return "Ready. Pick an example or write your own prompt."
        }
        return "Model files are not ready yet."
    }

    var showPreparationHint: Bool {
        !allWeightsReady && !weightStatuses.isEmpty
    }

    var weightSummary: String {
        guard !weightStatuses.isEmpty else { return "Checking" }
        let readyCount = weightStatuses.filter(\.isReady).count
        return "\(readyCount)/\(weightStatuses.count) ready"
    }

    private var allWeightsReady: Bool {
        !weightStatuses.isEmpty && weightStatuses.allSatisfy(\.isReady)
    }

    func bootstrap() {
        guard !didBootstrap else { return }
        didBootstrap = true
        inspectWeights()
        pipelineStatus = allWeightsReady ? "Ready" : "Missing weights"

        Task {
            let result = await Task.detached(priority: .userInitiated) {
                MLXSmokeTest.run()
            }.value

            switch result {
            case .success(let report):
                runtimeDevice = report.device
                runtimeStatus = "Ready"
                runtimeResult = report.summary
            case .failure(let error):
                runtimeStatus = "Failed"
                runtimeResult = error.localizedDescription
            }
        }
    }

    func runDiagnostics() {
        guard !isRunning else { return }
        isRunning = true
        runtimeStatus = "Running"
        runtimeResult = "-"

        Task {
            let result = await Task.detached(priority: .userInitiated) {
                MLXSmokeTest.run()
            }.value

            switch result {
            case .success(let report):
                runtimeDevice = report.device
                runtimeStatus = "Ready"
                runtimeResult = report.summary
            case .failure(let error):
                runtimeStatus = "Failed"
                runtimeResult = error.localizedDescription
            }

            isRunning = false
        }
    }

    func inspectWeights() {
        weightStatuses = inspector.inspect()
    }

    func applyPreset(_ preset: PromptPreset) {
        prompt = preset.prompt
        durationSeconds = preset.duration
        stepCount = preset.steps
        pipelineStatus = allWeightsReady ? "Ready" : "Missing weights"
    }

    func loadDecoderWeights() {
        guard !isRunning else { return }
        isRunning = true
        tensorLoadStatus = "Loading"

        Task {
            let result = await Task.detached(priority: .userInitiated) {
                WeightTensorLoader.load(fileName: "same_s_decoder_f32.safetensors")
            }.value

            switch result {
            case .success(let report):
                tensorLoadStatus = "\(report.tensorCount) tensors, \(report.sample)"
            case .failure(let error):
                tensorLoadStatus = error.localizedDescription
            }

            isRunning = false
        }
    }

    func runT5Test() {
        guard !isRunning else { return }
        isRunning = true
        t5Status = "Running"
        let currentPrompt = prompt

        Task {
            let result = await Task.detached(priority: .userInitiated) {
                T5GemmaLoadTester.runPromptForward(prompt: currentPrompt)
            }.value

            switch result {
            case .success(let report):
                t5Status = "\(report.tensorCount) tensors, \(report.shape), \(String(format: "%.2f", report.elapsedSeconds))s"
            case .failure(let error):
                t5Status = error.localizedDescription
            }

            isRunning = false
        }
    }

    func generate() {
        guard !isRunning else { return }
        inspectWeights()
        guard allWeightsReady else {
            pipelineStatus = "Missing weights"
            return
        }

        isRunning = true
        pipelineStatus = "Starting"
        timingStatus = "Running"
        let currentPrompt = prompt
        let currentDuration = durationSeconds
        let currentSteps = stepCount

        Task {
            let result: Swift.Result<StableAudioPipeline.Result, Error>
            do {
                let startedAt = Date()
                let output = try await pipeline.generate(prompt: currentPrompt, seconds: currentDuration, steps: currentSteps) { stage in
                    let elapsedMilliseconds = Int((Date().timeIntervalSince(startedAt) * 1000).rounded())
                    print("[SA3][UI] \(stage) total=\(elapsedMilliseconds)ms")
                    Task { @MainActor in
                        self.pipelineStatus = "\(stage) \(elapsedMilliseconds)ms"
                    }
                }
                result = .success(output)
            } catch {
                result = .failure(error)
            }

            switch result {
            case .success(let output):
                pipelineStatus = "Done \(String(format: "%.1f", output.duration))s, Tlat \(output.latentLength)"
                timingStatus = "\(Int((output.elapsedSeconds * 1000).rounded()))ms total"
                do {
                    audioPlayer = try AVAudioPlayer(contentsOf: output.url)
                    audioPlayer?.prepareToPlay()
                    audioPlayer?.play()
                } catch {
                    pipelineStatus = "WAV ready, playback failed"
                }
            case .failure(let error):
                pipelineStatus = error.localizedDescription
                timingStatus = "Failed"
            }

            isRunning = false
        }
    }
}
