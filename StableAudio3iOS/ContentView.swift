import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = StableAudioViewModel()

    private let presets: [PromptPreset] = [
        PromptPreset(
            title: "Lo-fi House",
            subtitle: "120 BPM loop",
            prompt: "lofi house loop, 120 BPM",
            duration: 2,
            steps: 4
        ),
        PromptPreset(
            title: "Festival House",
            subtitle: "sunny 124 BPM",
            prompt: "House music that encapsulates the feeling of being at a festival in the sunny weather with all your friends 124 BPM",
            duration: 5,
            steps: 8
        ),
        PromptPreset(
            title: "Piano Build",
            subtitle: "cinematic climb",
            prompt: "A beautiful piano arpeggio grows into a cinematic climax",
            duration: 5,
            steps: 8
        ),
        PromptPreset(
            title: "Ambient Drone",
            subtitle: "soft texture",
            prompt: "ambient drone",
            duration: 2,
            steps: 4
        ),
    ]

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Generate music and sound effects on iPhone.")
                            .font(.title2.weight(.semibold))
                            .fixedSize(horizontal: false, vertical: true)

                        Text(viewModel.heroStatus)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.vertical, 4)
                }

                Section("Try A Prompt") {
                    ForEach(presets) { preset in
                        Button {
                            viewModel.applyPreset(preset)
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: preset.iconName)
                                    .font(.title3)
                                    .frame(width: 30)
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(preset.title)
                                        .font(.headline)
                                    Text(preset.subtitle)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.tertiary)
                            }
                        }
                    }
                }

                Section("Prompt") {
                    TextField("Describe the audio you want", text: $viewModel.prompt, axis: .vertical)
                        .lineLimit(4...8)
                }

                Section("Length") {
                    Picker("Duration", selection: $viewModel.durationSeconds) {
                        Text("1s").tag(Float(1))
                        Text("2s").tag(Float(2))
                        Text("5s").tag(Float(5))
                        Text("10s").tag(Float(10))
                        Text("15s").tag(Float(15))
                    }
                    .pickerStyle(.segmented)
                }

                Section("Quality") {
                    Picker("Mode", selection: $viewModel.stepCount) {
                        Label("Fast", systemImage: "bolt.fill").tag(4)
                        Label("Better", systemImage: "sparkles").tag(8)
                    }
                    .pickerStyle(.segmented)
                }

                Section {
                    Button {
                        viewModel.generate()
                    } label: {
                        HStack {
                            Spacer()
                            if viewModel.isRunning {
                                ProgressView()
                            } else {
                                Image(systemName: "play.fill")
                            }
                            Text(viewModel.generateButtonTitle)
                                .font(.headline)
                            Spacer()
                        }
                        .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!viewModel.canGenerate)

                    if viewModel.showPreparationHint {
                        Label("Prepare the model files before generating.", systemImage: "exclamationmark.triangle")
                            .font(.footnote)
                            .foregroundStyle(.orange)
                    }
                }

                Section("Now") {
                    LabeledContent("Status", value: viewModel.pipelineStatus)
                    LabeledContent("Last run", value: viewModel.timingStatus)
                }

                Section {
                    DisclosureGroup("Advanced") {
                        LabeledContent("Model", value: "small-music")
                        LabeledContent("Decoder", value: "same-s")
                        LabeledContent("Runtime", value: viewModel.runtimeStatus)
                        LabeledContent("Weights", value: viewModel.weightSummary)

                        ForEach(viewModel.weightStatuses) { status in
                            WeightStatusRow(status: status)
                        }
                    }
                }
            }
            .navigationTitle("Stable Audio 3")
            .task {
                viewModel.bootstrap()
            }
        }
    }
}

struct PromptPreset: Identifiable {
    var id: String { title }
    let title: String
    let subtitle: String
    let prompt: String
    let duration: Float
    let steps: Int

    var iconName: String {
        switch title {
        case "Lo-fi House", "Festival House":
            "music.note"
        case "Piano Build":
            "pianokeys"
        default:
            "waveform"
        }
    }
}

private struct WeightStatusRow: View {
    let status: WeightStatus

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Image(systemName: status.isReady ? "checkmark.circle.fill" : "exclamationmark.circle")
                .foregroundStyle(status.isReady ? .green : .orange)
            VStack(alignment: .leading, spacing: 3) {
                Text(status.role)
                    .font(.subheadline.weight(.semibold))
                Text(status.sizeSummary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

#Preview {
    ContentView()
}
