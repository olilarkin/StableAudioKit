import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = StableAudioViewModel()

    var body: some View {
        NavigationStack {
            List {
                Section("Generate") {
                    TextField("Prompt", text: $viewModel.prompt, axis: .vertical)
                        .lineLimit(3...6)

                    Picker("Duration", selection: $viewModel.durationSeconds) {
                        Text("1s").tag(Float(1))
                        Text("2s").tag(Float(2))
                        Text("5s").tag(Float(5))
                        Text("10s").tag(Float(10))
                        Text("15s").tag(Float(15))
                    }
                    .pickerStyle(.segmented)

                    Picker("Steps", selection: $viewModel.stepCount) {
                        Text("4").tag(4)
                        Text("8").tag(8)
                    }
                    .pickerStyle(.segmented)

                    LabeledContent("Model", value: "small-music")
                    LabeledContent("Decoder", value: "same-s")
                    LabeledContent("Status", value: viewModel.pipelineStatus)
                    LabeledContent("Timing", value: viewModel.timingStatus)

                    Button {
                        viewModel.generate()
                    } label: {
                        Label(viewModel.generateButtonTitle, systemImage: "waveform")
                    }
                    .disabled(!viewModel.canGenerate)
                }

                Section("Ready") {
                    LabeledContent("Device", value: viewModel.runtimeDevice)
                    LabeledContent("MLX", value: viewModel.runtimeStatus)
                    LabeledContent("Weights", value: viewModel.weightSummary)

                    ForEach(viewModel.weightStatuses) { status in
                        WeightStatusRow(status: status)
                    }
                }
            }
            .navigationTitle("Stable Audio 3")
            .toolbar {
                if viewModel.isRunning {
                    ProgressView()
                }
            }
            .task {
                viewModel.bootstrap()
            }
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
                    .font(.headline)
                Text(status.fileName)
                    .font(.caption)
                    .foregroundStyle(.secondary)
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
