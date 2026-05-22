import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = StableAudioViewModel()

    private let presets: [PromptPreset] = [
        PromptPreset(
            title: "Kick",
            subtitle: "short hit",
            prompt: "Single punchy acoustic kick drum hit, dry studio sound, sharp transient, short decay, no rhythm, no melody",
            duration: 1,
            steps: 4
        ),
        PromptPreset(
            title: "Snare",
            subtitle: "crisp hit",
            prompt: "Single crisp snare drum hit, tight room sound, sharp attack, short decay, no rhythm, no melody",
            duration: 1,
            steps: 4
        ),
        PromptPreset(
            title: "Hi-Hat",
            subtitle: "closed tick",
            prompt: "Single closed hi-hat tick, bright metallic click, very short decay, no rhythm, no melody",
            duration: 1,
            steps: 4
        ),
        PromptPreset(
            title: "Tom",
            subtitle: "low hit",
            prompt: "Single low tom drum hit, resonant body, short room decay, no rhythm, no melody",
            duration: 1,
            steps: 4
        ),
        PromptPreset(
            title: "Cymbal",
            subtitle: "short crash",
            prompt: "Single short crash cymbal hit, bright metallic shimmer, quick decay, no rhythm, no melody",
            duration: 1,
            steps: 4
        ),
        PromptPreset(
            title: "Lo-fi",
            subtitle: "120 BPM",
            prompt: "lofi house loop, 120 BPM",
            duration: 2,
            steps: 4
        ),
        PromptPreset(
            title: "Festival",
            subtitle: "house",
            prompt: "House music that encapsulates the feeling of being at a festival in the sunny weather with all your friends 124 BPM",
            duration: 5,
            steps: 8
        ),
        PromptPreset(
            title: "Piano",
            subtitle: "cinematic",
            prompt: "A beautiful piano arpeggio grows into a cinematic climax",
            duration: 5,
            steps: 8
        ),
    ]

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                header
                presetScroller
                promptEditor
                controls
                generateButton
                status
                Spacer(minLength: 0)
            }
            .padding(20)
            .navigationTitle("Stable Audio 3")
            .navigationBarTitleDisplayMode(.inline)
            .task {
                viewModel.bootstrap()
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Generate audio on iPhone")
                .font(.title2.weight(.semibold))
            Text(viewModel.heroStatus)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
    }

    private var presetScroller: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Examples")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(presets) { preset in
                        Button {
                            viewModel.applyPreset(preset)
                        } label: {
                            VStack(alignment: .leading, spacing: 4) {
                                Image(systemName: preset.iconName)
                                    .font(.headline)
                                Text(preset.title)
                                    .font(.subheadline.weight(.semibold))
                                Text(preset.subtitle)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .frame(width: 92, height: 76, alignment: .leading)
                            .padding(10)
                            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var promptEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prompt")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            TextField("Describe the audio you want", text: $viewModel.prompt, axis: .vertical)
                .lineLimit(4...5)
                .textFieldStyle(.roundedBorder)
        }
    }

    private var controls: some View {
        VStack(spacing: 10) {
            Picker("Length", selection: $viewModel.durationSeconds) {
                Text("1s").tag(Float(1))
                Text("2s").tag(Float(2))
                Text("5s").tag(Float(5))
                Text("10s").tag(Float(10))
                Text("15s").tag(Float(15))
            }
            .pickerStyle(.segmented)

            Picker("Quality", selection: $viewModel.stepCount) {
                Text("Fast").tag(4)
                Text("Better").tag(8)
            }
            .pickerStyle(.segmented)
        }
    }

    private var generateButton: some View {
        Button {
            viewModel.generate()
        } label: {
            HStack {
                Spacer()
                if viewModel.isRunning {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "play.fill")
                }
                Text(viewModel.generateButtonTitle)
                    .font(.headline)
                Spacer()
            }
            .frame(height: 48)
        }
        .buttonStyle(.borderedProminent)
        .disabled(!viewModel.canGenerate)
    }

    private var status: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(viewModel.shortStatus)
                    .font(.subheadline.weight(.medium))
                Spacer()
                Text(viewModel.timingStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if viewModel.showPreparationHint {
                Text("Model files are missing. Run the weight preparation script first.")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
        .padding(12)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
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
        case "Kick", "Snare", "Hi-Hat", "Tom", "Cymbal":
            "drumsticks"
        case "Piano":
            "pianokeys"
        case "Festival", "Lo-fi":
            "music.note"
        default:
            "waveform"
        }
    }
}

#Preview {
    ContentView()
}
