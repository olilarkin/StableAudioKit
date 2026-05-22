import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = StableAudioViewModel()
    @State private var showsCustomPrompt = false

    private let musicPrompts: [PromptPreset] = [
        PromptPreset(
            title: "Lo-fi House",
            subtitle: "lofi house loop, 120 BPM",
            prompt: "lofi house loop, 120 BPM",
            duration: 2,
            steps: 4,
            iconName: "music.note"
        ),
        PromptPreset(
            title: "Festival House",
            subtitle: "sunny friends, 124 BPM",
            prompt: "House music that encapsulates the feeling of being at a festival in the sunny weather with all your friends 124 BPM",
            duration: 5,
            steps: 8,
            iconName: "music.note"
        ),
        PromptPreset(
            title: "Piano Build",
            subtitle: "cinematic arpeggio climax",
            prompt: "A beautiful piano arpeggio grows into a cinematic climax",
            duration: 5,
            steps: 8,
            iconName: "pianokeys"
        ),
        PromptPreset(
            title: "Ambient Drone",
            subtitle: "soft sustained texture",
            prompt: "ambient drone",
            duration: 2,
            steps: 4,
            iconName: "waveform"
        ),
    ]

    private let drumHits: [PromptPreset] = [
        PromptPreset(
            title: "Kick",
            subtitle: "short punch",
            prompt: "Single punchy acoustic kick drum hit, dry studio sound, sharp transient, short decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "drumsticks"
        ),
        PromptPreset(
            title: "Snare",
            subtitle: "crisp hit",
            prompt: "Single crisp snare drum hit, tight room sound, sharp attack, short decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "drumsticks"
        ),
        PromptPreset(
            title: "Hi-Hat",
            subtitle: "closed tick",
            prompt: "Single closed hi-hat tick, bright metallic click, very short decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "drumsticks"
        ),
        PromptPreset(
            title: "Tom",
            subtitle: "low hit",
            prompt: "Single low tom drum hit, resonant body, short room decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "drumsticks"
        ),
        PromptPreset(
            title: "Cymbal",
            subtitle: "short crash",
            prompt: "Single short crash cymbal hit, bright metallic shimmer, quick decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "drumsticks"
        ),
    ]

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 14) {
                header
                musicPromptScroller
                drumHitScroller
                status
                customPrompt
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

    private var musicPromptScroller: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Music Prompts")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 10) {
                    ForEach(musicPrompts) { preset in
                        presetButton(preset, width: 190, height: 86)
                    }
                }
            }
        }
    }

    private var drumHitScroller: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Drum Hits")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(drumHits) { preset in
                        presetButton(preset, width: 88, height: 74)
                    }
                }
            }
        }
    }

    private func sectionTitle(_ text: String) -> some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .foregroundStyle(.secondary)
    }

    private func presetButton(_ preset: PromptPreset, width: CGFloat, height: CGFloat) -> some View {
        Button {
            viewModel.generatePreset(preset)
        } label: {
            VStack(alignment: .leading, spacing: 4) {
                Image(systemName: preset.iconName)
                    .font(.headline)
                Text(preset.title)
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                Text(preset.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            .frame(width: width, height: height, alignment: .leading)
            .padding(10)
            .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 10))
        }
        .buttonStyle(.plain)
        .disabled(!viewModel.canGenerate)
    }

    private var status: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(viewModel.shortStatus)
                    .font(.subheadline.weight(.medium))
                    .lineLimit(1)
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

    private var customPrompt: some View {
        DisclosureGroup("Custom Prompt", isExpanded: $showsCustomPrompt) {
            VStack(alignment: .leading, spacing: 10) {
                TextField("Describe the audio you want", text: $viewModel.prompt, axis: .vertical)
                    .lineLimit(3...4)
                    .textFieldStyle(.roundedBorder)

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
                    .frame(height: 44)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.canGenerate)
            }
            .padding(.top, 10)
        }
        .font(.subheadline.weight(.semibold))
    }
}

struct PromptPreset: Identifiable {
    var id: String { title }
    let title: String
    let subtitle: String
    let prompt: String
    let duration: Float
    let steps: Int
    let iconName: String
}

#Preview {
    ContentView()
}
