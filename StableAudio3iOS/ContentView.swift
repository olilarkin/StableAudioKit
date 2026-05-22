import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = StableAudioViewModel()
    @State private var showsCustomPrompt = false

    private let waveBars: [CGFloat] = [0.28, 0.62, 0.42, 0.86, 0.55, 0.35, 0.74, 0.48, 0.92, 0.66, 0.38, 0.58, 0.82, 0.44, 0.7, 0.32]

    private let musicPrompts: [PromptPreset] = [
        PromptPreset(
            title: "Lo-fi House",
            subtitle: "120 BPM loop",
            prompt: "lofi house loop, 120 BPM",
            duration: 2,
            steps: 4,
            iconName: "music.note",
            accent: Color(red: 0.26, green: 0.88, blue: 0.68)
        ),
        PromptPreset(
            title: "Festival",
            subtitle: "sunny house",
            prompt: "House music that encapsulates the feeling of being at a festival in the sunny weather with all your friends 124 BPM",
            duration: 5,
            steps: 8,
            iconName: "sun.max.fill",
            accent: Color(red: 1.0, green: 0.74, blue: 0.28)
        ),
        PromptPreset(
            title: "Piano Build",
            subtitle: "cinematic",
            prompt: "A beautiful piano arpeggio grows into a cinematic climax",
            duration: 5,
            steps: 8,
            iconName: "pianokeys",
            accent: Color(red: 0.42, green: 0.7, blue: 1.0)
        ),
        PromptPreset(
            title: "Ambient",
            subtitle: "soft drone",
            prompt: "ambient drone",
            duration: 2,
            steps: 4,
            iconName: "waveform",
            accent: Color(red: 0.82, green: 0.54, blue: 1.0)
        ),
    ]

    private let drumHits: [PromptPreset] = [
        PromptPreset(
            title: "Kick",
            subtitle: "punch",
            prompt: "Single punchy acoustic kick drum hit, dry studio sound, sharp transient, short decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "circle.fill",
            accent: Color(red: 0.24, green: 0.86, blue: 0.58)
        ),
        PromptPreset(
            title: "Snare",
            subtitle: "crisp",
            prompt: "Single crisp snare drum hit, tight room sound, sharp attack, short decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "asterisk",
            accent: Color(red: 1.0, green: 0.45, blue: 0.45)
        ),
        PromptPreset(
            title: "Hi-Hat",
            subtitle: "tick",
            prompt: "Single closed hi-hat tick, bright metallic click, very short decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "sparkle",
            accent: Color(red: 0.98, green: 0.88, blue: 0.36)
        ),
        PromptPreset(
            title: "Tom",
            subtitle: "low",
            prompt: "Single low tom drum hit, resonant body, short room decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "circle.dashed",
            accent: Color(red: 0.42, green: 0.7, blue: 1.0)
        ),
        PromptPreset(
            title: "Cymbal",
            subtitle: "crash",
            prompt: "Single short crash cymbal hit, bright metallic shimmer, quick decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "rays",
            accent: Color(red: 0.96, green: 0.6, blue: 0.26)
        ),
    ]

    var body: some View {
        ZStack {
            background

            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 18) {
                    topBar
                    playerPanel
                    promptRow
                    drumPad
                    statusCard
                    customPrompt
                }
                .padding(.horizontal, 20)
                .padding(.top, 18)
                .padding(.bottom, 28)
            }
        }
        .preferredColorScheme(.dark)
        .task {
            viewModel.bootstrap()
        }
    }

    private var background: some View {
        LinearGradient(
            colors: [
                Color(red: 0.05, green: 0.055, blue: 0.07),
                Color(red: 0.015, green: 0.017, blue: 0.022),
            ],
            startPoint: .top,
            endPoint: .bottom
        )
        .ignoresSafeArea()
    }

    private var topBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Stable Audio 3")
                    .font(.title2.weight(.bold))
                Text("Generate audio on iPhone")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Image(systemName: "waveform.circle.fill")
                .font(.system(size: 34))
                .foregroundStyle(Color(red: 0.26, green: 0.88, blue: 0.68))
        }
    }

    private var playerPanel: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(viewModel.shortStatus)
                        .font(.headline)
                        .lineLimit(1)
                    Text(viewModel.heroStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer()
                if viewModel.isRunning {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "play.fill")
                        .foregroundStyle(.black)
                        .font(.headline)
                        .frame(width: 38, height: 38)
                        .background(Color(red: 0.26, green: 0.88, blue: 0.68), in: Circle())
                }
            }

            HStack(alignment: .center, spacing: 5) {
                ForEach(Array(waveBars.enumerated()), id: \.offset) { index, value in
                    Capsule()
                        .fill(waveColor(index))
                        .frame(width: 8, height: 78 * value)
                        .frame(maxHeight: 78, alignment: .center)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 4)

            HStack {
                Text(viewModel.timingStatus)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(Int(viewModel.durationSeconds))s")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(viewModel.stepCount == 4 ? "Fast" : "Better")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(18)
        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 22))
        .overlay(
            RoundedRectangle(cornerRadius: 22)
                .stroke(Color.white.opacity(0.08), lineWidth: 1)
        )
    }

    private var promptRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            label("Music Prompts")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(musicPrompts) { preset in
                        musicCard(preset)
                    }
                }
            }
        }
    }

    private var drumPad: some View {
        VStack(alignment: .leading, spacing: 10) {
            label("Drum Hits")
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 10), count: 5), spacing: 10) {
                ForEach(drumHits) { preset in
                    drumButton(preset)
                }
            }
        }
    }

    private var statusCard: some View {
        Group {
            if viewModel.showPreparationHint {
                Text("Model files are missing. Run the weight preparation script first.")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.orange)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
            }
        }
    }

    private var customPrompt: some View {
        DisclosureGroup("Custom Prompt", isExpanded: $showsCustomPrompt) {
            VStack(alignment: .leading, spacing: 12) {
                TextField("Describe the audio you want", text: $viewModel.prompt, axis: .vertical)
                    .lineLimit(3...4)
                    .padding(12)
                    .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))

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
                        Image(systemName: viewModel.isRunning ? "waveform" : "play.fill")
                        Text(viewModel.generateButtonTitle)
                            .font(.headline)
                        Spacer()
                    }
                    .frame(height: 44)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!viewModel.canGenerate)
            }
            .padding(.top, 12)
        }
        .font(.subheadline.weight(.semibold))
        .padding(14)
        .background(Color.white.opacity(0.07), in: RoundedRectangle(cornerRadius: 16))
    }

    private func musicCard(_ preset: PromptPreset) -> some View {
        Button {
            viewModel.generatePreset(preset)
        } label: {
            VStack(alignment: .leading, spacing: 10) {
                Image(systemName: preset.iconName)
                    .font(.title3)
                    .foregroundStyle(preset.accent)
                Spacer(minLength: 0)
                Text(preset.title)
                    .font(.headline)
                    .lineLimit(1)
                Text(preset.subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
            .frame(width: 168, height: 116, alignment: .leading)
            .padding(14)
            .background(
                LinearGradient(
                    colors: [preset.accent.opacity(0.24), Color.white.opacity(0.07)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 18)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18)
                    .stroke(preset.accent.opacity(0.28), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .disabled(!viewModel.canGenerate)
    }

    private func drumButton(_ preset: PromptPreset) -> some View {
        Button {
            viewModel.generatePreset(preset)
        } label: {
            VStack(spacing: 6) {
                Image(systemName: preset.iconName)
                    .font(.headline)
                    .foregroundStyle(preset.accent)
                Text(preset.title)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
            }
            .frame(height: 62)
            .frame(maxWidth: .infinity)
            .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
            .overlay(
                RoundedRectangle(cornerRadius: 14)
                    .stroke(preset.accent.opacity(0.35), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .disabled(!viewModel.canGenerate)
    }

    private func label(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.caption2.weight(.bold))
            .foregroundStyle(.secondary)
            .tracking(1.2)
    }

    private func waveColor(_ index: Int) -> Color {
        let colors = [
            Color(red: 0.26, green: 0.88, blue: 0.68),
            Color(red: 0.42, green: 0.7, blue: 1.0),
            Color(red: 1.0, green: 0.74, blue: 0.28),
            Color(red: 1.0, green: 0.45, blue: 0.45),
        ]
        return colors[index % colors.count]
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
    let accent: Color
}

#Preview {
    ContentView()
}
