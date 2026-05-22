import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = StableAudioViewModel()

    private let waveBars: [CGFloat] = [0.35, 0.62, 0.44, 0.82, 0.52, 0.38, 0.74, 0.48, 0.88, 0.58, 0.42, 0.68]

    private let musicPrompts: [PromptPreset] = [
        PromptPreset(
            title: "Lo-fi House",
            subtitle: "10s loop",
            model: .smallMusic,
            prompt: "lofi house loop, 120 BPM",
            duration: 10,
            steps: 4,
            iconName: "music.note",
            accent: Color(red: 0.05, green: 0.68, blue: 0.48)
        ),
        PromptPreset(
            title: "Festival",
            subtitle: "10s house",
            model: .smallMusic,
            prompt: "House music that encapsulates the feeling of being at a festival in the sunny weather with all your friends 124 BPM",
            duration: 10,
            steps: 8,
            iconName: "sun.max.fill",
            accent: Color(red: 0.94, green: 0.58, blue: 0.12)
        ),
        PromptPreset(
            title: "Piano Build",
            subtitle: "10s cinematic",
            model: .smallMusic,
            prompt: "A beautiful piano arpeggio grows into a cinematic climax",
            duration: 10,
            steps: 8,
            iconName: "pianokeys",
            accent: Color(red: 0.24, green: 0.48, blue: 0.9)
        ),
        PromptPreset(
            title: "Ambient",
            subtitle: "10s drone",
            model: .smallMusic,
            prompt: "ambient drone",
            duration: 10,
            steps: 4,
            iconName: "waveform",
            accent: Color(red: 0.62, green: 0.38, blue: 0.85)
        ),
    ]

    private let sfxPrompts: [PromptPreset] = [
        PromptPreset(
            title: "Kick",
            subtitle: "punch",
            model: .smallSFX,
            prompt: "Single punchy acoustic kick drum hit, dry studio sound, sharp transient, short decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "circle.fill",
            accent: Color(red: 0.07, green: 0.66, blue: 0.44)
        ),
        PromptPreset(
            title: "Snare",
            subtitle: "crisp",
            model: .smallSFX,
            prompt: "Single crisp snare drum hit, tight room sound, sharp attack, short decay, no rhythm, no melody",
            duration: 1,
            steps: 4,
            iconName: "asterisk",
            accent: Color(red: 0.92, green: 0.28, blue: 0.34)
        ),
        PromptPreset(
            title: "Whoosh",
            subtitle: "quick",
            model: .smallSFX,
            prompt: "Short cinematic whoosh pass by, clean air movement, quick rise and decay, no music, no voice",
            duration: 1,
            steps: 4,
            iconName: "wind",
            accent: Color(red: 0.88, green: 0.68, blue: 0.05)
        ),
        PromptPreset(
            title: "Footstep",
            subtitle: "wood",
            model: .smallSFX,
            prompt: "Single sneaker footstep on wooden floor, close microphone, short natural room tail, no music, no voice",
            duration: 1,
            steps: 4,
            iconName: "figure.walk",
            accent: Color(red: 0.16, green: 0.48, blue: 0.8)
        ),
        PromptPreset(
            title: "Glass",
            subtitle: "break",
            model: .smallSFX,
            prompt: "Small glass bottle breaking on concrete, sharp impact, scattered tiny shards, dry room, no music, no voice",
            duration: 1,
            steps: 4,
            iconName: "sparkles",
            accent: Color(red: 0.95, green: 0.48, blue: 0.12)
        ),
    ]

    var body: some View {
        ZStack {
            background

            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 24) {
                    topBar
                    playerPanel
                    promptComposer
                    musicPromptRow
                    sfxPromptPad
                    statusHint
                }
                .padding(.horizontal, 22)
                .padding(.bottom, 32)
            }
            .safeAreaPadding(.top, 18)
        }
        .preferredColorScheme(.light)
        .task {
            viewModel.bootstrap()
        }
    }

    private var background: some View {
        LinearGradient(
            colors: [
                Color(red: 0.98, green: 0.97, blue: 0.94),
                Color(red: 0.93, green: 0.97, blue: 0.95),
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
        .ignoresSafeArea()
    }

    private var topBar: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Stable Audio 3")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(Color(red: 0.08, green: 0.09, blue: 0.11))
                Text("Music and sound effects on iPhone")
                    .font(.caption)
                    .foregroundStyle(Color.black.opacity(0.48))
            }
            Spacer()
            Image(systemName: "waveform")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 46, height: 46)
                .background(Color(red: 0.05, green: 0.68, blue: 0.48), in: Circle())
        }
    }

    private var playerPanel: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(viewModel.shortStatus)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Color(red: 0.08, green: 0.09, blue: 0.11))
                        .lineLimit(1)
                    Text(viewModel.heroStatus)
                        .font(.caption)
                        .foregroundStyle(Color.black.opacity(0.48))
                        .lineLimit(2)
                }
                Spacer()
                playbackGlyph
            }

            HStack(alignment: .center, spacing: 6) {
                ForEach(Array(waveBars.enumerated()), id: \.offset) { index, value in
                    Capsule()
                        .fill(waveColor(index))
                        .frame(width: 7, height: 58 * value)
                        .frame(maxHeight: 58, alignment: .center)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 4)

            HStack {
                Text(viewModel.timingStatus)
                Spacer()
                Text("\(Int(viewModel.durationSeconds))s")
                Text(viewModel.selectedModel.title)
                Text(viewModel.stepCount == 4 ? "Fast" : "Better")
            }
            .font(.caption.weight(.medium))
            .foregroundStyle(Color.black.opacity(0.46))
        }
        .padding(18)
        .background(.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 22))
        .shadow(color: Color.black.opacity(0.08), radius: 18, y: 8)
    }

    private var playbackGlyph: some View {
        Group {
            if viewModel.isRunning {
                ProgressView()
            } else {
                Image(systemName: "play.fill")
                    .font(.headline)
                    .foregroundStyle(.white)
            }
        }
        .frame(width: 44, height: 44)
        .background(modelAccent(viewModel.selectedModel), in: Circle())
    }

    private var promptComposer: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                sectionTitle("Prompt")
                Spacer()
                HStack(spacing: 6) {
                    modelChip(.smallMusic)
                    modelChip(.smallSFX)
                    qualityChip("Fast", steps: 4)
                    qualityChip("Better", steps: 8)
                }
            }

            HStack(spacing: 7) {
                durationChip(1)
                durationChip(5)
                durationChip(10)
                durationChip(15)
                durationChip(30)
                Spacer(minLength: 0)
            }

            HStack(alignment: .center, spacing: 10) {
                TextField("Write your own prompt", text: $viewModel.prompt, axis: .vertical)
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(Color(red: 0.08, green: 0.09, blue: 0.11))
                    .lineLimit(2...3)
                    .padding(.vertical, 10)
                    .padding(.horizontal, 12)
                    .background(.white.opacity(0.9), in: RoundedRectangle(cornerRadius: 16))
                    .overlay(
                        RoundedRectangle(cornerRadius: 16)
                            .stroke(Color.black.opacity(0.06), lineWidth: 1)
                    )

                Button {
                    viewModel.generate()
                } label: {
                    Image(systemName: viewModel.isRunning ? "waveform" : "play.fill")
                        .font(.subheadline.weight(.bold))
                        .foregroundStyle(.white)
                        .frame(width: 46, height: 46)
                        .background(modelAccent(viewModel.selectedModel), in: Circle())
                }
                .buttonStyle(.plain)
                .disabled(!viewModel.canGenerate)
                .opacity(viewModel.canGenerate ? 1 : 0.45)
            }
        }
        .padding(14)
        .background(.white.opacity(0.76), in: RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .stroke(Color.white.opacity(0.72), lineWidth: 1)
        )
    }

    private var musicPromptRow: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Music Prompts")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 14) {
                    ForEach(musicPrompts) { preset in
                        musicCard(preset)
                    }
                }
            }
        }
    }

    private var sfxPromptPad: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle("Sound Effects")
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 12), count: 5), spacing: 12) {
                ForEach(sfxPrompts) { preset in
                    sfxButton(preset)
                }
            }
        }
    }

    private var statusHint: some View {
        Group {
            if viewModel.showPreparationHint {
                Text("Model files are missing. Run the weight preparation script first.")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(Color(red: 0.75, green: 0.34, blue: 0.08))
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(red: 1.0, green: 0.9, blue: 0.78), in: RoundedRectangle(cornerRadius: 14))
            }
        }
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
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(Color(red: 0.08, green: 0.09, blue: 0.11))
                    .lineLimit(1)
                Text(preset.subtitle)
                    .font(.caption)
                    .foregroundStyle(Color.black.opacity(0.45))
                    .lineLimit(2)
            }
            .frame(width: 148, height: 96, alignment: .leading)
            .padding(14)
            .background(preset.accent.opacity(0.13), in: RoundedRectangle(cornerRadius: 18))
            .overlay(
                RoundedRectangle(cornerRadius: 18)
                    .stroke(preset.accent.opacity(0.28), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .disabled(!viewModel.canGenerate)
    }

    private func sfxButton(_ preset: PromptPreset) -> some View {
        Button {
            viewModel.generatePreset(preset)
        } label: {
            VStack(spacing: 6) {
                Image(systemName: preset.iconName)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(preset.accent)
                Text(preset.title)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(Color(red: 0.08, green: 0.09, blue: 0.11))
                    .lineLimit(1)
            }
            .frame(height: 56)
            .frame(maxWidth: .infinity)
            .background(.white.opacity(0.78), in: RoundedRectangle(cornerRadius: 16))
            .overlay(
                RoundedRectangle(cornerRadius: 16)
                    .stroke(preset.accent.opacity(0.25), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .disabled(!viewModel.canGenerate)
    }

    private func sectionTitle(_ text: String) -> some View {
        Text(text.uppercased())
            .font(.caption2.weight(.bold))
            .foregroundStyle(Color.black.opacity(0.42))
            .tracking(1.4)
    }

    private func modelChip(_ model: StableAudioModelKind) -> some View {
        Button {
            viewModel.selectedModel = model
        } label: {
            Text(model.title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(viewModel.selectedModel == model ? .white : Color.black.opacity(0.55))
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(
                    viewModel.selectedModel == model
                        ? modelAccent(model)
                        : Color.white.opacity(0.72),
                    in: Capsule()
                )
        }
        .buttonStyle(.plain)
    }

    private func modelAccent(_ model: StableAudioModelKind) -> Color {
        switch model {
        case .smallMusic:
            return Color(red: 0.05, green: 0.68, blue: 0.48)
        case .smallSFX:
            return Color(red: 0.74, green: 0.35, blue: 0.86)
        }
    }

    private func durationChip(_ seconds: Int) -> some View {
        Button {
            viewModel.durationSeconds = Float(seconds)
        } label: {
            Text("\(seconds)s")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(viewModel.durationSeconds == Float(seconds) ? .white : Color.black.opacity(0.55))
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(
                    viewModel.durationSeconds == Float(seconds)
                        ? Color(red: 0.05, green: 0.68, blue: 0.48)
                        : Color.white.opacity(0.72),
                    in: Capsule()
                )
        }
        .buttonStyle(.plain)
    }

    private func qualityChip(_ title: String, steps: Int) -> some View {
        Button {
            viewModel.stepCount = steps
        } label: {
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(viewModel.stepCount == steps ? .white : Color.black.opacity(0.55))
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(
                    viewModel.stepCount == steps
                        ? Color(red: 0.94, green: 0.58, blue: 0.12)
                        : Color.white.opacity(0.72),
                    in: Capsule()
                )
        }
        .buttonStyle(.plain)
    }

    private func waveColor(_ index: Int) -> Color {
        let colors = [
            Color(red: 0.05, green: 0.68, blue: 0.48),
            Color(red: 0.24, green: 0.48, blue: 0.9),
            Color(red: 0.94, green: 0.58, blue: 0.12),
            Color(red: 0.92, green: 0.28, blue: 0.34),
            Color(red: 0.62, green: 0.38, blue: 0.85),
        ]
        return colors[index % colors.count]
    }
}

struct PromptPreset: Identifiable {
    var id: String { title }
    let title: String
    let subtitle: String
    let model: StableAudioModelKind
    let prompt: String
    let duration: Float
    let steps: Int
    let iconName: String
    let accent: Color
}

#Preview {
    ContentView()
}
