# StableAudioKit

Swift package and CLI/demo app for running Stable Audio 3 MLX inference on Apple Silicon without a Python runtime dependency.

Supported text-to-audio variants:

- **`smallMusic`** and **`smallSFX`** — ~0.6 B params, SAME-S decoder. Runs on Apple Silicon Mac, iOS, and visionOS.
- **`medium`** — 1.4 B params, SAME-L decoder. macOS only (~5–6.5 GB peak memory). Higher quality.

Components shared across variants:

- T5Gemma prompt encoding
- Stable Audio 3 conditioning (text + duration)
- DiT sampling (differential attention for medium)
- SAME-S / SAME-L decoder
- WAV writing for CLI/demo output

The package is structured so audio-to-audio, in-painting, masks, and future Stable Audio Open 3 features can be added without changing the public loading model: apps create a `StableAudioPipeline` from a prepared model directory, then pass typed generation requests.

## Requirements

- macOS 14+ on Apple Silicon for the CLI/demo
- Xcode with SwiftPM
- Prepared Stable Audio 3 MLX weights in safetensors format
- For weight download: access granted to the gated Stability AI model on Hugging Face

The Swift library does not depend on Python at runtime. `Scripts/prepare_weights.py` is a developer preparation tool used to download and convert official NPZ weights into local safetensors files.
Python 3.10 or newer is fine for preparation; use a virtual environment so the `hf` CLI and its dependencies are installed consistently.

### MLX Metal library

`swift build` does not compile `.metal` shader files, so the MLX Metal library must be built once as a separate step before the CLI will run. `Scripts/compile-mlx-metallib.sh` handles this — see the CLI setup instructions below.

## Package Layout

- `Sources/StableAudioKit` - reusable Swift library
- `Sources/StableAudioCLI` - command line generator
- `Example/SA3CLI` - tiny package using the library as a dependency
- `Scripts/prepare_weights.py` - authorized download and conversion helper
- `Scripts/compile-mlx-metallib.sh` - compiles MLX Metal shaders for SwiftPM CLI builds

## Model Authorization and Preparation

Stable Audio 3 weights are gated. Before downloading, accept the upstream model terms on Hugging Face for the Stability AI repo, then authenticate locally.

Recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
hf auth login
```

Alternatively, set `HF_TOKEN` to a token that has access to the gated repo.

Download and convert:

```bash
python Scripts/prepare_weights.py --download
```

If you already have the official optimized MLX NPZ files locally:

```bash
python Scripts/prepare_weights.py \
  --source Models/stable-audio-3-optimized/MLX \
  --destination Resources/Weights
```

By default the script prepares the small variants. To also prepare medium (macOS only):

```bash
python Scripts/prepare_weights.py --download --variants small medium
```

Generated model files are ignored by git. The expected prepared files are:

Common:

- `t5gemma_f16.safetensors`
- `t5gemma_tokenizer.model`
- `manifest.json`

Small (`smallMusic`, `smallSFX`):

- `dit_sm-music_f16.safetensors`
- `dit_sm-sfx_f16.safetensors`
- `same_s_decoder_f32.safetensors`
- `sa3_conditioner_sm-music.safetensors`
- `sa3_conditioner_sm-sfx.safetensors`

Medium (macOS only):

- `dit_medium_f16.safetensors` (~2.9 GB)
- `same_l_decoder_f32.safetensors` (~1.7 GB)
- `sa3_conditioner_medium.safetensors`

## CLI Usage

### First-time setup

Build the CLI binary, then compile the MLX Metal shaders. The compiled
`mlx.metallib` is placed next to the binary and found automatically at runtime.

```bash
swift build --product StableAudioCLI
./Scripts/compile-mlx-metallib.sh
```

Re-run `compile-mlx-metallib.sh` whenever the mlx-swift dependency is updated.
The metallib is not model-specific — it contains MLX's core GPU kernels and
is reusable across any project on the same mlx-swift version.

### Generating audio

```bash
swift run StableAudioCLI \
  --model smallMusic \
  --prompt "lofi house loop, 120 BPM" \
  --duration 10 \
  --steps 8 \
  --seed 42 \
  --model-path Resources/Weights \
  -o output.wav
```

For sound effects:

```bash
swift run StableAudioCLI \
  --model smallSFX \
  --prompt "single crisp snare drum hit, dry studio room" \
  --duration 1 \
  --steps 4 \
  -o snare.wav
```

For the medium model (macOS only):

```bash
swift run StableAudioCLI \
  --model medium \
  --prompt "warm arpeggios over a house beat, 124 BPM" \
  --duration 10 \
  --steps 8 \
  --seed 42 \
  --model-path Resources/Weights \
  -o output.wav
```

## Swift API

```swift
import StableAudioKit

let pipeline = try StableAudioPipeline.load(from: modelDirectoryURL)
let request = StableAudioGenerationRequest(
    model: .smallMusic,
    prompt: "warm arpeggios over a house beat, 124 BPM",
    seconds: 10,
    steps: 8,
    seed: 42
)

let result = try await pipeline.generate(request) { progress in
    print(progress)
}

try AudioWriter.write(result, to: outputURL)
```

### Audio-to-audio

Pass an existing audio file (or already-decoded PCM) as `initAudio` and a noise level in `[0, 1]` to restyle a recording with a text prompt, mirroring the upstream Python `model.generate(init_audio=..., init_noise_level=0.9, ...)` API. A level of 1.0 ignores the input (equivalent to text-to-audio); 0.0 returns roughly the input audio through the autoencoder.

```swift
let request = StableAudioGenerationRequest(
    model: .smallMusic,
    prompt: "bossa nova bassline",
    seconds: 30,
    initAudio: .url(sourceAudioURL),
    initNoiseLevel: 0.9
)
let result = try await pipeline.generate(request)
```

Or from the CLI:

```bash
swift run StableAudioCLI \
  --prompt "bossa nova bassline" \
  --init-audio source.wav \
  --init-noise-level 0.9 \
  --duration 30 \
  -o restyled.wav
```

Audio-to-audio requires the SAME encoder weights in addition to the standard text-to-audio files. Prepare them with `--with-encoder`:

```bash
python Scripts/prepare_weights.py --download --with-encoder
```

The encoder file is loaded lazily on first audio-to-audio request; text-to-audio generation does not need it.

### Inpainting / continuation

Pass `initAudio` together with one or more `InpaintRegion`s to regenerate specific time ranges while keeping the rest of the source intact. This mirrors the upstream Python `model.generate(inpaint_audio=..., inpaint_mask_start_seconds=..., inpaint_mask_end_seconds=..., duration=...)` flow and reuses the SAME encoder. Regions are given over the *output* timeline in seconds; they may be unordered and overlap (they are sorted and merged automatically). When `inpaintRegions` is set, `initNoiseLevel` is ignored and the full diffusion schedule runs — masked frames are regenerated, unmasked frames are reseeded from the encoded source at every step.

```swift
// Regenerate seconds 4–8 of a 30 s source.
let request = StableAudioGenerationRequest(
    model: .smallMusic,
    prompt: "punchy kick drum fill",
    seconds: 30,
    initAudio: .url(sourceAudioURL),
    inpaintRegions: [InpaintRegion(startSeconds: 4, endSeconds: 8)]
)
```

Continuation is just an inpaint region starting at the source's length and ending at the requested duration. Short sources are zero-padded internally to the requested duration before encoding, so the trailing region behaves as pure regeneration conditioned on the leading audio:

```swift
// Extend an 8 s source out to 30 s.
let request = StableAudioGenerationRequest(
    model: .smallMusic,
    prompt: "warm pad continues",
    seconds: 30,
    initAudio: .url(eightSecondAudioURL),
    inpaintRegions: [InpaintRegion(startSeconds: 8, endSeconds: 30)]
)
```

From the CLI, regions are supplied as parallel lists:

```bash
# Inpaint a single region.
swift run StableAudioCLI \
  --prompt "punchy kick drum fill" \
  --init-audio source.wav \
  --duration 30 \
  --inpaint-mask-start 4 \
  --inpaint-mask-end 8 \
  -o inpainted.wav

# Continuation: extend an 8 s clip out to 30 s.
swift run StableAudioCLI \
  --prompt "warm pad continues" \
  --init-audio src.wav \
  --duration 30 \
  --inpaint-mask-start 8 \
  --inpaint-mask-end 30 \
  -o continued.wav

# Multiple non-contiguous regions.
swift run StableAudioCLI \
  --prompt "punchy kick drum fill" \
  --init-audio source.wav \
  --duration 30 \
  --inpaint-mask-start 4 16 \
  --inpaint-mask-end 8 20 \
  -o inpainted.wav
```

Inpainting requires the same SAME encoder weights as audio-to-audio — prepare them once with `python Scripts/prepare_weights.py --download --with-encoder`.

## Current Scope

The package API intentionally separates model loading, request configuration, sampling, and audio writing. Inference modes currently supported: text-to-audio, audio-to-audio, and inpainting / continuation. All three modes are exposed through both the Swift `StableAudioPipeline` API and the C ABI declared in `Scripts/xcframework-resources/StableAudioKit.h` (entry points `stable_audio_generate`, `stable_audio_generate_a2a`, `stable_audio_generate_inpaint`). Both surfaces ship in every slice of the `StableAudioKit.xcframework` produced by `Scripts/build-xcframework.sh` (macOS, iOS, iOS-simulator, visionOS, visionOS-simulator).
