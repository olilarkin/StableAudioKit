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

## Current Scope

The package API intentionally separates model loading, request configuration, sampling, and audio writing. The current inference implementation is text-to-audio only. Audio-to-audio, in-painting, negative conditioning, and mask-conditioned generation should be added as new request modes backed by additional encoder/conditioning components, not by changing where apps store or authorize model files.
