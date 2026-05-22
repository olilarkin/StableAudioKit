# stableaudio3-ios

[English](README.md) | [简体中文](README.zh-CN.md)

Run Stable Audio 3 Small Music locally on iPhone with MLX Swift.

This project is an iOS app and runtime for text-to-audio generation on Apple
devices. It converts the official Stable Audio 3 optimized MLX checkpoints into
app-bundled safetensors, loads them with MLX Swift, generates audio on device,
writes a WAV file, and plays it back in the app.

## What It Can Do

- Generate stereo audio from a text prompt directly on iPhone
- Run the Stable Audio 3 `small-music` path with the `same-s` decoder
- Test short creative loops such as drum grooves, texture beds, riffs, and music cues
- Choose short duration presets for latency testing: 1s, 2s, 5s, 10s, 15s
- Choose 4 or 8 sampling steps
- Keep the pipeline warm after the first run, so later generations reuse loaded weights
- Print per-stage timing logs in Xcode for T5, DiT sampling, decoder, and WAV writing

Current scope:

- Text-to-audio only
- Stable Audio 3 Small Music only
- iOS 17+ device target
- Local on-device inference, no server required

## Quick Start

```bash
git clone https://github.com/kellyvv/StableAudio3-IOS.git
cd StableAudio3-IOS
```

Install the small local tools:

```bash
brew install xcodegen
python3 -m venv .venv
source .venv/bin/activate
pip install -U mlx numpy huggingface_hub
```

Log in to Hugging Face with an account that has accepted the upstream Stable
Audio 3 and Gemma terms:

```bash
hf auth login
```

Download the official optimized MLX checkpoints:

```bash
hf download stabilityai/stable-audio-3-optimized \
  --include "MLX/t5gemma_f16.npz" \
  --include "MLX/dit_sm-music_f16.npz" \
  --include "MLX/same_s_decoder_f32.npz" \
  --local-dir Models/stable-audio-3-optimized
```

Convert the weights for the iOS app:

```bash
python3 Scripts/prepare_weights.py
```

Generate and open the Xcode project:

```bash
xcodegen generate
open StableAudio3iOS.xcodeproj
```

In Xcode, select your development team, choose a real iPhone target, then run
the `StableAudio3iOS` scheme. The app will show the staged weights and a
`Generate & Play` button.

## Command-Line Build Check

You can verify the project without signing:

```bash
xcodebuild -quiet \
  -project StableAudio3iOS.xcodeproj \
  -scheme StableAudio3iOS \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

## Weight Files

This repository does not include model weights. After conversion, the app uses:

```text
Resources/Weights/t5gemma_f16.safetensors
Resources/Weights/dit_sm-music_f16.safetensors
Resources/Weights/same_s_decoder_f32.safetensors
Resources/Weights/t5gemma_tokenizer.model
Resources/Weights/sa3_conditioner.safetensors
Resources/Weights/manifest.json
```

These generated files are ignored by git.

## Timing Logs

Run from Xcode and tap `Generate & Play`. The first run includes model loading:

```text
[SA3] cache miss DiT, loading weights
[SA3] step 1/4 320ms total=...
[SA3] total 1800ms prompt="..." seconds=1.0 steps=4 latentLength=...
```

Run again to measure warm latency. Warm runs should show cache hits:

```text
[SA3] cache hit tokenizer
[SA3] cache hit T5Gemma
[SA3] cache hit DiT
[SA3] cache hit decoder
```

For backend-style experiments, use 1s / 4 steps first, then increase duration or
steps if the quality/latency tradeoff is acceptable.

## Project Layout

```text
StableAudio3iOS/          SwiftUI app and MLX Swift runtime
Scripts/prepare_weights.py  NPZ -> safetensors conversion
Resources/Weights/        Local generated weights, ignored by git
project.yml               XcodeGen project spec
```

## Notes

- Use a real iPhone. Simulator does not represent the target MLX/Metal path.
- The first generation is a cold run. Use the second or third run for latency numbers.
- The app bundle is large when weights are staged locally, roughly 1.6 GB before app overhead.
- This is a practical prototype runtime, not a full Stable Audio 3 product surface.

## License

Repository code is licensed under MIT.

Model weights are not included and are not covered by this repository license.
Stable Audio 3 weights are subject to the Stability AI Community License.
T5Gemma components are subject to the Gemma Terms of Use.

Read `NOTICE` and `THIRD_PARTY_LICENSES.md` before downloading, converting, or
distributing any model weights.
