# stableaudio3-ios

Stable Audio 3 Small Music on iOS with MLX Swift.

This repository contains an iOS demo/runtime, weight conversion script, and
Swift MLX ports for the small-music text-to-audio path:

```text
prompt -> T5Gemma -> conditioning -> DiT sampler -> SAME-S decoder -> WAV playback
```

The default latency-test preset is 1 second, 4 sampling steps. The first
generation loads weights into a long-lived pipeline actor; later generations
reuse tokenizer, T5Gemma, conditioner, DiT, and decoder weights and print stage
timing to the Xcode console.

## What Is Included

- SwiftUI iOS app target
- MLX Swift implementation for T5Gemma, SA3 conditioning, DiT small-music,
  SAME-S decoder, sampler, WAV writer, and playback
- Python conversion script from official MLX NPZ checkpoints to app-bundled
  safetensors resources
- XcodeGen project spec

## What Is Not Included

Model weights are not included in this repository.

Generated files such as these are git-ignored:

```text
Resources/Weights/t5gemma_f16.safetensors
Resources/Weights/dit_sm-music_f16.safetensors
Resources/Weights/same_s_decoder_f32.safetensors
Resources/Weights/t5gemma_tokenizer.model
Resources/Weights/sa3_conditioner.safetensors
Resources/Weights/manifest.json
```

Download and use model weights only after accepting the upstream Stability AI
and Gemma terms.

## Requirements

- macOS with Xcode 16+
- iOS 17+ device with Apple GPU
- XcodeGen
- Python environment with `mlx` and `numpy`
- Hugging Face access to `stabilityai/stable-audio-3-optimized`

## Prepare Weights

Install the Hugging Face CLI and log in with an account that has accepted the
upstream model terms.

Download the official optimized MLX checkpoints into the local ignored
`Models/` directory:

```bash
hf download stabilityai/stable-audio-3-optimized \
  --include "MLX/t5gemma_f16.npz" \
  --include "MLX/dit_sm-music_f16.npz" \
  --include "MLX/same_s_decoder_f32.npz" \
  --local-dir Models/stable-audio-3-optimized
```

Convert the checkpoints for the iOS app:

```bash
python3 Scripts/prepare_weights.py
```

The script writes the generated resources under `Resources/Weights/`.

## Build

Generate the Xcode project:

```bash
xcodegen generate
```

Build from the command line:

```bash
xcodebuild -quiet \
  -project StableAudio3iOS.xcodeproj \
  -scheme StableAudio3iOS \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

For a real device install, open `StableAudio3iOS.xcodeproj`, select your
development team, and run the `StableAudio3iOS` scheme on an iPhone.

## Timing Logs

Run the app from Xcode and tap `Generate & Play`. The console prints cold and
warm timing:

```text
[SA3] cache miss DiT, loading weights
[SA3] step 1/4 320ms total=...
[SA3] total 1800ms prompt="..." seconds=1.0 steps=4 latentLength=...
```

Run a second or third generation to measure warm pipeline latency. Warm runs
should show cache hits for tokenizer, T5Gemma, conditioner, DiT, and decoder.

## License

Repository code is licensed under MIT. Model weights are not included and are
not covered by this repository license.

See `NOTICE` and `THIRD_PARTY_LICENSES.md` before downloading, converting, or
distributing any model weights.
