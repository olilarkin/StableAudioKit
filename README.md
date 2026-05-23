# stableaudio3-ios

[English](README.md) | [简体中文](README.zh-CN.md)

Generate music and sound effects on iPhone with Stable Audio 3.

This is an iOS app and runtime for running Stable Audio 3 locally with MLX
Swift. You type a prompt, tap play, and the app generates a short
stereo WAV on the phone. No cloud server is needed.

The current app runs `small-music` and `small-sfx`. The next practical target is
`medium`. The largest models are not the target here.

## Demo

https://github.com/user-attachments/assets/5435773b-fb4b-492e-ac8e-33a8d211979b

## Why This Exists

Stable Audio 3 already has an official MLX path on Mac. This project shows how
to move that path onto iPhone:

- convert the official MLX weights into iOS-friendly files
- load the model in an iOS app
- generate audio fully on device
- keep the model warm so repeated generations are faster
- log generation time for real device testing

## What You Can Try

- drum grooves
- short music loops
- sound effects
- ambient textures
- one-shot audio ideas

Start with `1s` and `4 steps` to test latency. Use longer durations or more
steps when quality matters more than speed.

## Quick Start

Clone the repo:

```bash
git clone https://github.com/kellyvv/StableAudio3-IOS.git
cd StableAudio3-IOS
```

Install tools:

```bash
brew install xcodegen
python3 -m venv .venv
source .venv/bin/activate
pip install -U mlx numpy huggingface_hub
```

Log in to Hugging Face:

```bash
hf auth login
```

You need access to the Stable Audio 3 weights on Hugging Face and must accept
the upstream Stability AI and Gemma terms first.

Download the official MLX weights:

```bash
hf download stabilityai/stable-audio-3-optimized \
  --include "MLX/t5gemma_f16.npz" \
  --include "MLX/dit_sm-music_f16.npz" \
  --include "MLX/dit_sm-sfx_f16.npz" \
  --include "MLX/same_s_decoder_f32.npz" \
  --local-dir Models/stable-audio-3-optimized
```

Convert them for the iOS app:

```bash
python3 Scripts/prepare_weights.py
```

Open the app:

```bash
xcodegen generate
open StableAudio3iOS.xcodeproj
```

In Xcode, choose your development team, select a real iPhone, and run. When the
app opens, choose Music or SFX, then tap play.

## What Gets Generated Locally

The conversion script creates these files under `Resources/Weights/`:

```text
t5gemma_f16.safetensors
dit_sm-music_f16.safetensors
dit_sm-sfx_f16.safetensors
same_s_decoder_f32.safetensors
t5gemma_tokenizer.model
sa3_conditioner_sm-music.safetensors
sa3_conditioner_sm-sfx.safetensors
manifest.json
```

They are ignored by git. This repo does not ship model weights.

## Timing

The first generation loads the model, so it is slower. Run again to measure the
warm speed. Xcode logs look like this:

```text
[SA3] cache hit DiT
[SA3] step 1/4 320ms total=...
[SA3] total 1800ms model=Small SFX prompt="..." seconds=1.0 steps=4 latentLength=...
```

## Build Check

You can check the project without signing:

```bash
xcodebuild -quiet \
  -project StableAudio3iOS.xcodeproj \
  -scheme StableAudio3iOS \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO \
  build
```

## Notes

- Use a real iPhone. The simulator is not useful for this.
- The current working models are `small-music` and `small-sfx`.
- `medium` is the next target.
- The app becomes large when local weights are added, roughly 2.5 GB for both small models.

## License

The code in this repo is MIT licensed.

Model weights are not included. Stable Audio 3 weights use the Stability AI
Community License. T5Gemma uses the Gemma Terms of Use. Read `NOTICE` and
`THIRD_PARTY_LICENSES.md` before downloading, converting, or distributing
weights.
