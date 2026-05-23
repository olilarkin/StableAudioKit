# stableaudio3-ios

On-device Stable Audio 3 generation for iPhone, powered by MLX Swift.

[English](README.md) | [简体中文](README.zh-CN.md)

<table align="center">
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/5435773b-fb4b-492e-ac8e-33a8d211979b" controls width="360"></video>
    </td>
  </tr>
</table>

## What It Does

This repo contains an iOS app and runtime for running Stable Audio 3 locally on
iPhone. Type a prompt, choose `Music` or `SFX`, tap play, and the phone generates
a stereo WAV locally.

- No server
- No streaming backend
- Music loops, drum one-shots, and sound effects
- `small-music` and `small-sfx` supported now
- `medium` is the next practical target

The app uses a shared T5Gemma text encoder and SAME-S decoder. Switching between
`Music` and `SFX` switches the DiT model.

## Quick Start

Clone the repo:

```bash
git clone https://github.com/kellyvv/StableAudio3-IOS.git
cd StableAudio3-IOS
```

Install local tools:

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

Generate the Xcode project and run on a real iPhone:

```bash
xcodegen generate
open StableAudio3iOS.xcodeproj
```

In Xcode, select your development team and run the app on device.

## App Modes

| Mode | Model | Best For |
| --- | --- | --- |
| Music | `dit_sm-music_f16` | loops, grooves, tonal ideas |
| SFX | `dit_sm-sfx_f16` | sound effects, drum hits, short Foley |

Quality options use the same sampler:

| Option | Steps | Use |
| --- | ---: | --- |
| Fast | 4 | quick tests |
| Better | 8 | default quality |
| Best | 16 | slower, cleaner generations |

Drum hit presets use 2 steps so you can test very low latency one-shots.

## Local Weight Files

`Scripts/prepare_weights.py` creates these files under `Resources/Weights/`:

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

## Timing Logs

The first generation loads weights, so it is slower. Run again to measure warm
speed. Xcode logs look like this:

```text
[SA3] cache hit DiT Small SFX
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

- Use a real iPhone. The simulator is not useful for this runtime.
- Bundling both small models makes the app large, roughly 2.5 GB of local model files.
- The largest Stable Audio 3 models are not the target of this project.

## License

The code in this repo is MIT licensed.

Model weights are not included. Stable Audio 3 weights use the Stability AI
Community License. T5Gemma uses the Gemma Terms of Use. Read `NOTICE` and
`THIRD_PARTY_LICENSES.md` before downloading, converting, distributing, or using
weights.
