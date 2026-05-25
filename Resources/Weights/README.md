# Weights

This directory is intentionally committed without model weights.

The CLI app expects the official `stable-audio-3-optimized` small-music and
small-sfx MLX weights converted from NPZ to safetensors:

- `t5gemma_f16.safetensors`
- `dit_sm-music_f16.safetensors`
- `dit_sm-sfx_f16.safetensors`
- `same_s_decoder_f32.safetensors`
- `t5gemma_tokenizer.model`
- `sa3_conditioner_sm-music.safetensors`
- `sa3_conditioner_sm-sfx.safetensors`

Download the official weights after accepting the upstream licenses, then run
`Scripts/prepare_weights.py` from the project folder. Generated files stay local
and are ignored by git.
