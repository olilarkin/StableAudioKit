# Third-Party Licenses

This repository contains application/runtime code only. It does not include
model weights.

## Runtime Dependencies

- MLX Swift: https://github.com/ml-explore/mlx-swift
- swift-sentencepiece: https://github.com/jkrukowski/swift-sentencepiece

Swift package revisions are pinned in `project.yml`.

## Model Dependencies

The app can run converted weights derived from:

- Stability AI Stable Audio 3 Small Music
- Stability AI Stable Audio 3 Optimized MLX checkpoints
- Google T5Gemma, redistributed with Stable Audio 3 for text conditioning

These model files are not licensed under this repository's MIT license. Users
must accept and follow the upstream licenses before downloading or converting
weights:

- Stability AI Community License: https://stability.ai/license
- Gemma Terms of Use: https://ai.google.dev/gemma/terms

Generated files under `Resources/Weights/` are intentionally ignored by git.
