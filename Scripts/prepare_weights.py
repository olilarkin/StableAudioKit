#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np


FILES = [
    {
        "role": "T5Gemma text encoder",
        "sourceFileName": "t5gemma_f16.npz",
        "fileName": "t5gemma_f16.safetensors",
    },
    {
        "role": "DiT small-music",
        "sourceFileName": "dit_sm-music_f16.npz",
        "fileName": "dit_sm-music_f16.safetensors",
    },
    {
        "role": "same-s decoder",
        "sourceFileName": "same_s_decoder_f32.npz",
        "fileName": "same_s_decoder_f32.safetensors",
    },
]

TOKENIZER_FILE = "t5gemma_tokenizer.model"
CONDITIONER_FILE = "sa3_conditioner.safetensors"


def convert_file(source: Path, target: Path) -> int:
    print(f"loading {source}")
    arrays = dict(mx.load(str(source)))
    print(f"saving {target} ({len(arrays)} tensors)")
    mx.save_safetensors(str(target), arrays)
    return target.stat().st_size


def extract_tokenizer(source: Path, target: Path) -> int:
    print(f"extracting tokenizer {target}")
    with np.load(source) as archive:
        tokenizer = archive["TOKENIZER_MODEL"].tobytes()
    target.write_bytes(tokenizer)
    return target.stat().st_size


def extract_conditioner(source: Path, target: Path) -> int:
    print(f"extracting conditioner {target}")
    source_arrays = dict(mx.load(str(source)))
    conditioner = {
        key: source_arrays[key]
        for key in (
            "cond.padding_embedding",
            "cond.seconds_total_weight",
            "cond.seconds_total_bias",
        )
    }
    mx.save_safetensors(str(target), conditioner)
    return target.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Models/stable-audio-3-optimized/MLX"),
        help="Directory containing official optimized/mlx NPZ files.",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("Resources/Weights"),
        help="Directory where iOS safetensors resources will be written.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rewrite safetensors files that already exist.",
    )
    args = parser.parse_args()

    source_dir = args.source.resolve()
    destination_dir = args.destination.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    manifest_files = []
    for item in FILES:
        source = source_dir / item["sourceFileName"]
        target = destination_dir / item["fileName"]
        if not source.exists():
            raise FileNotFoundError(source)

        if args.skip_existing and target.exists():
            size = target.stat().st_size
        else:
            size = convert_file(source, target)

        manifest_files.append(
            {
                "role": item["role"],
                "fileName": item["fileName"],
                "minimumBytes": size,
                "sourceFileName": item["sourceFileName"],
            }
        )

    tokenizer_source = source_dir / "t5gemma_f16.npz"
    tokenizer_target = destination_dir / TOKENIZER_FILE
    if args.skip_existing and tokenizer_target.exists():
        tokenizer_size = tokenizer_target.stat().st_size
    else:
        tokenizer_size = extract_tokenizer(tokenizer_source, tokenizer_target)

    conditioner_source = source_dir / "dit_sm-music_f16.npz"
    conditioner_target = destination_dir / CONDITIONER_FILE
    if args.skip_existing and conditioner_target.exists():
        conditioner_size = conditioner_target.stat().st_size
    else:
        conditioner_size = extract_conditioner(conditioner_source, conditioner_target)

    manifest = {
        "model": "stabilityai/stable-audio-3-small-music",
        "format": "safetensors",
        "tokenizer": {
            "fileName": TOKENIZER_FILE,
            "bytes": tokenizer_size,
            "tokenOffset": 0,
        },
        "conditioner": {
            "fileName": CONDITIONER_FILE,
            "bytes": conditioner_size,
        },
        "files": manifest_files,
    }
    manifest_path = destination_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
