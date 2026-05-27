#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import mlx.core as mx
import numpy as np


COMMON_FILES = [
    {
        "role": "T5Gemma text encoder",
        "sourceFileName": "t5gemma_f16.npz",
        "fileName": "t5gemma_f16.safetensors",
    },
]

SMALL_FILES = [
    {
        "role": "DiT small-music",
        "sourceFileName": "dit_sm-music_f16.npz",
        "fileName": "dit_sm-music_f16.safetensors",
    },
    {
        "role": "DiT small-sfx",
        "sourceFileName": "dit_sm-sfx_f16.npz",
        "fileName": "dit_sm-sfx_f16.safetensors",
    },
    {
        "role": "same-s decoder",
        "sourceFileName": "same_s_decoder_f32.npz",
        "fileName": "same_s_decoder_f32.safetensors",
    },
]

MEDIUM_FILES = [
    {
        "role": "DiT medium",
        "sourceFileName": "dit_medium_f16.npz",
        "fileName": "dit_medium_f16.safetensors",
    },
    {
        "role": "same-l decoder",
        "sourceFileName": "same_l_decoder_f32.npz",
        "fileName": "same_l_decoder_f32.safetensors",
    },
]

TOKENIZER_FILE = "t5gemma_tokenizer.model"
LEGACY_CONDITIONER_FILE = "sa3_conditioner.safetensors"

SMALL_CONDITIONER_FILES = [
    {
        "role": "Conditioner small-music",
        "sourceFileName": "dit_sm-music_f16.npz",
        "fileName": "sa3_conditioner_sm-music.safetensors",
    },
    {
        "role": "Conditioner small-sfx",
        "sourceFileName": "dit_sm-sfx_f16.npz",
        "fileName": "sa3_conditioner_sm-sfx.safetensors",
    },
]

MEDIUM_CONDITIONER_FILES = [
    {
        "role": "Conditioner medium",
        "sourceFileName": "dit_medium_f16.npz",
        "fileName": "sa3_conditioner_medium.safetensors",
    },
]


def ensure_huggingface_auth() -> None:
    if os.environ.get("HF_TOKEN"):
        return
    hf = shutil.which("hf")
    if not hf:
        raise RuntimeError(
            "Gated model download requires authorization. Install the Hugging Face CLI "
            "(`hf`) and run `hf auth login`, or set HF_TOKEN to a token that has access "
            "to the Stability AI gated model."
        )
    result = subprocess.run([hf, "auth", "whoami"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Gated model download requires authorization. Run `hf auth login` after "
            "accepting the model terms on Hugging Face, or set HF_TOKEN to an authorized token."
        )


def download_weights(repo_id: str, target: Path, revision: str | None) -> None:
    ensure_huggingface_auth()
    hf = shutil.which("hf")
    if not hf:
        raise RuntimeError("Hugging Face CLI `hf` is required for downloads.")

    command = [
        hf,
        "download",
        repo_id,
        "--include",
        "MLX/*",
        "--local-dir",
        str(target),
    ]
    if revision:
        command.extend(["--revision", revision])
    print("downloading gated weights with Hugging Face authorization")
    subprocess.run(command, check=True)


def convert_file(source: Path, target: Path) -> int:
    print(f"loading {source}")
    arrays = dict(mx.load(str(source)))
    # SAME-L stores mapping.weight as PyTorch Conv1d [out, in, 1]; flatten to
    # nn.Linear [out, in] so the Swift loader can use it as a regular matmul.
    if "mapping.weight" in arrays:
        w = arrays["mapping.weight"]
        if w.ndim == 3 and w.shape[-1] == 1:
            arrays["mapping.weight"] = w.reshape(w.shape[0], w.shape[1])
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


def can_skip(target: Path) -> bool:
    return target.exists() and target.stat().st_size > 0


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
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download gated weights with `hf download` before conversion.",
    )
    parser.add_argument(
        "--repo-id",
        default="stabilityai/stable-audio-3-optimized",
        help="Hugging Face repo id containing the official MLX weights.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, tag, or commit to download.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["small", "medium"],
        default=["small"],
        help="Model size(s) to prepare. Defaults to the small variants.",
    )
    args = parser.parse_args()

    variants = set(args.variants)

    source_dir = args.source.resolve()
    destination_dir = args.destination.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_target = source_dir.parent
        download_target.mkdir(parents=True, exist_ok=True)
        download_weights(args.repo_id, download_target, args.revision)

    files_to_convert = list(COMMON_FILES)
    conditioner_sources = []
    if "small" in variants:
        files_to_convert.extend(SMALL_FILES)
        conditioner_sources.extend(SMALL_CONDITIONER_FILES)
    if "medium" in variants:
        files_to_convert.extend(MEDIUM_FILES)
        conditioner_sources.extend(MEDIUM_CONDITIONER_FILES)

    manifest_files = []
    for item in files_to_convert:
        source = source_dir / item["sourceFileName"]
        target = destination_dir / item["fileName"]
        if not source.exists():
            raise FileNotFoundError(source)

        if args.skip_existing and can_skip(target):
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
    if args.skip_existing and can_skip(tokenizer_target):
        tokenizer_size = tokenizer_target.stat().st_size
    else:
        tokenizer_size = extract_tokenizer(tokenizer_source, tokenizer_target)
    manifest_files.append(
        {
            "role": "T5Gemma tokenizer",
            "fileName": TOKENIZER_FILE,
            "minimumBytes": tokenizer_size,
            "sourceFileName": "t5gemma_f16.npz:TOKENIZER_MODEL",
        }
    )

    conditioner_manifest = []
    for item in conditioner_sources:
        conditioner_source = source_dir / item["sourceFileName"]
        conditioner_target = destination_dir / item["fileName"]
        if not conditioner_source.exists():
            raise FileNotFoundError(conditioner_source)

        if args.skip_existing and can_skip(conditioner_target):
            conditioner_size = conditioner_target.stat().st_size
        else:
            conditioner_size = extract_conditioner(conditioner_source, conditioner_target)

        manifest_item = {
            "role": item["role"],
            "fileName": item["fileName"],
            "minimumBytes": conditioner_size,
            "sourceFileName": item["sourceFileName"],
        }
        manifest_files.append(manifest_item)
        conditioner_manifest.append(manifest_item)

        if item["fileName"] == "sa3_conditioner_sm-music.safetensors":
            legacy_target = destination_dir / LEGACY_CONDITIONER_FILE
            if not (args.skip_existing and can_skip(legacy_target)):
                legacy_target.write_bytes(conditioner_target.read_bytes())

    model_segments = []
    if "small" in variants:
        model_segments.append("stable-audio-3-small-music+stable-audio-3-small-sfx")
    if "medium" in variants:
        model_segments.append("stable-audio-3-medium")
    manifest = {
        "model": "stabilityai/" + "+".join(model_segments) if model_segments else "stabilityai/none",
        "format": "safetensors",
        "tokenizer": {
            "fileName": TOKENIZER_FILE,
            "bytes": tokenizer_size,
            "tokenOffset": 0,
        },
        "conditioners": conditioner_manifest,
        "files": manifest_files,
    }
    manifest_path = destination_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
