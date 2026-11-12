#!/usr/bin/env python3
"""Fail fast on common mistakes before committing the release folder."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 20 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".h5",
    ".hdf5", ".zip", ".7z", ".tar", ".gz", ".pyc",
}
TEXT_SUFFIXES = {".py", ".json", ".sh", ".md", ".txt", ".yaml", ".yml", ".toml"}
PRIVATE_PATTERNS = {
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "local Windows user path": re.compile(r"[A-Za-z]:\\Users\\[^\\]+"),
    "internal AutoDL path": re.compile(r"/root/" + r"autodl-tmp"),
    "server hostname": re.compile(r"connect\.[A-Za-z0-9.-]+"),
}
REQUIRED = [
    "pyproject.toml",
    "partial_relkd/models/visual_adapter.py",
    "partial_relkd/data/degradations.py",
    "partial_relkd/losses/distillation.py",
    "configs/stage1/geometry_quality.json",
    "configs/stage2/partial_relkd.json",
    "scripts/train_stage1.py",
    "scripts/train_partial_relkd.py",
    "scripts/evaluate_lofo.py",
    "experiments/legacy_stages/stage6_7_partial_hybrid/runner/train.py",
    "experiments/legacy_stages/stage6_7_partial_hybrid/runner/evaluate.py",
    "experiments/legacy_stages/stage10_25_lchg_ablation/configs/encoder_wo_lchg.json",
    "baselines/wrappers/stage10_17_scnet_lofo/runner/model.py",
    "baselines/wrappers/stage10_18_rmnet_lofo/runner/model.py",
    "baselines/wrappers/stage10_19_scnet_multidataset/run_dataset.sh",
    "baselines/wrappers/stage10_20_rmnet_multidataset/run_dataset.sh",
    "docs/partial_relkd_overview.png",
    "docs/severity_unseen_curves.png",
]


def main() -> None:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    config_pairs = {
        "configs/stage1/geometry_quality.json":
            "experiments/legacy_stages/stage6_6_m2_backbones/configs/dino_partial_encoder_geometry.json",
        "configs/stage2/partial_relkd.json":
            "experiments/legacy_stages/stage6_7_partial_hybrid/configs/p3_partial_relation.json",
        "configs/stage2/partial_relkd_mask_free.json":
            "experiments/legacy_stages/stage10_3_p3_nomask/configs/p3_nomask.json",
    }
    for public, validated in config_pairs.items():
        public_path, validated_path = ROOT / public, ROOT / validated
        if public_path.is_file() and validated_path.is_file():
            if public_path.read_bytes() != validated_path.read_bytes():
                errors.append(f"public config drift: {public} != {validated}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size > MAX_BYTES:
            errors.append(f"file exceeds 20 MiB: {relative} ({size} bytes)")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden artifact type: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 text file: {relative}")
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} found in {relative}")

    if errors:
        print("RELEASE_AUDIT_FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    tracked_files = [
        path for path in ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts
    ]
    count = len(tracked_files)
    total = sum(path.stat().st_size for path in tracked_files)
    print(f"RELEASE_AUDIT_OK files={count} bytes={total}")


if __name__ == "__main__":
    main()
