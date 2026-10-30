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
TEXT_SUFFIXES = {".py", ".json", ".sh", ".md", ".txt", ".yaml", ".yml"}
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
    "research_code/stage6_5_calibrated_v3/degradation/calibrated_v3.py",
    "research_code/stage6_7_partial_hybrid/runner/train.py",
    "research_code/stage6_7_partial_hybrid/runner/evaluate.py",
    "research_code/stage6_7_partial_hybrid/configs/p3_partial_relation.json",
    "research_code/stage10_7_severity_curves/degradation/calibrated_v3_curve.py",
    "docs/partial_relkd_overview.png",
]


def main() -> None:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

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
