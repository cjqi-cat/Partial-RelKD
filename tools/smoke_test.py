#!/usr/bin/env python3
"""Data-free smoke test for a fresh clone of the public release.

Full inference requires external datasets, upstream code, and checkpoints that
are intentionally excluded. This test verifies path materialization and the
deterministic online degradation operators supplied by the repository.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_degradations():
    path = ROOT / "research_code" / "stage6_5_calibrated_v3" / "degradation" / "calibrated_v3.py"
    spec = importlib.util.spec_from_file_location("calibrated_v3_release", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import degradation module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_materialization() -> None:
    with tempfile.TemporaryDirectory(
        prefix="partial_relkd_smoke_", dir=ROOT.parent, ignore_cleanup_errors=True
    ) as temp:
        base = Path(temp)
        output = base / "materialized"
        command = [
            sys.executable,
            str(ROOT / "tools" / "materialize_workspace.py"),
            "--output", str(output),
            "--data-root", str(base / "data"),
            "--change-agent-root", str(base / "Change-Agent" / "Multi_change"),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                "Materialization failed:\n"
                + completed.stdout
                + completed.stderr
            )
        required = output / "stage6_7_partial_hybrid" / "runner" / "train.py"
        if not required.is_file():
            raise RuntimeError(f"Materialization omitted core file: {required}")
        tokens = ("__PROJECT_ROOT__", "__DATA_ROOT__", "__CHANGE_AGENT_ROOT__")
        unresolved = []
        text_suffixes = {".py", ".json", ".sh", ".md", ".txt", ".yaml", ".yml"}
        for path in output.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in text_suffixes:
                continue
            if any(token in path.read_text(encoding="utf-8") for token in tokens):
                unresolved.append(str(path.relative_to(output)))
        if unresolved:
            raise RuntimeError(f"Unresolved path tokens: {unresolved}")


def check_degradations() -> None:
    module = load_degradations()
    yy, xx = np.mgrid[0:64, 0:64]
    array = np.stack(
        ((xx * 4) % 256, (yy * 4) % 256, ((xx + yy) * 2) % 256), axis=-1
    ).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    for name in module.CORRUPTIONS:
        first, metadata = module.apply_degradation(
            image, name, 20260928, "smoke.png", "A", severity=2
        )
        second, _ = module.apply_degradation(
            image, name, 20260928, "smoke.png", "A", severity=2
        )
        if first.mode != "RGB" or first.size != image.size:
            raise RuntimeError(f"Invalid output for {name}: {first.mode} {first.size}")
        if not np.array_equal(np.asarray(first), np.asarray(second)):
            raise RuntimeError(f"Non-deterministic output for {name}")
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Invalid metadata for {name}")


def main() -> None:
    check_materialization()
    check_degradations()
    print("SMOKE_TEST_OK materialization=1 degradations=10 deterministic=1")


if __name__ == "__main__":
    main()
