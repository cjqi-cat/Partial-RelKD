#!/usr/bin/env python3
"""Run the existing DINO Stage-1 trainer with a strict LOFO corruption list."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


ALL_DEGRADATIONS = (
    "cloud", "haze", "gaussian_noise", "blur", "jpeg",
    "low_resolution", "pixel_dropout", "misregistration",
    "radiometric_shift", "mixed",
)
HOLDOUTS = {
    "geometry_quality": {"blur", "low_resolution", "misregistration", "mixed"},
    "atmospheric": {"cloud", "haze", "mixed"},
    "noise_loss": {"gaussian_noise", "jpeg", "pixel_dropout", "mixed"},
}


parser = argparse.ArgumentParser()
parser.add_argument("--source-train", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--family", default="geometry_quality")
parser.add_argument("--resume")
args = parser.parse_args()

source = Path(args.source_train).resolve()
sys.path.insert(0, str(source.parent))
spec = importlib.util.spec_from_file_location("stage66_source_dino_train", source)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {source}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
expected = tuple(x for x in ALL_DEGRADATIONS if x not in HOLDOUTS[args.family])
configured = tuple(cfg.get("allowed_degradations", expected))
if configured != expected:
    raise RuntimeError(f"LOFO mismatch: configured={configured}, expected={expected}")
module.CORRUPTIONS = configured
sys.argv = [str(source), "--config", args.config, "--output", args.output]
if args.resume:
    sys.argv += ["--resume", args.resume]
print("strict LOFO corruptions:", module.CORRUPTIONS, flush=True)
module.main()
