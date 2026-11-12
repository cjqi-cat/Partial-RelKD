#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from common import load_json, load_operators


parser = argparse.ArgumentParser()
parser.add_argument("--config", action="append", required=True)
args = parser.parse_args()
failed = []
for config_path in args.config:
    cfg = load_json(config_path)
    required = [
        cfg["dataset_root"], cfg["metadata_root"], cfg["change_agent_root"],
        cfg["degradation_script"], cfg["dino_model_path"],
        cfg["dino_caption_model_path"], cfg["dino_encoder_checkpoint"],
        cfg["dino_caption_checkpoint"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    failed.extend(f"{config_path}: missing {path}" for path in missing)
    operators = load_operators(cfg["degradation_script"])
    if set(cfg["allowed_degradations"]) - set(operators):
        failed.append(f"{config_path}: operator mapping incomplete")
    if set(cfg["allowed_degradations"]) & set(cfg["heldout_degradations"]):
        failed.append(f"{config_path}: LOFO leakage")
    if not missing:
        print(f"OK {config_path}: {cfg['run_name']}")
if failed:
    raise SystemExit("\n".join(failed))
print("PREFLIGHT_OK")
