#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import load_json, load_operators


parser = argparse.ArgumentParser()
parser.add_argument("--config", action="append", required=True)
args = parser.parse_args()

failed = []
for config_path in args.config:
    cfg = load_json(config_path)
    required = [
        cfg["dataset_root"], cfg["metadata_root"], cfg["change_agent_root"],
        cfg["degradation_script"],
    ]
    if cfg["architecture"] == "change_agent":
        required += [
            cfg["ca_checkpoint"],
            str(Path(cfg["change_agent_root"]) / "model" / "model_encoder_att.py"),
            str(Path(cfg["change_agent_root"]) / "model" / "model_decoder.py"),
            str(Path(cfg["change_agent_root"]) / "model" / "segformer.py"),
            str(Path(cfg["change_agent_root"]) / "model" / "pretrained" / "mit_b1.pth"),
        ]
    else:
        required += [
            cfg["dino_model_path"], cfg["dino_caption_model_path"],
            cfg["dino_encoder_checkpoint"], cfg["dino_caption_checkpoint"],
        ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        failed.extend(f"{config_path}: missing {path}" for path in missing)
        continue
    operators = load_operators(cfg["degradation_script"])
    if set(cfg["allowed_degradations"]) - set(operators):
        failed.append(f"{config_path}: operator mapping incomplete")
    if set(cfg["allowed_degradations"]) & set(cfg["heldout_degradations"]):
        failed.append(f"{config_path}: LOFO leakage")
    if cfg["architecture"] == "change_agent":
        checkpoint = torch.load(cfg["ca_checkpoint"], map_location="cpu", weights_only=False)
        keys = {"encoder_dict", "encoder_trans_dict", "decoder_dict"}
        if not keys.issubset(checkpoint):
            failed.append(f"{config_path}: Change-Agent checkpoint keys={sorted(checkpoint)}")
    print(f"OK {config_path}: architecture={cfg['architecture']} allowed={cfg['allowed_degradations']}")

if failed:
    raise SystemExit("\n".join(failed))
print("PREFLIGHT_OK")
