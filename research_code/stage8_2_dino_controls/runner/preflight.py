#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("__PROJECT_ROOT__/stage8_2_dino_controls")
for dataset in ("LEVIR_MCI", "SECOND_CC", "WHU_CDC"):
    for name in ("dino_clean", "dino_aug"):
        path = ROOT / "configs" / dataset / f"{name}.json"
        cfg = json.loads(path.read_text(encoding="utf-8"))
        for field in (
            "dataset_root", "metadata_root", "change_agent_root",
            "degradation_script", "dino_model_path",
            "dino_caption_model_path", "dino_encoder_checkpoint",
            "dino_caption_checkpoint",
        ):
            target = Path(cfg[field])
            if not target.exists():
                raise FileNotFoundError(f"{dataset} {field}: {target}")
        allowed = set(cfg["allowed_degradations"])
        heldout = set(cfg["heldout_degradations"])
        if allowed & heldout:
            raise RuntimeError(f"LOFO leakage in {dataset}: {sorted(allowed & heldout)}")
    checkpoint = ROOT / "runs" / dataset / "dino_clean.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
print("PREFLIGHT_OK", flush=True)
