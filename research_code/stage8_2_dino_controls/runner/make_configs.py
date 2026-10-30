#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("__PROJECT_ROOT__/stage8_2_dino_controls")
SOURCES = {
    "LEVIR_MCI": Path("__PROJECT_ROOT__/stage6_6_m2_backbones/configs/dino_partial_m2_geometry.json"),
    "SECOND_CC": Path("__PROJECT_ROOT__/stage8_1_multidataset_full/configs/SECOND_CC/m2.json"),
    "WHU_CDC": Path("__PROJECT_ROOT__/stage8_1_multidataset_full/configs/WHU_CDC/m2.json"),
}


def dump(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    for dataset, source in SOURCES.items():
        cfg = json.loads(source.read_text(encoding="utf-8"))
        cfg["run_name"] = f"{dataset.lower()}_dino_aug_ce"
        # Variant m1 in Stage6.6 uses caption CE only.  Keep the same
        # clean/dirty sampling, initialization, optimizer, and validation
        # protocol as the matching M2 run so token KD is the sole difference.
        dump(cfg, ROOT / "configs" / dataset / "dino_aug.json")
        clean_cfg = dict(cfg)
        clean_cfg["run_name"] = f"{dataset.lower()}_dino_clean_control"
        dump(clean_cfg, ROOT / "configs" / dataset / "dino_clean.json")
        clean_checkpoint = {
            "architecture": clean_cfg["architecture"],
            "variant": "dino_clean_control",
            "epoch": None,
            "best_val_loss": None,
            "config": clean_cfg,
            # build_model already loads the target encoder and clean caption
            # teacher named in the config.  An empty overlay preserves them.
            "trainable_state": {},
        }
        checkpoint_path = ROOT / "runs" / dataset / "dino_clean.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(clean_checkpoint, checkpoint_path)
    print("CONFIGS_AND_CLEAN_CHECKPOINTS_OK", flush=True)


if __name__ == "__main__":
    main()
