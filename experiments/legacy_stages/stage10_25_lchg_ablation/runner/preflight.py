#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path("__PROJECT_ROOT__")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    encoder = load(ROOT / "configs" / "encoder_wo_lchg.json")
    p3 = load(ROOT / "configs" / "p3_wo_lchg.json")

    assert encoder["lambda_change"] == 0.0, "L_chg was not disabled"
    assert encoder["lambda_patch_cls"] == 0.5, "L_patch must remain enabled"
    assert encoder["lambda_invariance"] == 1.0
    assert encoder["lambda_distill"] == 0.1
    assert encoder["lambda_adversarial"] == 0.15
    assert encoder["lambda_backbone_anchor"] == 0.001
    expected = ["cloud", "haze", "gaussian_noise", "jpeg", "pixel_dropout", "radiometric_shift"]
    assert encoder["allowed_degradations"] == expected
    assert p3["allowed_degradations"] == expected

    required = [
        Path(encoder["dataset_root"]),
        Path(encoder["degradation_script"]),
        PROJECT / "dino_finetune_step1" / "runner" / "train.py",
        PROJECT / "stage6_6_m2_backbones" / "runner" / "run_dino_lofo.py",
        PROJECT / "stage4_caption_dino_finetune" / "runner" / "prepare_features.py",
        PROJECT / "stage4_caption_dino_finetune" / "runner" / "train_caption.py",
        PROJECT / "stage6_7_partial_hybrid" / "runner" / "train.py",
        PROJECT / "stage6_7_partial_hybrid" / "runner" / "evaluate.py",
        Path(p3["metadata_root"]) / "train.txt",
        Path(p3["metadata_root"]) / "vocab.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

    print({
        "ablation": "P3 minus L_chg only",
        "lambda_change": encoder["lambda_change"],
        "lambda_patch_cls": encoder["lambda_patch_cls"],
        "family": p3["family"],
        "seed": p3["seed"],
    })
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
