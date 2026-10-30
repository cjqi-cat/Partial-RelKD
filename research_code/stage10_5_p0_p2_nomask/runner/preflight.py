#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {"cloud", "haze", "gaussian_noise", "jpeg", "pixel_dropout", "radiometric_shift"}
FORBIDDEN = {"blur", "low_resolution", "misregistration", "mixed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="__PROJECT_ROOT__/stage10_5_p0_p2_nomask")
    args = parser.parse_args()
    root = Path(args.root)
    configs = [root / "configs" / f"p{i}_nomask.json" for i in range(3)]
    for path in configs:
        cfg = json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("mask_supervision") is not False:
            raise RuntimeError(f"{path}: mask_supervision must be false")
        if set(cfg["allowed_degradations"]) != EXPECTED:
            raise RuntimeError(f"{path}: allowed degradation mismatch")
        if set(cfg["heldout_degradations"]) != FORBIDDEN:
            raise RuntimeError(f"{path}: heldout degradation mismatch")
        for key in ("dataset_root", "metadata_root", "change_agent_root", "degradation_script",
                    "dino_model_path", "dino_caption_model_path", "dino_encoder_checkpoint",
                    "dino_caption_checkpoint"):
            if not Path(cfg[key]).exists():
                raise FileNotFoundError(f"{path}: missing {key}={cfg[key]}")
        print({"config": str(path), "run": cfg["run_name"], "mask_supervision": False})

    p0, p1, p2 = [json.loads(p.read_text(encoding="utf-8")) for p in configs]
    assert p0["dino_train_encoder"] is False
    assert p1["dino_train_encoder"] is True and p1["lambda_clean"] == 0 and p1["lambda_relation"] == 0
    assert p2["dino_train_encoder"] is True and p2["lambda_clean"] > 0 and p2["lambda_relation"] == 0
    assert len({p["dino_encoder_checkpoint"] for p in (p0, p1, p2)}) == 1
    assert len({p["dino_caption_checkpoint"] for p in (p0, p1, p2)}) == 1

    stage67_data = Path("__PROJECT_ROOT__/stage6_7_partial_hybrid/runner/data.py")
    source = stage67_data.read_text(encoding="utf-8")
    for marker in (' / "label" / ', ' / "mask" / ', ' / "sem" / '):
        if marker in source:
            raise RuntimeError(f"GT mask path detected in caption loader: {marker}")
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
