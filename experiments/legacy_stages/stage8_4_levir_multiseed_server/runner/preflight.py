#!/usr/bin/env python3
import json
from pathlib import Path

P = Path("__PROJECT_ROOT__"); R = P / "stage8_4_levir_multiseed_server"
required = [P / "dino_finetune_step1/runner/train.py",
            P / "stage4_caption_dino_finetune/runner/prepare_features.py",
            P / "stage6_6_m2_backbones/runner/run_dino_lofo.py",
            P / "stage6_7_partial_hybrid/runner/train.py",
            Path("__DATA_ROOT__/LEVIR-MCI-dataset")]
missing = [str(x) for x in required if not x.exists()]
if missing: raise SystemExit("Missing:\n" + "\n".join(missing))
for seed in (20261018, 20261028):
    for name in ("partial_encoder", "dino_aug", "p3"):
        path = R / "configs" / f"seed_{seed}" / f"{name}.json"
        if not path.exists(): raise SystemExit(f"Missing {path}")
        data = json.loads(path.read_text())
        if int(data.get("seed", seed)) != seed: raise SystemExit(f"Bad seed in {path}")
print("PREFLIGHT_OK")
