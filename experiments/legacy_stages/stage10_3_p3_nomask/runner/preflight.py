#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
ALLOWED={"cloud","haze","gaussian_noise","jpeg","pixel_dropout","radiometric_shift"}
HELDOUT={"blur","low_resolution","misregistration","mixed"}
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path("__PROJECT_ROOT__/stage10_3_p3_nomask")); args=parser.parse_args(); root=args.root
    configs=[root/"configs/partial_encoder_nomask.json",root/"configs/p3_nomask.json"]
    for path in configs:
        cfg=json.loads(path.read_text(encoding="utf-8"))
        if cfg.get("mask_supervision") is not False: raise RuntimeError(f"{path}: mask supervision must be false")
        if set(cfg.get("allowed_degradations",ALLOWED))!=ALLOWED: raise RuntimeError(f"{path}: allowed degradation mismatch")
        if not HELDOUT.issubset(cfg.get("heldout_degradations",HELDOUT)): raise RuntimeError(f"{path}: holdout mismatch")
    required=[Path("__DATA_ROOT__/LEVIR-MCI-dataset"),Path("__PROJECT_ROOT__/stage6_5_calibrated_v3/degradation/calibrated_v3.py"),Path("__PROJECT_ROOT__/stage6_7_partial_hybrid/runner/train.py")]
    missing=[str(path) for path in required if not path.exists()]
    if missing: raise FileNotFoundError("missing: "+", ".join(missing))
    source=(root/"runner/train_encoder_nomask.py").read_text(encoding="utf-8")
    if ' / "label" / ' in source or ' / "mask" / ' in source: raise RuntimeError("static mask path detected")
    print("PREFLIGHT_OK")
if __name__=="__main__": main()
