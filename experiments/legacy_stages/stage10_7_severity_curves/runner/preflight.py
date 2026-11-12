#!/usr/bin/env python3
from pathlib import Path
import argparse

p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path("__PROJECT_ROOT__")); a=p.parse_args()
required=[
 a.root/"stage3_caption_retrain/vendor/Change-Agent/Multi_change/models_ckpt/MCI_model.pth",
 a.root/"stage6_6_m2_backbones/configs/ca_m2_geometry.json",
 a.root/"stage6_6_m2_backbones/runner/models.py",
 a.root/"stage6_7_partial_hybrid/runs/p3_partial_relation/best.pt",
 a.root/"stage10_3_p3_nomask/runs/p3_nomask/best.pt",
 a.root/"stage6_5_calibrated_v3/degradation/calibrated_v3.py",
 a.root/"stage10_7_severity_curves/degradation/calibrated_v3_curve.py",
]
missing=[str(x) for x in required if not x.is_file() or x.stat().st_size==0]
if missing: raise SystemExit("Missing:\n"+"\n".join(missing))
print({"models":["Change-Agent","P3","P3-NoMask"],"severities":[1,2,3,4,5],"conditions_per_severity":30,"seed":20260928})
print("PREFLIGHT_OK")
