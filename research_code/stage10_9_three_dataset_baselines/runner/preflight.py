import json
from pathlib import Path
ROOT=Path("__PROJECT_ROOT__/stage10_9_three_dataset_baselines")
required=[Path("__PROJECT_ROOT__/stage10_authoritative_baselines/runner/models.py"),
          Path("__PROJECT_ROOT__/stage6_5_calibrated_v3/degradation/calibrated_v3.py")]
for p in required:
 if not p.exists(): raise FileNotFoundError(p)
configs=sorted((ROOT/"configs").glob("*/*.json"))
if len(configs)!=8: raise RuntimeError(f"expected 8 configs, found {len(configs)}")
for path in configs:
 cfg=json.loads(path.read_text());
 if set(cfg["allowed_degradations"])&set(cfg["heldout_degradations"]): raise RuntimeError(f"LOFO leakage: {path}")
 for key in ("dataset_root","metadata_root","change_agent_root","degradation_script","stage10_root"):
  if not Path(cfg[key]).exists(): raise FileNotFoundError(f"{path}: {key}={cfg[key]}")
 vendor={"chg2cap":"chg2cap","mccformer_d":"sen","rsiccformer":"sen","sen":"sen"}[cfg["model"]]
 if not (Path(cfg["stage10_root"])/"vendor"/vendor).is_dir(): raise FileNotFoundError(f"missing vendor/{vendor}")
print("PREFLIGHT_OK")
