#!/usr/bin/env python3
from pathlib import Path
import json


ROOT = Path("__PROJECT_ROOT__/stage8_1_multidataset_full")
required = [
    "__PROJECT_ROOT__/stage8_1_multidataset_full/metadata/SECOND_CC/vocab.json",
    "__PROJECT_ROOT__/stage8_1_multidataset_full/metadata/WHU_CDC/vocab.json",
    "__PROJECT_ROOT__/stage6_5_calibrated_v3/degradation/calibrated_v3.py",
    "__PROJECT_ROOT__/dino_finetune_step1/runner/train.py",
    "__PROJECT_ROOT__/stage6_7_partial_hybrid/runner/train.py",
    "__CHANGE_AGENT_ROOT__/models_ckpt/MCI_model.pth",
]
missing = [x for x in required if not Path(x).is_file()]
if missing:
    raise FileNotFoundError("Missing:\n" + "\n".join(missing))
for dataset in ("SECOND_CC", "WHU_CDC"):
    view = ROOT / "data_views" / dataset
    for split in ("train", "val", "test"):
        for side in ("A", "B", "label"):
            path = view / "images" / split / side
            if not path.is_dir():
                raise FileNotFoundError(path)
    if not (view / "LevirCCcaptions.json").is_file():
        raise FileNotFoundError(view / "LevirCCcaptions.json")
    cfg_root = ROOT / "configs" / dataset
    for name in ("change_agent", "partial_encoder", "m2", "p3"):
        path = cfg_root / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        json.loads(path.read_text(encoding="utf-8"))
print("PREFLIGHT_OK", flush=True)
