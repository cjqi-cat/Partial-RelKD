#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED_SHA256 = "62b99fd098aaf61b739f678d7effbeaceab9ef10c2b04a69d823332d98471320"
EXPECTED = ("cloud","haze","gaussian_noise","blur","jpeg","low_resolution","pixel_dropout","misregistration","radiometric_shift","mixed")


def load(path):
    spec=importlib.util.spec_from_file_location("calibrated_v3",path); module=importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot load {path}")
    spec.loader.exec_module(module); return module


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); a=p.parse_args()
    stage=a.root/"project/stage6_5_calibrated_v3"; operator=stage/"degradation/calibrated_v3.py"
    digest=hashlib.sha256(operator.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256: raise RuntimeError(f"operator sha256 mismatch: {digest}")
    module=load(operator)
    if tuple(module.OPERATORS) != EXPECTED: raise RuntimeError("operator order/name mismatch")
    image=Image.fromarray(np.random.default_rng(7).integers(0,256,(64,64,3),dtype=np.uint8))
    for name,fn in module.OPERATORS.items():
        a1,_=fn(image,3,np.random.default_rng(123)); a2,_=fn(image,3,np.random.default_rng(123))
        if not np.array_equal(np.asarray(a1),np.asarray(a2)): raise RuntimeError(f"nondeterministic: {name}")
    required=[
        a.root/"data/LEVIR-MCI-dataset/images/train/A",
        a.root/"data/LEVIR-MCI-dataset/images/val/A",
        a.root/"data/LEVIR-MCI-dataset/images/test/A",
        stage/"runner/prepare_patch_features.py",
        a.root/"project/dino_finetune_step1/runner/train.py",
        a.root/"project/stage3_caption_retrain/caption_features/manifest.json",
        a.root/"project/stage3_caption_retrain/vendor/Change-Agent/Multi_change/data/LEVIR_MCI/vocab.json",
    ]
    missing=[str(x) for x in required if not x.exists()]
    if missing: raise RuntimeError("missing required paths:\n"+"\n".join(missing))
    free=shutil.disk_usage("__STORAGE_ROOT__").free
    if free < 14*1024**3: raise RuntimeError(f"need >=14 GiB free before adapter cache; found {free/1024**3:.1f} GiB")
    payload={"status":"PASS","operator_sha256":digest,"operators":list(module.OPERATORS),"free_gib":round(free/1024**3,2)}
    (stage/"preflight.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload))


if __name__=="__main__": main()
