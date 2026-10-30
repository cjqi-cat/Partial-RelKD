#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from PIL import Image

KINDS=("cloud","haze","gaussian_noise","blur","jpeg","low_resolution","pixel_dropout","misregistration","radiometric_shift","mixed")
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def seed(base,*parts): return int.from_bytes(hashlib.sha256("|".join(map(str,(base,*parts))).encode()).digest()[:8],"little")
def main():
    p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,default=Path("__PROJECT_ROOT__"));p.add_argument("--dataset-root",type=Path,default=Path("__DATA_ROOT__/LEVIR-MCI-dataset"));p.add_argument("--output",type=Path,required=True);p.add_argument("--samples",type=int,default=24);a=p.parse_args()
    old=load(a.project_root/"stage6_5_calibrated_v3/degradation/calibrated_v3.py","old_v3");new=load(a.project_root/"stage10_7_severity_curves/degradation/calibrated_v3_curve.py","curve_v3")
    names=[x.strip() for x in (a.project_root/"stage3_caption_retrain/vendor/Change-Agent/Multi_change/data/LEVIR_MCI/test.txt").read_text().splitlines() if x.strip()][:a.samples]
    report={"profile":"calibrated_v3_curve_s2_anchor","samples":len(names),"s2_byte_exact":{},"mean_absolute_pixel_difference":{}}
    for kind in KINDS:
        exact=True;values={str(s):[] for s in range(1,6)}
        for name in names:
            image=Image.open(a.dataset_root/"images/test/A"/name).convert("RGB");base=np.asarray(image,dtype=np.float32)
            rng_seed=seed(20260928,kind,"A",name,"A")
            old_img,_=old.OPERATORS[kind](image,2,np.random.default_rng(rng_seed));new2,_=new.OPERATORS[kind](image,2,np.random.default_rng(rng_seed))
            exact &= np.array_equal(np.asarray(old_img),np.asarray(new2))
            for s in range(1,6):
                out,_=new.OPERATORS[kind](image,s,np.random.default_rng(rng_seed));values[str(s)].append(float(np.abs(np.asarray(out,dtype=np.float32)-base).mean()))
        means={k:float(np.mean(v)) for k,v in values.items()};report["s2_byte_exact"][kind]=bool(exact);report["mean_absolute_pixel_difference"][kind]=means
        if not exact: raise RuntimeError(f"severity-2 compatibility failed: {kind}")
        if len({round(x,6) for x in means.values()})<5: raise RuntimeError(f"non-distinct severities: {kind} {means}")
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2));print("SEVERITY_AUDIT_OK")
if __name__=="__main__":main()
