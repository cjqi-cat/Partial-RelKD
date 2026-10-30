#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys, tempfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args()
    with tempfile.TemporaryDirectory(prefix="p3_nomask_smoke_") as out:
        subprocess.run([sys.executable,str(Path(__file__).with_name("train_encoder_nomask.py")),"--config",a.config,"--output",out,"--smoke-batches","1"],check=True)
        checkpoint=Path(out)/"checkpoints"/"best.pt"
        if not checkpoint.is_file(): raise RuntimeError("best.pt missing")
        import torch
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if payload.get("provenance",{}).get("mask_used") is not False: raise RuntimeError("provenance audit failed")
    print("SMOKE_OK")
if __name__=="__main__": main()
