#!/usr/bin/env python3
import argparse, json
from pathlib import Path

PROJECT = Path("__PROJECT_ROOT__")
ROOT = PROJECT / "stage8_4_levir_multiseed_server"
ALLOWED = ["cloud", "haze", "gaussian_noise", "jpeg", "pixel_dropout", "radiometric_shift"]
HELDOUT = ["blur", "low_resolution", "misregistration", "mixed"]

def load(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def dump(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True)
    seed = ap.parse_args().seed; tag = f"seed_{seed}"
    cfg = ROOT / "configs" / tag; run = ROOT / "runs" / tag
    enc = load(PROJECT / "stage6_6_m2_backbones/configs/dino_partial_encoder_geometry.json")
    enc.update(run_name=f"{tag}_partial_encoder", seed=seed)
    dump(enc, cfg / "partial_encoder.json")
    common = dict(seed=seed, allowed_degradations=ALLOWED, heldout_degradations=HELDOUT,
                  dino_encoder_checkpoint=str(run / "partial_encoder/checkpoints/best.pt"),
                  dino_caption_checkpoint=str(run / "partial_teacher/checkpoints/partial_last2_best.pt"))
    aug = load(PROJECT / "stage6_6_m2_backbones/configs/dino_partial_m2_geometry.json")
    aug.update(common, run_name=f"{tag}_dino_aug"); dump(aug, cfg / "dino_aug.json")
    p3 = load(PROJECT / "stage6_7_partial_hybrid/configs/p3_partial_relation.json")
    p3.update(common, run_name=f"{tag}_p3"); dump(p3, cfg / "p3.json")
    print(f"CONFIGS_OK {tag}")

if __name__ == "__main__": main()
