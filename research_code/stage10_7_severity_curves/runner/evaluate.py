#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ALL = ("cloud", "haze", "gaussian_noise", "blur", "jpeg", "low_resolution", "pixel_dropout", "misregistration", "radiometric_shift", "mixed")


def clean_ids(ids, tokens):
    ignored = {tokens.start, tokens.end, tokens.pad}
    return [int(x) for x in ids if int(x) not in ignored]


def score(root, references, hypotheses):
    root = str(root)
    if root not in sys.path: sys.path.insert(0, root)
    from eval_func.bleu.bleu import Bleu
    from eval_func.rouge.rouge import Rouge
    from eval_func.cider.cider import Cider
    gts = {i: [" ".join(map(str, x)) for x in group] for i, group in enumerate(references)}
    res = {i: [" ".join(map(str, x))] for i, x in enumerate(hypotheses)}
    bleu, _ = Bleu(4).compute_score(gts, res); rouge, _ = Rouge().compute_score(gts, res); cider, _ = Cider().compute_score(gts, res)
    return {"BLEU-1": float(bleu[0]), "BLEU-2": float(bleu[1]), "BLEU-3": float(bleu[2]), "BLEU-4": float(bleu[3]), "ROUGE-L": float(rouge), "CIDEr": float(cider)}


def save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def conditions(argument):
    if argument == "degraded": return [f"{k}__{side}" for k in ALL for side in ("A", "B", "both")]
    if argument == "clean": return ["clean"]
    return [x.strip() for x in argument.split(",") if x.strip()]


def load_stack(model_name, root):
    runner = root / ("stage6_6_m2_backbones/runner" if model_name == "change_agent" else "stage6_7_partial_hybrid/runner")
    sys.path.insert(0, str(runner))
    common = importlib.import_module("common"); data = importlib.import_module("data"); models = importlib.import_module("models")
    return common, data, models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("change_agent", "p3", "p3_nomask"), required=True)
    p.add_argument("--root", type=Path, default=Path("__PROJECT_ROOT__"))
    p.add_argument("--severity", type=int, required=True, choices=range(1, 6))
    p.add_argument("--conditions", default="degraded")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    common, data, models = load_stack(a.model, a.root)
    if a.model == "change_agent":
        cfg = common.load_json(a.root / "stage6_6_m2_backbones/configs/ca_m2_geometry.json")
        checkpoint = None; variant = "Change-Agent"
    else:
        ckpt_path = a.root / ("stage6_7_partial_hybrid/runs/p3_partial_relation/best.pt" if a.model == "p3" else "stage10_3_p3_nomask/runs/p3_nomask/best.pt")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = dict(checkpoint["config"]); variant = "P3" if a.model == "p3" else "P3-NoMask"
    cfg["severity"] = a.severity
    cfg["degradation_seed"] = 20260928
    cfg["degradation_script"] = str(a.root / "stage10_7_severity_curves/degradation/calibrated_v3_curve.py")
    common.seed_everything(int(cfg["seed"])); device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    tokens = data.CaptionTokens(cfg["metadata_root"], int(cfg.get("max_length", 41)))
    model = models.build_model(cfg, tokens.vocab)
    if checkpoint is not None:
        _, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
        if unexpected: raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    model = model.to(device).eval(); a.output.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for condition in conditions(a.conditions):
        target = a.output / f"{condition}.metrics.json"
        prediction_target = a.output / f"{condition}.predictions.json"
        if target.is_file() and prediction_target.is_file():
            all_metrics[condition] = json.loads(target.read_text(encoding="utf-8")); print(f"SKIP {a.model} s={a.severity} {condition}", flush=True); continue
        ds = data.OnlineCaptionPairs(cfg, "test", tokens, training=False, condition=condition)
        if a.limit:
            ds.names = ds.names[:a.limit]
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=device.type == "cuda", collate_fn=data.collate_eval)
        refs_all, hyps, rows = [], [], []; started = time.time()
        with torch.inference_mode():
            for step, batch in enumerate(loader, 1):
                seqs = model.generate(batch["a"].to(device, non_blocking=True), batch["b"].to(device, non_blocking=True))
                for name, seq, refs in zip(batch["name"], seqs, batch["references"]):
                    pred = clean_ids(seq, tokens); ref_ids = [clean_ids(tokens.encode(words)[0].tolist(), tokens) for words in refs]
                    hyps.append(pred); refs_all.append(ref_ids); rows.append({"name": name, "prediction_ids": pred, "reference_ids": ref_ids})
                if step % 20 == 0 or step == len(loader): print(f"model={a.model} s={a.severity} condition={condition} batches={step}/{len(loader)} elapsed={time.time()-started:.1f}s", flush=True)
        payload = {"model": a.model, "variant": variant, "family": "geometry_quality", "condition": condition, "samples": len(rows), "severity": a.severity, "degradation_seed": 20260928, "operator_profile": "calibrated_v3_curve_s2_anchor", **score(cfg["change_agent_root"], refs_all, hyps)}
        save(payload, target); save(rows, prediction_target); all_metrics[condition] = payload
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    save(all_metrics, a.output / "all_metrics.json")


if __name__ == "__main__": main()
