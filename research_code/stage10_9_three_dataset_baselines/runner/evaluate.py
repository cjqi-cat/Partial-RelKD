from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import conditions, save_json, seed_everything
from data import CaptionPairs, CaptionTokens, collate_eval
from metrics import score
from models import build_model


def clean(ids, tok):
    ignored = {tok.start, tok.end, tok.pad}; out = []
    for x in ids:
        x = int(x)
        if x == tok.end: break
        if x not in ignored: out.append(x)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--conditions", default="all"); args = ap.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False); cfg = state["config"]
    seed_everything(cfg["seed"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = CaptionTokens(cfg["metadata_root"], cfg.get("max_length", 41)); model = build_model(cfg, tok.vocab)
    model.load_state_dict(state["model_state"]); model.to(device).eval(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    names = conditions() if args.conditions == "all" else args.conditions.split(","); all_metrics = {}
    for condition in names:
        ds = CaptionPairs(cfg, "test", tok, False, condition)
        loader = DataLoader(ds, batch_size=cfg["eval_batch_size"], shuffle=False, num_workers=cfg.get("num_workers", 4),
                            pin_memory=True, collate_fn=collate_eval)
        refs, hyps, rows = [], [], []
        with torch.inference_mode():
            for i, batch in enumerate(loader, 1):
                a, b = batch["a"].to(device), batch["b"].to(device)
                seq = model.greedy(a, b, tok.start, tok.end, tok.pad, tok.max_length).cpu().tolist()
                for name, pred, words in zip(batch["name"], seq, batch["references"]):
                    p = clean(pred, tok); r = [clean(tok.encode(x)[0].tolist(), tok) for x in words]
                    refs.append(r); hyps.append(p); rows.append({"name": name, "prediction_ids": p, "reference_ids": r})
                if i % 20 == 0: print(f"{condition} batches={i}/{len(loader)} samples={len(rows)}", flush=True)
        payload = {"model": cfg["model"],
                   "family": cfg["family"], "condition": condition, "samples": len(rows),
                   "severity": cfg["severity"], "degradation_seed": cfg["degradation_seed"],
                   **score(cfg["change_agent_root"], refs, hyps)}
        save_json(payload, out / f"{condition}.metrics.json"); save_json(rows, out / f"{condition}.predictions.json")
        all_metrics[condition] = payload; print(json.dumps(payload), flush=True)
    save_json(all_metrics, out / "all_metrics.json"); print("EVALUATION_OK", flush=True)


if __name__ == "__main__": main()
