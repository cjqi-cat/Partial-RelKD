#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import ALL_DEGRADATIONS, load_json, save_json, seed_everything
from data import CaptionTokens, OnlineCaptionPairs, collate_eval
from models import build_model


def score(change_agent_root, references, hypotheses):
    root = str(Path(change_agent_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    from eval_func.bleu.bleu import Bleu
    from eval_func.rouge.rouge import Rouge
    from eval_func.cider.cider import Cider
    gts = {i: [" ".join(map(str, row)) for row in group] for i, group in enumerate(references)}
    res = {i: [" ".join(map(str, row))] for i, row in enumerate(hypotheses)}
    bleu, _ = Bleu(4).compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    return {
        "BLEU-1": float(bleu[0]), "BLEU-2": float(bleu[1]),
        "BLEU-3": float(bleu[2]), "BLEU-4": float(bleu[3]),
        "ROUGE-L": float(rouge), "CIDEr": float(cider),
    }


def clean_ids(ids, tokens):
    ignored = {tokens.start, tokens.end, tokens.pad}
    return [int(value) for value in ids if int(value) not in ignored]


def reference_ids(words, tokens):
    encoded, _ = tokens.encode(words)
    return clean_ids(encoded.tolist(), tokens)


def condition_list(argument):
    if argument == "all":
        return ["clean"] + [f"{kind}__{side}" for kind in ALL_DEGRADATIONS for side in ("A", "B", "both")]
    return [x.strip() for x in argument.split(",") if x.strip()]


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--conditions", default="all")
parser.add_argument("--output", required=True)
parser.add_argument("--batch-size", type=int, default=16)
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = checkpoint["config"]
seed_everything(int(cfg["seed"]))
device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
tokens = CaptionTokens(cfg["metadata_root"], int(cfg.get("max_length", 41)))
model = build_model(cfg, tokens.vocab)
missing, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
if unexpected:
    raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
model = model.to(device).eval()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
all_metrics = {}

for condition in condition_list(args.conditions):
    dataset = OnlineCaptionPairs(cfg, "test", tokens, training=False, condition=condition)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)), pin_memory=device.type == "cuda",
        collate_fn=collate_eval,
    )
    references, hypotheses, rows = [], [], []
    started = time.time()
    with torch.inference_mode():
        for step, batch in enumerate(loader, 1):
            sequences = model.generate(
                batch["a"].to(device, non_blocking=True),
                batch["b"].to(device, non_blocking=True),
            )
            for name, sequence, refs in zip(batch["name"], sequences, batch["references"]):
                pred_ids = clean_ids(sequence, tokens)
                ref_ids = [reference_ids(words, tokens) for words in refs]
                hypotheses.append(pred_ids)
                references.append(ref_ids)
                rows.append({"name": name, "prediction_ids": pred_ids, "reference_ids": ref_ids})
            if step % 10 == 0 or step == len(loader):
                print(
                    f"condition={condition} batches={step}/{len(loader)} "
                    f"samples={len(rows)}/{len(dataset)} elapsed={time.time()-started:.1f}s",
                    flush=True,
                )
    metrics = score(cfg["change_agent_root"], references, hypotheses)
    payload = {
        "architecture": cfg["architecture"], "variant": checkpoint["variant"],
        "family": cfg["family"], "condition": condition, "samples": len(rows),
        "severity": cfg.get("severity", 2),
        "degradation_seed": cfg["degradation_seed"], **metrics,
    }
    save_json(payload, output / f"{condition}.metrics.json")
    save_json(rows, output / f"{condition}.predictions.json")
    all_metrics[condition] = payload
    print(json.dumps(payload, ensure_ascii=False), flush=True)

save_json(all_metrics, output / "all_metrics.json")
