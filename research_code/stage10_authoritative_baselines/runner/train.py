from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from common import atomic_save, load_json, seed_everything
from data import CaptionPairs, CaptionTokens
from models import build_model


def caption_loss(output, label_smoothing=.05):
    scores, captions, lengths, _ = output
    targets = captions[:, 1:]
    scores = pack_padded_sequence(scores, lengths, batch_first=True, enforce_sorted=True).data
    targets = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=True).data
    return torch.nn.functional.cross_entropy(scores.float(), targets, label_smoothing=label_smoothing)


def run_epoch(model, loader, device, optimizer=None, scaler=None, smoothing=.05):
    train = optimizer is not None
    model.train(train)
    total, count = 0., 0
    for batch in loader:
        a, b = batch["a"].to(device), batch["b"].to(device)
        tokens, lengths = batch["tokens"].to(device), batch["length"].to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = caption_loss(model(a, b, tokens, lengths), smoothing)
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
            scaler.step(optimizer); scaler.update()
        total += float(loss) * a.size(0); count += a.size(0)
    return total / count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--resume")
    args = ap.parse_args()
    cfg = load_json(args.config); cfg["output_root"] = str(Path(args.output).resolve())
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens = CaptionTokens(cfg["metadata_root"], cfg.get("max_length", 41))
    model = build_model(cfg, tokens.vocab).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg.get("lr", 5e-5), weight_decay=cfg.get("weight_decay", .01))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start, best, bad = 0, float("inf"), 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"]); optimizer.load_state_dict(state["optimizer_state"])
        start, best, bad = state["epoch"] + 1, state["best_val"], state.get("bad_epochs", 0)
    train_ds = CaptionPairs(cfg, "train", tokens, training=True)
    val_ds = CaptionPairs(cfg, "val", tokens, training=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg.get("num_workers", 4), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["eval_batch_size"], shuffle=False,
                            num_workers=cfg.get("num_workers", 4), pin_memory=True)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    for epoch in range(start, cfg.get("epochs", 30)):
        tic = time.time(); train_ds.set_epoch(epoch); val_ds.set_epoch(0)
        tr = run_epoch(model, train_loader, device, optimizer, scaler, cfg.get("label_smoothing", .05))
        with torch.no_grad(): va = run_epoch(model, val_loader, device, smoothing=cfg.get("label_smoothing", .05))
        record = {"epoch": epoch, "train_loss": tr, "val_allowed_mix_loss": va, "seconds": time.time()-tic}
        with log.open("a", encoding="utf-8") as f: f.write(json.dumps(record) + "\n")
        print(record, flush=True)
        improved = va < best - cfg.get("min_delta", 1e-4)
        if improved: best, bad = va, 0
        else: bad += 1
        payload = {"config": cfg, "model": cfg["model"], "epoch": epoch, "best_val": best,
                   "bad_epochs": bad, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}
        atomic_save(payload, out / "last.pt")
        if improved: atomic_save(payload, out / "best.pt")
        if bad >= cfg.get("patience", 8): break
    print("TRAINING_OK", flush=True)


if __name__ == "__main__": main()
