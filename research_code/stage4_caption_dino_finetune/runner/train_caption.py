from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from caption_model import ChangeCaptioner
from common import caption_metrics, encode_caption, seed_everything, strip_special


class CaptionFeatureDataset(Dataset):
    def __init__(self, cache, split, token_root, vocab, max_length=41, training=False):
        self.a = np.load(cache / f"{split}_a.npy", mmap_mode="r")
        self.b = np.load(cache / f"{split}_b.npy", mmap_mode="r")
        self.names = np.load(cache / f"{split}_names.npy")
        self.token_root, self.vocab, self.max_length, self.training = token_root, vocab, max_length, training

    def __len__(self): return len(self.names)

    def refs(self, name):
        return json.loads((self.token_root / f"{Path(str(name)).stem}.txt").read_text(encoding="utf-8"))

    def __getitem__(self, index):
        name = str(self.names[index]); refs = self.refs(name)
        nochange = ["<START>", "the", "scene", "is", "the", "same", "as", "before", "<END>"]
        words = nochange if self.training and nochange in refs else random.choice(refs)
        tokens, length = encode_caption(words, self.vocab, self.max_length)
        all_refs = [encode_caption(row, self.vocab, self.max_length)[0] for row in refs]
        return torch.from_numpy(np.array(self.a[index], dtype=np.float32)), torch.from_numpy(np.array(self.b[index], dtype=np.float32)), torch.from_numpy(tokens), length, torch.from_numpy(np.stack(all_refs)), name


@torch.inference_mode()
def validate(model, loader, device, vocab, change_agent_root):
    model.eval(); references, hypotheses = [], []
    for a, b, _, _, all_refs, _ in loader:
        sequence = model.greedy(a.to(device), b.to(device)).cpu().tolist()
        for prediction, ref_set in zip(sequence, all_refs.tolist()):
            hypotheses.append(strip_special(prediction, vocab)); references.append([strip_special(row, vocab) for row in ref_set])
    return caption_metrics(change_agent_root, references, hypotheses)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["lora", "partial_last2"], required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--token-root", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--change-agent-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4); parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args(); seed_everything(args.seed)
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "logs").mkdir(exist_ok=True); (args.output / "checkpoints").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(args.output / "logs" / f"train_{args.variant}.log", encoding="utf-8"), logging.StreamHandler()])
    vocab = json.loads(args.vocab.read_text(encoding="utf-8")); model = ChangeCaptioner(args.change_agent_root, vocab).to(args.device)
    train = CaptionFeatureDataset(args.cache, "train", args.token_root, vocab, training=True)
    val = CaptionFeatureDataset(args.cache, "val", args.token_root, vocab)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, generator=generator)
    val_loader = DataLoader(val, batch_size=args.val_batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    best, stale, rows = -math.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train(); total, seen, started = 0.0, 0, time.time()
        for step, (a, b, tokens, _, _, _) in enumerate(train_loader, 1):
            a, b, tokens = a.to(args.device), b.to(args.device), tokens.to(args.device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                logits = model(a, b, tokens[:, :-1]); target = tokens[:, 1:]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=vocab["<NULL>"])
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(optimizer); scaler.update()
            total += float(loss) * len(a); seen += len(a)
            if step % 50 == 0: logging.info("variant=%s epoch=%d step=%d/%d loss=%.4f", args.variant, epoch, step, len(train_loader), total/seen)
            if args.smoke_batches and step >= args.smoke_batches: break
        score = validate(model, val_loader, args.device, vocab, args.change_agent_root)
        row = {"epoch": epoch, "train_loss": total/seen, **score, "elapsed_min": (time.time()-started)/60}; rows.append(row)
        logging.info("variant=%s epoch=%d loss=%.4f BLEU4=%.5f ROUGE_L=%.5f CIDEr=%.5f elapsed_min=%.1f", args.variant, epoch, row["train_loss"], score["Bleu_4"], score["ROUGE_L"], score["CIDEr"], row["elapsed_min"])
        if score["Bleu_4"] > best + 1e-5:
            best, stale = score["Bleu_4"], 0
            torch.save({"variant": args.variant, "epoch": epoch, "metrics": score, "model": model.state_dict(), "args": vars(args)}, args.output / "checkpoints" / f"{args.variant}_best.pt")
        else:
            stale += 1
            if stale >= args.patience: logging.info("early_stop variant=%s epoch=%d best_bleu4=%.5f", args.variant, epoch, best); break
        if args.smoke_batches: break
    with (args.output / f"history_{args.variant}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps({"status": "complete", "variant": args.variant, "best_bleu4": best, "epochs": rows[-1]["epoch"]}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
