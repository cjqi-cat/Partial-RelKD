#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ca_target import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    runner = Path("__PROJECT_ROOT__/stage6_6_m2_backbones/runner")
    sys.path.insert(0, str(runner))
    from data import CaptionTokens, OnlineCaptionPairs
    from losses import caption_loss
    tokens = CaptionTokens(cfg["metadata_root"], cfg["max_length"])
    dataset = OnlineCaptionPairs(cfg, "train", tokens, training=True)
    item = dataset[0]
    model = build_model(cfg, tokens.vocab).cuda().train()
    a, b = item["a"][None].cuda(), item["b"][None].cuda()
    caption = item["tokens"][None].cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(a, b, caption[:, :-1])
        loss = caption_loss(logits, caption[:, 1:], tokens.pad, 0.05)
    loss.backward()
    print({"dataset": Path(cfg["metadata_root"]).name, "samples": len(dataset),
           "vocab": len(tokens.vocab), "logits": tuple(logits.shape),
           "loss": float(loss.detach()), "allowed": cfg["allowed_degradations"]}, flush=True)
    print("SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
