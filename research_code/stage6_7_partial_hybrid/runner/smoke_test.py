#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from common import load_json, seed_everything
from data import CaptionTokens, OnlineCaptionPairs
from losses import caption_loss, relation_kd, token_kd
from models import build_model, frozen_teacher


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_json(args.config)
seed_everything(int(cfg["seed"]))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
device = torch.device("cuda")
tokens = CaptionTokens(cfg["metadata_root"], int(cfg.get("max_length", 41)))
item = OnlineCaptionPairs(cfg, "train", tokens, training=True)[0]
student = build_model(cfg, tokens.vocab).to(device).train()
teacher = frozen_teacher(student)
a = item["a"].unsqueeze(0).to(device)
b = item["b"].unsqueeze(0).to(device)
ca = item["clean_a"].unsqueeze(0).to(device)
cb = item["clean_b"].unsqueeze(0).to(device)
caption = item["tokens"].unsqueeze(0).to(device)
decoder_in, targets = caption[:, :-1], caption[:, 1:]
with torch.autocast("cuda", dtype=torch.float16):
    dirty_memory = student.memory(a, b)
    dirty_logits = student.decode_memory(dirty_memory, decoder_in)
    with torch.no_grad():
        teacher_memory = teacher.memory(ca, cb)
        teacher_logits = teacher.decode_memory(teacher_memory, decoder_in)
    clean_logits = student.decode_memory(student.memory(ca, cb), decoder_in)
    losses = {
        "dirty": caption_loss(dirty_logits, targets, tokens.pad),
        "clean": caption_loss(clean_logits, targets, tokens.pad),
        "token_kd": token_kd(dirty_logits, teacher_logits, targets, tokens.pad),
        "relation": relation_kd(dirty_memory, teacher_memory),
    }
    total = sum(losses.values())
total.backward()
print({
    "run_name": cfg["run_name"],
    "logits": tuple(dirty_logits.shape),
    "memory": tuple(dirty_memory.shape),
    "trainable": sum(p.numel() for p in student.parameters() if p.requires_grad),
    **{name: float(value.detach()) for name, value in losses.items()},
    "allowed": cfg["allowed_degradations"],
})
print("SMOKE_OK")
