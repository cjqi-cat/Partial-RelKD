#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from common import load_json, seed_everything
from data import CaptionTokens, OnlineCaptionPairs
from losses import caption_loss, token_kd
from models import build_model, frozen_teacher


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = load_json(args.config)
seed_everything(int(cfg["seed"]))
if not torch.cuda.is_available():
    raise RuntimeError("The smoke test requires CUDA because upstream Change-Agent uses CUDA positional indices")
device = torch.device("cuda")
tokens = CaptionTokens(cfg["metadata_root"], int(cfg.get("max_length", 41)))
dataset = OnlineCaptionPairs(cfg, "train", tokens, training=True)
item = dataset[0]
student = build_model(cfg, tokens.vocab).to(device).eval()
teacher = frozen_teacher(student)
a = item["a"].unsqueeze(0).to(device)
b = item["b"].unsqueeze(0).to(device)
ca = item["clean_a"].unsqueeze(0).to(device)
cb = item["clean_b"].unsqueeze(0).to(device)
caption = item["tokens"].unsqueeze(0).to(device)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    student_logits = student(a, b, caption[:, :-1])
    teacher_logits = teacher(ca, cb, caption[:, :-1])
cap = caption_loss(student_logits, caption[:, 1:], tokens.pad)
kd = token_kd(student_logits, teacher_logits, caption[:, 1:], tokens.pad)
print({
    "architecture": cfg["architecture"], "name": item["name"],
    "student_logits": tuple(student_logits.shape),
    "teacher_logits": tuple(teacher_logits.shape),
    "caption_loss": float(cap), "token_kd": float(kd),
    "allowed": dataset.allowed,
})
print("SMOKE_OK")
