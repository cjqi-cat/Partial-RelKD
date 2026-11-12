#!/usr/bin/env python3
from _delegate import run_entry

run_entry(
    "stage4_caption_dino_finetune/runner/train_caption.py",
    "Train the clean caption teacher.",
)
