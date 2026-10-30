from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stable_seed(base_seed: int, *parts: str):
    payload = "|".join((str(base_seed), *map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def encode_caption(words: list[str], vocab: dict[str, int], max_length=41):
    ids = [vocab.get(word, vocab["<UNK>"]) for word in words]
    if not ids or ids[0] != vocab["<START>"]: ids.insert(0, vocab["<START>"])
    if ids[-1] != vocab["<END>"]: ids.append(vocab["<END>"])
    ids = ids[:max_length]
    if ids[-1] != vocab["<END>"]: ids[-1] = vocab["<END>"]
    output = np.full(max_length, vocab["<NULL>"], dtype=np.int64)
    output[:len(ids)] = ids
    return output, len(ids)


def strip_special(row, vocab):
    ignored = {vocab["<START>"], vocab["<END>"], vocab["<NULL>"]}
    return [int(value) for value in row if int(value) not in ignored]


def caption_metrics(change_agent_root: Path, references, hypotheses):
    sys.path.insert(0, str(change_agent_root))
    from eval_func.bleu.bleu import Bleu
    from eval_func.rouge.rouge import Rouge
    from eval_func.cider.cider import Cider
    hypotheses_text = [[" ".join(map(str, row))] for row in hypotheses]
    references_text = [[" ".join(map(str, row)) for row in group] for group in references]
    bleu, _ = Bleu(4).compute_score(references_text, hypotheses_text)
    rouge, _ = Rouge().compute_score(references_text, hypotheses_text)
    cider, _ = Cider().compute_score(references_text, hypotheses_text)
    return {**{f"Bleu_{index + 1}": float(value) for index, value in enumerate(bleu)}, "ROUGE_L": float(rouge), "CIDEr": float(cider)}


def load_encoder(stage1_root: Path, checkpoint_path: Path, device: str):
    model_path = stage1_root / "runner" / "model.py"
    spec = importlib.util.spec_from_file_location("stage1_dino_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load encoder model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    DinoRobustModel = module.DinoRobustModel
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DinoRobustModel(checkpoint["config"])
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state"], strict=False)
    if unexpected: raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
    return model.to(device).eval(), checkpoint, len(missing)
