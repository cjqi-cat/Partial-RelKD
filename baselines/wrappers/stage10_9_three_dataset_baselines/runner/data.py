from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from common import SEEN, load_operators, stable_seed

MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)[:, None, None]
STD = np.asarray((0.229, 0.224, 0.225), np.float32)[:, None, None]


def image_tensor(image, size=256):
    image = image.convert("RGB")
    if image.size != (size, size): image = image.resize((size, size), Image.Resampling.BICUBIC)
    x = np.asarray(image, np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).copy())


class CaptionTokens:
    def __init__(self, root, max_length=41):
        self.root = Path(root); self.max_length = int(max_length)
        self.vocab = json.loads((self.root / "vocab.json").read_text(encoding="utf-8"))
        self.pad = self.vocab["<NULL>"]; self.start = self.vocab["<START>"]
        self.end = self.vocab["<END>"]; self.unk = self.vocab["<UNK>"]

    def references(self, name):
        return json.loads((self.root / "tokens" / f"{Path(name).stem}.txt").read_text(encoding="utf-8"))

    def encode(self, words):
        ids = [self.vocab.get(x, self.unk) for x in words]
        if not ids or ids[0] != self.start: ids.insert(0, self.start)
        if ids[-1] != self.end: ids.append(self.end)
        ids = ids[:self.max_length]; ids[-1] = self.end; length = len(ids)
        ids += [self.pad] * (self.max_length - length)
        return torch.tensor(ids), torch.tensor(length)


class CaptionPairs(Dataset):
    def __init__(self, cfg, split, tokens, training=False, condition="clean"):
        self.cfg, self.tokens, self.training, self.condition, self.epoch = cfg, tokens, training, condition, 0
        meta = Path(cfg["metadata_root"])
        self.names = [x.strip() for x in (meta / f"{split}.txt").read_text().splitlines() if x.strip()]
        self.image_root = Path(cfg["dataset_root"]) / "images" / split
        self.ops = load_operators(cfg["degradation_script"])
        self.allowed = list(cfg.get("allowed_degradations", SEEN))
        forbidden = set(cfg.get("heldout_degradations", ()))
        overlap = forbidden.intersection(self.allowed)
        if overlap: raise ValueError(f"LOFO leakage: {sorted(overlap)}")
        self.size = int(cfg.get("image_size", 256))

    def set_epoch(self, epoch): self.epoch = int(epoch)
    def __len__(self): return len(self.names)
    def _open(self, name, side): return Image.open(self.image_root / side / name).convert("RGB")

    def _apply(self, image, kind, seed):
        out, _ = self.ops[kind](image, int(self.cfg.get("severity", 2)), np.random.default_rng(seed))
        return out.convert("RGB") if isinstance(out, Image.Image) else Image.fromarray(np.asarray(out, np.uint8))

    def _view(self, image, name, side):
        if self.training:
            seed = stable_seed(self.cfg["seed"], self.epoch, name, side)
            rng = np.random.default_rng(seed)
            if rng.random() < float(self.cfg.get("clean_probability", .25)): return image
            kind = self.allowed[int(rng.integers(len(self.allowed)))]
            return self._apply(image, kind, stable_seed(seed, kind))
        if self.condition == "clean": return image
        kind, target = self.condition.rsplit("__", 1)
        if target not in (side, "both"): return image
        return self._apply(image, kind, stable_seed(self.cfg["degradation_seed"], kind, target, name, side))

    def __getitem__(self, index):
        name = self.names[index]; refs = self.tokens.references(name)
        row = {"name": name,
               "a": image_tensor(self._view(self._open(name, "A"), name, "A"), self.size),
               "b": image_tensor(self._view(self._open(name, "B"), name, "B"), self.size)}
        if self.training:
            canonical = ["<START>", "the", "scene", "is", "the", "same", "as", "before", "<END>"]
            chosen = canonical if canonical in refs else refs[stable_seed(self.cfg["seed"], self.epoch, name, "caption") % len(refs)]
            row["tokens"], row["length"] = self.tokens.encode(chosen)
        else: row["references"] = refs
        return row


def collate_eval(batch):
    return {"name": [x["name"] for x in batch], "a": torch.stack([x["a"] for x in batch]),
            "b": torch.stack([x["b"] for x in batch]), "references": [x["references"] for x in batch]}
