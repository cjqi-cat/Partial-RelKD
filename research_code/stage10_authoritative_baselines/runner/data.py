from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from common import allowed_degradations, load_operators, stable_seed

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), np.float32)[:, None, None]
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), np.float32)[:, None, None]
CLIP_MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), np.float32)[:, None, None]
CLIP_STD = np.asarray((0.26862954, 0.26130258, 0.27577711), np.float32)[:, None, None]


def image_tensor(image, size, normalization):
    image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    x = np.asarray(image, np.float32).transpose(2, 0, 1) / 255.0
    mean, std = (CLIP_MEAN, CLIP_STD) if normalization == "clip" else (IMAGENET_MEAN, IMAGENET_STD)
    return torch.from_numpy(((x - mean) / std).copy())


class CaptionTokens:
    def __init__(self, metadata_root, max_length=41):
        self.root = Path(metadata_root)
        self.token_root = self.root / "tokens"
        self.vocab = json.loads((self.root / "vocab.json").read_text(encoding="utf-8"))
        self.id_to_word = {int(v): k for k, v in self.vocab.items()}
        self.pad, self.start = self.vocab["<NULL>"], self.vocab["<START>"]
        self.end, self.unk = self.vocab["<END>"], self.vocab["<UNK>"]
        self.max_length = int(max_length)

    def references(self, name):
        return json.loads((self.token_root / f"{Path(name).stem}.txt").read_text(encoding="utf-8"))

    def encode(self, words):
        ids = [self.vocab.get(w, self.unk) for w in words]
        if not ids or ids[0] != self.start:
            ids.insert(0, self.start)
        if ids[-1] != self.end:
            ids.append(self.end)
        ids = ids[:self.max_length]
        ids[-1] = self.end
        length = len(ids)
        ids += [self.pad] * (self.max_length - length)
        return torch.tensor(ids), torch.tensor(length)


class CaptionPairs(Dataset):
    def __init__(self, cfg, split, tokens, training=False, condition="clean"):
        self.cfg, self.split, self.tokens = cfg, split, tokens
        self.training, self.condition, self.epoch = training, condition, 0
        meta = Path(cfg["metadata_root"])
        self.names = [x.strip() for x in (meta / f"{split}.txt").read_text().splitlines() if x.strip()]
        self.image_root = Path(cfg["dataset_root"]) / "images" / split
        self.ops = load_operators(cfg["degradation_script"])
        self.allowed = list(cfg.get("allowed_degradations") or allowed_degradations(cfg["family"]))
        heldout = set(cfg.get("heldout_degradations", []))
        if set(self.allowed) & heldout:
            raise ValueError(f"LOFO leakage: {sorted(set(self.allowed) & heldout)}")
        self.size = int(cfg.get("image_size", 256))
        self.normalization = cfg.get("normalization", "imagenet")

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
            if rng.random() < float(self.cfg.get("clean_probability", .25)):
                return image
            kind = self.allowed[int(rng.integers(len(self.allowed)))]
            return self._apply(image, kind, stable_seed(seed, kind))
        if self.condition == "clean": return image
        if self.condition == "seen_fixed":
            seed = stable_seed(self.cfg["degradation_seed"], "seen_fixed", name, side)
            kind = self.allowed[seed % len(self.allowed)]
            return self._apply(image, kind, stable_seed(seed, kind))
        kind, target = self.condition.rsplit("__", 1)
        if target not in (side, "both"): return image
        seed = stable_seed(self.cfg["degradation_seed"], kind, target, name, side)
        return self._apply(image, kind, seed)

    def __getitem__(self, index):
        name = self.names[index]
        a, b = self._open(name, "A"), self._open(name, "B")
        refs = self.tokens.references(name)
        row = {
            "name": name,
            "a": image_tensor(self._view(a, name, "A"), self.size, self.normalization),
            "b": image_tensor(self._view(b, name, "B"), self.size, self.normalization),
        }
        if self.training:
            canonical = ["<START>", "the", "scene", "is", "the", "same", "as", "before", "<END>"]
            chosen = canonical if canonical in refs else refs[stable_seed(self.cfg["seed"], self.epoch, name, "caption") % len(refs)]
            row["tokens"], row["length"] = self.tokens.encode(chosen)
        else:
            row["references"] = refs
        return row


def collate_eval(batch):
    return {"name": [x["name"] for x in batch], "a": torch.stack([x["a"] for x in batch]),
            "b": torch.stack([x["b"] for x in batch]), "references": [x["references"] for x in batch]}
