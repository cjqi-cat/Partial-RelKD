from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from common import allowed_degradations, load_operators, stable_seed


DINO_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
DINO_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]


def image_tensor(image):
    image = image.convert("RGB")
    if image.size != (224, 224):
        image = image.resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(((array - DINO_MEAN) / DINO_STD).copy())


class CaptionTokens:
    def __init__(self, metadata_root, max_length=41):
        self.root = Path(metadata_root)
        self.token_root = self.root / "tokens"
        self.vocab = json.loads((self.root / "vocab.json").read_text(encoding="utf-8"))
        self.id_to_word = {int(value): key for key, value in self.vocab.items()}
        self.pad = self.vocab["<NULL>"]
        self.start = self.vocab["<START>"]
        self.end = self.vocab["<END>"]
        self.unk = self.vocab["<UNK>"]
        self.max_length = max_length

    def references(self, name):
        return json.loads((self.token_root / f"{Path(name).stem}.txt").read_text(encoding="utf-8"))

    def encode(self, words):
        ids = [self.vocab.get(word, self.unk) for word in words][: self.max_length]
        if not ids or ids[0] != self.start:
            ids.insert(0, self.start)
        if ids[-1] != self.end:
            ids.append(self.end)
        ids = ids[: self.max_length]
        if ids[-1] != self.end:
            ids[-1] = self.end
        length = len(ids)
        ids += [self.pad] * (self.max_length - length)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(length, dtype=torch.long)


class OnlineCaptionPairs(Dataset):
    def __init__(self, cfg, split, tokens, training=True, condition="clean"):
        self.cfg = cfg
        self.split = split
        self.tokens = tokens
        self.training = training
        self.condition = condition
        self.epoch = 0
        metadata = Path(cfg["metadata_root"])
        self.names = [
            row.strip() for row in (metadata / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            if row.strip()
        ]
        self.image_root = Path(cfg["dataset_root"]) / "images" / split
        self.operators = load_operators(cfg["degradation_script"])
        self.allowed = list(cfg.get("allowed_degradations") or allowed_degradations(cfg["family"]))
        leakage = set(self.allowed) & set(cfg.get("heldout_degradations", []))
        if leakage:
            raise ValueError(f"Training degradation leakage: {sorted(leakage)}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.names)

    def _open(self, name, side):
        return Image.open(self.image_root / side / name).convert("RGB")

    def _apply(self, image, kind, seed):
        output, _ = self.operators[kind](
            image, int(self.cfg.get("severity", 2)), np.random.default_rng(seed)
        )
        return output.convert("RGB") if isinstance(output, Image.Image) else Image.fromarray(np.asarray(output, dtype=np.uint8))

    def _train_view(self, image, name, side):
        seed = stable_seed(self.cfg["seed"], self.epoch, name, side)
        rng = np.random.default_rng(seed)
        if rng.random() < float(self.cfg.get("clean_probability", 0.25)):
            return image.copy()
        kind = self.allowed[int(rng.integers(0, len(self.allowed)))]
        return self._apply(image, kind, stable_seed(seed, kind))

    def _test_view(self, image, name, side):
        if self.condition == "clean":
            return image
        kind, target = self.condition.rsplit("__", 1)
        if target not in (side, "both"):
            return image
        seed = stable_seed(self.cfg["degradation_seed"], kind, target, name, side)
        return self._apply(image, kind, seed)

    def __getitem__(self, index):
        name = self.names[index]
        clean_a, clean_b = self._open(name, "A"), self._open(name, "B")
        refs = self.tokens.references(name)
        if self.training:
            canonical = ["<START>", "the", "scene", "is", "the", "same", "as", "before", "<END>"]
            pick = canonical if canonical in refs else refs[
                stable_seed(self.cfg["seed"], self.epoch, name, "caption") % len(refs)
            ]
            encoded, length = self.tokens.encode(pick)
            return {
                "name": name,
                "a": image_tensor(self._train_view(clean_a, name, "A")),
                "b": image_tensor(self._train_view(clean_b, name, "B")),
                "clean_a": image_tensor(clean_a),
                "clean_b": image_tensor(clean_b),
                "tokens": encoded,
                "length": length,
            }
        return {
            "name": name,
            "a": image_tensor(self._test_view(clean_a, name, "A")),
            "b": image_tensor(self._test_view(clean_b, name, "B")),
            "references": refs,
        }


def collate_eval(batch):
    return {
        "name": [row["name"] for row in batch],
        "a": torch.stack([row["a"] for row in batch]),
        "b": torch.stack([row["b"] for row in batch]),
        "references": [row["references"] for row in batch],
    }
