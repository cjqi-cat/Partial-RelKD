from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch

ALL_DEGRADATIONS = (
    "cloud", "haze", "gaussian_noise", "blur", "jpeg", "low_resolution",
    "pixel_dropout", "misregistration", "radiometric_shift", "mixed",
)
SEEN = ("cloud", "haze", "gaussian_noise", "jpeg", "pixel_dropout", "radiometric_shift")
UNSEEN = ("blur", "low_resolution", "misregistration")
SIDES = ("A", "B", "both")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_seed(base_seed, *parts):
    raw = "|".join((str(base_seed), *(str(x) for x in parts))).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_operators(path):
    spec = importlib.util.spec_from_file_location("stage109_degradations", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return dict(module.OPERATORS)


def conditions():
    return ["clean"] + [f"{d}__{s}" for d in ALL_DEGRADATIONS for s in SIDES]


def atomic_save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp); tmp.replace(path)
