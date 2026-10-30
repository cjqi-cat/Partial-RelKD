from __future__ import annotations

import hashlib
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch


ALL_DEGRADATIONS = (
    "cloud", "haze", "gaussian_noise", "blur", "jpeg",
    "low_resolution", "pixel_dropout", "misregistration",
    "radiometric_shift", "mixed",
)
FAMILY_HOLDOUTS = {
    "geometry_quality": {"blur", "low_resolution", "misregistration", "mixed"},
    "atmospheric": {"cloud", "haze", "mixed"},
    "noise_loss": {"gaussian_noise", "jpeg", "pixel_dropout", "mixed"},
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_operators(path):
    return dict(load_module(path, "stage67_degradations").OPERATORS)


def stable_seed(base_seed, *parts):
    payload = "|".join((str(base_seed), *(str(x) for x in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def allowed_degradations(family):
    heldout = FAMILY_HOLDOUTS[family]
    return [name for name in ALL_DEGRADATIONS if name not in heldout]


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
