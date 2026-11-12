from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def balanced_names(dataset_root: Path, split: str, limit: int, seed: int):
    rows=json.loads((dataset_root/"LevirCCcaptions.json").read_text(encoding="utf-8"))["images"]
    groups={0:[],1:[]}
    for row in rows:
        if row["split"]==split: groups[int(row["changeflag"])].append(row["filename"])
    rng=random.Random(seed+(0 if split=="train" else 1000)); each=min(limit//2,len(groups[0]),len(groups[1]))
    chosen=[(name,label) for label in (0,1) for name in rng.sample(groups[label],each)]; rng.shuffle(chosen)
    return [x[0] for x in chosen],np.asarray([x[1] for x in chosen],dtype=np.int64)


def load_degradations(path: Path):
    spec=importlib.util.spec_from_file_location("calibrated_v3",path)
    if spec is None or spec.loader is None: raise RuntimeError(f"cannot import {path}")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.OPERATORS
