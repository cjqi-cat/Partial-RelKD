from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_names(dataset_root: Path, split: str, limit: int, seed: int):
    with (dataset_root / "LevirCCcaptions.json").open("r", encoding="utf-8") as handle:
        rows = json.load(handle)["images"]
    groups = {0: [], 1: []}
    for row in rows:
        if row["split"] == split:
            groups[int(row["changeflag"])].append(row["filename"])
    rng = random.Random(seed + (0 if split == "train" else 1000))
    each = min(limit // 2, len(groups[0]), len(groups[1]))
    chosen = [(name, label) for label in (0, 1) for name in rng.sample(groups[label], each)]
    rng.shuffle(chosen)
    return [item[0] for item in chosen], np.asarray([item[1] for item in chosen], dtype=np.int64)


def load_degradations(script: Path):
    spec = importlib.util.spec_from_file_location("levir_degradations", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import degradation script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OPERATORS


def atomic_torch_save(payload, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in names}
