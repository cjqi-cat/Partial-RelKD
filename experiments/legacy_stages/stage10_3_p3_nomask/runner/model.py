"""Architecture shim for the mask-free Partial-RelKD encoder."""
from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = Path("__PROJECT_ROOT__/stage6_7_partial_hybrid/runner/model.py")
spec = importlib.util.spec_from_file_location("stage103_architecture", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import architecture source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

LoRALinear = module.LoRALinear
GradientReverse = module.GradientReverse
DinoRobustModel = module.DinoRobustModel
