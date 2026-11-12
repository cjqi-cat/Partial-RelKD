"""Compatibility import for the canonical visual adapter implementation."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from partial_relkd.models.visual_adapter import (  # noqa: E402,F401
    DinoRobustModel,
    GradientReverse,
    LoRALinear,
)

__all__ = ["DinoRobustModel", "GradientReverse", "LoRALinear"]
