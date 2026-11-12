"""Compatibility import for canonical caption and distillation losses."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from partial_relkd.losses.distillation import (  # noqa: E402,F401
    caption_loss,
    relation_kd,
    token_kd,
)

__all__ = ["caption_loss", "relation_kd", "token_kd"]
