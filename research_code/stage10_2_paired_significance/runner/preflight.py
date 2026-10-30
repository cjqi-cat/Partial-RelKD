#!/usr/bin/env python3
from __future__ import annotations

import sys

from bootstrap_p3 import CONDITIONS, METRIC_ROOT, MODELS


def main():
    if not METRIC_ROOT.exists():
        raise FileNotFoundError(METRIC_ROOT)
    for model, root in MODELS.items():
        metrics = root / "all_metrics.json"
        if not metrics.is_file():
            raise FileNotFoundError(metrics)
        missing = [c for c in CONDITIONS if not (root / f"{c}.predictions.json").is_file()]
        if missing:
            raise FileNotFoundError(f"{model}: missing {missing}")
        print(f"OK {model}: {len(CONDITIONS)} unseen conditions", flush=True)
    print("PREFLIGHT_OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PREFLIGHT_FAILED: {exc}", file=sys.stderr)
        raise
