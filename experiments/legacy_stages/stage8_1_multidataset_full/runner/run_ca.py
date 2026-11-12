#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from ca_target import build_model


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("train", "evaluate"), required=True)
    parser.add_argument("--stage66-runner", required=True)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--resume")
    args = parser.parse_args()
    runner = Path(args.stage66_runner).resolve()
    sys.path.insert(0, str(runner))
    if args.action == "train":
        module = load(runner / "train_m2.py", "stage81_ca_train")
        module.build_model = build_model
        argv = ["train_m2.py", "--config", args.config, "--variant", "m1", "--output", args.output]
        if args.resume:
            raise ValueError("Target Change-Agent runner currently resumes by rerunning from best; omit --resume")
    else:
        argv = ["evaluate.py", "--checkpoint", args.checkpoint, "--conditions", args.conditions, "--output", args.output]
        sys.argv = argv
        import models as stage66_models
        stage66_models.build_model = build_model
        # The Stage6.6 evaluator is a script and runs while imported.
        load(runner / "evaluate.py", "stage81_ca_eval")
        return
    sys.argv = argv
    module.main()


if __name__ == "__main__":
    main()
