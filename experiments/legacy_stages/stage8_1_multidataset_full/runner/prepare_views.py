#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


DATASETS = {
    "SECOND_CC": {
        "raw": "__DATA_ROOT__/SECOND-CC-AUG",
        "metadata": "__PROJECT_ROOT__/stage8_1_multidataset_full/metadata/SECOND_CC",
        "layout": "second",
    },
    "WHU_CDC": {
        "raw": "__DATA_ROOT__/WHU_CDC/whu_CDC_dataset",
        "metadata": "__PROJECT_ROOT__/stage8_1_multidataset_full/metadata/WHU_CDC",
        "layout": "whu",
    },
}


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"Wrong symlink: {target} -> {target.resolve()}")
        return
    if target.exists():
        raise RuntimeError(f"Refusing to replace existing path: {target}")
    os.symlink(source, target, target_is_directory=True)


def names(metadata: Path, split: str) -> list[str]:
    return [x.strip() for x in (metadata / f"{split}.txt").read_text(encoding="utf-8").splitlines() if x.strip()]


def prepare_second(raw: Path, metadata: Path, output: Path) -> None:
    rows = []
    for split in ("train", "val", "test"):
        link(raw / split / "rgb" / "A", output / "images" / split / "A")
        link(raw / split / "rgb" / "B", output / "images" / split / "B")
        label_root = output / "images" / split / "label"
        label_root.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(names(metadata, split), 1):
            target = label_root / name
            if not target.exists():
                with Image.open(raw / split / "sem" / "A" / name) as a_img:
                    a = np.asarray(a_img)
                with Image.open(raw / split / "sem" / "B" / name) as b_img:
                    b = np.asarray(b_img)
                if a.shape != b.shape:
                    raise RuntimeError(f"Semantic shape mismatch: {name}: {a.shape} vs {b.shape}")
                if a.ndim == 3:
                    changed = np.any(a != b, axis=2)
                else:
                    changed = a != b
                Image.fromarray((changed.astype(np.uint8) * 255), mode="L").save(target, optimize=True)
            with Image.open(target) as mask:
                changeflag = int(np.asarray(mask.convert("L")).max() > 0)
            rows.append({"filename": name, "split": split, "changeflag": changeflag})
            if index % 1000 == 0:
                print(f"SECOND_CC {split}: {index}", flush=True)
    (output / "LevirCCcaptions.json").write_text(json.dumps({"images": rows}), encoding="utf-8")


def prepare_whu(raw: Path, metadata: Path, output: Path) -> None:
    rows = []
    base = raw / "images" if (raw / "images").is_dir() else raw
    for split in ("train", "val", "test"):
        for side in ("A", "B", "label"):
            link(base / split / side, output / "images" / split / side)
        for name in names(metadata, split):
            with Image.open(base / split / "label" / name) as mask:
                changeflag = int(np.asarray(mask.convert("L")).max() > 0)
            rows.append({"filename": name, "split": split, "changeflag": changeflag})
    (output / "LevirCCcaptions.json").write_text(json.dumps({"images": rows}), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="__PROJECT_ROOT__/stage8_1_multidataset_full")
    parser.add_argument("--dataset", choices=tuple(DATASETS), action="append")
    args = parser.parse_args()
    root = Path(args.root)
    for dataset in args.dataset or list(DATASETS):
        spec = DATASETS[dataset]
        raw, metadata = Path(spec["raw"]), Path(spec["metadata"])
        output = root / "data_views" / dataset
        output.mkdir(parents=True, exist_ok=True)
        if spec["layout"] == "second":
            prepare_second(raw, metadata, output)
        else:
            prepare_whu(raw, metadata, output)
        print(f"VIEW_OK {dataset} {output}", flush=True)


if __name__ == "__main__":
    main()
