from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image

from common import load_encoder, seed_everything

MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--list-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def image_tensor(path: Path):
    with Image.open(path) as opened:
        image = opened.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1) - MEAN) / STD


def extract(model, dataset_root, list_root, split, output, batch_size, device, limit=None):
    names = [line.strip() for line in (list_root / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit:
        names = names[:limit]
    arrays = {side: open_memmap(output / f"{split}_{side.lower()}.npy", mode="w+", dtype=np.float16, shape=(len(names), 256, 768)) for side in ("A", "B")}
    pending, locations, started = [], [], time.time()

    def flush():
        if not pending: return
        images = torch.stack(pending).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            _, features = model.encode_images(images)
        values = features.cpu().numpy().astype(np.float16)
        for (row, side), value in zip(locations, values): arrays[side][row] = value
        pending.clear(); locations.clear()

    for row, name in enumerate(names):
        for side in ("A", "B"):
            pending.append(image_tensor(dataset_root / "images" / split / side / name)); locations.append((row, side))
            if len(pending) >= batch_size: flush()
        if (row + 1) % 250 == 0 or row + 1 == len(names):
            logging.info("split=%s pairs=%d/%d elapsed_min=%.1f", split, row + 1, len(names), (time.time()-started)/60)
    flush()
    for array in arrays.values(): array.flush()
    np.save(output / f"{split}_names.npy", np.asarray(names))
    return len(names)


def main():
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True); (args.output / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(args.output / "logs" / "prepare.log", encoding="utf-8"), logging.StreamHandler()])
    seed_everything(args.seed)
    model, checkpoint, missing = load_encoder(args.stage1_root, args.checkpoint, args.device)
    logging.info("variant=%s checkpoint_epoch=%s frozen_missing=%d", args.variant, checkpoint.get("epoch"), missing)
    counts = {split: extract(model, args.dataset_root, args.list_root, split, args.output, args.batch_size, args.device, args.limit) for split in args.splits}
    (args.output / "manifest.json").write_text(json.dumps({"status": "complete", "variant": args.variant, "checkpoint_epoch": checkpoint.get("epoch"), "counts": counts, "shape": "N x 256 x 768", "dtype": "float16"}, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
