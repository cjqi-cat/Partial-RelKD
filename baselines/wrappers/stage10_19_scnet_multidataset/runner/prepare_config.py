from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if "image_mean" in cached and "image_std" in cached:
            print({"dataset": cached["dataset"], "image_mean": cached["image_mean"], "image_std": cached["image_std"], "cached": True})
            print("NORMALIZATION_OK")
            return

    metadata_root = Path(config["metadata_root"])
    image_root = Path(config["dataset_root"]) / "images" / "train"
    names = [line.strip() for line in (metadata_root / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    size = int(config["image_size"])
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    pixels = 0
    images = 0
    for name in names:
        for side in ("A", "B"):
            with Image.open(image_root / side / name) as image:
                image = image.convert("RGB")
                if image.size != (size, size):
                    image = image.resize((size, size), Image.Resampling.BICUBIC)
                array = np.asarray(image, dtype=np.float64).reshape(-1, 3)
            total += array.sum(axis=0)
            total_sq += np.square(array).sum(axis=0)
            pixels += array.shape[0]
            images += 1
        if images % 1000 == 0:
            print(f"normalization images={images}/{2 * len(names)}", flush=True)

    mean = total / pixels
    variance = np.maximum(total_sq / pixels - np.square(mean), 1e-12)
    config["image_mean"] = mean.tolist()
    config["image_std"] = np.sqrt(variance).tolist()
    output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({"dataset": config["dataset"], "images": images, "pixels": pixels, "image_mean": config["image_mean"], "image_std": config["image_std"], "cached": False})
    print("NORMALIZATION_OK")


if __name__ == "__main__":
    main()
