"""Self-contained deterministic degradations used by the DINO fine-tuning stage."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter


def _u8(array):
    return np.clip(array, 0, 255).astype(np.uint8)


def cloud(image, severity, rng):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    coarse = rng.random((max(4, h // 32), max(4, w // 32)), dtype=np.float32)
    field = Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
    field = field.filter(ImageFilter.GaussianBlur(radius=(5, 9, 14)[severity - 1]))
    f = np.asarray(field, dtype=np.float32) / 255.0
    coverage = (0.18, 0.32, 0.48)[severity - 1]
    threshold = float(np.quantile(f, 1.0 - coverage))
    mask = np.clip((f - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    alpha = (mask ** 0.72) * (0.55, 0.72, 0.88)[severity - 1]
    out = arr * (1.0 - alpha[..., None]) + np.array([242.0, 245.0, 248.0], dtype=np.float32) * alpha[..., None]
    shadow = np.roll(mask, shift=(4 * severity, 3 * severity), axis=(0, 1)) * (0.06 * severity)
    return Image.fromarray(_u8(out * (1.0 - shadow[..., None]))), {"coverage": coverage}


def haze(image, severity, rng):
    del rng
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    radial = np.sqrt(((xx - w * 0.5) / max(w, 1)) ** 2 + ((yy - h * 0.42) / max(h, 1)) ** 2)
    strength = (0.18, 0.32, 0.48)[severity - 1]
    transmission = np.clip(1.0 - strength * (1.18 - radial), 0.34, 1.0)
    out = arr * transmission[..., None] + np.array([225.0, 232.0, 238.0]) * (1.0 - transmission[..., None])
    return Image.fromarray(_u8(out)), {"strength": strength}


def gaussian_noise(image, severity, rng):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    sigma = (7.0, 15.0, 28.0)[severity - 1]
    return Image.fromarray(_u8(arr + rng.normal(0.0, sigma, size=arr.shape))), {"sigma": sigma}


def blur(image, severity, rng):
    del rng
    radius = (0.8, 1.8, 3.2)[severity - 1]
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius)), {"radius": radius}


def jpeg(image, severity, rng):
    del rng
    quality = (55, 28, 12)[severity - 1]
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        output = decoded.convert("RGB").copy()
    return output, {"quality": quality}


def low_resolution(image, severity, rng):
    del rng
    scale = (0.65, 0.40, 0.24)[severity - 1]
    w, h = image.size
    small = image.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BICUBIC), {"scale": scale}


def pixel_dropout(image, severity, rng):
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    fraction = (0.004, 0.012, 0.030)[severity - 1]
    dead = rng.random((h, w)) < fraction
    arr[dead] = rng.choice(np.array([0, 255], dtype=np.uint8), size=(int(dead.sum()), 1))
    for _ in range((1, 3, 6)[severity - 1]):
        x = int(rng.integers(0, w)); width = int(rng.integers(1, 2 + severity))
        arr[:, x:min(w, x + width), :] = int(rng.choice([0, 255]))
    for _ in range((1, 2, 4)[severity - 1]):
        bw = int(rng.integers(max(4, w // 24), max(5, w // (10 - severity))))
        bh = int(rng.integers(max(4, h // 24), max(5, h // (10 - severity))))
        x = int(rng.integers(0, max(1, w - bw + 1))); y = int(rng.integers(0, max(1, h - bh + 1)))
        arr[y:y + bh, x:x + bw] = 0
    return Image.fromarray(arr), {"dead_fraction": fraction}


def misregistration(image, severity, rng):
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    max_shift = (2, 5, 10)[severity - 1]
    dx = int(rng.integers(-max_shift, max_shift + 1)); dy = int(rng.integers(-max_shift, max_shift + 1))
    if dx == 0 and dy == 0:
        dx = max_shift
    pad = max_shift + 1
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = arr.shape[:2]
    return Image.fromarray(padded[pad - dy:pad - dy + h, pad - dx:pad - dx + w]), {"dx": dx, "dy": dy}


def radiometric_shift(image, severity, rng):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    spread = (0.10, 0.20, 0.34)[severity - 1]
    gains = rng.uniform(1.0 - spread, 1.0 + spread, size=(1, 1, 3)).astype(np.float32)
    bias = float(rng.uniform(-spread * 0.18, spread * 0.18))
    gamma = float(rng.uniform(1.0 - spread, 1.0 + spread))
    return Image.fromarray(_u8((np.clip(arr * gains + bias, 0.0, 1.0) ** gamma) * 255.0)), {"gamma": gamma}


def mixed(image, severity, rng):
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=4, dtype=np.uint32)
    output, _ = haze(image, severity, np.random.default_rng(int(seeds[0])))
    output, _ = cloud(output, max(1, severity - 1), np.random.default_rng(int(seeds[1])))
    output, _ = gaussian_noise(output, severity, np.random.default_rng(int(seeds[2])))
    output, _ = blur(output, max(1, severity - 1), np.random.default_rng(int(seeds[3])))
    output, _ = jpeg(output, severity, rng)
    return output, {"composition": "haze+cloud+noise+blur+jpeg"}


OPERATORS = {
    "cloud": cloud, "haze": haze, "gaussian_noise": gaussian_noise, "blur": blur,
    "jpeg": jpeg, "low_resolution": low_resolution, "pixel_dropout": pixel_dropout,
    "misregistration": misregistration, "radiometric_shift": radiometric_shift, "mixed": mixed,
}
