"""Portable calibrated-v3 degradation operators for LEVIR-MCI.

This module is intentionally self-contained. Training and evaluation code load
the same ``OPERATORS`` mapping; no degraded image files need to be stored.
The profile matches the audited Stage 6.4 calibrated-v3 test generator.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image, ImageFilter


PROFILE = "calibrated_v3"
CORRUPTIONS = (
    "cloud",
    "haze",
    "gaussian_noise",
    "blur",
    "jpeg",
    "low_resolution",
    "pixel_dropout",
    "misregistration",
    "radiometric_shift",
    "mixed",
)


def stable_seed(base_seed: int, *parts: str) -> int:
    payload = "|".join((str(base_seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def to_uint8(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0, 255).astype(np.uint8)


def cloud(image: Image.Image, severity: int, rng: np.random.Generator):
    # The calibrated profile uses the former severity-3 cloud parameters.
    del severity
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    coarse = rng.random((max(4, h // 32), max(4, w // 32)), dtype=np.float32)
    field = Image.fromarray((coarse * 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.BICUBIC
    )
    radius = 14
    field = field.filter(ImageFilter.GaussianBlur(radius=radius))
    f = np.asarray(field, dtype=np.float32) / 255.0
    coverage = 0.48
    threshold = float(np.quantile(f, 1.0 - coverage))
    mask = np.clip((f - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    alpha_scale = 0.88
    alpha = (mask**0.72) * alpha_scale
    tint = np.array([242.0, 245.0, 248.0], dtype=np.float32)
    out = arr * (1.0 - alpha[..., None]) + tint * alpha[..., None]
    shadow = np.roll(mask, shift=(12, 9), axis=(0, 1)) * 0.18
    out *= 1.0 - shadow[..., None]
    return Image.fromarray(to_uint8(out)), {
        "profile": PROFILE,
        "coverage_target": coverage,
        "alpha_scale": alpha_scale,
    }


def haze(image: Image.Image, severity: int, rng: np.random.Generator):
    del severity, rng
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    radial = np.sqrt(
        ((xx - w * 0.5) / max(w, 1)) ** 2
        + ((yy - h * 0.42) / max(h, 1)) ** 2
    )
    strength = 0.62
    transmission = np.clip(1.0 - strength * (1.18 - radial), 0.30, 1.0)
    light = np.array([225.0, 232.0, 238.0], dtype=np.float32)
    out = arr * transmission[..., None] + light * (1.0 - transmission[..., None])
    return Image.fromarray(to_uint8(out)), {"profile": PROFILE, "strength": strength}


def gaussian_noise(image: Image.Image, severity: int, rng: np.random.Generator):
    del severity
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    sigma = 44.0
    out = arr + rng.normal(0.0, sigma, size=arr.shape)
    return Image.fromarray(to_uint8(out)), {"profile": PROFILE, "sigma": sigma}


def blur(image: Image.Image, severity: int, rng: np.random.Generator):
    del severity, rng
    radius = 2.55
    out = image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))
    return out, {"profile": PROFILE, "radius": radius}


def jpeg(image: Image.Image, severity: int, rng: np.random.Generator):
    del severity, rng
    quality = 7
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer, format="JPEG", quality=quality, subsampling=2, optimize=False
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        out = decoded.convert("RGB").copy()
    return out, {
        "profile": PROFILE,
        "quality": quality,
        "subsampling": 2,
    }


def low_resolution(image: Image.Image, severity: int, rng: np.random.Generator):
    del severity, rng
    scale = 0.18
    w, h = image.size
    small = image.convert("RGB").resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        Image.Resampling.BILINEAR,
    )
    out = small.resize((w, h), Image.Resampling.BICUBIC)
    return out, {"profile": PROFILE, "scale": scale}


def pixel_dropout(image: Image.Image, severity: int, rng: np.random.Generator):
    del severity
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    dead_fraction = 0.018
    dead = rng.random((h, w)) < dead_fraction
    arr[dead] = rng.choice(np.array([0, 255], dtype=np.uint8), size=(int(dead.sum()), 1))
    stripes = []
    for _ in range(4):
        x = int(rng.integers(0, w))
        width = int(rng.integers(1, 4))
        value = int(rng.choice([0, 255]))
        arr[:, x : min(w, x + width), :] = value
        stripes.append([x, width, value])
    blocks = []
    for _ in range(3):
        bw = int(rng.integers(max(4, w // 28), max(5, w // 9)))
        bh = int(rng.integers(max(4, h // 28), max(5, h // 9)))
        x = int(rng.integers(0, max(1, w - bw + 1)))
        y = int(rng.integers(0, max(1, h - bh + 1)))
        arr[y : y + bh, x : x + bw] = 0
        blocks.append([x, y, bw, bh])
    return Image.fromarray(arr), {
        "profile": PROFILE,
        "dead_fraction": dead_fraction,
        "stripes": stripes,
        "blocks": blocks,
    }


def misregistration(image: Image.Image, severity: int, rng: np.random.Generator):
    del severity
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    max_shift = 20
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))
    if dx == 0 and dy == 0:
        dx = max_shift
    pad = max_shift + 1
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = arr.shape[:2]
    out = padded[pad - dy : pad - dy + h, pad - dx : pad - dx + w]
    return Image.fromarray(out), {
        "profile": PROFILE,
        "dx": dx,
        "dy": dy,
        "max_shift": max_shift,
    }


def radiometric_shift(image: Image.Image, severity: int, rng: np.random.Generator):
    del severity
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    spread = 0.70
    gains = rng.uniform(1.0 - spread, 1.0 + spread, size=(1, 1, 3)).astype(np.float32)
    bias = float(rng.uniform(-spread * 0.16, spread * 0.16))
    gamma = float(rng.uniform(max(0.45, 1.0 - spread), 1.0 + spread))
    out = np.clip(arr * gains + bias, 0.0, 1.0) ** gamma
    return Image.fromarray(to_uint8(out * 255.0)), {
        "profile": PROFILE,
        "gains": gains.reshape(-1).round(5).tolist(),
        "bias": round(bias, 5),
        "gamma": round(gamma, 5),
        "spread": spread,
    }


def mixed(image: Image.Image, severity: int, rng: np.random.Generator):
    del severity
    child = rng.integers(0, np.iinfo(np.uint32).max, size=4, dtype=np.uint32)
    # These two steps deliberately match the audited v3 composition, which
    # retained the base severity-2 haze/cloud behavior inside mixed.
    out, p_haze = _base_haze(image, 2, np.random.default_rng(int(child[0])))
    out, p_cloud = _base_cloud(out, 2, np.random.default_rng(int(child[1])))
    arr = np.asarray(out.convert("RGB"), dtype=np.float32)
    sigma = 20.0
    arr += np.random.default_rng(int(child[2])).normal(0.0, sigma, size=arr.shape)
    out = Image.fromarray(to_uint8(arr))
    radius = 1.15
    out = out.filter(ImageFilter.GaussianBlur(radius=radius))
    out, p_jpeg = _base_jpeg(out, 2, np.random.default_rng(int(child[3])))
    return out, {
        "profile": PROFILE,
        "haze": p_haze,
        "cloud": p_cloud,
        "noise": {"sigma": sigma},
        "blur": {"radius": radius},
        "jpeg": p_jpeg,
    }


def _base_cloud(image: Image.Image, severity: int, rng: np.random.Generator):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    coarse = rng.random((max(4, h // 32), max(4, w // 32)), dtype=np.float32)
    field = Image.fromarray((coarse * 255).astype(np.uint8)).resize(
        (w, h), Image.Resampling.BICUBIC
    )
    radius = (5, 9, 14)[severity - 1]
    field = field.filter(ImageFilter.GaussianBlur(radius=radius))
    f = np.asarray(field, dtype=np.float32) / 255.0
    coverage = (0.18, 0.32, 0.48)[severity - 1]
    threshold = float(np.quantile(f, 1.0 - coverage))
    mask = np.clip((f - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    alpha_scale = (0.55, 0.72, 0.88)[severity - 1]
    alpha = (mask**0.72) * alpha_scale
    tint = np.array([242.0, 245.0, 248.0], dtype=np.float32)
    out = arr * (1.0 - alpha[..., None]) + tint * alpha[..., None]
    shadow = np.roll(mask, shift=(4 * severity, 3 * severity), axis=(0, 1)) * (0.06 * severity)
    out *= 1.0 - shadow[..., None]
    return Image.fromarray(to_uint8(out)), {
        "coverage_target": coverage,
        "alpha_scale": alpha_scale,
    }


def _base_haze(image: Image.Image, severity: int, rng: np.random.Generator):
    del rng
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    radial = np.sqrt(
        ((xx - w * 0.5) / max(w, 1)) ** 2
        + ((yy - h * 0.42) / max(h, 1)) ** 2
    )
    strength = (0.18, 0.32, 0.48)[severity - 1]
    transmission = np.clip(1.0 - strength * (1.18 - radial), 0.34, 1.0)
    light = np.array([225.0, 232.0, 238.0], dtype=np.float32)
    out = arr * transmission[..., None] + light * (1.0 - transmission[..., None])
    return Image.fromarray(to_uint8(out)), {"strength": strength}


def _base_jpeg(image: Image.Image, severity: int, rng: np.random.Generator):
    del rng
    quality = (55, 28, 12)[severity - 1]
    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer, format="JPEG", quality=quality, subsampling=2, optimize=False
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        out = decoded.convert("RGB").copy()
    return out, {"quality": quality, "subsampling": 2}


OPERATORS = {
    "cloud": cloud,
    "haze": haze,
    "gaussian_noise": gaussian_noise,
    "blur": blur,
    "jpeg": jpeg,
    "low_resolution": low_resolution,
    "pixel_dropout": pixel_dropout,
    "misregistration": misregistration,
    "radiometric_shift": radiometric_shift,
    "mixed": mixed,
}


def apply_degradation(
    image: Image.Image,
    corruption: str,
    base_seed: int,
    *seed_parts: str,
    severity: int = 3,
):
    """Apply one deterministic degradation using the shared seed convention."""
    if corruption not in OPERATORS:
        raise KeyError(f"Unknown corruption: {corruption}")
    rng = np.random.default_rng(stable_seed(base_seed, corruption, *seed_parts))
    return OPERATORS[corruption](image, severity, rng)
