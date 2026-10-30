"""Five-level extension of the audited calibrated-v3 stress operators.

Severity 2 is byte-for-byte compatible with stage6_5_calibrated_v3's fixed
operators.  Levels 1 and 3--5 only extend the stress axis for post-training
evaluation; they are never used for training or model selection.
"""
from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageFilter

PROFILE = "calibrated_v3_curve_s2_anchor"
CORRUPTIONS = (
    "cloud", "haze", "gaussian_noise", "blur", "jpeg",
    "low_resolution", "pixel_dropout", "misregistration",
    "radiometric_shift", "mixed",
)


def _check(severity: int) -> int:
    severity = int(severity)
    if severity not in (1, 2, 3, 4, 5):
        raise ValueError(f"severity must be in 1..5, got {severity}")
    return severity


def _pick(values, severity):
    return values[_check(severity) - 1]


def _u8(array):
    return np.clip(array, 0, 255).astype(np.uint8)


def cloud(image: Image.Image, severity: int, rng: np.random.Generator):
    s = _check(severity)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    coarse = rng.random((max(4, h // 32), max(4, w // 32)), dtype=np.float32)
    field = Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
    radius = _pick((9, 14, 18, 22, 26), s)
    field = field.filter(ImageFilter.GaussianBlur(radius=radius))
    f = np.asarray(field, dtype=np.float32) / 255.0
    coverage = _pick((0.30, 0.48, 0.58, 0.68, 0.78), s)
    threshold = float(np.quantile(f, 1.0 - coverage))
    mask = np.clip((f - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    alpha_scale = _pick((0.70, 0.88, 0.94, 0.98, 1.00), s)
    alpha = (mask ** 0.72) * alpha_scale
    tint = np.array([242.0, 245.0, 248.0], dtype=np.float32)
    out = arr * (1.0 - alpha[..., None]) + tint * alpha[..., None]
    # S2 exactly matches calibrated-v3: shift=(12,9), multiplier=.18.
    shadow_shift = _pick(((8, 6), (12, 9), (16, 12), (20, 15), (24, 18)), s)
    shadow_scale = _pick((0.11, 0.18, 0.22, 0.26, 0.30), s)
    shadow = np.roll(mask, shift=shadow_shift, axis=(0, 1)) * shadow_scale
    out *= 1.0 - shadow[..., None]
    return Image.fromarray(_u8(out)), {
        "profile": PROFILE, "severity": s, "coverage_target": coverage,
        "alpha_scale": alpha_scale, "radius": radius,
    }


def haze(image: Image.Image, severity: int, rng: np.random.Generator):
    del rng
    s = _check(severity)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    radial = np.sqrt(((xx - w * .5) / max(w, 1)) ** 2 + ((yy - h * .42) / max(h, 1)) ** 2)
    strength = _pick((0.40, 0.62, 0.72, 0.82, 0.92), s)
    transmission = np.clip(1.0 - strength * (1.18 - radial), 0.30, 1.0)
    light = np.array([225.0, 232.0, 238.0], dtype=np.float32)
    out = arr * transmission[..., None] + light * (1.0 - transmission[..., None])
    return Image.fromarray(_u8(out)), {"profile": PROFILE, "severity": s, "strength": strength}


def gaussian_noise(image: Image.Image, severity: int, rng: np.random.Generator):
    s = _check(severity)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    sigma = _pick((24.0, 44.0, 60.0, 78.0, 100.0), s)
    out = arr + rng.normal(0.0, sigma, size=arr.shape)
    return Image.fromarray(_u8(out)), {"profile": PROFILE, "severity": s, "sigma": sigma}


def blur(image: Image.Image, severity: int, rng: np.random.Generator):
    del rng
    s = _check(severity)
    radius = _pick((1.30, 2.55, 3.80, 5.40, 7.20), s)
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius)), {
        "profile": PROFILE, "severity": s, "radius": radius,
    }


def jpeg(image: Image.Image, severity: int, rng: np.random.Generator):
    del rng
    s = _check(severity)
    quality = _pick((18, 7, 4, 2, 1), s)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        out = decoded.convert("RGB").copy()
    return out, {"profile": PROFILE, "severity": s, "quality": quality, "subsampling": 2}


def low_resolution(image: Image.Image, severity: int, rng: np.random.Generator):
    del rng
    s = _check(severity)
    scale = _pick((0.32, 0.18, 0.12, 0.08, 0.05), s)
    w, h = image.size
    small = image.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
    out = small.resize((w, h), Image.Resampling.BICUBIC)
    return out, {"profile": PROFILE, "severity": s, "scale": scale}


def pixel_dropout(image: Image.Image, severity: int, rng: np.random.Generator):
    s = _check(severity)
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    h, w = arr.shape[:2]
    dead_fraction = _pick((0.008, 0.018, 0.035, 0.060, 0.100), s)
    dead = rng.random((h, w)) < dead_fraction
    arr[dead] = rng.choice(np.array([0, 255], dtype=np.uint8), size=(int(dead.sum()), 1))
    stripes = []
    for _ in range(_pick((2, 4, 6, 9, 12), s)):
        x = int(rng.integers(0, w)); width = int(rng.integers(1, 4)); value = int(rng.choice([0, 255]))
        arr[:, x:min(w, x + width), :] = value; stripes.append([x, width, value])
    blocks = []
    for _ in range(_pick((1, 3, 5, 7, 10), s)):
        bw = int(rng.integers(max(4, w // 28), max(5, w // 9)))
        bh = int(rng.integers(max(4, h // 28), max(5, h // 9)))
        x = int(rng.integers(0, max(1, w - bw + 1))); y = int(rng.integers(0, max(1, h - bh + 1)))
        arr[y:y + bh, x:x + bw] = 0; blocks.append([x, y, bw, bh])
    return Image.fromarray(arr), {
        "profile": PROFILE, "severity": s, "dead_fraction": dead_fraction,
        "stripes": stripes, "blocks": blocks,
    }


def misregistration(image: Image.Image, severity: int, rng: np.random.Generator):
    s = _check(severity)
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    max_shift = _pick((10, 20, 30, 40, 52), s)
    # Draw at the calibrated-v3 S2 range, then scale the same latent offset.
    # This both reproduces S2 exactly and prevents a larger severity from
    # accidentally drawing a smaller displacement merely due to RNG remapping.
    base_dx = int(rng.integers(-20, 21)); base_dy = int(rng.integers(-20, 21))
    if base_dx == 0 and base_dy == 0: base_dx = 20
    if s == 2:
        dx, dy = base_dx, base_dy
    else:
        dx = int(round(base_dx * max_shift / 20.0)); dy = int(round(base_dy * max_shift / 20.0))
    pad = max_shift + 1
    padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = arr.shape[:2]
    out = padded[pad - dy:pad - dy + h, pad - dx:pad - dx + w]
    return Image.fromarray(out), {
        "profile": PROFILE, "severity": s, "dx": dx, "dy": dy, "max_shift": max_shift,
    }


def radiometric_shift(image: Image.Image, severity: int, rng: np.random.Generator):
    s = _check(severity)
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    spread = _pick((0.40, 0.70, 0.85, 1.00, 1.20), s)
    gains = rng.uniform(1.0 - spread, 1.0 + spread, size=(1, 1, 3)).astype(np.float32)
    bias = float(rng.uniform(-spread * .16, spread * .16))
    gamma_floor = _pick((0.60, 0.45, 0.36, 0.30, 0.24), s)
    gamma = float(rng.uniform(max(gamma_floor, 1.0 - spread), 1.0 + spread))
    out = np.clip(arr * gains + bias, 0.0, 1.0) ** gamma
    return Image.fromarray(_u8(out * 255.0)), {
        "profile": PROFILE, "severity": s, "gains": gains.reshape(-1).round(5).tolist(),
        "bias": round(bias, 5), "gamma": round(gamma, 5), "spread": spread,
    }


def mixed(image: Image.Image, severity: int, rng: np.random.Generator):
    s = _check(severity)
    child = rng.integers(0, np.iinfo(np.uint32).max, size=4, dtype=np.uint32)
    # Mixed in the original calibrated-v3 intentionally used the *base*
    # severity-2 cloud/haze/JPEG parameters rather than the strengthened
    # public operators.  The dedicated tables below retain those exact S2
    # anchors while extending the composition to five levels.
    arr0 = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr0.shape[:2]
    haze_strength = _pick((0.20, 0.32, 0.45, 0.58, 0.72), s)
    yy, xx = np.mgrid[0:h, 0:w]
    radial = np.sqrt(((xx - w * .5) / max(w, 1)) ** 2 + ((yy - h * .42) / max(h, 1)) ** 2)
    transmission = np.clip(1.0 - haze_strength * (1.18 - radial), 0.34, 1.0)
    light = np.array([225.0, 232.0, 238.0], dtype=np.float32)
    out = Image.fromarray(_u8(arr0 * transmission[..., None] + light * (1.0 - transmission[..., None])))
    p_haze = {"strength": haze_strength}
    cloud_rng = np.random.default_rng(int(child[1]))
    arr = np.asarray(out.convert("RGB"), dtype=np.float32)
    coarse = cloud_rng.random((max(4, h // 32), max(4, w // 32)), dtype=np.float32)
    field = Image.fromarray((coarse * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BICUBIC)
    cloud_radius = _pick((5, 9, 14, 18, 22), s)
    field = field.filter(ImageFilter.GaussianBlur(radius=cloud_radius))
    f = np.asarray(field, dtype=np.float32) / 255.0
    cloud_coverage = _pick((0.18, 0.32, 0.48, 0.60, 0.72), s)
    threshold = float(np.quantile(f, 1.0 - cloud_coverage))
    mask = np.clip((f - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
    cloud_alpha = _pick((0.55, 0.72, 0.88, 0.95, 1.00), s)
    alpha = (mask ** .72) * cloud_alpha
    out_arr = arr * (1.0 - alpha[..., None]) + np.array([242.0, 245.0, 248.0], dtype=np.float32) * alpha[..., None]
    shadow = np.roll(mask, shift=(4 * s, 3 * s), axis=(0, 1)) * (0.06 * s)
    out = Image.fromarray(_u8(out_arr * (1.0 - shadow[..., None])))
    p_cloud = {"coverage_target": cloud_coverage, "alpha_scale": cloud_alpha, "radius": cloud_radius}
    arr = np.asarray(out.convert("RGB"), dtype=np.float32)
    sigma = _pick((10.0, 20.0, 30.0, 42.0, 56.0), s)
    arr += np.random.default_rng(int(child[2])).normal(0.0, sigma, size=arr.shape)
    out = Image.fromarray(_u8(arr))
    radius = _pick((0.65, 1.15, 1.80, 2.70, 3.80), s)
    out = out.filter(ImageFilter.GaussianBlur(radius=radius))
    # Dedicated mixed JPEG values; S2 quality=7 equals calibrated-v3 base JPEG.
    quality = _pick((45, 28, 15, 7, 2), s)
    buffer = io.BytesIO(); out.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False); buffer.seek(0)
    with Image.open(buffer) as decoded: out = decoded.convert("RGB").copy()
    return out, {
        "profile": PROFILE, "severity": s, "haze": p_haze, "cloud": p_cloud,
        "noise": {"sigma": sigma}, "blur": {"radius": radius}, "jpeg": {"quality": quality, "subsampling": 2},
    }


OPERATORS = {
    "cloud": cloud, "haze": haze, "gaussian_noise": gaussian_noise, "blur": blur,
    "jpeg": jpeg, "low_resolution": low_resolution, "pixel_dropout": pixel_dropout,
    "misregistration": misregistration, "radiometric_shift": radiometric_shift, "mixed": mixed,
}
