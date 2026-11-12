"""Canonical leave-one-degradation-family-out protocol definitions."""

ALL_DEGRADATIONS = (
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

HOLDOUTS = {
    "atmospheric": frozenset({"cloud", "haze", "mixed"}),
    "geometry_quality": frozenset(
        {"blur", "low_resolution", "misregistration", "mixed"}
    ),
    "noise_loss": frozenset(
        {"gaussian_noise", "jpeg", "pixel_dropout", "mixed"}
    ),
}


def allowed_degradations(family: str) -> tuple[str, ...]:
    """Return the ordered training set for a strict LOFO family."""
    if family not in HOLDOUTS:
        raise KeyError(f"Unknown LOFO family: {family}")
    return tuple(name for name in ALL_DEGRADATIONS if name not in HOLDOUTS[family])
