"""Metrics for §6 — and why every cosine here is noise-corrected.

A raw cosine between two estimated profiles is confounded with signal-to-noise. In 6,144
dimensions, with signal concentrated in some coordinates and estimation noise spread over
all of them, the observed cosine between two independent estimates of the same profile is
roughly

    cos_obs  ~  cos_true * sqrt((1 - v_a / |a|^2) * (1 - v_b / |b|^2))

where v is the expected squared norm of the noise, i.e. the summed squared standard errors.
So a convergence curve reported as a raw cosine largely retraces the arithmetic of adding
prompts, and a construct flagged as nu-fragile may only be a construct with a weak profile.

Two things therefore travel with every cosine: the **disattenuated** value, and the
**pure-noise expectation** — what the cosine would be if both vectors were noise alone,
which is 0 with a standard deviation of about 1/sqrt(d). Without that line on the plot a
cosine of 0.05 looks like weak agreement rather than none.
"""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(a @ b / (na * nb))


def noise_norm_sq(se: np.ndarray) -> float:
    """Expected squared norm of the estimation noise: the summed squared SEs."""
    return float((np.asarray(se) ** 2).sum())


def reliability(vec: np.ndarray, se: np.ndarray) -> float:
    """Share of a vector's squared norm that is signal rather than estimation noise."""
    total = float((np.asarray(vec) ** 2).sum())
    if total <= 0:
        return 0.0
    return float(max(0.0, 1.0 - noise_norm_sq(se) / total))


def disattenuated(a: np.ndarray, se_a: np.ndarray, b: np.ndarray,
                  se_b: np.ndarray) -> dict:
    """Cosine corrected for the noise in both estimates.

    Capped at 1.0 and reported alongside the raw value and both reliabilities, because a
    correction that exceeds 1 means the noise estimate was too large — which is itself worth
    seeing rather than hiding behind a clip.
    """
    raw = cosine(a, b)
    ra, rb = reliability(a, se_a), reliability(b, se_b)
    denom = np.sqrt(ra * rb)
    corrected = raw / denom if denom > 0 else float("nan")
    return {"cosine": raw, "disattenuated": float(min(corrected, 1.0))
            if np.isfinite(corrected) else float("nan"),
            "uncapped": float(corrected) if np.isfinite(corrected) else float("nan"),
            "reliability_a": ra, "reliability_b": rb,
            "over_unity": bool(np.isfinite(corrected) and corrected > 1.0)}


def pure_noise_cosine(d: int) -> dict:
    """What a cosine between two independent noise vectors looks like in d dimensions."""
    return {"mean": 0.0, "sd": float(1.0 / np.sqrt(d)),
            "abs_95": float(1.96 / np.sqrt(d))}


def relative_distance(a: np.ndarray, b: np.ndarray) -> float:
    """|a - b| / mean(|a|, |b|) — a scale-free companion to the cosine.

    Cosine ignores magnitude entirely, so two profiles pointing the same way at different
    strengths look identical to it. Twin noise is reported in both.
    """
    scale = (np.linalg.norm(a) + np.linalg.norm(b)) / 2
    if scale == 0:
        return float("nan")
    return float(np.linalg.norm(a - b) / scale)


def in_twin_units(value: float, twin_value: float) -> float:
    """Every sensitivity in §6 is reported as a multiple of twin disagreement."""
    if not np.isfinite(twin_value) or twin_value == 0:
        return float("nan")
    return float(value / twin_value)
