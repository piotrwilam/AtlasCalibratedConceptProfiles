"""The validation checks of §6.

Each returns a plain dict — the licence table's row — and every sensitivity is reported in
**twin units**, because "the profile changed" is meaningless without the scale of the change
the same generator produces twice.
"""

from __future__ import annotations

import numpy as np

from ..profiles.profile import build
from .metrics import (cosine, disattenuated, in_twin_units, pure_noise_cosine,
                      relative_distance)


def twins(a, b) -> dict:
    """Check 1 — the noise floor. Two batteries sampling one declared distribution.

    Their disagreement is the unit every other check is reported in. Both a cosine and a
    relative distance, because cosine ignores magnitude and two profiles pointing the same way
    at different strengths look identical to it.
    """
    layers = a.mean.shape[0]
    per_layer = []
    for l in range(layers):
        d = disattenuated(a.mean[l], a.se[l], b.mean[l], b.se[l])
        per_layer.append({"layer": l, **d,
                          "distance": relative_distance(a.mean[l], b.mean[l]),
                          "norm_a": float(np.linalg.norm(a.mean[l])),
                          "norm_b": float(np.linalg.norm(b.mean[l]))})
    return {"check": "twins", "construct": a.construct, "n_a": a.n, "n_b": b.n,
            "cores_a": len(a.core_means), "cores_b": len(b.core_means),
            "pure_noise": pure_noise_cosine(a.mean.shape[1]),
            "per_layer": per_layer,
            "median_cosine": float(np.median([p["cosine"] for p in per_layer])),
            "median_disattenuated": float(np.median([p["disattenuated"]
                                                     for p in per_layer])),
            "median_distance": float(np.median([p["distance"] for p in per_layer]))}


def convergence(primary, rows, ys, construct, ns=(15, 30, 60, 120, 180),
                n_rep: int = 8, seed: int = 0) -> dict:
    """Check 2 — how many prompts a profile needs.

    Subsamples are drawn **by core**, not by prompt: a subsample of 15 prompts all from one
    core would converge quickly to that core's profile and tell us nothing about the construct.
    """
    full = build(primary, rows, ys, construct, "A")
    rng = np.random.default_rng(seed)
    by_core: dict[str, list[int]] = {}
    for i, r in enumerate(rows[:primary.shape[0]]):
        by_core.setdefault(r.get("core_id", "?"), []).append(i)
    cores = sorted(by_core)
    out = []
    for n in ns:
        if n > full.n:
            continue
        per_core = max(1, n // len(cores))
        cos, dis = [], []
        for _ in range(n_rep):
            pick = []
            for c in cores:
                idx = by_core[c]
                pick.extend(rng.choice(idx, size=min(per_core, len(idx)),
                                       replace=False).tolist())
            sub = build(primary[pick], [rows[i] for i in pick], ys, construct, "A")
            mid = full.mean.shape[0] // 2
            d = disattenuated(sub.mean[mid], sub.se[mid], full.mean[mid], full.se[mid])
            cos.append(d["cosine"])
            dis.append(d["disattenuated"])
        out.append({"n": len(pick), "cosine": float(np.mean(cos)),
                    "disattenuated": float(np.mean(dis)),
                    "cosine_sd": float(np.std(cos))})
    return {"check": "convergence", "construct": construct, "n_full": full.n,
            "curve": out, "pure_noise": pure_noise_cosine(full.mean.shape[1])}


def length_sensitivity(primary, rows, ys, construct) -> dict:
    """Check 3 — the same cores at three padding levels.

    Not by deleting text, which breaks parsing, roles and sites, but by the grid's own padding
    dimension. The per-unit index is the spread of a unit's profile value across the three
    lengths, in units of its own SE.
    """
    by_pad: dict[int, list[int]] = {}
    for i, r in enumerate(rows[:primary.shape[0]]):
        by_pad.setdefault(int(r.get("padding", 0)), []).append(i)
    profiles = {}
    for pad, idx in sorted(by_pad.items()):
        if len(idx) < 8:
            continue
        profiles[pad] = build(primary[idx], [rows[i] for i in idx], ys, construct, "A")
    if len(profiles) < 2:
        return {"check": "length", "construct": construct, "error": "fewer than two levels"}
    pads = sorted(profiles)
    mid = profiles[pads[0]].mean.shape[0] // 2
    pairs = []
    for i in range(len(pads) - 1):
        a, b = profiles[pads[i]], profiles[pads[i + 1]]
        pairs.append({"from": pads[i], "to": pads[i + 1],
                      **disattenuated(a.mean[mid], a.se[mid], b.mean[mid], b.se[mid])})
    stack = np.stack([profiles[p].mean[mid] for p in pads])
    se = np.stack([profiles[p].se[mid] for p in pads]).mean(axis=0)
    index = stack.std(axis=0, ddof=1) / np.maximum(se, 1e-12)
    return {"check": "length", "construct": construct,
            "levels": {str(p): profiles[p].n for p in pads}, "pairs": pairs,
            "index_median": float(np.median(index)),
            "flagged_units": int((index > 3).sum()),
            "flagged_share": float((index > 3).mean())}


def nu_sensitivity(base, other, label: str, twin_distance: float) -> dict:
    """Check 4 — a second generator menu, and real corpus code, against the twin floor."""
    layers = base.mean.shape[0]
    mid = layers // 2
    d = disattenuated(base.mean[mid], base.se[mid], other.mean[mid], other.se[mid])
    dist = relative_distance(base.mean[mid], other.mean[mid])
    return {"check": "nu", "arm": label, "construct": base.construct,
            "n_base": base.n, "n_other": other.n, **d, "distance": dist,
            "distance_in_twin_units": in_twin_units(dist, twin_distance)}


def calibration(a, b, boot_a=None, boot_b=None, z: float = 1.96) -> dict:
    """Check 6 — do the stated uncertainties cover the twin differences?

    Nominal coverage at z = 1.96 is 95%. Both the prompt-level SE and the core bootstrap are
    tested, because they estimate different things and the programme uses the second.
    """
    diff = a.mean - b.mean
    out = {"check": "calibration", "construct": a.construct, "z": z, "nominal": 0.95}
    se = np.sqrt(a.se ** 2 + b.se ** 2)
    out["prompt_se_coverage"] = float((np.abs(diff) <= z * se).mean())
    out["prompt_se_inflation"] = float(np.median(np.abs(diff) / np.maximum(se, 1e-12)) /
                                       0.6745)
    if boot_a is not None and boot_b is not None:
        bse = np.sqrt(boot_a ** 2 + boot_b ** 2)
        out["bootstrap_coverage"] = float((np.abs(diff) <= z * bse).mean())
        out["bootstrap_inflation"] = float(
            np.median(np.abs(diff) / np.maximum(bse, 1e-12)) / 0.6745)
    return out


def null_split_half(primary, rows, ys, construct, n_rep: int = 20,
                    seed: int = 0) -> dict:
    """Check 7b — pure sampling noise: one A set split in half, by core."""
    rng = np.random.default_rng(seed)
    by_core: dict[str, list[int]] = {}
    for i, r in enumerate(rows[:primary.shape[0]]):
        by_core.setdefault(r.get("core_id", "?"), []).append(i)
    sizes = []
    for _ in range(n_rep):
        left, right = [], []
        for idx in by_core.values():
            perm = rng.permutation(idx)
            left.extend(perm[: len(perm) // 2].tolist())
            right.extend(perm[len(perm) // 2:].tolist())
        a = build(primary[left], [rows[i] for i in left], ys, construct, "A")
        b = build(primary[right], [rows[i] for i in right], ys, construct, "A")
        sizes.append(np.linalg.norm(a.mean - b.mean, axis=1))   # per layer
    arr = np.stack(sizes)                                       # [n_rep, layers]
    mid = arr.shape[1] // 2
    # Per layer, because the bar is only a null for the layer it was computed at. The
    # verdict may be read at the v1 home layer (21-22 for most constructs), and layer 0
    # carries a token-embedding floor about three times the mid-stack norm — comparing a
    # layer-21 signal with a layer-12 bar would be meaningless in either direction.
    return {"check": "null_split_half", "construct": construct, "n_rep": n_rep,
            "median_norm": float(np.median(arr[:, mid])),
            "p95_norm": float(np.percentile(arr[:, mid], 95)),
            "layer": mid,
            "median_by_layer": np.median(arr, axis=0).tolist(),
            "p95_by_layer": np.percentile(arr, 95, axis=0).tolist()}


def companion_leakage(concept: np.ndarray, asymmetry: dict, layer: int) -> dict:
    """Check 10 — the measured version of the pre-run companion cap.

    Regresses the concept profile's per-layer norm on the A-minus-B companion asymmetry. The
    cap is a proxy applied before the sweep; this is what the readouts say.
    """
    if not asymmetry:
        return {"check": "leakage", "error": "no asymmetry vector"}
    keys = sorted(asymmetry)
    x = np.array([asymmetry[k] for k in keys], dtype=float)
    return {"check": "leakage", "layer": layer,
            "concept_norm": float(np.linalg.norm(concept[layer])),
            "asymmetry_l2": float(np.linalg.norm(x)),
            "asymmetry_max": float(np.max(np.abs(x))),
            "companions": keys}


def eps_sensitivity(build_at, twin_distance: float, grid=None) -> dict:
    """Check 8 — how much the smooth floor's open parameter moves the reading.

    sigma0 is floored as sqrt(sigma0^2 + (eps * median sigma0)^2), and eps is a declared
    recipe parameter rather than something the data fixes. So the reading has to be
    quoted against its own sensitivity to it: rebuild the profile at each floor and
    measure the displacement from the default, in twin units. A displacement that rivals
    twin disagreement means the floor, not the construct, is setting the answer.

    `build_at(eps)` returns a Profile at that floor, so this stays indifferent to whether
    the caller used the per-prompt or the accumulator route.
    """
    from ..profiles.yardstick import DEFAULT_EPS, EPS_GRID

    grid = tuple(grid or EPS_GRID)
    profiles = {e: build_at(e) for e in grid}
    base = profiles.get(DEFAULT_EPS)
    if base is None:
        return {"check": "eps", "error": f"default eps {DEFAULT_EPS} not in grid {grid}"}
    vs = {}
    for e, p in profiles.items():
        if e == DEFAULT_EPS:
            continue
        d = relative_distance(base.mean, p.mean)
        vs[str(e)] = {"relative_distance": d,
                      "in_twin_units": in_twin_units(d, twin_distance),
                      "cosine": cosine(base.mean.ravel(), p.mean.ravel())}
    worst = max((v["in_twin_units"] for v in vs.values()), default=float("nan"))
    return {"check": "eps", "default": DEFAULT_EPS, "grid": list(grid),
            "vs_default": vs, "worst_in_twin_units": worst}


def adjacent(now, before, twin_distance: float, layer: int) -> dict:
    """Check 9 — smoothness between two adjacent checkpoints.

    A date is read off a trajectory, so the trajectory has to be a curve and not a
    scatter. The test is whether one 4.19B-token step moves the concept profile by an
    amount comparable to the noise floor rather than by an amount that makes any
    interpolation between checkpoints meaningless. Quoted in twin units, like every
    other sensitivity: a step of many twin units is not fatal — the model really is
    changing — but a step that is *smaller* than twin noise means the two checkpoints
    are indistinguishable and no date can separate them.
    """
    a, b = now[layer], before[layer]
    step = relative_distance(a, b)
    return {"check": "adjacent", "layer": layer,
            "cosine": cosine(a, b),
            "relative_step": step,
            "step_in_twin_units": in_twin_units(step, twin_distance),
            "norm_now": float(np.linalg.norm(a)),
            "norm_before": float(np.linalg.norm(b))}


def null_scrambled(primary_a, rows_a, primary_b, rows_b, ys, construct,
                   n_rep: int = 20, seed: int = 0) -> dict:
    """Check 7a — the false-discovery control: a contrast that is not there.

    Split-half (7b) asks what a *meaningless* split of A produces. This asks the sharper
    question: pool A and B, relabel at random keeping the set sizes, and take the
    difference of the two profiles. Every prompt keeps its own text, cell and yardstick;
    only the label moves. Whatever concept profile survives that is manufactured by the
    pipeline rather than by the contrast, so a real concept profile must stand above it.

    Prompts are pooled and relabelled without regard to core, because the null being
    tested is "there is no A-versus-B distinction at all", not "the cores are exchangeable".
    """
    rng = np.random.default_rng(seed)
    n_a = primary_a.shape[0]
    pool = np.concatenate([primary_a[:len(rows_a)], primary_b[:len(rows_b)]])
    rows = list(rows_a[:primary_a.shape[0]]) + list(rows_b[:primary_b.shape[0]])
    sizes = []
    for _ in range(n_rep):
        perm = rng.permutation(len(pool))
        left, right = perm[:n_a], perm[n_a:]
        a = build(pool[left], [rows[i] for i in left], ys, construct, "A")
        b = build(pool[right], [rows[i] for i in right], ys, construct, "B")
        sizes.append(np.linalg.norm(a.mean - b.mean, axis=1))
    arr = np.stack(sizes)
    mid = arr.shape[1] // 2
    return {"check": "null_scrambled", "construct": construct, "n_rep": n_rep,
            "median_norm": float(np.median(arr[:, mid])),
            "p95_norm": float(np.percentile(arr[:, mid], 95)),
            "layer": mid,
            "median_by_layer": np.median(arr, axis=0).tolist(),
            "p95_by_layer": np.percentile(arr, 95, axis=0).tolist()}
