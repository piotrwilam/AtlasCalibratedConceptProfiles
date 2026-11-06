"""Profiles — §5 steps 3 to 5.

`pi(S) = mean_p z_p` with a per-unit standard error, a median twin, and a bootstrap over
**cores** rather than prompts. Three things here are not the obvious implementation.

**Each prompt is standardised against the cell its own read site falls in.** Prompts in one
set do not share a yardstick: a `for` at band 0 and a `for` at band 3 are read at different
relative positions, which is precisely what the cells exist to separate. Standardising a whole
set against one cell would put the position difference back into the profile.

**The bootstrap resamples cores, not prompts.** Templates leak — two prompts from one core
share their surrounding statement, so they are not independent observations of the construct.
Resampling prompts would report an error bar for a quantity nobody wants.

**The median twin is computed per unit, not per prompt.** Its job is to flag units where mean
and median disagree, which is where a unit is bimodal across contexts and the mean describes
no actual prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cells import band, cell_key, token_class
from .yardstick import Yardstick


def recorder_order(rows: list[dict]) -> list[dict]:
    """The prompt rows in the order the **per-prompt arrays** hold them.

    `sweep_anchored` sorts each set by length before batching, to cut padding
    (`profilebattery/record/sweep.py:137`), and writes the per-prompt vectors in that order;
    the jsonl is in its own. `build` pairs row *i* with vector *i*, so rows must be put into
    this order before they reach it, or every prompt's core, cell and padding level is attached
    to another prompt's vector.

    The fault is invisible to profile-level checks -- a set's mean is a mean over all its
    prompts and does not depend on order -- so it survived until a per-(core, cell) comparison
    was made. Under this order the per-prompt route reproduces the accumulator route exactly
    (cosine 1.000000, carrier overlap 1.000000); in jsonl order it gives 0.982489 and 0.977350,
    and the between-core SE only 0.630775.
    """
    return sorted(rows, key=lambda r: r.get("n_tokens") or len(r.get("text", "")))


def cell_of(row: dict) -> str:
    """The yardstick cell of a prompt's construct-token site, from stored fields only."""
    return cell_key(band(row["site_construct_token"], row["n_tokens"]),
                    token_class(row["anchor_piece"]))


@dataclass
class Profile:
    construct: str
    set_kind: str
    subtype: str | None
    n: int
    mean: np.ndarray            # [layers, units]
    se: np.ndarray              # REPORTED: between-core spread of core means / sqrt(cores)
    median: np.ndarray
    raw_mean: np.ndarray        # unstandardised, so a re-standardisation stays arithmetic
    core_means: dict            # core_id -> [layers, units], for the bootstrap
    cells: dict = field(default_factory=dict)
    skipped: int = 0
    #: per-prompt spread / sqrt(n) — kept for the comparison that justified the core-level
    #: SE, and None on the accumulator route, which never sees individual prompts
    se_prompt: np.ndarray | None = None

    def norm(self) -> np.ndarray:
        return np.linalg.norm(self.mean, axis=1)

    def top_share(self, k: int = 64) -> np.ndarray:
        """Share of each layer's profile norm carried by its k largest coordinates."""
        out = []
        for layer in self.mean:
            mag = np.sort(np.abs(layer))[::-1]
            out.append(float((mag[:k] ** 2).sum() / max((mag ** 2).sum(), 1e-12)))
        return np.asarray(out)


def core_se(core_means: dict) -> np.ndarray:
    """The reported standard error: between-core spread of core means / sqrt(cores).

    The per-prompt SE treats 720 prompts as independent draws. They are not: they come
    from eight structurally distinct cores, and prompts within a core share a shape, so
    the effective sample size is nearer the number of cores than the number of prompts.
    Measured on the untrained reference, the per-prompt interval covered 0.903 against a
    nominal 0.95 — understated, in the direction this predicts.

    Cores are equally weighted, which is right while the grid is balanced (8 cores x 90
    prompts). If a core ever loses prompts to the validity filters the weighting should
    follow; the cell census in `Profile.cells` is where that would show.

    With one core the between-core spread is undefined and this returns zeros rather
    than a nan, so a single-core diagnostic set does not poison a whole table.
    """
    stack = np.stack([core_means[c] for c in sorted(core_means)])
    if len(stack) < 2:
        return np.zeros_like(stack[0])
    return stack.std(axis=0, ddof=1) / np.sqrt(len(stack))


def build(primary: np.ndarray, rows: list[dict], ys: Yardstick, construct: str,
          set_kind: str, subtype: str | None = None) -> Profile:
    """primary: [n_prompts, layers, units] as recorded at the construct-token site."""
    n_prompts, n_layers, width = primary.shape
    usable, cells = [], {}
    for i, row in enumerate(rows[:n_prompts]):
        c = cell_of(row)
        if not all(ys.has(l, c) for l in range(n_layers)):
            continue
        usable.append((i, c))
        cells[c] = cells.get(c, 0) + 1
    if not usable:
        raise ValueError(f"{construct}/{set_kind}: no prompt has a supported cell")

    z = np.empty((len(usable), n_layers, width), dtype=np.float64)
    raw = np.empty_like(z)
    for j, (i, c) in enumerate(usable):
        x = primary[i].astype(np.float64)
        raw[j] = x
        for layer in range(n_layers):
            z[j, layer] = (x[layer] - ys.mu(layer, c)) / ys.sigma(layer, c)

    core_means: dict[str, np.ndarray] = {}
    by_core: dict[str, list[int]] = {}
    for j, (i, _) in enumerate(usable):
        by_core.setdefault(rows[i].get("core_id", "?"), []).append(j)
    for core, idx in by_core.items():
        core_means[core] = z[idx].mean(axis=0)

    return Profile(construct=construct, set_kind=set_kind, subtype=subtype,
                   n=len(usable), mean=z.mean(axis=0),
                   se=core_se(core_means),
                   se_prompt=z.std(axis=0, ddof=1) / np.sqrt(len(usable)),
                   median=np.median(z, axis=0), raw_mean=raw.mean(axis=0),
                   core_means=core_means, cells=cells,
                   skipped=n_prompts - len(usable))


def core_bootstrap(profile: Profile, n_boot: int = 500, seed: int = 0) -> np.ndarray:
    """Bootstrap SE over core means. Returns [layers, units]."""
    cores = list(profile.core_means)
    if len(cores) < 2:
        return np.full_like(profile.mean, np.nan)
    stack = np.stack([profile.core_means[c] for c in cores])
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot,) + profile.mean.shape)
    for b in range(n_boot):
        pick = rng.integers(0, len(cores), len(cores))
        draws[b] = stack[pick].mean(axis=0)
    return draws.std(axis=0, ddof=1)


def concept(a: Profile, b: Profile) -> dict:
    """pi(A) - pi(B), with the error bars added in quadrature."""
    return {"construct": a.construct, "subtype": b.subtype,
            "delta": a.mean - b.mean,
            "se": np.sqrt(a.se ** 2 + b.se ** 2),
            "n_a": a.n, "n_b": b.n}


def pooled(b_profiles: list[Profile], weights: dict[str, float] | None = None) -> Profile:
    """Pool B subtypes under declared weights; equal weights when none are given.

    Secondary other-role subtypes enter at weight 0 by the symbol table's rule — registering
    them as a second subtype removes the frame confound from the concept profile only if they
    do not carry weight in the pooled B.
    """
    if not b_profiles:
        raise ValueError("no B profiles to pool")
    w = np.array([(weights or {}).get(p.subtype or "", 1.0) for p in b_profiles],
                 dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(b_profiles))
    w = w / w.sum()
    mean = sum(wi * p.mean for wi, p in zip(w, b_profiles))
    se = np.sqrt(sum((wi * p.se) ** 2 for wi, p in zip(w, b_profiles)))
    median = sum(wi * p.median for wi, p in zip(w, b_profiles))
    raw = sum(wi * p.raw_mean for wi, p in zip(w, b_profiles))
    merged: dict[str, np.ndarray] = {}
    for p in b_profiles:
        merged.update({f"{p.subtype}:{k}": v for k, v in p.core_means.items()})
    return Profile(construct=b_profiles[0].construct, set_kind="B", subtype="pooled",
                   n=sum(p.n for p in b_profiles), mean=mean, se=se, median=median,
                   raw_mean=raw, core_means=merged)


def from_accumulators(acc, ys: Yardstick, construct: str, set_kind: str,
                      site: str = "token", stream: str = "mk") -> Profile:
    """A profile from per-core, **per-cell** accumulators.

    This is the only route for m_k and for the span-end and last-token sites, where
    per-prompt vectors are not kept. Keying the accumulators by cell is what makes it correct:
    each cell's accumulator is standardised against that cell's own yardstick and the set's
    profile is the count-weighted combination. Standardising a pooled accumulator against one
    cell — the first implementation — is wrong wherever a set's prompts span cells, and at the
    last-token site they certainly do, since a prompt's final token is sometimes a dedent and
    sometimes a symbol.
    """
    want = f"{stream}|"
    keys = [k[:-2] for k in acc.files
            if k.endswith("|n") and k.startswith(want) and f"|{site}|" in k]
    if not keys:
        raise ValueError(f"no {stream} accumulators for site {site}")
    # core -> layer -> cell -> (n, mean)
    tree: dict[str, dict[int, dict[str, tuple[int, np.ndarray]]]] = {}
    for k in keys:
        parts = k.split("|")
        if len(parts) != 5:
            raise ValueError(f"accumulator key {k!r} is not cell-keyed; re-record")
        _, core, _, layer, cell = parts
        li = int(layer[1:])
        tree.setdefault(core, {}).setdefault(li, {})[cell] = (
            int(acc[k + "|n"][0]), acc[k + "|mean"])
    layers = sorted({li for v in tree.values() for li in v})

    core_means, raw_layers, cell_counts = {}, [], {}
    for core, per_layer in tree.items():
        rows = []
        for li in layers:
            cells = per_layer.get(li, {})
            total = sum(n for n, _ in cells.values()) or 1
            z = np.zeros_like(next(iter(cells.values()))[1], dtype=np.float64)
            for cell, (n, mean) in cells.items():
                if not ys.has(li, cell):
                    continue
                z += (n / total) * (mean - ys.mu(li, cell)) / ys.sigma(li, cell)
                cell_counts[cell] = cell_counts.get(cell, 0) + n
            rows.append(z)
        core_means[core] = np.stack(rows)
    stack = np.stack([core_means[c] for c in sorted(core_means)])
    for li in layers:
        per_core = []
        for core in sorted(tree):
            cells = tree[core].get(li, {})
            tot = sum(n for n, _ in cells.values()) or 1
            per_core.append(sum((n / tot) * m for n, m in cells.values()))
        raw_layers.append(np.mean(per_core, axis=0))
    n_total = sum(n for v in tree.values() for c in v.get(layers[0], {}).values()
                  for n in (c[0],))
    return Profile(construct=construct, set_kind=set_kind, subtype=stream,
                   n=n_total, mean=stack.mean(axis=0),
                   se=core_se(core_means),
                   median=np.median(stack, axis=0),
                   raw_mean=np.stack(raw_layers),
                   core_means=core_means, cells=cell_counts)
