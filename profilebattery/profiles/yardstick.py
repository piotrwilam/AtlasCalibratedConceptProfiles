"""The yardstick — §5 step 2, built from C's cell accumulators.

Per (checkpoint, layer, cell) rather than per site. Decision 2's reasoning: C has no
construct token, so standardising A's construct-token vectors against C's *last-token*
statistics compares different positions. Bands control for context length; the token class
controls for what the token is, which is what actually drives a unit's ordinary behaviour.

Two things are kept alongside the standardised numbers and both matter.

**Raw means and sigma0 itself are first-class outputs.** sigma0 is developmental — it grows
as the model trains — so standardising by a checkpoint-specific sigma0 removes exactly that
growth. A profile therefore measures *distinguishability from ordinary code*, not activation
strength, and the raw quantities are what let the other reading be recovered.

**The floor is smooth**, sigma0 <- sqrt(sigma0^2 + (eps * median)^2), not a hard clip. A
construct-selective unit is quiet on generic code by construction, so it has a small sigma0 by
construction, and a hard clip makes an open parameter decide the magnitude of exactly the
units the study is about. The smooth form has no cliff, and eps-stability is check 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cells import all_cells

DEFAULT_EPS = 0.01
EPS_GRID = (0.003, 0.01, 0.03)


@dataclass
class Cell:
    n: int
    mean: np.ndarray
    var: np.ndarray
    active: np.ndarray

    def sd(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.var, 0.0))


class Yardstick:
    """mu0 and sigma0 per (layer, cell), with support counts and the smooth floor."""

    def __init__(self, cells: dict[tuple[int, str], Cell], eps: float = DEFAULT_EPS):
        self.cells = cells
        self.eps = eps
        self.layers = sorted({l for l, _ in cells})
        self._floored: dict[tuple[int, str], np.ndarray] = {}

    @classmethod
    def from_npz(cls, path, eps: float = DEFAULT_EPS, stream: str = "C") -> "Yardstick":
        """The yardstick for one stream of the C pass.

        The C recording holds three kinds of key, and they are not interchangeable:
        `C|L{layer}|{cell}|{stat}` for the gated product, `M|...` for m_k — whose spread
        is the variance of a linear combination and so is not recoverable from C — and
        `X|L{layer}|{cell}|{token}|{stat}` for the excluded subsets the masked variant is
        recovered from. Selecting by prefix *and* arity is what keeps them apart: reading
        every `|n` key as a four-part C key crashes on the five-part X keys, and silently
        mixing M into C would put an m_k denominator under an h profile.
        """
        z = np.load(path)
        want = f"{stream}|"
        out: dict[tuple[int, str], Cell] = {}
        for key in z.files:
            if not (key.startswith(want) and key.endswith("|n")):
                continue
            parts = key.split("|")
            if len(parts) != 4:
                continue
            _, layer, cell, _ = parts
            li = int(layer[1:])
            n = int(z[key][0])
            base = f"{stream}|{layer}|{cell}"
            m2 = z[f"{base}|m2"]
            out[(li, cell)] = Cell(n=n, mean=z[f"{base}|mean"],
                                   var=m2 / max(n - 1, 1),
                                   active=z[f"{base}|active"])
        if not out:
            raise ValueError(f"no {stream!r} cells in {path}")
        return cls(out, eps)

    def sigma(self, layer: int, cell: str) -> np.ndarray:
        """Floored sigma0. Smooth, so no unit sits on a cliff set by an open parameter."""
        key = (layer, cell)
        if key in self._floored:
            return self._floored[key]
        c = self.cells[key]
        sd = c.sd()
        floor = self.eps * float(np.median(sd)) if sd.size else 0.0
        out = np.sqrt(sd ** 2 + floor ** 2)
        self._floored[key] = out
        return out

    def rel_se_sigma(self, layer: int, cell: str) -> float:
        """Relative standard error of sigma0 in this cell: 1 / sqrt(2(n-1)).

        The sampling error of a normal sample's standard deviation. The C pass stores `n` per
        (layer, cell), so this is known exactly rather than estimated: on the test record it runs
        from 0.71% in the identifier cells to 2.19% in the thinnest keyword cell, the keyword
        cells being the ones a 1,000-prompt C pass leaves thinnest.
        """
        n = self.cells[(layer, cell)].n
        return 1.0 / np.sqrt(2.0 * max(n - 1, 1))

    def jittered(self, rng) -> "Yardstick":
        """A copy with sigma0 perturbed by its own sampling error, for the core bootstrap.

        The bootstrap resamples cores, but the yardstick is estimated from the C pass, which has
        no core, shard or prompt-group axis — so it cannot be resampled alongside them. This
        stands in for that: each cell's sigma0 is multiplied by (1 + r * N(0,1)) per unit, with r
        the cell's own relative standard error, which keeps the yardstick's contribution in the
        interval without the C pass having to be sharded.

        Measured cost of *omitting* it (`experiments/yardstick_jitter.py`): the perturbed top-5%
        set holds 0.966-0.980 Jaccard against the unperturbed one, which is 6-8% of the
        instability the twin ceiling already absorbs. Sharding the C pass to capture that
        directly would cost 56-112 GB and, by splitting the thin keyword cells eight ways, would
        raise sigma0's own error from 2% to 5-7% - paying a great deal for a worse yardstick.
        """
        out = {}
        for key, c in self.cells.items():
            layer, cell = key
            r = self.rel_se_sigma(layer, cell)
            factor = np.maximum(1.0 + r * rng.standard_normal(c.var.shape), 1e-6)
            out[key] = Cell(n=c.n, mean=c.mean, var=c.var * factor ** 2, active=c.active)
        return Yardstick(out, self.eps)

    def mu(self, layer: int, cell: str) -> np.ndarray:
        return self.cells[(layer, cell)].mean

    def has(self, layer: int, cell: str) -> bool:
        return (layer, cell) in self.cells

    def support(self, layer: int, cell: str) -> int:
        return self.cells[(layer, cell)].n if self.has(layer, cell) else 0

    def thin_units(self, layer: int, cell: str, min_active: int) -> np.ndarray:
        """Units whose spread in this cell rests on too few active observations.

        Replaces the global near-dead flag: support is a property of (unit, cell), not of a
        unit, and a unit quiet in one band may be the busiest in another.
        """
        if not self.has(layer, cell):
            return np.zeros(0, dtype=bool)
        return self.cells[(layer, cell)].active < min_active

    def summary(self) -> dict:
        rows = {}
        for (layer, cell), c in self.cells.items():
            rows.setdefault(cell, []).append(c.n)
        return {"layers": len(self.layers), "cells": len(rows),
                "missing": [c for c in all_cells() if c not in rows],
                "support_min": {c: int(min(v)) for c, v in sorted(rows.items())}}


def standardise(x: np.ndarray, ys: Yardstick, layer: int, cell: str) -> np.ndarray:
    """z = (x - mu0) / sigma0 for one layer and one yardstick cell."""
    return (x - ys.mu(layer, cell)) / ys.sigma(layer, cell)


def remove(total: Cell, part: Cell) -> Cell:
    """Statistics of `total` with the observations of `part` removed.

    The reverse of Chan's merge. This is what makes the per-construct masked yardstick
    affordable: rather than a full masked accumulator per construct — 30 x 24 x 20 of them —
    only the *excluded* positions are accumulated, and they are a small fraction of C.
    """
    n = total.n - part.n
    if n <= 0:
        return Cell(n=0, mean=np.zeros_like(total.mean), var=np.zeros_like(total.var),
                    active=np.zeros_like(total.active))
    mean = (total.n * total.mean - part.n * part.mean) / n
    delta = part.mean - mean
    m2_total = total.var * max(total.n - 1, 1)
    m2_part = part.var * max(part.n - 1, 1)
    m2 = m2_total - m2_part - (delta ** 2) * (n * part.n / total.n)
    return Cell(n=n, mean=mean, var=np.maximum(m2, 0.0) / max(n - 1, 1),
                active=total.active - part.active)


#: above this excluded share the reverse merge subtracts near-equal quantities and its
#: precision collapses. Measured on the recorded C the worst share is 0.196 (`=` in
#: b4:symbol), so this is a guard rather than a live problem — but generated C is
#: attribute-poor, and against harvested C a `.` cell would sit far higher.
MAX_EXCLUDED_SHARE = 1 / 3


def masked(ys: "Yardstick", excluded: dict[tuple[int, str], Cell],
           eps: float = DEFAULT_EPS) -> tuple["Yardstick", dict]:
    """A yardstick with one construct's own token positions removed, and its audit.

    Recovering masked statistics by reversing Chan's merge is exact while the excluded subset
    is small. Where it is not — a frequent symbol whose token is a large share of its cell —
    the subtraction is a difference of near-equal quantities and the precision goes. Those
    cells fall back to the shared yardstick, flagged, and the excluded share is reported per
    cell because the deflation report wants it anyway.
    """
    cells, audit = {}, {}
    for key, cell in ys.cells.items():
        part = excluded.get(key)
        share = (part.n / cell.n) if (part is not None and cell.n) else 0.0
        entry = {"excluded": part.n if part is not None else 0, "total": cell.n,
                 "share": round(share, 4), "fallback": False}
        if part is None or not part.n:
            cells[key] = cell
        elif share > MAX_EXCLUDED_SHARE:
            cells[key] = cell
            entry["fallback"] = "share above MAX_EXCLUDED_SHARE; shared yardstick used"
        else:
            got = remove(cell, part)
            if got.n <= 0 or np.any(got.var < 0):
                cells[key] = cell
                entry["fallback"] = "reverse merge produced a negative variance"
            else:
                cells[key] = got
        audit[f"L{key[0]}|{key[1]}"] = entry
    return Yardstick(cells, eps), audit


def deflation(shared: "Yardstick", mask: "Yardstick", layer: int,
              cell: str) -> float:
    """Median ratio sigma0(masked) / sigma0(shared) — the z-deflation the shared yardstick
    imposes on a construct whose own tokens sit inside C at their corpus rate."""
    if not (shared.has(layer, cell) and mask.has(layer, cell) and mask.support(layer, cell)):
        return float("nan")
    return float(np.median(mask.sigma(layer, cell) / shared.sigma(layer, cell)))
