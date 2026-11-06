"""Which axes are corpus-weighted, and how the weights are derived.

**The rule** (decided 20 Sept, stated so future axes need no further decision):

  An axis whose levels *determine* companion presence is corpus-weighted.
  An axis that exists to cancel a confound stays balanced.

`body` determines companions — two of its four levels emit `BinOp` in every single
prompt, so balancing them puts `BinOp` at ~0.5 whatever the construct's corpus rate is,
and for `With` that is 0.58 against 0.13. Domain, scope, padding and operand exist to
cancel names, context and position and to cover the construct's forms, so they stay
balanced and the rare operand levels keep their coverage.

Why it matters, and it is not the A−B cap: companions cancel in A − B at any level, but
they do not cancel in **A − C**, which is the primary object for the structural
constructs with no B, for `Tuple`, for the compound family's span end, and for every
statistic in §8. `With` carrying `BinOp` at 0.58 against C's 0.13 puts 0.45 of `BinOp`'s
own profile into `With`'s A − C.

Weights are realised by **allocating cores** to levels rather than by weighting cores, so
every core still stands for 1/n of its set and the core bootstrap needs no change — which
is the one place unequal weights could have silently corrupted an error bar. The cost is
resolution: with n cores the realised weight is a multiple of 1/n, so n = 8 realises a
weight only to about ±0.06 and the 24-core fitting sample to ±0.02.
"""

from __future__ import annotations

import collections
import contextlib

import numpy as np

#: axis -> "determines" (corpus-weighted) | "cancels" (balanced)
#:
#: Applying the rule needs one clarification, because *several* axes determine companion
#: presence as a side effect: a comprehension's `tuple` element level emits `Tuple` in
#: every prompt, as `mutate` emits `BinOp`. The line that separates them is whether the
#: axis exists to **cancel a confound** (domain, scope, padding) or to **cover the
#: construct's own grammatical forms** (operand, variant, element, suite, host, source).
#: Both of those stay balanced — the first because balance is what cancels, the second
#: because coverage of the rare forms is the point.
#:
#: `body` is the only axis that is neither: what a suite *does* is a free stylistic choice,
#: not a form of the construct and not a confound being cancelled. So it is the one that
#: is corpus-weighted, and the residual excess on the coverage axes is reported as
#: characterisation rather than chased.
AXIS_KIND = {
    "body": "determines",
    "access": "cancels",      # introduced to reach corpus rates; balance suffices
    "operand": "cancels",     # covers the construct's forms, rare ones included
    "suite": "cancels",       # a form of the construct
    "variant": "cancels",     # a form of the construct
    "element": "cancels",     # a form of the comprehension
    "source": "cancels",      # a form of the comprehension
    "host": "cancels",        # where an expression sits; rotated for coverage
    "neighbourhood": "cancels",
    "shape": "cancels",
}
WEIGHTED_AXES = tuple(a for a, k in AXIS_KIND.items() if k == "determines")

#: no level may vanish: a level sampled at zero cannot be reweighted back, checked, or
#: audited later. Costs a little fidelity, keeps the grid's coverage property.
FLOOR = 0.05


def allocate(levels: tuple[str, ...], n: int, weights: dict[str, float],
             floor: float = FLOOR) -> list[str]:
    """n core slots assigned to levels, honouring weights and the floor.

    Largest-remainder allocation after reserving the floor, so the realised distribution
    is the closest multiple of 1/n to the target and every level keeps at least one core.
    """
    k = len(levels)
    if n < k:
        raise ValueError(f"{n} cores cannot cover {k} levels at a {floor} floor")
    reserved = max(1, int(round(floor * n)))
    if reserved * k > n:
        reserved = 1
    left = n - reserved * k
    w = np.array([max(weights.get(l, 0.0), 0.0) for l in levels], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
    exact = w * left
    counts = np.floor(exact).astype(int)
    for i in np.argsort(-(exact - counts))[:left - counts.sum()]:
        counts[i] += 1
    out: list[str] = []
    for level, c in zip(levels, counts + reserved):
        out.extend([level] * int(c))
    return out


def load(path: str | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """construct -> axis -> level -> weight, from the fitted file; empty if absent.

    An absent file means balanced axes everywhere, which is the honest fallback: nothing
    is silently weighted by a stale table.
    """
    import json
    import os
    import pathlib

    p = pathlib.Path(path or os.path.expanduser(
        "$ATLAS_DATA/calibration/core_weights.json"))
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("weights", {})


_CACHE: dict | None = None

#: while true, `for_construct` reports no weights, so cores are allocated to levels evenly
_BALANCED = False


@contextlib.contextmanager
def balanced():
    """Build cores as if no fitted weights existed, for the duration of the block.

    **Why this exists, and it is the fix for a feedback loop.** `fit_core_weights` writes
    `core_weights.json`; `for_construct` reads it; `spread` allocates cores to levels by it;
    and `body_signatures` -- the measurement the fit is computed *from* -- builds its cores
    through exactly that path. So the fit's input depended on the fit's own previous output,
    and running the chain twice gave two different answers.

    Measured before the fix (1 Oct): from the committed state one run moved a weight by
    **0.364**, a third of the probability mass, and iterating did not converge -- eight runs
    gave eight states in a **limit cycle of period 3** (per-step moves 0.081, 0.143, 0.224,
    repeating). The core weights were therefore not identified by the procedure: which weights
    the battery was built with depended on how many times the script had been run.

    The measurement is *meant* to be per level -- "what each `body` level emits, by rendering"
    -- so it must not depend on the weights at all. The leak was that a skewed weight starves a
    level of cores, so that level's signature is measured on fewer prompts or missing entirely.
    Allocating evenly while measuring makes the fit a function of the corpus alone, which makes
    it idempotent and reproducible.
    """
    global _BALANCED
    prev = _BALANCED
    _BALANCED = True
    try:
        yield
    finally:
        _BALANCED = prev


def for_construct(construct: str) -> dict[str, dict[str, float]]:
    global _CACHE
    if _BALANCED:
        return {}
    if _CACHE is None:
        _CACHE = load()
    return _CACHE.get(construct, {})


def body_signatures(construct: str, menu, cells, n_probe: int = 3) -> dict:
    """Per body level, the share of prompts in which each companion appears.

    Measured by rendering, not declared: the mapping from a body to the companions it
    emits is a property of the generator and drifts when the bodies change.
    """
    from .assemble import assemble
    from .audit import present
    from .families import build_a_cores

    from .seeds import stable_seed

    tally: dict[str, list] = collections.defaultdict(lambda: [0, collections.Counter()])
    # `balanced()` is what breaks the feedback loop: without it the cores this measurement
    # runs on are allocated by the weights it is computing. `stable_seed` replaces the
    # salted `hash()`, which returned a different number in every interpreter.
    with balanced():
        cores = build_a_cores(construct, 24, np.random.default_rng(7))
    for core in cores:
        level = core.axes.get("body")
        if level is None:
            continue
        for cell in cells[:n_probe]:
            p = assemble(core, menu, cell, seed=stable_seed(core.core_id, cell.key()))
            if not p.valid:
                continue
            tally[level][0] += 1
            tally[level][1].update(present(p.text, construct, p.body_start))
    return {lvl: {c: v / n for c, v in counter.items()}
            for lvl, (n, counter) in tally.items() if n}


def fit_body_weights(signatures: dict, target: dict[str, float],
                     floor: float = FLOOR) -> dict[str, float]:
    """Weights over body levels whose mixture best matches the corpus companion rates.

    A small constrained least squares: minimise ||S w - t|| over the companions both
    sides have, subject to w >= floor and sum(w) = 1. Projected gradient rather than a
    solver dependency — the problem has at most a handful of levels.
    """
    levels = sorted(signatures)
    if not levels:
        return {}
    companions = sorted({c for s in signatures.values() for c in s} & set(target))
    if not companions:
        return {l: 1.0 / len(levels) for l in levels}
    S = np.array([[signatures[l].get(c, 0.0) for c in companions] for l in levels]).T
    t = np.array([target[c] for c in companions])
    w = np.full(len(levels), 1.0 / len(levels))
    step = 0.5
    for _ in range(4000):
        grad = 2 * S.T @ (S @ w - t)
        w = _project(w - step * grad, floor)
    return {l: float(round(v, 4)) for l, v in zip(levels, w)}


def _project(w: np.ndarray, floor: float) -> np.ndarray:
    """Onto {w >= floor, sum(w) = 1}."""
    k = len(w)
    if floor * k > 1:
        return np.full(k, 1.0 / k)
    w = np.maximum(w, floor)
    excess = w.sum() - 1.0
    for _ in range(64):
        if abs(excess) < 1e-12:
            break
        free = w > floor + 1e-12
        if excess > 0 and free.any():
            take = np.minimum(excess * (w[free] - floor) / (w[free] - floor).sum(),
                              w[free] - floor)
            w[free] -= take
        else:
            w += -excess / k
            w = np.maximum(w, floor)
        excess = w.sum() - 1.0
    return w / w.sum()
