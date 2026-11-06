"""Cores: the distinct construct-bearing shapes a construct is measured through.

A *core* is one syntactic realisation of the construct — a different surrounding
statement, a different body, a different operand form. It is not a menu value and
not a grid cell: the grid wraps a core in domains, scopes and padding, and menu
draws fill its identifiers. v1 and the pre-addendum handout had one core per
construct, so twin noise measured draw variance only and the bootstrap had one
group (report addendum item 3).

Contract: >=5 distinct cores per construct x applicable subtype is the licence
floor, build 8; the bootstrap resamples cores; core ID is logged per prompt.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .menu import Cell, Menu


@dataclass(frozen=True)
class Core:
    """One construct-bearing shape. `render` returns unindented source lines."""

    core_id: str
    construct: str
    family: str
    axes: dict[str, str]
    frame_class: str
    companions: tuple[str, ...]
    render: Callable[[Menu, Cell, np.random.Generator], list[str]] = field(compare=False,
                                                                          repr=False)
    #: "node" = the target is an AST node type (families 1-4); "name" = the target is a
    #: builtin's name (family 5), anchored on the Name node and verified by identifier.
    target_kind: str = "node"

    def __post_init__(self):
        if not self.core_id or "/" not in self.core_id:
            raise ValueError(f"core_id must be '<Construct>/<slug>': {self.core_id!r}")


def spread(levels: dict[str, tuple[str, ...]], n: int,
           rng: np.random.Generator,
           weights: dict[str, dict[str, float]] | None = None,
           invalid=None) -> list[dict[str, str]]:
    """Choose n distinct axis combinations.

    Axes named in `weights` are **corpus-weighted**: their level counts across the n cores
    are allocated to the weights (largest remainder, every level floored), so the realised
    distribution is the nearest multiple of 1/n to the target. Every other axis stays
    **balanced** and is chosen greedily for coverage, so its levels — including the rare
    ones — all appear. See weights.py for why the split falls where it does.

    Because the weights are realised by allocating cores rather than by weighting them,
    all n cores remain exchangeable and the core bootstrap is unchanged.

    `invalid(combination) -> bool` rejects combinations the family declares meaningless, and it
    exists because the axes are not always independent (item 95). `AugAssign`'s `scope_use`
    neighbourhood appends a use of the name the statement binds — and its `subscript_target`
    variant, `rows[0] += 1`, binds no name at all. Nothing in the allocator knew that, so the
    greedy coverage pass put **all sixteen** of the smoke build's `scope_use` cores on exactly
    that variant and the construct ended with zero scope positions, having had four before. The
    pairing is deterministic, so it was a systematic collision rather than bad luck.
    """
    from .weights import allocate

    names = list(levels)
    weights = weights or {}
    # An axis can only be weighted if n can honour the floor — at least one core per
    # level. Below that the request is a probe, not a battery, so the axis falls back to
    # balanced rather than raising: the floor is a property of a real set.
    wnames = [a for a in names if a in weights and 1 < len(levels[a]) <= n]
    bnames = [a for a in names if a not in wnames]

    if n > int(np.prod([len(levels[k]) for k in names])):
        raise ValueError(f"n={n} exceeds the number of distinct combinations")

    pool = [dict(zip(bnames, combo))
            for combo in itertools.product(*(levels[k] for k in bnames))] or [{}]
    if invalid is not None and bnames:
        kept = [c for c in pool if not invalid(c)]
        # only applied when something survives: a predicate that rejected everything would be
        # a declaration bug, and failing loudly here beats silently building a different battery
        if not kept:
            raise ValueError("`invalid` rejected every combination")
        pool = kept

    # A weighted level cannot take more cores than it has distinct partners: with eight
    # unweighted combinations, a level allocated twelve slots would have to repeat one.
    # Excess is redistributed to levels with headroom, and the realised weight is
    # reported by the caller rather than assumed to equal the target.
    alloc = {}
    for a in wnames:
        counts = collections.Counter(allocate(levels[a], n, weights[a]))
        cap = len(pool)
        excess = sum(max(0, c - cap) for c in counts.values())
        for lvl in counts:
            counts[lvl] = min(counts[lvl], cap)
        while excess > 0:
            room = [l for l in levels[a] if counts[l] < cap]
            if not room:
                raise ValueError(f"n={n} exceeds {cap * len(levels[a])} combinations")
            for lvl in sorted(room, key=lambda l: -weights[a].get(l, 0.0)):
                if excess == 0:
                    break
                counts[lvl] += 1
                excess -= 1
        seq = [l for l in levels[a] for _ in range(counts[l])]
        rng.shuffle(seq)
        alloc[a] = seq
    order = list(rng.permutation(len(pool)))
    pool = [pool[i] for i in order]
    uncovered = {k: set(levels[k]) for k in bnames}

    out: list[dict[str, str]] = []
    seen: set[tuple] = set()
    for i in range(n):
        fixed = {a: alloc[a][i] for a in wnames}
        best, best_gain = None, -1
        for j, cand in enumerate(pool):
            full = cand | fixed
            key = tuple(full[k] for k in names)
            if key in seen:
                continue
            if invalid is not None and invalid(full):
                continue
            gain = sum(1 for k in bnames if cand[k] in uncovered[k])
            if gain > best_gain:
                best, best_gain = j, gain
            if gain == len(bnames):
                break
        if best is None:
            raise RuntimeError(f"n={n} exceeds the distinct combinations available")
        # the pool is *not* consumed: the same unweighted combination paired with a
        # different weighted level is a different core, and popping globally capped the
        # run at the size of the unweighted product
        cand = pool[best]
        for k in bnames:
            uncovered[k].discard(cand[k])
        combo = cand | fixed
        seen.add(tuple(combo[k] for k in names))
        out.append(combo)
    return out
