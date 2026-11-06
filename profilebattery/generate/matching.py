"""Fit a B set's companion marginals to its A set's.

Two steps. First measure the A set's companion shares. Then choose, per companion, the
probability of adding the garnish that supplies it, so the B set's expected share
matches. Garnishes create more than one node type each, so the fit is greedy over
companions in descending order of gap, updating the expected baseline as it goes.

**Marginals, not the joint** — the same discipline the report applies to the generator's
calibration (S6): matching every joint would mean building a model of code, so the
declared object is a list of low-order statistics and the residual is measured rather
than assumed. Here the residual is exactly what the post-fit audit reports.
"""

from __future__ import annotations
from profilebattery.paths import data_root

import collections

from .seeds import stable_seed

import numpy as np

from .assemble import assemble, assemble_b
from .audit import SITES, present, scope_for
from .garnish import GARNISHES, SUPPLIERS, garnishes_for
from .menu import Cell, Menu

#: quasi-obligatory in any Python, and matched rather than capped
EXEMPT = ("Assign",)


def corpus_profile(construct: str) -> dict[str, float]:
    """The §2 companion rates for one construct — what A is calibrated *to*."""
    import json
    import os
    import pathlib

    from .audit import COUNTED

    D = (data_root() / "calibration")
    comp = json.loads((D / "companions.json").read_text()).get(construct, {})
    tot = json.loads((D / "construct_rates.json").read_text())[construct]["occurrences"]
    if not tot:
        return {}
    return {k: v / tot for k, v in comp.items() if k in COUNTED and k != construct}


def companion_profile(cores, menu: Menu, cells: list[Cell], construct: str,
                      seed: int = 7, site: str = "last",
                      pre: dict[str, float] | None = None,
                      post: dict[str, float] | None = None) -> dict[str, float]:
    """Share of a set's prompts carrying each companion, scoped to one read site."""
    count, n = collections.Counter(), 0
    for core in cores:
        for cell in cells:
            # a per-prompt seed: a constant one makes every draw — menu values and
            # garnish choices alike — identical across the set, which silently turns a
            # Bernoulli scheme into an all-or-nothing one
            p = assemble(core, menu, cell,
                         seed=stable_seed(core.core_id, cell.key(), seed),
                         pre_probs=pre, post_probs=post)
            if not p.valid:
                continue
            n += 1
            start, before = scope_for(p, site)
            count.update(present(p.text, construct, start, before))
    return {k: v / n for k, v in count.items()} if n else {}


def b_profile(bcores, token: str, menu: Menu, cells: list[Cell], construct: str,
              seed: int = 7, pre: dict[str, float] | None = None,
              post: dict[str, float] | None = None,
              site: str = "last") -> dict[str, float]:
    count, n = collections.Counter(), 0
    for core in bcores:
        for cell in cells:
            p = assemble_b(core, token, menu, cell,
                           seed=stable_seed(core.core_id, cell.key(), seed),
                           pre_probs=pre, post_probs=post)
            if not p.valid:
                continue
            n += 1
            start, before = scope_for(p, site)
            count.update(present(p.text, construct, start, before))
    return {k: v / n for k, v in count.items()} if n else {}


def fit_two_placements(cores, bcores, token: str, menu: Menu, cells: list[Cell],
                       construct: str, components: tuple[str, ...], cap: float,
                       seed: int = 7):
    """Fit garnishes at both placements: before the carrier, then after it.

    The prefix fit targets the A set's *token-site* profile, which is what the causal
    read at the construct token can see. The suffix fit then targets the whole-core
    profile with the prefix already in place, so the last-token site matches too. One B
    set, matched at both ends.
    """
    a_tok = companion_profile(cores, menu, cells, construct, seed, site="token")
    b_tok = b_profile(bcores, token, menu, cells, construct, seed, site="token")
    pre = fit_garnishes(a_tok, b_tok, construct, components, cap=cap)

    a_last = companion_profile(cores, menu, cells, construct, seed, site="last")
    b_last = b_profile(bcores, token, menu, cells, construct, seed, pre=pre, site="last")
    post = fit_garnishes(a_last, b_last, construct, components, cap=cap)

    # Corrective passes. The forward fit can only raise a share, so it cannot undo an
    # overshoot — and with two placements each allowed its own side-effect headroom,
    # overshoot is exactly what happened at the last-token site. Here the measured
    # result feeds back: any garnish whose creations sit more than the cap above target
    # has its probability scaled down, at both placements, until it fits or three
    # passes are spent.
    for _ in range(3):
        over = _overshoot(a_last,
                          b_profile(bcores, token, menu, cells, construct, seed,
                                    pre=pre, post=post, site="last"),
                          construct, components, cap)
        if not over:
            break
        pre = _relax(pre, over)
        post = _relax(post, over)
    return pre, post


def _overshoot(target: dict[str, float], got: dict[str, float], construct: str,
               components: tuple[str, ...], cap: float) -> dict[str, float]:
    """companion -> how far B exceeds A beyond the cap."""
    out = {}
    for c, v in got.items():
        if c == construct or c in components or c in EXEMPT:
            continue
        excess = v - target.get(c, 0.0) - cap * SAFETY
        if excess > 0.01:
            out[c] = excess
    return out


def _relax(probs: dict[str, float], over: dict[str, float]) -> dict[str, float]:
    out = dict(probs)
    for g, p in probs.items():
        worst = max((over.get(made, 0.0) for made in GARNISHES[g][1]), default=0.0)
        if worst > 0:
            out[g] = max(0.0, p * max(0.2, 1.0 - worst / max(p, 1e-6) * 0.5))
    return {g: p for g, p in out.items() if p > 0.02}


#: how much of the cap the overshoot guard is allowed to consume, leaving room for the
#: Bernoulli sampling noise of a finite set. Two placements are fitted, each allowed
#: this much headroom on its side-effects, so the budget is halved relative to a
#: single-placement fit.
SAFETY = 0.45


def _needed(target_share: float, have: float) -> float:
    """Bernoulli probability that lifts a *union* share from `have` to `target_share`.

    Shares are P(the prompt contains the companion at all), not counts, so a garnish
    added to a prompt that already carries the companion raises nothing. Treating the
    two as additive was under-shooting every fit: the correction is
    p = (target - have) / (1 - have).
    """
    if have >= 1.0 or target_share <= have:
        return 0.0
    return (target_share - have) / (1.0 - have)


def _union(have: float, p: float) -> float:
    return have + p * (1.0 - have)


def fit_garnishes(target: dict[str, float], baseline: dict[str, float],
                  construct: str, components: tuple[str, ...] = (),
                  cap: float = 0.15) -> dict[str, float]:
    """Garnish -> probability, so B's expected companion shares match A's.

    Every garnish creates more than one node type, so raising one companion to target
    pushes its side-effects up too. The probability is therefore clipped by the
    tightest side-effect headroom: a garnish is used only as far as it does not push
    any companion it creates more than `cap` above that companion's own target. This is
    what stops the fit from trading one asymmetry for a larger opposite one.
    """
    usable = set(garnishes_for(construct))
    expected = dict(baseline)
    probs: dict[str, float] = {}
    gaps = sorted(((c, target.get(c, 0.0) - baseline.get(c, 0.0)) for c in target),
                  key=lambda kv: -kv[1])
    for companion, _ in gaps:
        if companion == construct or companion in components:
            continue
        # Several suppliers may serve one companion, and each is clipped by a different
        # side-effect ceiling. Using only the roomiest one left large gaps where every
        # supplier was individually constrained, so they are applied in turn until the
        # want is met or all of them are exhausted.
        for _ in range(len(SUPPLIERS.get(companion, ()))):
            want = _needed(target.get(companion, 0.0), expected.get(companion, 0.0))
            if want <= 0.01:
                break
            best, best_room = None, 0.0
            for g in SUPPLIERS.get(companion, ()):
                if g not in usable or g in probs:
                    continue
                room = want
                for made in GARNISHES[g][1]:
                    if made == companion or made in components or made in EXEMPT:
                        continue
                    ceiling = target.get(made, 0.0) + cap * SAFETY
                    room = min(room, _needed(ceiling, expected.get(made, 0.0)))
                if room > best_room:
                    best, best_room = g, room
            if best is None or best_room <= 0.01:
                break
            p = float(min(1.0, best_room))
            probs[best] = p
            for made in GARNISHES[best][1]:
                expected[made] = _union(expected.get(made, 0.0), p)
    return probs


def worst_gap(a: dict[str, float], b: dict[str, float], construct: str,
              components: tuple[str, ...] = ()) -> tuple[str, float]:
    """Largest |A-B| companion difference, ignoring components and exempt names."""
    keys = (set(a) | set(b)) - {construct} - set(components) - set(EXEMPT)
    if not keys:
        return ("-", 0.0)
    # `sorted` before `max`, and it is not cosmetic: on a tie `max` returns whichever
    # candidate the *set* yielded first, and Python salts string hashing per process, so the
    # reported companion changed between runs. `NamedExpr` was the one that tied -- `BinOp`
    # and `Compare` at the same |gap| -- and it alone made `a_calibration.json` differ
    # byte-for-byte across runs while every probability and every pass/fail stayed identical.
    worst = max(sorted(keys), key=lambda k: abs(a.get(k, 0.0) - b.get(k, 0.0)))
    return (worst, a.get(worst, 0.0) - b.get(worst, 0.0))
