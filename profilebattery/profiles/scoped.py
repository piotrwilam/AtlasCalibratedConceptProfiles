"""Scope-restricted profiles from the recorded `hs` accumulators — A24, S15.

The sweep recorded a second, additive accumulator family keyed by core *and* scope,
`hs|{core}@{scope}|token|L{i}|{cell}`, at every construct of every reference
(`SCOPE_SCHEDULE.json` is `mode: all`, option A, not the D-plus subset its docstring
describes). Nothing here re-records anything: the scope key is already in the record, and a
key not written at recording time could not be recovered by any later compute.

**Why the indented profile is a separate thing and not the average of two stored ones.**
Cowork's read-site rule (1–6 Oct, NOTES §3a item 2, option iii): for the constructs whose
construct-token piece differs between module and indented scope — `for` at module level
against `▁for` indented, two vocabulary ids — the *primary* profile is the indented scopes,
function and method pooled, and the module profile is a separate TOKEN reading. Measured on
`For` at `pretrain_iter_0484865`, the module profile sits at cosine 0.8335 to the pooled-over-
all-scopes one at layer 12, so this is not a rounding question.
"""
from __future__ import annotations

import numpy as np

#: the three scopes the recorder keys by
SCOPES = ("module", "function", "method")
#: the primary profile for the scope-varying constructs: function and method pooled
INDENTED = ("function", "method")
#: every grouping S15 stores, in the order it stores them
GROUPS = {"module": ("module",), "function": ("function",),
          "method": ("method",), "indented": INDENTED}


class ScopeView:
    """`acc` restricted to `scopes`, scope stripped off the core, collisions combined.

    **The trap this exists to close.** `from_accumulators` reads `acc.files` and `acc[key]`
    and builds `tree[core][layer][cell] = (n, mean)` *by assignment*. Stripping `@function`
    and `@method` off two keys makes them collide on exactly that triple, so handing them
    over unmerged would keep whichever arrived last, silently halve the set, and report a
    core count of 8 and a profile that looks entirely reasonable. They are combined here
    instead, by count-weighted mean — exact for means, since `(n1*m1 + n2*m2) / (n1+n2)` is
    what a single Welford over the union would have produced:

        >>> # one core's function and method cells become one indented cell
        >>> # n = n_f + n_m, mean = the count-weighted combination

    Only `|n` and `|mean` are exposed, which is all `from_accumulators` reads; the spread
    fields are the C yardstick's and are not written for the `hs` family.
    """

    def __init__(self, acc, scopes, stream: str = "hs"):
        want = tuple(f"@{s}" for s in scopes)
        merged: dict[str, tuple[int, np.ndarray]] = {}
        for k in acc.files:
            if not (k.startswith(f"{stream}|") and k.endswith("|n")):
                continue
            base = k[:-2]
            parts = base.split("|")
            if len(parts) != 5:
                raise ValueError(f"accumulator key {base!r} is not cell-keyed; re-record")
            core = parts[1]
            if "@" not in core or not core.endswith(want):
                continue
            parts[1] = core.rsplit("@", 1)[0]
            key = "|".join(parts)
            n = int(acc[base + "|n"][0])
            m = np.asarray(acc[base + "|mean"], dtype=np.float64)
            if key in merged:
                n0, m0 = merged[key]
                tot = n0 + n
                merged[key] = (tot, (n0 * m0 + n * m) / tot) if tot else (0, m0)
            else:
                merged[key] = (n, m)
        if not merged:
            raise ValueError(
                f"no {stream} accumulators for scopes {tuple(scopes)}. This reference was "
                f"recorded without the scope key, so the scope-restricted profile cannot be "
                f"computed from it and cannot be recovered later either -- re-record or skip.")
        self._merged = merged
        self.files = [k + suffix for k in merged for suffix in ("|n", "|mean")]

    def __getitem__(self, key: str):
        base, _, field = key.rpartition("|")
        n, mean = self._merged[base]
        return np.array([n]) if field == "n" else mean


def scoped(acc, ys, construct: str, kind: str, scopes) -> "object":
    """The profile of one set restricted to `scopes`."""
    from .profile import from_accumulators
    return from_accumulators(ScopeView(acc, scopes), ys, construct, kind, stream="hs")


def scope_varying(path=None) -> list[str]:
    """The 24 constructs whose construct-token piece differs between module and indented.

    Read from `SCOPE_SCHEDULE.json` rather than listed here: the schedule is derived from the
    battery and the tokeniser and carries the battery's own hash, and a list in a source file
    would go stale the next time the battery is rebuilt with nothing to say so.
    """
    import json
    import os
    import pathlib
    p = pathlib.Path(path or os.path.expanduser(os.environ.get(
        "PB_SCOPE_SCHEDULE",
        "$ATLAS_DATA/validation/battery/SCOPE_SCHEDULE.json")))
    return sorted(json.loads(p.read_text())["varies_by_scope"])
