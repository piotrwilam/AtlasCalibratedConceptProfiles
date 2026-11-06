"""Deep/shallow tier assignment: the union of four declared sets.

The deep tier is where phase 1 and phase 2 make their claims, so it is a union of
the sets those claims need, not a frequency cut. Frequency fills the rest, up to
the knee of the cumulative-coverage curve. The criterion and the outcome are
different numbers and both are recorded: the frequency criterion is the **top 40**
(the knee, 97.5% of mass; past it each member adds under 0.12%), and the realised
union is the **outcome**.

The four input sets:

1. `bridge` — the v1 concepts carrying a published per-concept number, so v2 can be
   read against v1 at the last-token site in the m_k basis. Two disjoint groups in
   the v1 repo qualify and both are taken (see BRIDGE below); the programme
   documents call this "the bridge-20", but no enumeration of twenty was ever
   written down, and the grounded members come to 22.
2. `claim` — the composite/atomic class: v1 compositional level 1 or 2. This is the
   class the no-deeper-no-later claim is about, so it is in whatever its frequency.
3. `polysemous` — constructs sharing a surface token with another construct, plus
   the symbol other-role table. These are what phase 2 dates.
4. `top_n` — the top N by corpus rate over the whole inventory, AST constructs and
   builtin names together, N = FREQUENCY_N.

Nothing here is a frequency threshold in disguise: 36 of the union's members sit
outside the top 40 and are there because a claim needs them.
"""
import csv
import json
import pathlib

#: the frequency criterion — the knee of the cumulative-coverage curve
FREQUENCY_N = 40

#: leaf nodes with no surface of their own; they head the rate table and would
#: otherwise consume the frequency slots
RATE_LEAVES = frozenset({"Name", "Constant"})

#: v1 concepts with a published per-concept number. Two groups, both grounded in
#: the No-Deeper-No-Later repository and neither a superset of the other:
#:   - `data/probe_onset.json` — probe accuracy and circuit date per concept
#:   - the concepts named by `analysis/t1_riders.py` (COMPREH, plus the loop and
#:     Try constructs its robustness cuts turn on) and by
#:     `analysis/isolated_sentinel.py` (the isolated single-token statements)
#: They overlap only in ListComp and GeneratorExp.
BRIDGE_PROBE = frozenset({
    "AnnAssign", "Assign", "Call", "Exception", "GeneratorExp", "ListComp",
    "RuntimeError", "Set", "Slice", "StopIteration", "abs", "next",
})
BRIDGE_NAMED = frozenset({
    "For", "Try", "ListComp", "SetComp", "DictComp", "GeneratorExp",
    "Assert", "Break", "Continue", "Import", "ImportFrom", "Pass",
})
BRIDGE = BRIDGE_PROBE | BRIDGE_NAMED


def rates(calibration):
    """Corpus rate per inventory member, AST constructs and builtin names together."""
    d = pathlib.Path(calibration)
    ast_r = json.loads((d / "construct_rates.json").read_text())
    bi_r = json.loads((d / "builtin_rates.json").read_text())["rates"]
    out = {c: v["per_file"] for c, v in ast_r.items() if c not in RATE_LEAVES}
    out.update({c: v["per_file"] for c, v in bi_r.items()})
    return out


def top_n(rate, n=FREQUENCY_N):
    """The n most frequent members. Ties broken by name, so the set is stable."""
    return frozenset(c for c, _ in sorted(rate.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def claim_class(compositional_level_csv):
    """v1 compositional level 1 (composed) or 2 (higher-order) — the claim's class."""
    rows = csv.DictReader(open(compositional_level_csv))
    return frozenset(r["concept"] for r in rows
                     if r["language"] == "python" and r["compositional_level"] in ("1", "2"))


def polysemous(sharers, other_role):
    """Constructs sharing a surface token, plus the symbol other-role table."""
    return frozenset(c for group in sharers.values() for c in group) | frozenset(other_role)


def assign(rate, claim, poly, bridge=BRIDGE, n=FREQUENCY_N):
    """Return (deep, sets) — the union and the four inputs, for PROVENANCE."""
    top = top_n(rate, n)
    sets = {"bridge": bridge, "claim": claim, "polysemous": poly, "top_n": top}
    deep = frozenset().union(*sets.values())
    return deep, sets
