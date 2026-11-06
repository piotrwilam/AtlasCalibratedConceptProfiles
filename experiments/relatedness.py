r"""The three relatedness measures for paper 2's test 6, fixed before any reading.

`PAPER-3-Options` §2.11 (g): relatedness R(j, k) between constructs, "three measures fixed
before any reading and reported separately: same grammar family (the inventory's categories);
shared tokens at the read site or in the cores; co-occurrence rate in the reference corpus (held
in nu as companion rates)".

They are defined here, written to `validation/paper2/RELATEDNESS.json` with their coverage, and
not touched again. Test 6 imports them. The point of fixing them first is that each has a free
choice inside it -- which family field, which token set, what counts as co-occurring -- and a
choice made after seeing which constructs rise is not a measure.

  **family**      the inventory's `a_family`, the field that names a construct's grammatical
                  family (compound, comprehensions, expressions, simple, builtin). Related when
                  equal and not `builtin`, which is a catch-all of 143 names rather than a
                  family.
  **tokens**      the inventory's `tokens` list, which is the construct's surface tokens at the
                  read site and in its cores. Related when the two lists intersect.
  **co-occurrence** both constructs in one **function** of the calibration corpus, observed
                  against what their marginal shares would give independently; related when the
                  ratio exceeds 2, with the graded ratio reported beside. Built by
                  `experiments/cooccurrence.py`. Replaces nu's companion rates (21 Oct), which
                  covered 90 of P1's 14,196 pairs.

    python experiments/relatedness.py
"""
from __future__ import annotations

import json

import numpy as np

from experiments.atlas import D, provenance

OUT2 = D / "validation/paper2"
CATCH_ALL = "builtin"
#: a co-occurrence ratio resting on fewer functions than this is not evidence either way
MIN_COOC_SUPPORT = 20


def inventory() -> dict:
    return json.loads((D / "inventory/inventory_v2.json").read_text())["constructs"]


def companions() -> dict:
    c = json.loads((D / "calibration/companions.json").read_text())
    return c.get("rates", c)


class Relatedness:
    """R(j, k) under three measures, each a bool or None where the measure has no data."""

    def __init__(self):
        from profilebattery.profiles.names import contract_name
        self.inv = inventory()
        self.comp = companions()
        self._cn = contract_name
        import json as _j
        p = OUT2 / "COOCCURRENCE.json"
        self._cooc = _j.loads(p.read_text())["rows"] if p.exists() else {}
        self.median_rate = None

    def _e(self, c):
        return self.inv.get(self._cn(c)) or self.inv.get(c) or {}

    def kind(self, c):
        """A construct's family: `a_family` for grammar elements, the builtin's own kind for
        builtins. The inventory's `a_family` puts all 143 builtins in one bucket, which would
        make two arbitrary exception classes 'related'; the builtins carry a finer field,
        `builtin_kind`, and the warning classes are separable from the other exceptions by
        name. Amendment of 21 Oct."""
        e = self._e(c)
        if e.get("member_kind") == "ast-node":
            f = e.get("a_family")
            return None if f in (None, CATCH_ALL) else f"grammar:{f}"
        bk = e.get("builtin_kind")
        if bk is None:
            return None
        if bk == "exception":
            return ("builtin:warning class" if self._cn(c).endswith("Warning")
                    else "builtin:exception class")
        return f"builtin:{bk}"

    def family(self, j, k):
        a, b = self.kind(j), self.kind(k)
        if a is None or b is None:
            return None
        return a == b

    def tokens(self, j, k):
        a = set(self._e(j).get("tokens") or ([self._e(j).get("token")] if self._e(j).get("token") else []))
        b = set(self._e(k).get("tokens") or ([self._e(k).get("token")] if self._e(k).get("token") else []))
        if not a or not b:
            return None
        return bool(a & b)

    def tokens_at_read_site(self, j, k):
        """Shared token **at the read site**: the construct's own surface, not its surroundings."""
        a, b = self._e(j).get("token"), self._e(k).get("token")
        return None if a is None or b is None else bool(a == b)

    def tokens_in_cores_only(self, j, k):
        """Shared token only **in the cores** -- the surroundings, not the read site."""
        t = self.tokens(j, k)
        if t is None:
            return None
        if not t:
            return False
        rs = self.tokens_at_read_site(j, k)
        return bool(rs is False)

    def cooccurrence(self, j, k):
        """Co-occurrence within one function, observed against independence; related above 2.

        Replaces nu's companion rates (21 Oct), which covered 90 of P1's 14,196 pairs. Built by
        `experiments/cooccurrence.py` from the calibration corpus. `graded` returns the ratio.
        """
        e = self._cooc.get(f"{j}|{k}") or self._cooc.get(f"{k}|{j}")
        if e is None:
            return None
        if e.get("functions_with_both", 0) < MIN_COOC_SUPPORT:
            return None                       # not measured, not 'unrelated'
        return bool(e["ratio"] > 2.0)

    def cooccurrence_ratio(self, j, k):
        e = self._cooc.get(f"{j}|{k}") or self._cooc.get(f"{k}|{j}")
        return None if e is None else float(e["ratio"])

    def all(self, j, k) -> dict:
        return {"family": self.family(j, k), "tokens": self.tokens(j, k),
                "cooccurrence": self.cooccurrence(j, k)}


def main() -> None:
    from experiments.atlas import p1_set
    R = Relatedness()
    names = sorted(p1_set())
    cov, base = {}, {}
    for m in ("family", "tokens", "cooccurrence"):
        vals = [getattr(R, m)(j, k) for i, j in enumerate(names) for k in names[i + 1:]]
        have = [v for v in vals if v is not None]
        cov[m] = {"pairs_total": len(vals), "pairs_with_data": len(have),
                  "coverage": len(have) / len(vals) if vals else None}
        base[m] = float(np.mean(have)) if have else None
    OUT2.mkdir(parents=True, exist_ok=True)
    doc = {"what": "the three relatedness measures for paper 2's test 6, fixed before any "
                   "reading (PAPER-3-Options 2.11 g)",
           "measures": {
               "family": "the inventory's `a_family`; related when equal and not the `builtin` "
                         "catch-all of 143 names",
               "tokens": "the inventory's `tokens` list -- the surface tokens at the read site "
                         "and in the cores; related when the two lists intersect",
               "cooccurrence": "both constructs in one function of the calibration corpus "
                               "(shards 1, 18, 36), observed against independence; related when "
                               f"the ratio exceeds 2 AND at least {MIN_COOC_SUPPORT} functions "
                               "hold both. Below that support the pair is not measured, not "
                               "unrelated, and is counted separately."},
           "rules_fixed_21_Oct": {
               "reading_rule": {
                   "statistic": "per measure, the ratio of the observed rate of related rises "
                                "to the chance rate, with a bootstrap interval and the count of "
                                "related pairs beside it",
                   "bootstrap_unit": "the construct. The programme's intervals resample cores, "
                                     "but a departure's relatedness is fixed by the pair "
                                     "(construct left, construct risen), so every departure of "
                                     "one construct shares its relatedness profile and the "
                                     "construct is the unit that is exchangeable here. Named "
                                     "rather than substituted silently.",
                   "branching": "ratio > 1 with the interval excluding 1 on tokens AND on "
                                "family, and co-occurrence not below 1",
                   "displacement": "rises above chance in any construct, not concentrated in "
                                   "related ones",
                   "drift": "rises at chance",
                   "no_measure_is_promoted_beyond_this": True,
                   "pattern_outside_the_rule": "a lift clear of 1 on tokens but not on family "
                                               "is none of the three verdicts and is reported "
                                               "as outside them, not as one of them"},
               "diagnostics_fixed_21_Oct_before_pairs_2_and_3": {
                   "displacement_number": "the share of departures followed by a rise in ANY "
                                          "other construct, against the matched chance "
                                          "construct's share, with a construct-level interval. "
                                          "Displacement predicts this above chance while the "
                                          "relatedness lifts sit at 1.",
                   "tokens_split_by_where_the_token_sits": "the tokens lift computed twice -- "
                                                           "for pairs sharing the token AT THE "
                                                           "READ SITE, and for pairs sharing "
                                                           "one only IN THE CORES -- each with "
                                                           "its related-pair count. The first "
                                                           "is the construct's own surface, the "
                                                           "second its surroundings.",
                   "cross_twin": "the tokens and family lifts with the departure taken from "
                                 "twin 1 and the rise from twin 2. A lift that survives the "
                                 "twins is not an artefact of one battery draw."},
               "shards": "all three calibration shards (1, 18, 36), fixed before coverage was "
                         "seen",
               "min_support": MIN_COOC_SUPPORT,
               "related_requires": "ratio > 2 and support >= 20"},
           "median_companion_rate": R.median_rate,
           "population": len(names), "pairs": len(names) * (len(names) - 1) // 2,
           "coverage": cov,
           "base_rate_of_related_pairs": base,
           "note": "co-occurrence covers only the constructs nu's companion table holds; pairs "
                   "outside it are marked absent and never imputed",
           "provenance": provenance(["inventory/inventory_v2.json",
                                     "calibration/companions.json"])}
    (OUT2 / "RELATEDNESS.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"{len(names)} constructs, {doc['pairs']:,} pairs")
    for m in cov:
        print(f"  {m:13s} coverage {cov[m]['coverage']:6.1%}  "
              f"({cov[m]['pairs_with_data']:,} pairs)   base rate of 'related' "
              f"{base[m]:.4f}" if base[m] is not None else f"  {m}: no data")
    print(f"-> {OUT2 / 'RELATEDNESS.json'}")


if __name__ == "__main__":
    main()
