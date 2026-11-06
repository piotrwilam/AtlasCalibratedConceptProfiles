"""The second generator menu — the alternative ν of validation check 4.

A different *draw* from the same menu is a twin, not an alternative: check 4 asks what
happens to a profile when the generator's declared choices change. So the alternative has to
differ in its declarations while staying comparable in diversity, or the check measures
vocabulary size rather than ν.

Three declarations are inverted, each of which was a real decision in the calibrated menu:

* **which end of the vocabulary.** The calibrated menu draws from the whole of each domain
  table weighted by damped frequency; the alternative draws from the tail half, uniformly.
  The tails are *subsets*, not disjoint sets, so the honest figure is the realised one:
  measured over 3,000 draws per slot, about **half** of the calibrated menu's draw mass
  falls on words the alternative also produces, and half on words it never does. The
  frequency weighting — itself a choice — is gone entirely.
* **the literal distribution.** Calibrated samples integers at corpus frequency, which is
  dominated by 0, 1 and 2. The alternative samples uniformly over the same support.
* **the padding sub-pool.** The second half of the pool rather than the whole of it.

Using `StubMenu` as the alternative was the obvious alternative and is the wrong one: its
five declared nouns per role would make the comparison a test of vocabulary size.
"""

from __future__ import annotations

import numpy as np

from .calibrated_menu import CalibratedMenu


class AlternativeMenu(CalibratedMenu):
    """Same tables, inverted declarations."""

    PROVISIONAL = False
    NU = "alternative"

    def __init__(self, path=None):
        super().__init__(path)
        # tail half of every naming table, uniform weights
        for slot, per_domain in self.tables.items():
            for dom, vals in per_domain.items():
                if len(vals) >= 8:
                    tail = vals[len(vals) // 2:]
                    per_domain[dom] = [(name, 1.0) for name, _ in tail]
                else:
                    per_domain[dom] = [(name, 1.0) for name, _ in vals]
            self.sources[slot] = {d: (v + " [tail, uniform]" if v == "corpus" else v)
                                  for d, v in self.sources[slot].items()}
        vals, _ = self._ints
        self._ints = (vals, np.ones(len(vals)))
        lens, _ = self._slen
        self._slen = (lens, np.ones(len(lens)))
        self.sources["int_literal"] = {d: "corpus support, uniform"
                                       for d in self.sources["int_literal"]}
        half = len(self.padding_pool) // 2
        self.padding_pool = self.padding_pool[half:] or self.padding_pool
        self.sources["padding"] = {d: f"corpus pool, second half ({len(self.padding_pool)})"
                                   for d in self.sources["padding"]}

    def float_literal(self, rng):
        """Uniform over a wider range than the calibrated menu's small factors."""
        return f"{int(rng.integers(0, 100))}.{int(rng.integers(0, 10))}"

    def provenance(self) -> dict:
        p = super().provenance()
        p["nu"] = self.NU
        p["inverted"] = ["vocabulary tail, uniform weights",
                         "uniform literals over the corpus support",
                         "second half of the padding pool",
                         "wider uniform floats"]
        return p


def alternative_menu(path=None) -> AlternativeMenu:
    return AlternativeMenu(path)
