"""The generator's menu: every lexical choice a core needs, injected.

Cores declare which menu slots they use; the menu supplies values. This split is
the point: at generation time the menu is a *stub* (below), and when the
calibration tables of handout S2 exist the same interface is backed by measured
corpus distributions without touching a single core.

Nothing here may consult the model. Data in the loop is fine; the measured model
in the loop is never (report S6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import keyword

import numpy as np

DOMAINS = ("finance", "biology", "gaming", "physics", "ecommerce")
SCOPES = ("module", "function", "method")


@dataclass(frozen=True)
class Cell:
    """One grid cell plus its draw index (handout S3: 5 x 3 x padding x 4)."""

    domain: str
    scope: str
    padding: int
    draw: int

    def key(self) -> str:
        return f"{self.domain}/{self.scope}/p{self.padding}/d{self.draw}"


class Menu(Protocol):
    """What a core may ask for. Implementations must be deterministic given rng."""

    def name(self, domain: str, role: str, rng: np.random.Generator) -> str: ...
    def verb(self, domain: str, rng: np.random.Generator) -> str: ...
    def field(self, domain: str, rng: np.random.Generator) -> str: ...
    def int_literal(self, rng: np.random.Generator) -> str: ...
    def float_literal(self, rng: np.random.Generator) -> str: ...
    def text(self, domain: str, rng: np.random.Generator) -> str: ...
    def padding(self, domain: str, n_lines: int, rng: np.random.Generator) -> list[str]: ...


_PLURAL = {
    "finance": ("ledger_entries", "transactions", "invoices", "positions", "accruals"),
    "biology": ("genome_reads", "samples", "sequences", "cultures", "assays"),
    "gaming": ("player_scores", "sessions", "inventories", "matches", "spawns"),
    "physics": ("measurements", "trajectories", "detectors", "spectra", "lattices"),
    "ecommerce": ("order_lines", "baskets", "shipments", "listings", "returns"),
}
_SINGULAR = {
    "finance": ("entry", "transaction", "invoice", "position", "accrual"),
    "biology": ("read", "sample", "sequence", "culture", "assay"),
    "gaming": ("score", "session", "item", "match", "spawn"),
    "physics": ("reading", "trajectory", "detector", "spectrum", "site"),
    "ecommerce": ("line", "basket", "shipment", "listing", "refund"),
}
_VERB = {
    "finance": ("reconcile", "settle", "post", "accrue", "audit"),
    "biology": ("align", "assemble", "normalise", "cluster", "anneal"),
    "gaming": ("respawn", "score", "matchmake", "render", "tally"),
    "physics": ("integrate", "calibrate", "propagate", "smooth", "bucket"),
    "ecommerce": ("fulfil", "price", "restock", "refund", "bundle"),
}
_FIELD = {
    "finance": ("amount", "currency", "settled", "book", "counterparty"),
    "biology": ("length", "quality", "strand", "locus", "coverage"),
    "gaming": ("points", "level", "streak", "region", "loadout"),
    "physics": ("energy", "sigma", "channel", "timestamp", "gain"),
    "ecommerce": ("price", "quantity", "sku", "carrier", "status"),
}
_TEXT = {
    "finance": ("trade booked", "ledger closed", "awaiting settlement"),
    "biology": ("run complete", "low coverage", "waiting for barcode"),
    "gaming": ("match started", "player idle", "loading assets"),
    "physics": ("detector armed", "drift corrected", "waiting for trigger"),
    "ecommerce": ("order placed", "stock low", "awaiting carrier pickup"),
}
_CLASSNAME = {
    "finance": "LedgerEngine", "biology": "GenomePipeline", "gaming": "MatchRunner",
    "physics": "DetectorStage", "ecommerce": "OrderRouter",
}


def _assert_no_reserved_words() -> None:
    """The declared words must not collide with a builtin or keyword.

    Checked at import, because the corpus filter only guards *drawn* names and the
    declared fallback slipped past it: `bin` shipped as a physics verb, and `bin` is a
    builtin in the inventory — a core writing `bin = 3` would both shadow it and corrupt
    `bin`'s own A and B sets.
    """
    import builtins as _b

    reserved = set(dir(_b)) | set(keyword.kwlist)
    for table in (_PLURAL, _SINGULAR, _VERB, _FIELD, _TEXT):
        for words in table.values():
            bad = sorted(set(words) & reserved)
            assert not bad, f"declared words collide with builtins: {bad}"
    assert not set(_CLASSNAME.values()) & reserved


_assert_no_reserved_words()


class StubMenu:
    """Placeholder menu. Replaced by calibration-backed values at handout S2.

    Marked PROVISIONAL so no released battery is ever generated from it by accident.
    """

    PROVISIONAL = True

    def _pick(self, table: dict, domain: str, rng: np.random.Generator) -> str:
        opts = table[domain]
        return str(opts[int(rng.integers(len(opts)))])

    def name(self, domain, role, rng):
        if role == "plural":
            return self._pick(_PLURAL, domain, rng)
        if role == "singular":
            return self._pick(_SINGULAR, domain, rng)
        if role == "class":
            return _CLASSNAME[domain]
        if role == "total":
            return f"{self._pick(_SINGULAR, domain, rng)}_total"
        raise KeyError(role)

    def verb(self, domain, rng):
        return self._pick(_VERB, domain, rng)

    def field(self, domain, rng):
        return self._pick(_FIELD, domain, rng)

    def int_literal(self, rng):
        return str(int(rng.integers(2, 64)))

    def float_literal(self, rng):
        return f"{float(rng.integers(1, 99)) / 10:.1f}"

    def text(self, domain, rng):
        return self._pick(_TEXT, domain, rng)

    def padding(self, domain, n_lines, rng):
        """Stub stand-in for the corpus padding pool (handout S2)."""
        out = []
        for _ in range(n_lines):
            shape = int(rng.integers(3))
            if shape == 0:
                out.append(f"{self.name(domain, 'plural', rng).upper()}_LIMIT = "
                           f"{self.int_literal(rng)}")
            elif shape == 1:
                out.append(f"import json")
            else:
                out.append(f"{self.name(domain, 'singular', rng)}_seen = set()")
        return out
