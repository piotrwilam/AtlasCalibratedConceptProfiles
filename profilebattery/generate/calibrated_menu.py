"""The calibrated menu — §2's tables in place of `StubMenu`.

The handout's rule is that nothing generated on the stub menu stamps the licence. This is
what replaces it, and it reports **per slot and per domain** whether the values came from
the corpus or from a declared fallback, because the corpus does not fill every slot
equally:

* **nouns** (`Name` ids) and **fields** (`Attribute.attr`) — corpus, all five domains;
* **verbs** (`FunctionDef` names) — corpus for finance and ecommerce only; the other three
  domains yield 0, 2 and 4 usable names, so they fall back;
* **class names** — corpus for finance only; the rest fall back;
* **integer literals** — corpus distribution, sampled by frequency;
* **floats** and **string contents** — *not* collected by the §2 walk. Floats are
  synthesised; strings are composed from domain nouns to a **corpus-drawn length**, so the
  length is calibrated and the content is not. Both are recorded as such.
* **padding** — real corpus blocks from the pool.

Two filters matter more than they look. A name that collides with a builtin would corrupt
that builtin's own A and B sets — `list = 5` in a core both shadows the builtin and makes
`list`'s prompts wrong — so keywords and builtins are excluded from every naming slot.
And sampling is damped by a square root of the count, or one head name would take half of
a domain's draws.
"""

from __future__ import annotations
from profilebattery.paths import data_root

import builtins as _builtins
import json
import keyword
import os
import pathlib

import numpy as np

from .menu import DOMAINS, StubMenu

CAL = (data_root() / "calibration")
#: below this many usable values a slot falls back to the declared words
MIN_USABLE = 20
RESERVED = frozenset(dir(_builtins)) | frozenset(keyword.kwlist) | {"self", "cls"}


def default_menu(path: pathlib.Path | None = None):
    """The calibrated menu where the tables exist, the stub otherwise.

    Never a silent mix: the stub reports `PROVISIONAL = True`, so a caller can always tell
    whether what it generated may stamp the licence.
    """
    d = pathlib.Path(path or CAL)
    if (d / "identifiers.json").exists() and (d / "vocabulary.json").exists():
        return CalibratedMenu(d)
    return StubMenu()


def _clean(names: dict, plural: bool | None = None) -> list[tuple[str, float]]:
    """(name, weight) pairs, filtered and damped."""
    out = []
    for name, info in names.items():
        if not name.isidentifier() or name in RESERVED or name.startswith("_"):
            continue
        if len(name) < 3 or len(name) > 28:
            continue
        looks_plural = name.endswith("s") and not name.endswith(("ss", "is", "us", "as"))
        if plural is True and not looks_plural:
            continue
        if plural is False and looks_plural:
            continue
        n = info["count"] if isinstance(info, dict) else int(info)
        out.append((name, float(n) ** 0.5))
    return out


class CalibratedMenu(StubMenu):
    """Menu backed by the §2 tables, falling back to StubMenu's declared words."""

    PROVISIONAL = False

    def __init__(self, path: pathlib.Path | None = None):
        d = pathlib.Path(path or CAL)
        load = lambda n: json.loads((d / f"{n}.json").read_text())
        ids, voc = load("identifiers"), load("vocabulary")["vocabulary"]
        lit, self.padding_pool = load("literals"), load("padding_pool")
        self.manifest = load("MANIFEST")

        self.tables: dict[str, dict[str, list]] = {}
        self.sources: dict[str, dict[str, str]] = {}
        for slot, raw, plural in (("plural", ids, True), ("singular", ids, False),
                                  ("field", voc["attribute"], None),
                                  ("verb", voc["verb"], None),
                                  ("class", voc["classname"], None)):
            self.tables[slot], self.sources[slot] = {}, {}
            for dom in DOMAINS:
                entry = raw.get(dom, {})
                table = entry.get("detail", entry) if "source" in entry else entry
                vals = _clean(table, plural)
                if len(vals) >= MIN_USABLE:
                    self.tables[slot][dom] = vals
                    self.sources[slot][dom] = "corpus"
                else:
                    self.tables[slot][dom] = []
                    self.sources[slot][dom] = f"declared ({len(vals)} usable)"

        # Literals use the raw counts, not the damped ones. Damping exists to stop one
        # head *name* taking half a domain's draws; applied to literals it flattened a
        # distribution that genuinely is head-heavy — the corpus is dominated by 0, 1 and 2
        # — and only 38% of draws came out at or below 10.
        ints = lit["int_values"]
        self._ints = ([int(k) for k in ints], np.array(list(ints.values()), float))
        lens = {int(k): v for k, v in lit["str_lengths"].items() if 4 <= int(k) <= 60}
        self._slen = (list(lens), np.array(list(lens.values()), float))
        self.sources["int_literal"] = {d: "corpus" for d in DOMAINS}
        self.sources["float_literal"] = {d: "synthesised (not collected)" for d in DOMAINS}
        self.sources["text"] = {d: "composed at a corpus-drawn length" for d in DOMAINS}
        self.sources["padding"] = {d: f"corpus pool ({len(self.padding_pool)} blocks)"
                                   for d in DOMAINS}

    # --- Menu protocol ----------------------------------------------------------

    def _draw(self, slot: str, domain: str, rng) -> str | None:
        vals = self.tables[slot].get(domain) or []
        if not vals:
            return None
        names = [v[0] for v in vals]
        w = np.array([v[1] for v in vals])
        return str(names[int(rng.choice(len(names), p=w / w.sum()))])

    def name(self, domain, role, rng):
        if role == "total":
            base = self._draw("singular", domain, rng)
            return f"{base}_total" if base else super().name(domain, role, rng)
        got = self._draw("plural" if role == "plural" else
                         "class" if role == "class" else "singular", domain, rng)
        return got if got else super().name(domain, role, rng)

    def verb(self, domain, rng):
        got = self._draw("verb", domain, rng)
        return got if got else super().verb(domain, rng)

    def field(self, domain, rng):
        got = self._draw("field", domain, rng)
        return got if got else super().field(domain, rng)

    def int_literal(self, rng):
        vals, w = self._ints
        return str(vals[int(rng.choice(len(vals), p=w / w.sum()))])

    def float_literal(self, rng):
        """Synthesised: the §2 walk counted float *types* but not float *values*.

        Kept small deliberately — chaining off `int_literal` produced 250.5 and 500.4,
        and corpus floats are overwhelmingly small scaling factors.
        """
        return f"{int(rng.integers(0, 20)) + int(rng.integers(1, 10)) / 10:.1f}"

    def text(self, domain, rng):
        """Domain nouns composed to a corpus-drawn string length."""
        lens, w = self._slen
        target = lens[int(rng.choice(len(lens), p=w / w.sum()))]
        words, out = [], 0
        while out < target:
            word = self._draw("singular", domain, rng) or "value"
            words.append(word)
            out += len(word) + 1
        return " ".join(words)[:max(target, 4)]

    def padding(self, domain, n_lines, rng):
        """Whole blocks only — never a slice.

        Each pooled block was validated as parseable on its own, but slicing one to fit a
        line budget can cut a balanced triple-quoted string in half, and the fragment then
        swallows the core. That produced every one of the 39 invalid prompts in the first
        calibrated run. Blocks are taken whole and the budget may be overshot, which
        matters not at all: padding length is a grid level, not an exact count.
        """
        out: list[str] = []
        if not self.padding_pool:
            return super().padding(domain, n_lines, rng)
        for _ in range(12):
            if len(out) >= n_lines:
                break
            block = self.padding_pool[int(rng.integers(len(self.padding_pool)))]
            if len(block) <= n_lines * 2:
                out.extend(block)
        return out or super().padding(domain, n_lines, rng)

    def provenance(self) -> dict:
        return {"provisional": self.PROVISIONAL, "sources": self.sources,
                "calibration_manifest": self.manifest}
