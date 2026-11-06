r"""CORE_COUNTS.md -- the size of each construct's core family, to test "at least 24".

A *core* is one syntactic realisation of the construct; the battery draws **eight** of them
per construct and the bootstrap resamples them. The draft says the eight are "drawn from a
family of at least 24". This measures the family.

**What the family is.** `generate/core.py:spread()` chooses n distinct combinations of the
construct's axes, and refuses when `n > prod(len(levels[axis]))`. That product IS the family:
the number of distinct cores the generator could build for this construct. It is read here by
intercepting `spread` and recording the levels it is handed, so the number comes from the
generator rather than from a reimplementation of it.

A construct whose family splits by subtype calls `spread` more than once; each call is a
separate pool and the per-call sizes are reported, with their **minimum** as the construct's
figure -- the conservative reading, since a draw must come from one of them.

Note what this does NOT test. The generator's own contract (`core.py`) is "**>= 5** distinct
cores per construct x applicable subtype is the licence floor, build 8". That floor and the
draft's "at least 24" are different claims about different things, and only the second is
checked here.

    python experiments/core_counts.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from experiments.atlas import D, provenance

OUT = D / "validation/paper1"
CLAIM = 24


def _family_modules() -> list:
    """Every module under generate/ that holds its own reference to `spread`."""
    import importlib
    import pkgutil

    import profilebattery.generate as g
    out = []
    for mi in pkgutil.iter_modules(g.__path__):
        try:
            out.append(importlib.import_module(f"profilebattery.generate.{mi.name}"))
        except Exception:                                              # noqa: BLE001
            continue
    return out


def main() -> None:
    t0 = time.time()
    from profilebattery.generate import core as core_mod
    from profilebattery.generate.families import build_a_cores, family_of

    inv = json.loads((D / "inventory/inventory_v2.json").read_text())
    cs = inv["constructs"]
    items = list(cs.items()) if isinstance(cs, dict) else [(c.get("token"), c) for c in cs]
    built = [(k, c) for k, c in items if c.get("a_family")]

    seen: list[int] = []
    real_spread = core_mod.spread

    def spy(levels, n, rng, weights=None, invalid=None):
        seen.append(int(np.prod([len(v) for v in levels.values()])) if levels else 0)
        return real_spread(levels, n, rng, weights, invalid)

    rows, failed = [], []
    for key, c in built:
        seen.clear()
        core_mod.spread = spy
        saved = [(m, m.spread) for m in _family_modules() if hasattr(m, "spread")]
        for m, _ in saved:
            m.spread = spy
        try:
            build_a_cores(key, 8, np.random.default_rng(20261102))
            pools = sorted(x for x in seen if x > 0)
            rows.append({"construct": key, "family_kind": family_of(key),
                         "n_pools": len(pools), "pool_sizes": pools,
                         "family_size_min": (min(pools) if pools else None)})
        except Exception as exc:                                       # noqa: BLE001
            failed.append({"construct": key, "error": f"{type(exc).__name__}: {exc}"[:120]})
        finally:
            core_mod.spread = real_spread
            for m, old in saved:
                if old is not None:
                    m.spread = old

    sized = [r for r in rows if r["family_size_min"] is not None]
    mins = [r["family_size_min"] for r in sized]
    below = [r for r in sized if r["family_size_min"] < CLAIM]
    doc = {"what": "the size of each construct's core family, against the draft's "
                   f"'at least {CLAIM}'",
           "definition": {
               "core": "one syntactic realisation of the construct; eight are built and the "
                       "bootstrap resamples them",
               "family_size": "prod(len(levels[axis])) inside generate/core.py:spread(), the "
                              "number of distinct cores the generator could build; read by "
                              "intercepting spread, not by reimplementing it",
               "per_construct_figure": "the MINIMUM over that construct's pools, since a draw "
                                       "comes from one pool",
               "not_tested": "the generator's own licence floor is '>= 5 distinct cores per "
                             "construct x applicable subtype'; that is a different claim"},
           "n_constructs_built": len(built), "n_measured": len(sized),
           "n_not_measured": len(failed),
           "claim": CLAIM,
           "claim_holds": bool(sized) and min(mins) >= CLAIM,
           "min_family_size": (min(mins) if mins else None),
           "max_family_size": (max(mins) if mins else None),
           "median_family_size": (float(np.median(mins)) if mins else None),
           "n_below_claim": len(below),
           "constructs_below_claim": [r["construct"] for r in below],
           "per_construct": rows, "not_measured": failed,
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["inventory/inventory_v2.json",
                                     "profilebattery/generate/core.py"])}
    (OUT / "CORE_COUNTS.json").write_text(json.dumps(doc, indent=1, default=float))

    md = [f"# Core family size per construct",
          "",
          f"A *core* is one syntactic realisation of a construct. The battery builds **eight** "
          f"per construct and the bootstrap resamples them. The draft says the eight are "
          f"\"drawn from a family of at least {CLAIM}\"; this table is that family.",
          "",
          f"The family size is `prod(len(levels[axis]))` inside "
          f"`generate/core.py:spread()` — the number of distinct cores the generator could "
          f"build — read by intercepting `spread`, so the figure comes from the generator "
          f"itself. Where a construct's cores split by subtype, `spread` is called once per "
          f"pool; the **minimum** over pools is the construct's figure, since a draw comes "
          f"from one pool.",
          "",
          f"**Measured for {len(sized)} of the {len(built)} built constructs.** "
          + (f"Minimum **{min(mins)}**, median {np.median(mins):.0f}, maximum {max(mins)}. "
             if mins else "")
          + (f"**The claim holds: every measured family has at least {CLAIM} members.**"
             if doc["claim_holds"] else
             f"**The claim does NOT hold: {len(below)} constructs have a family smaller "
             f"than {CLAIM}.**"),
          ""]
    if failed:
        md += [f"{len(failed)} constructs could not be measured and are listed at the end.", ""]
    md += ["| construct | family kind | pools | family size (min) |",
           "|---|---|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["family_size_min"] is None,
                                         r["family_size_min"] or 0, r["construct"])):
        fs = r["family_size_min"]
        md.append(f"| `{r['construct']}` | {r['family_kind']} | {r['n_pools']} | "
                  + (f"**{fs}**" if fs is not None and fs < CLAIM else f"{fs if fs else '—'}")
                  + " |")
    if failed:
        md += ["", "## Not measured", "", "| construct | reason |", "|---|---|"]
        md += [f"| `{f['construct']}` | {f['error']} |" for f in failed]
    (OUT / "CORE_COUNTS.md").write_text("\n".join(md) + "\n")

    print(f"  measured {len(sized)}/{len(built)} built constructs")
    if mins:
        print(f"  family size: min {min(mins)}, median {np.median(mins):.0f}, max {max(mins)}")
        print(f"  the draft's 'at least {CLAIM}': "
              f"{'HOLDS' if doc['claim_holds'] else f'FAILS for {len(below)} constructs'}")
        if below:
            print("   below: " + ", ".join(f"{r['construct']}({r['family_size_min']})"
                                           for r in below[:12]))
    if failed:
        print(f"  not measured: {len(failed)} -- {failed[0]['construct']}: "
              f"{failed[0]['error'][:60]}")
    print(f"-> {OUT / 'CORE_COUNTS.md'}, CORE_COUNTS.json")


if __name__ == "__main__":
    main()
