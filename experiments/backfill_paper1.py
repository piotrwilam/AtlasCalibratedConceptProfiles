r"""Store the one summary of paper 1's that no artefact held: B's grid, by scope and padding.

\S A.2 item 4 says A's grid "is balanced by construction (240 prompts per scope, 240 per
padding level) and B's is not (219/234/267 by scope, 175/262/283 by padding for `for`)". Those
six counts were in no artefact: `B_SUBTYPES.json` described what B's subtypes are, not how B's
prompts fall across the cells. They are recomputed here from the battery files, CPU-only, for
every construct, with `for`'s checked against the figures the paper prints.

**The other numbers CC had called unstored turned out to be stored**, in artefacts CC's index
had skipped (`_abl_dl/validation/ablation/FULL_*.json`) or under keys CC had not read
(`LOCATE_BY_CLASS.json :: by_contrast_class`, `LENGTH_BAND_HELD.json :: constructs.<c>`).
Those are corrected in the manifest rather than written again here. Two groups are left
deliberately unstored and say so in the manifest: Appendix A.7's wall-clock hours, which
cannot be recomputed without re-running the recordings on a GPU, and Appendix D's pre-fix
comparator, which belongs to a superseded run.

    python experiments/backfill_paper1.py
"""
from __future__ import annotations

import collections
import glob
import gzip
import json
import time

from experiments.atlas import D, provenance

BAT = D / "phase_battery"
OUT = D / "validation/paper1"
# what section A.2 prints, for `for`
EXPECT = {"B by scope": {"module": 219, "function": 234, "method": 267},
          "B by padding": {0: 175, 1: 262, 2: 283},
          "A by scope": {"module": 240, "function": 240, "method": 240},
          "A by padding": {0: 240, 1: 240, 2: 240}}


def read(path: str) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(l) for l in fh]


def grid(rows: list[dict]) -> dict:
    out = {}
    for field, key in (("scope", "by_scope"), ("padding", "by_padding")):
        if rows and field in rows[0]:
            c = collections.Counter(r[field] for r in rows)
            out[key] = {str(k): v for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
            out[key + "_balanced"] = len(set(c.values())) == 1
    return out


def main() -> None:
    t0 = time.time()
    constructs = sorted({p.split("/")[-1].rsplit("_", 1)[0]
                         for p in glob.glob(str(BAT / "twin1" / "*_A.jsonl.gz"))})
    per = {}
    for c in constructs:
        row = {}
        for s in ("A", "B"):
            f = BAT / "twin1" / f"{c}_{s}.jsonl.gz"
            if f.exists():
                row[s] = grid(read(str(f)))
        if row:
            per[c] = row

    # the check: `for`'s six counts must be the ones section A.2 prints
    f = per.get("For", {})
    got = {"B by scope": {k: v for k, v in f.get("B", {}).get("by_scope", {}).items()},
           "B by padding": {k: v for k, v in f.get("B", {}).get("by_padding", {}).items()},
           "A by scope": {k: v for k, v in f.get("A", {}).get("by_scope", {}).items()},
           "A by padding": {k: v for k, v in f.get("A", {}).get("by_padding", {}).items()}}
    for name, want in EXPECT.items():
        have = got.get(name, {})
        for k, v in want.items():
            if have.get(str(k)) != v:
                raise SystemExit(
                    f"REFUSED: {name}, {k} recomputes to {have.get(str(k))!r}, but the paper "
                    f"prints {v!r}. Nothing written.")

    n_unbal = sum(1 for v in per.values()
                  if v.get("B", {}).get("by_scope_balanced") is False)
    doc = {"what": "how each set's prompts fall across the grid, by scope and by padding level",
           "why": "section A.2 item 4 quotes A's grid as balanced by construction and B's as "
                  "not; the counts themselves were in no artefact before this",
           "definition": {
               "counts": "twin1 only, which is the twin the section's figures are taken from",
               "balanced": "every cell of that axis holds the same number of prompts"},
           "checked_against_the_paper": {k: EXPECT[k] for k in EXPECT},
           "n_constructs": len(per),
           "n_constructs_with_unbalanced_B_scope": n_unbal,
           "for_example": got,
           "per_construct": per,
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance(["phase_battery/twin1/*_{A,B}.jsonl.gz"])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SET_GRIDS.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"  {len(per)} constructs; B's scope grid is unbalanced for {n_unbal}")
    print(f"  `for`: B by scope {got['B by scope']}, by padding {got['B by padding']}")
    print(f"  all four of section A.2's printed grids reproduce")
    print(f"-> {OUT / 'SET_GRIDS.json'}  ({doc['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
