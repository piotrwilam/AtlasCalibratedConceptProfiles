"""The twin ceiling for every construct at every reference, from the released carrier sets.

The ceiling is what the instrument can agree with itself on: twin 1's carrier set against
twin 2's at the SAME reference, as a share of K. Every turnover and overlap in either paper
is read against it, and until now it was stored only three ways, none of them per construct
across the record: per construct at the four per-prompt references
(`YARDSTICK_coarse.json :: rows.<ref>|<c>.alternatives.twenty.twin_ceiling`), as a median
over the 169 per reference (`B12_TWIN_TURNOVER.json`), and for `For` and `With` alone
(`CHECK6_TURNOVER.json`). Appendix A's artefact sentence names the ceilings beside the
carrier sets, so they are computed here into one file.

**It reads nothing but the release.** `validation/carriers/<ref>.npz` holds both twins' sets
for all 188 constructs in both spaces, so this is arithmetic on published bytes and runs
CPU-only in under a minute -- which is the point: the sentence can promise the ceilings
without promising the record.

The four references that YARDSTICK_coarse covers are recomputed and compared against its
stored values; a disagreement anywhere stops the write rather than publishing two numbers
for one quantity.

    python experiments/twin_ceilings.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from experiments.atlas import D, OUT, provenance

#: the two spaces the carrier sets are cut in, with their sizes
SPACES = {"product": 7373, "layer": 307}
#: the stored values this is checked against, where they exist
CHECK_AGAINST = "validation/paper1/YARDSTICK_coarse.json"
#: a recomputed ceiling must match the stored one to this many places
TOL = 1e-9


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """|A n B| / |A| -- the form the papers report, at equal set sizes."""
    return float(np.intersect1d(a, b, assume_unique=False).size / a.size)


def main() -> None:
    t0 = time.time()
    src = D / "validation/carriers"
    refs = sorted(p.stem for p in src.glob("*.npz"))
    if not refs:
        raise SystemExit(f"twin_ceilings: no carrier files under {src}")

    per_ref: dict[str, dict] = {}
    for ref in refs:
        with np.load(src / f"{ref}.npz") as z:
            keys = set(z.files)
            constructs = sorted({k.split("/", 1)[1].split("|")[0] for k in keys
                                 if k.startswith("twin1/")})
            rows = {}
            for c in constructs:
                row = {}
                for space, k in SPACES.items():
                    a, b = f"twin1/{c}|{space}", f"twin2/{c}|{space}"
                    if a not in keys or b not in keys:
                        continue
                    sa, sb = z[a], z[b]
                    if sa.size != k or sb.size != k:
                        raise SystemExit(f"twin_ceilings: {ref} {c} {space} has "
                                         f"{sa.size}/{sb.size} entries, not {k}")
                    row[space] = overlap(sa, sb)
                if row:
                    rows[c] = row
        per_ref[ref] = rows
        print(f"  {ref:<26} {len(rows):>3} constructs", flush=True)

    # the stored values, where they exist: four references x 169 constructs
    stored = json.loads((D / CHECK_AGAINST).read_text())["rows"]
    checked, bad = 0, []
    for key, v in stored.items():
        ref, c = key.split("|")
        want = ((v.get("alternatives") or {}).get("twenty") or {}).get("twin_ceiling")
        got = (per_ref.get(ref) or {}).get(c, {}).get("product")
        if want is None or got is None:
            continue
        checked += 1
        if abs(want - got) > TOL:
            bad.append((ref, c, want, got))
    if bad:
        for ref, c, want, got in bad[:10]:
            print(f"  DISAGREES {ref} {c}: stored {want:.6f}, recomputed {got:.6f}")
        raise SystemExit(f"twin_ceilings: {len(bad)} of {checked} disagree with "
                         f"{CHECK_AGAINST}; nothing written")
    print(f"  {checked} recomputed ceilings match {CHECK_AGAINST} exactly")

    def summary(space: str) -> dict:
        out = {}
        for ref, rows in per_ref.items():
            vals = sorted(r[space] for r in rows.values() if space in r)
            if not vals:
                continue
            q = lambda f: float(np.quantile(vals, f))
            lo = min(rows, key=lambda c: rows[c][space])
            out[ref] = {"n": len(vals), "median": q(0.5), "q1": q(0.25), "q3": q(0.75),
                        "min": vals[0], "min_at": lo, "max": vals[-1]}
        return out

    doc = {
        "what": "the twin ceiling of every construct at every reference: twin 1's carrier "
                "set against twin 2's at the same reference, as a share of K",
        "definition": {
            "ceiling": "|twin1 n twin2| / K, the overlap form the papers report",
            "spaces": {"product": "the top 7,373 of 24 x 6,144 by squared value, the "
                                  "paper's own 5% cut and the primary",
                       "layer": "the top 307 of 6,144 at the construct's verdict layer"},
            "convention": "plug-in: computed from the stored sets, not bootstrapped. The "
                          "core bootstrap's mean sits below it, and is reported where a "
                          "figure needs it (LICENCE_TABLE.json)",
            "read_from": "validation/carriers/<ref>.npz -- published bytes only, no record",
            "checked_against": CHECK_AGAINST + " :: rows.<ref>|<c>.alternatives.twenty."
                               "twin_ceiling, which covers the four per-prompt references",
        },
        "n_references": len(per_ref),
        "n_constructs": max(len(r) for r in per_ref.values()),
        "n_checked_against_stored": checked,
        "summary": {space: summary(space) for space in SPACES},
        "per_reference": per_ref,
        "minutes": (time.time() - t0) / 60,
        "provenance": provenance(["validation/carriers/<ref>.npz"]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "TWIN_CEILINGS.json").write_text(json.dumps(doc, indent=1, default=float))
    print(f"-> {OUT / 'TWIN_CEILINGS.json'}  "
          f"({len(per_ref)} references x {doc['n_constructs']} constructs, "
          f"{doc['minutes']:.2f} min)")


if __name__ == "__main__":
    main()
