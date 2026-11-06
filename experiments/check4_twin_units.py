r"""Check 4 (re-weighting) per construct, in TWIN UNITS -- the column B.1 is missing.

B.1's check-4 column is a bare cosine (median 0.91). Section 4.1 reports the same check as a
median 2.34 twin units. They are not two views of one number: a cosine has no noise reference,
and 0.91 reads as "barely moved" while 2.34 twin units reads as "moved by more than twice what
redrawing the battery does". The check exists to ask whether the profile depends on the
generator's mixture, so the noise-referenced figure is the one that answers it, and B.1 should
carry it per construct.

**How the twin unit is taken.** `LICENCE_TABLE.json` stores, per construct, check 4's cosine
against the base profile and check 1's `median_distance` -- the distance between the two twins,
which is the instrument's own noise. The re-weighting distance is the chord distance implied by
the cosine, `sqrt(2(1 - cos))`, and the reported figure is that divided by the construct's own
twin distance. The identity is checked here on check 1 itself, where both the cosine and the
distance are stored: the chord distance recomputed from check 1's cosine must reproduce check
1's stored distance.

**The disattenuated value must not be used.** It is clipped at 1.0 for 155 of 188 constructs,
so its median of 1.000 is a ceiling, not a measurement.

    python experiments/check4_twin_units.py
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from experiments.atlas import D, provenance

SRC = D / "validation/paper1/LICENCE_TABLE.json"
OUT = D / "validation/paper1"


def chord(cos: float) -> float:
    """Distance between two unit vectors with this cosine."""
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - cos))))


def main() -> None:
    t0 = time.time()
    src = json.loads(SRC.read_text())
    cs = src["constructs"]
    items = list(cs.items()) if isinstance(cs, dict) else [(r["construct"], r) for r in cs]

    # the identity, on check 1 where both forms are stored
    resid = []
    for _, r in items:
        t = r.get("check1_twin_unit") or {}
        if t.get("median_cosine") is not None and t.get("median_distance"):
            resid.append(abs(chord(t["median_cosine"]) - t["median_distance"]))
    rel = []
    for _, r in items:
        t = r.get("check1_twin_unit") or {}
        if t.get("median_cosine") is not None and t.get("median_distance"):
            rel.append(abs(chord(t["median_cosine"]) - t["median_distance"])
                       / t["median_distance"])
    rel = np.array(rel)
    if rel.size and np.median(rel) > 0.05:
        raise SystemExit(
            f"REFUSED: recomputing check 1's distance from its cosine is off by a median "
            f"{np.median(rel):.1%}, too much to carry the same conversion to check 4.")

    rows, ratios = [], []
    for name, r in items:
        a = (r.get("check4_nu") or [None])[0]
        t = r.get("check1_twin_unit") or {}
        if not a or a.get("cosine") is None or not t.get("median_distance"):
            continue
        d4, dt = chord(a["cosine"]), t["median_distance"]
        ratios.append(d4 / dt)
        rows.append({"construct": name, "in_p1": r.get("in_p1"),
                     "check4_cosine": a["cosine"],
                     "check4_distance": d4,
                     "twin_distance": dt,
                     "check4_twin_units": d4 / dt,
                     "within_one_twin_unit": bool(d4 / dt <= 1.0),
                     "check4_disattenuated_NOT_REPORTED": a.get("disattenuated"),
                     "arm": a.get("arm")})
    ratios = np.array(ratios)
    dis = np.array([r["check4_disattenuated_NOT_REPORTED"] for r in rows
                    if r["check4_disattenuated_NOT_REPORTED"] is not None])

    doc = {
        "what": "check 4 (re-weighting) per construct, in twin units",
        "definition": {
            "twin_unit": "check 1's median distance between the two twins, per construct",
            "check4_distance": "sqrt(2(1 - cosine)) from check 4's stored cosine against the "
                               "base profile",
            "check4_twin_units": "check4_distance / twin_distance, per construct",
            "identity_checked": "the chord distance recomputed from check 1's cosine "
                                "reproduces check 1's stored distance",
            "disattenuated": "stored but NOT reported: clipped at 1.0 for most constructs"},
        "fixed_before_computing": True,
        "n": int(ratios.size),
        "conversion_error_vs_stored_distance": {
            "median_relative": float(np.median(rel)) if rel.size else None,
            "p95_relative": float(np.percentile(rel, 95)) if rel.size else None,
            "max_relative": float(rel.max()) if rel.size else None,
            "note": "the column is DERIVED from the stored cosine, not the stored distance: "
                    "the chord identity holds per pair, and the artefact stores a median of "
                    "cosines beside a median of distances. The 7 Oct suite computed the "
                    "distance directly and reported a median 2.34 twin units; this "
                    "reconstruction gives 2.25, consistent with the error above."},
        "median_twin_units": float(np.median(ratios)),
        "quartiles": [float(np.percentile(ratios, 25)), float(np.percentile(ratios, 75))],
        "max_twin_units": float(ratios.max()),
        "n_within_one_twin_unit": int((ratios <= 1).sum()),
        "median_cosine": float(np.median([r["check4_cosine"] for r in rows])),
        "disattenuated_clipped_at_1": int((dis == 1.0).sum()) if dis.size else None,
        "disattenuated_n": int(dis.size),
        "arm": "designed arm only; the natural-corpus arm was never built",
        "per_construct": rows,
        "minutes": (time.time() - t0) / 60,
        "provenance": provenance(["validation/paper1/LICENCE_TABLE.json"])}
    (OUT / "CHECK4_TWIN_UNITS.json").write_text(json.dumps(doc, indent=1, default=float))

    cols = ["construct", "in_p1", "check4_cosine", "check4_twin_units",
            "within_one_twin_unit", "twin_distance"]
    with open(OUT / "CHECK4_TWIN_UNITS.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows([[r.get(c) for c in cols] for r in rows])

    print(f"  check 4, {ratios.size} constructs, designed arm only")
    print(f"    twin units: median {np.median(ratios):.2f}, "
          f"quartiles {np.percentile(ratios, [25, 75]).round(2)}, max {ratios.max():.1f}")
    print(f"    within one twin unit: {(ratios <= 1).sum()} of {ratios.size}")
    print(f"    (the cosine B.1 currently carries: median "
          f"{np.median([r['check4_cosine'] for r in rows]):.4f})")
    if dis.size:
        print(f"    disattenuated clipped at 1.0 for {(dis == 1.0).sum()} of {dis.size} "
              f"-- not reported")
    print(f"-> {OUT / 'CHECK4_TWIN_UNITS.csv'}, CHECK4_TWIN_UNITS.json")


if __name__ == "__main__":
    main()
