r"""Check 4 per construct in twin units, on the suite's OWN metric -- replacing the chord.

Table B.1's check-4 twin-unit column, as I first built it, was `sqrt(2(1 - cos))` over check
1's twin distance. That made it the only figure in the paper on that metric: check 4's headline
2.34, check 8's floor, the resolution of section 4.2 and check 6's re-weighting are all
`relative_distance` -- `||a - b|| / mean(||a||, ||b||)` -- over the same denominator, which is
itself a median over layers of `relative_distance` between the twins. A chord numerator over a
relative-distance denominator is two metrics in one ratio, and on check 1, where both forms are
stored, they differ by a median 1.5% and by as much as 18.6%.

So this rebuilds the quantity the suite computed and discarded. `nu_sensitivity` takes
`relative_distance(base.mean[mid], other.mean[mid])` and divides by the twin distance;
`LICENCE_TABLE.json` kept only the cosine. Both arms are rebuilt here from their per-core,
per-cell accumulators -- the route the suite itself used for all but the 31 constructs with
per-prompt arrays -- so base and nu arm are built by one path and the ratio is of like with
like.

**The acceptance test is the suite's own median of 2.34 twin units**, reported by the 7 October
run over 188 constructs. If this does not land there, it is not the same quantity and the
column should not be shipped.

    .venv/bin/python experiments/check4_relative.py
"""
from __future__ import annotations

import csv
import json
import time

import numpy as np

from experiments.atlas import D, LAST, provenance
from profilebattery.profiles.profile import from_accumulators
from profilebattery.profiles.yardstick import Yardstick
from profilebattery.validation.metrics import in_twin_units, relative_distance
from profilebattery.paths import dataset_repo

OUT = D / "validation/paper1"
REPO = dataset_repo()


def grab(name: str):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, f"record/{LAST}/{name}", repo_type="dataset")


def main() -> None:
    t0 = time.time()
    lic = json.loads((OUT / "LICENCE_TABLE.json").read_text())["constructs"]
    lic = lic if isinstance(lic, dict) else {r["construct"]: r for r in lic}
    ys = Yardstick.from_npz(grab("twin1__C_cells.npz"))

    rows, skipped = [], []
    for i, (c, r) in enumerate(sorted(lic.items()), 1):
        tw = (r.get("check1_twin_unit") or {}).get("median_distance")
        a4 = (r.get("check4_nu") or [None])[0]
        if not tw or not a4 or a4.get("cosine") is None:
            skipped.append((c, "no check 1 distance or no check 4 arm"))
            continue
        try:
            base = from_accumulators(np.load(grab(f"twin1__{c}_A_acc.npz")), ys, c, "A",
                                     stream="h")
            other = from_accumulators(np.load(grab(f"alt__{c}_A_acc.npz")), ys, c, "A",
                                      stream="h")
        except Exception as e:                       # noqa: BLE001 -- any fetch/build failure
            skipped.append((c, f"{type(e).__name__}: {e}"))
            continue
        mid = base.mean.shape[0] // 2
        d = relative_distance(base.mean[mid], other.mean[mid])
        rows.append({"construct": c, "in_p1": r.get("in_p1"),
                     "check4_cosine": a4["cosine"],
                     "relative_distance": d,
                     "twin_distance": tw,
                     "check4_twin_units": in_twin_units(d, tw),
                     "within_one_twin_unit": bool(d / tw <= 1.0),
                     "n_base": base.n, "n_other": other.n})
        if i % 40 == 0:
            print(f"  {i}/{len(lic)} ... {(time.time()-t0)/60:.1f} min", flush=True)

    # Every construct SKIPs when the accumulators are absent, and the aggregation then
    # indexed an empty array -- an IndexError a reader would have to decode. The
    # release carries the record for one construct, so this says what it needs
    # (clean-environment run, 5 Nov).
    if len(rows) < 2:
        raise SystemExit(
            f"check4_relative: read {len(rows)} of {len(lic)} constructs"
            + (f" ({skipped[0][1]})" if skipped else "")
            + f". This needs the per-core accumulators, record/{LAST}/"
              f"{{twin1,alt}}__<c>_A_acc.npz, which the public dataset does not carry "
              f"-- it holds one worked example. Nothing written.")
    v = np.array([x["check4_twin_units"] for x in rows])
    med = float(np.median(v))
    doc = {"what": "check 4 per construct in twin units, on relative_distance -- the metric "
                   "the suite itself used and did not store",
           "definition": {
               "numerator": "relative_distance(base A profile, nu-arm A profile) at the "
                            "middle layer, as checks.nu_sensitivity computes it",
               "denominator": "check 1's median_distance, itself a median over layers of "
                              "relative_distance between the twins",
               "both arms": "rebuilt from per-core, per-cell accumulators by one path",
               "supersedes": "the chord form sqrt(2(1-cos))/twin, whose median was 2.25 and "
                             "which was the only chord figure in the paper",
               "acceptance": "the 7 October suite's median of 2.34 twin units over 188"},
           "fixed_before_computing": True,
           "reference": LAST, "n": len(rows), "n_skipped": len(skipped),
           "skipped": skipped[:20],
           "median_twin_units": med,
           "quartiles": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
           "min": float(v.min()), "max": float(v.max()),
           "n_within_one_twin_unit": int((v <= 1).sum()),
           "suite_median_7_Oct": 2.34,
           "matches_suite": bool(abs(med - 2.34) <= 0.05),
           "per_construct": rows,
           "minutes": (time.time() - t0) / 60,
           "provenance": provenance([f"record/{LAST}/twin1__*_A_acc.npz",
                                     f"record/{LAST}/alt__*_A_acc.npz",
                                     f"record/{LAST}/twin1__C_cells.npz",
                                     "validation/paper1/LICENCE_TABLE.json"],
                                    reference=LAST)}
    (OUT / "CHECK4_RELATIVE.json").write_text(json.dumps(doc, indent=1, default=float))
    cols = ["construct", "in_p1", "check4_cosine", "check4_twin_units",
            "within_one_twin_unit", "relative_distance", "twin_distance"]
    with open(OUT / "CHECK4_RELATIVE.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows([[x.get(k) for k in cols] for x in rows])

    print(f"\n  {len(rows)} constructs, {len(skipped)} skipped")
    print(f"  median {med:.2f} twin units, quartiles "
          f"{np.percentile(v, [25, 75]).round(2)}, max {v.max():.1f}")
    print(f"  within one twin unit: {(v <= 1).sum()} of {len(v)}")
    print(f"  the suite's 7 Oct median was 2.34 -> "
          f"{'MATCHES' if doc['matches_suite'] else 'DOES NOT MATCH'}")
    print(f"  ({doc['minutes']:.1f} min)")
    print(f"-> {OUT / 'CHECK4_RELATIVE.csv'}, CHECK4_RELATIVE.json")


if __name__ == "__main__":
    main()
