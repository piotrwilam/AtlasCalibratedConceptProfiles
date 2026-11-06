r"""Does the companion asymmetry change which neurons carry the concept, or when they settle?

The asymmetry's size in twin units (`check6_reweight.py`: `For` 2.95, `With` 5.50) says the
profile moves. It does not say whether the *carrier set* moves, and the carrier set is what
every claim about formation and turnover is written in. So this reads the turnover test both
ways: carrier-set overlap between each pair of the four per-prompt references, from the profile
as recorded and from the profile with B re-weighted to A's companion shares, against the twin
ceiling in each condition.

**The quantities.** The carrier set is the top 5% of the product space, 7,373 of 24 x 6,144,
by squared value -- `paper1._carriers`, the same function the paper's own readings use.
Overlap is `|intersection| / K`, the plug-in form `paper1.b3` reports, with the Jaccard beside
it (`|inter| / |union|`; for two sets of equal size the two are monotone, `J = o / (2 - o)`).
The ceiling at a reference is twin 1's carrier set against twin 2's at that same reference:
two batteries of the same declared distribution do not agree perfectly, and a turnover figure
read against 1 rather than against that ceiling cannot be interpreted.

**The weights are the battery's, not the model's.** The battery is fixed across references, so
the maximum-entropy weights solve once per (construct, twin) and are then applied at all four
references. That is what makes the comparison clean: the only thing differing between the two
conditions at a reference is which B prompts carry weight.

**Only `For` and `With` can be read this way,** and the reason is in `check6_reweight.py`: of
the eight constructs Table B.3 reports, four have no per-prompt activations, and `If` and
`While` are provably infeasible to re-weight.

    .venv/bin/python experiments/check6_turnover.py
"""
from __future__ import annotations

import gzip
import itertools
import json
import time

import numpy as np

from experiments.check6_reweight import (shard, solve, standardise, targets, usable,
                                         weighted_mean, yardstick)
from experiments.atlas import D, K, REFS4, _carriers, provenance
from profilebattery.profiles.profile import recorder_order

BAT = D / "phase_battery"
OUT = D / "validation/paper1"
PAIR = ("For", "With")
TWINS = ("twin1", "twin2")


def overlap(a: np.ndarray, b: np.ndarray) -> dict:
    inter = int(np.intersect1d(a, b).size)
    union = a.size + b.size - inter
    return {"intersection": inter, "overlap": inter / K, "jaccard": inter / union}


def main() -> None:
    t0 = time.time()
    inv = json.loads((D / "inventory/inventory_v2.json").read_text())["constructs"]
    doc = {"what": "carrier-set turnover across the four per-prompt references, with and "
                   "without the companion re-weighting, against the twin ceiling",
           "definition": {
               "carrier_set": f"the top {K} of 24 x 6144 by squared value "
                              f"(paper1._carriers), the paper's own 5% cut",
               "overlap": "|intersection| / K, the form paper1.b3 reports",
               "jaccard": "|intersection| / |union|; monotone in the overlap at equal sizes",
               "ceiling": "twin 1's carrier set against twin 2's at the SAME reference -- the "
                          "instrument's own reproducibility, computed separately in each "
                          "condition",
               "ratio": "turnover overlap / the mean of the two references' ceilings; both "
                        "ceilings are stored so another convention can be applied",
               "reweighted": "B re-weighted to A's companion shares by maximum-entropy "
                             "weights, solved once per (construct, twin) on the fixed "
                             "battery and applied at every reference",
               "references": list(REFS4)},
           "fixed_before_computing": True, "per_construct": {}}

    for c in PAIR:
        rows = {k: {t: recorder_order([json.loads(l) for l in
                                       gzip.open(BAT / t / f"{c}_{k}.jsonl.gz", "rt")])
                    for t in TWINS} for k in ("A", "B")}
        entry = {"weights": {}, "per_reference": {}, "pairs": {}}
        keep, w = {}, {}

        # **The prompt set is the intersection across the four references.** Cell support is
        # a property of each reference's own C recording, so a prompt usable at one reference
        # need not be usable at another; taking the intersection is what makes one weight
        # vector, and the carrier sets built from it, comparable across all four.
        yss = {ref: yardstick(ref) for ref in REFS4}
        for t in TWINS:
            keep[t] = {}
            for k in ("A", "B"):
                per = [set(usable(rows[k][t], yss[ref], c)) for ref in REFS4]
                both = set.intersection(*per)
                keep[t][k] = sorted(both)
                if any(len(x) != len(both) for x in per):
                    entry.setdefault("cell_support_differs_by_reference", {})[f"{t}/{k}"] = {
                        "usable_per_reference": {r: len(x) for r, x in zip(REFS4, per)},
                        "intersection": len(both)}

        # **One target list for both twins.** It is selected on the two pooled -- the list
        # part 3 reports -- and then applied to each twin against that twin's own A shares.
        # Selecting it per twin instead makes the two twins carry different interventions,
        # and a twin-against-twin ceiling then measures that difference as well as the noise.
        pooled = {k: [rows[k][t][i] for t in TWINS for i, _ in keep[t][k]]
                  for k in ("A", "B")}
        shared, _, _ = targets(c, pooled, inv)
        entry["target_list"] = {
            "companions": shared,
            "selected_on": "both twins pooled, so each twin carries the same intervention"}
        for t in TWINS:
            rs = {k: [rows[k][t][i] for i, _ in keep[t][k]] for k in ("A", "B")}
            names, tgt, X = targets(c, rs, inv, names=shared)
            w[t], info = solve(X, tgt)
            if w[t] is None:
                raise SystemExit(f"REFUSED: {c}/{t}: the re-weighting is infeasible on this "
                                 f"twin alone ({info})")
            entry["weights"][t] = {"n_targets": len(names), "targets": names,
                                   "ess": info["ess"], "ess_fraction": info["ess_fraction"],
                                   "max_moment_error": info["max_moment_error"]}

        # the profiles
        car = {cond: {} for cond in ("as_recorded", "reweighted")}
        for ref in REFS4:
            ys = yss[ref]
            for t in TWINS:
                a = weighted_mean(standardise(shard(f"{t}__{c}_A.npz", ref), keep[t]["A"], ys),
                                  np.ones(len(keep[t]["A"])))
                zb = standardise(shard(f"{t}__{c}_B.npz", ref), keep[t]["B"], ys)
                b0 = weighted_mean(zb, np.ones(len(keep[t]["B"])))
                b1 = weighted_mean(zb, w[t])
                del zb
                car["as_recorded"][(ref, t)] = _carriers(a - b0)
                car["reweighted"][(ref, t)] = _carriers(a - b1)
            e = {}
            for cond in car:
                e[cond] = {"ceiling": overlap(car[cond][(ref, "twin1")],
                                              car[cond][(ref, "twin2")])}
            for t in TWINS:
                e[f"as_recorded_vs_reweighted_{t}"] = overlap(
                    car["as_recorded"][(ref, t)], car["reweighted"][(ref, t)])
            entry["per_reference"][ref] = e
            print(f"  {c:<5s} {ref:<22s} ceiling {e['as_recorded']['ceiling']['overlap']:.4f} "
                  f"-> {e['reweighted']['ceiling']['overlap']:.4f}   "
                  f"recorded vs re-weighted, twin1 "
                  f"{e['as_recorded_vs_reweighted_twin1']['overlap']:.4f}")

        for r1, r2 in itertools.combinations(REFS4, 2):
            key = f"{r1}|{r2}"
            p = {}
            for cond in car:
                o = overlap(car[cond][(r1, "twin1")], car[cond][(r2, "twin1")])
                c1 = entry["per_reference"][r1][cond]["ceiling"]["overlap"]
                c2 = entry["per_reference"][r2][cond]["ceiling"]["overlap"]
                p[cond] = {**o, "ceiling_at_earlier": c1, "ceiling_at_later": c2,
                           "ratio_to_mean_ceiling": o["overlap"] / ((c1 + c2) / 2)}
                o2 = overlap(car[cond][(r1, "twin2")], car[cond][(r2, "twin2")])
                p[cond]["overlap_twin2"] = o2["overlap"]
            p["change_in_overlap"] = (p["reweighted"]["overlap"]
                                      - p["as_recorded"]["overlap"])
            p["change_in_ratio"] = (p["reweighted"]["ratio_to_mean_ceiling"]
                                    - p["as_recorded"]["ratio_to_mean_ceiling"])
            entry["pairs"][key] = p
        doc["per_construct"][c] = entry

    doc["summary"] = {
        "ceiling_note": "re-weighting LOWERS the twin ceiling, because the weighted profile "
                        "rests on 20-34% of the prompts (the ESS above). The correction is "
                        "therefore not free: it buys companion balance at the price of a "
                        "noisier instrument, and at the untrained reference, where the "
                        "ceiling is already near chance, it costs most.",
        "reading": "if the re-weighted turnover and its ratio to the ceiling track the "
                   "recorded ones, the companion asymmetry moves the profile without moving "
                   "the carrier set, and the turnover claims stand as written; if they "
                   "diverge, the asymmetry reaches the quantity the formation claims rest on",
        "per_construct": {c: {
            "max_abs_change_in_overlap": float(max(
                abs(p["change_in_overlap"]) for p in v["pairs"].values())),
            "max_abs_change_in_ratio": float(max(
                abs(p["change_in_ratio"]) for p in v["pairs"].values())),
            "ceiling_as_recorded": {r: v["per_reference"][r]["as_recorded"]["ceiling"]
                                    ["overlap"] for r in REFS4},
            "ceiling_reweighted": {r: v["per_reference"][r]["reweighted"]["ceiling"]
                                   ["overlap"] for r in REFS4},
            "same_reference_recorded_vs_reweighted": {
                r: v["per_reference"][r]["as_recorded_vs_reweighted_twin1"]["overlap"]
                for r in REFS4},
            # does re-weighting move the carrier set by more than redrawing the battery
            # does? The recorded set and the re-weighted one are compared against the band
            # between the two conditions' own ceilings at that reference: inside the band is
            # within the instrument's noise, below it is more than noise.
            "carrier_move_vs_noise": {
                r: ("within the twin noise"
                    if v["per_reference"][r]["as_recorded_vs_reweighted_twin1"]["overlap"]
                    >= min(v["per_reference"][r]["as_recorded"]["ceiling"]["overlap"],
                           v["per_reference"][r]["reweighted"]["ceiling"]["overlap"])
                    else "beyond the twin noise")
                for r in REFS4},
            "ceiling_cost_of_reweighting": {
                r: (v["per_reference"][r]["reweighted"]["ceiling"]["overlap"]
                    - v["per_reference"][r]["as_recorded"]["ceiling"]["overlap"])
                for r in REFS4}}
            for c, v in doc["per_construct"].items()}}
    doc["minutes"] = (time.time() - t0) / 60
    doc["provenance"] = provenance(
        [f"record/{{{','.join(REFS4)}}}/twin{{1,2}}__{{For,With}}_{{A,B}}.npz",
         "record/*/twin1__C_cells.npz",
         "phase_battery/twin{1,2}/{For,With}_{A,B}.jsonl.gz",
         "inventory/inventory_v2.json"])
    (OUT / "CHECK6_TURNOVER.json").write_text(json.dumps(doc, indent=1, default=float))

    for c, v in doc["per_construct"].items():
        print(f"\n  {c}: carrier overlap between references, twin 1 "
              f"(as recorded -> re-weighted; ratio to the mean ceiling)")
        for key, p in v["pairs"].items():
            r1, r2 = key.split("|")
            print(f"    {r1:<22s} {r2:<22s} "
                  f"{p['as_recorded']['overlap']:.4f} -> {p['reweighted']['overlap']:.4f}   "
                  f"ratio {p['as_recorded']['ratio_to_mean_ceiling']:.3f} -> "
                  f"{p['reweighted']['ratio_to_mean_ceiling']:.3f}")
        s = doc["summary"]["per_construct"][c]
        print(f"    largest change: overlap {s['max_abs_change_in_overlap']:+.4f}, "
              f"ratio {s['max_abs_change_in_ratio']:+.4f}")
    print(f"\n  {doc['minutes']:.1f} min")
    print(f"-> {OUT / 'CHECK6_TURNOVER.json'}")


if __name__ == "__main__":
    main()
